package computing

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"k8s.io/klog/v2"
)

const (
	erlUpdateInterval = 500 * time.Millisecond

	// Token bucket defaults.
	burstWindow = 0.5      // capacity = rate × burstWindow (seconds)
	rateMin     = 10.0     // minimum refill rate (tokens/s)
	rateMax     = 200000.0 // maximum refill rate
	capacityMin = 200.0    // minimum token capacity
	capacityMax = 200000.0

	// Feedback-control parameters.
	utilAlpha           = 0.25 // utilization smoothing (EMA)
	utilDeadband        = 0.03 // ignore tiny error to reduce oscillation
	kp                  = 0.9
	ki                  = 0.35
	kd                  = 0.10
	integralDecayFactor = 0.85
	integralClamp       = 1.5

	// Rate slew limiting avoids abrupt oscillation when NVML samples jump.
	maxRateIncreaseRatio = 0.35 // max +35% per control interval
	maxRateDecreaseRatio = 0.25 // max -25% per control interval

	// Keep a modest token reserve for bursts while smoothing bucket release.
	tokenReserveRatio = 0.35
	tokenDrainRatio   = 0.80 // drain up to 80% of capacity/sec when oversupplied
	tokenDrainMin     = 25.0
)

// WorkerInfoSnapshot holds the worker info needed for ERL updates
type WorkerInfoSnapshot struct {
	Namespace  string
	WorkerName string
	Devices    []DeviceSnapshot
}

// DeviceSnapshot holds per-device info for a worker
type DeviceSnapshot struct {
	DeviceUUID string
	DeviceIdx  int
	UpLimit    uint32
}

// erlState tracks per-device rate control state
type erlState struct {
	currentRate  float64
	smoothedUtil float64
	integralErr  float64
	lastError    float64
	initialized  bool
}

type erlConfig struct {
	burstWindow         float64
	rateMin             float64
	rateMax             float64
	capacityMin         float64
	capacityMax         float64
	utilAlpha           float64
	kp                  float64
	ki                  float64
	kd                  float64
	integralDecayFactor float64
}

type Controller struct {
	deviceController framework.DeviceController
	backend          framework.Backend

	mu      sync.RWMutex
	running bool
	stopCh  chan struct{}

	workerInfoFn func() map[string]*WorkerInfoSnapshot
	shmHandleFn  func(workerUID string) *workerstate.SharedMemoryHandle

	// Per-worker per-device AIMD state: key = workerUID + ":" + deviceUUID
	erlStates map[string]*erlState
	config    erlConfig
}

func NewQuotaController(
	deviceController framework.DeviceController,
	backend framework.Backend,
) framework.QuotaController {
	return &Controller{
		deviceController: deviceController,
		backend:          backend,
		stopCh:           make(chan struct{}),
		erlStates:        make(map[string]*erlState),
		config:           defaultERLConfig(),
	}
}

func defaultERLConfig() erlConfig {
	return erlConfig{
		burstWindow:         burstWindow,
		rateMin:             rateMin,
		rateMax:             rateMax,
		capacityMin:         capacityMin,
		capacityMax:         capacityMax,
		utilAlpha:           utilAlpha,
		kp:                  kp,
		ki:                  ki,
		kd:                  kd,
		integralDecayFactor: integralDecayFactor,
	}
}

func parsePositiveFloat(raw string, fallback float64) float64 {
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil || v <= 0 {
		return fallback
	}
	return v
}

func loadERLConfigFromEnv() erlConfig {
	cfg := defaultERLConfig()

	raw := os.Getenv(constants.HypervisorSchedulingConfigEnv)
	if raw == "" {
		return cfg
	}

	var scheduling tfv1.HypervisorScheduling
	if err := json.Unmarshal([]byte(raw), &scheduling); err != nil {
		klog.Warningf("Failed to parse %s: %v", constants.HypervisorSchedulingConfigEnv, err)
		return cfg
	}

	erl := scheduling.ElasticRateLimitParameters
	cfg.rateMax = parsePositiveFloat(erl.MaxRefillRate, cfg.rateMax)
	cfg.rateMin = parsePositiveFloat(erl.MinRefillRate, cfg.rateMin)
	cfg.utilAlpha = parsePositiveFloat(erl.FilterAlpha, cfg.utilAlpha)
	cfg.kp = parsePositiveFloat(erl.Kp, cfg.kp)
	cfg.ki = parsePositiveFloat(erl.Ki, cfg.ki)
	cfg.kd = parsePositiveFloat(erl.Kd, cfg.kd)
	cfg.burstWindow = parsePositiveFloat(erl.BurstWindow, cfg.burstWindow)
	cfg.capacityMin = parsePositiveFloat(erl.CapacityMin, cfg.capacityMin)
	cfg.capacityMax = parsePositiveFloat(erl.CapacityMax, cfg.capacityMax)
	cfg.integralDecayFactor = parsePositiveFloat(erl.IntegralDecayFactor, cfg.integralDecayFactor)

	if cfg.rateMin > cfg.rateMax {
		cfg.rateMin = cfg.rateMax
	}
	if cfg.capacityMin > cfg.capacityMax {
		cfg.capacityMin = cfg.capacityMax
	}
	cfg.utilAlpha = clampFloat64(cfg.utilAlpha, 0.01, 0.95)
	cfg.integralDecayFactor = clampFloat64(cfg.integralDecayFactor, 0.01, 0.999)
	return cfg
}

func (c *Controller) SetWorkerInfoProvider(fn func() map[string]*WorkerInfoSnapshot) {
	c.workerInfoFn = fn
}

func (c *Controller) SetShmHandleProvider(fn func(workerUID string) *workerstate.SharedMemoryHandle) {
	c.shmHandleFn = fn
}

func (c *Controller) SetQuota(workerUID string) error {
	return nil
}

func (c *Controller) StartSoftQuotaLimiter() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.running {
		return nil
	}

	vendor := c.deviceController.GetAcceleratorVendor()
	if !strings.EqualFold(vendor, "NVIDIA") {
		return fmt.Errorf("soft isolation mode is only supported on NVIDIA GPUs, current vendor: %s", vendor)
	}
	c.config = loadERLConfigFromEnv()

	c.running = true
	c.stopCh = make(chan struct{})

	go c.runERLUpdateLoop()
	klog.Infof(
		"Soft quota limiter started (feedback-controlled ERL): "+
			"rate=[%.2f, %.2f] burstWindow=%.2f cap=[%.2f, %.2f] "+
			"alpha=%.2f kp=%.2f ki=%.2f kd=%.2f decay=%.3f",
		c.config.rateMin, c.config.rateMax,
		c.config.burstWindow,
		c.config.capacityMin, c.config.capacityMax,
		c.config.utilAlpha,
		c.config.kp, c.config.ki, c.config.kd,
		c.config.integralDecayFactor,
	)
	return nil
}

func (c *Controller) StopSoftQuotaLimiter() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.running {
		return nil
	}
	close(c.stopCh)
	c.running = false
	klog.Info("Soft quota limiter stopped")
	return nil
}

func (c *Controller) GetWorkerQuotaStatus(workerUID string) error {
	return nil
}

func (c *Controller) runERLUpdateLoop() {
	ticker := time.NewTicker(erlUpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.updateERLControllers()
		}
	}
}

func (c *Controller) getOrCreateERLState(key string) *erlState {
	st, ok := c.erlStates[key]
	if !ok {
		st = &erlState{
			currentRate:  100.0, // Start low to avoid immediate overshoot.
			smoothedUtil: 0,
			integralErr:  0,
			lastError:    0,
			initialized:  false,
		}
		c.erlStates[key] = st
	}
	return st
}

// TODO(known-issue): erlStates map entries are added per (workerUID,deviceUUID)
// on first ERL tick but never deleted when the worker pod is removed. Across
// pod churn the map grows and — more importantly for enforcement accuracy — the
// PID state (currentRate/integralErr/smoothedUtil) on an un-GC'd device entry
// leaks forward and biases the next pod scheduled on the same device. This
// manifests as the measured 60% limit drifting to ~74% time-average after a
// few pods have come and gone on the same GPU; restarting the hypervisor pod
// restores strict 60% enforcement (see commit message for the investigation).
// Fix direction: wire a worker-delete callback (from WorkerController /
// /process-delete handler) that calls `delete(c.erlStates, key)` and also
// resets the per-device shm bucket state.

func clampFloat64(v, lo, hi float64) float64 {
	return math.Min(math.Max(v, lo), hi)
}

func slewRate(current, target, upRatio, downRatio float64) float64 {
	if target > current {
		return math.Min(target, current*(1.0+upRatio))
	}
	return math.Max(target, current*(1.0-downRatio))
}

func computeDesiredRate(currentRate, targetUtil, smoothedUtil, dt float64, es *erlState, cfg erlConfig) float64 {
	if smoothedUtil <= 0.01 {
		// Near idle or no sample: ramp up conservatively but steadily.
		return math.Min(currentRate*(1.0+maxRateIncreaseRatio), cfg.rateMax)
	}

	error := targetUtil - smoothedUtil
	if math.Abs(error) < utilDeadband {
		es.integralErr *= cfg.integralDecayFactor
		return currentRate
	}

	es.integralErr = clampFloat64(es.integralErr*cfg.integralDecayFactor+error*dt, -integralClamp, integralClamp)
	derivative := 0.0
	if dt > 0 {
		derivative = (error - es.lastError) / dt
	}
	es.lastError = error

	// Feed-forward: estimate the rate required to move current utilization to target.
	feedForwardRate := currentRate * (targetUtil / math.Max(smoothedUtil, 0.05))
	controlFactor := 1.0 + cfg.kp*error + cfg.ki*es.integralErr + cfg.kd*derivative
	controlFactor = clampFloat64(controlFactor, 0.5, 1.5)

	desiredRate := clampFloat64(feedForwardRate*controlFactor, cfg.rateMin, cfg.rateMax)
	return slewRate(currentRate, desiredRate, maxRateIncreaseRatio, maxRateDecreaseRatio)
}

func rebalanceTokenBucket(
	deviceInfo *workerstate.SharedDeviceInfoV2,
	nowSecs, refillRate, capacity, targetUtil, smoothedUtil float64,
) float64 {
	currentTokens, lastUpdate := deviceInfo.LoadERLTokenState()
	if lastUpdate > 0 {
		refillElapsed := nowSecs - lastUpdate
		if refillElapsed > 0 && refillElapsed < 5.0 {
			deviceInfo.FetchAddERLTokens(refillRate * refillElapsed)
			currentTokens = deviceInfo.GetERLCurrentTokens()
		}
	}

	reserveTarget := clampFloat64(capacity*tokenReserveRatio, 0.0, capacity)
	if currentTokens > capacity {
		drainAmount := math.Max(tokenDrainMin, capacity*tokenDrainRatio) * (erlUpdateInterval.Seconds())
		currentTokens = math.Max(capacity, currentTokens-drainAmount)
		deviceInfo.SetERLCurrentTokens(currentTokens)
	} else if smoothedUtil > targetUtil+utilDeadband && currentTokens > reserveTarget {
		// Smoothly bleed extra burst budget when utilization is already above target.
		drainAmount := math.Max(tokenDrainMin, capacity*tokenDrainRatio) * (erlUpdateInterval.Seconds())
		currentTokens = math.Max(reserveTarget, currentTokens-drainAmount)
		deviceInfo.SetERLCurrentTokens(currentTokens)
	}

	deviceInfo.SetERLLastTokenUpdate(nowSecs)
	return currentTokens
}

func (c *Controller) updateERLControllers() {
	if c.workerInfoFn == nil || c.shmHandleFn == nil {
		return
	}

	workers := c.workerInfoFn()
	if len(workers) == 0 {
		return
	}

	// Get device-level GPU utilization via NVML
	deviceUtilization := make(map[string]float64)
	deviceMetrics, metricsErr := c.deviceController.GetDeviceMetrics()
	if metricsErr == nil {
		for uuid, m := range deviceMetrics {
			deviceUtilization[strings.ToLower(uuid)] = m.ComputePercentage
		}
	}

	timestampMicros := uint64(time.Now().UnixMicro())
	nowSecs := float64(timestampMicros) / 1e6
	dt := erlUpdateInterval.Seconds()

	for workerUID, workerInfo := range workers {
		handle := c.shmHandleFn(workerUID)
		if handle == nil {
			continue
		}
		state := handle.GetState()
		if state == nil {
			continue
		}

		for _, dev := range workerInfo.Devices {
			deviceUUID := strings.ToLower(dev.DeviceUUID)

			if state.V2 == nil || !state.HasDevice(dev.DeviceIdx) {
				continue
			}
			deviceInfo := &state.V2.Devices[dev.DeviceIdx].DeviceInfo

			stateKey := workerUID + ":" + deviceUUID
			es := c.getOrCreateERLState(stateKey)

			targetUtil := float64(dev.UpLimit) / 100.0

			// Get NVML GPU utilization (0-100 → 0-1)
			nvmlUtil := 0.0
			if u, ok := deviceUtilization[deviceUUID]; ok {
				nvmlUtil = u / 100.0
			}

			// EMA smooth utilization
			if !es.initialized {
				es.smoothedUtil = nvmlUtil
				es.initialized = true
			} else {
				es.smoothedUtil = c.config.utilAlpha*nvmlUtil + (1-c.config.utilAlpha)*es.smoothedUtil
			}

			es.currentRate = computeDesiredRate(es.currentRate, targetUtil, es.smoothedUtil, dt, es, c.config)

			// Dynamic capacity
			newCapacity := clampFloat64(es.currentRate*c.config.burstWindow, c.config.capacityMin, c.config.capacityMax)

			// Write to shm
			deviceInfo.SetERLTokenRefillRate(es.currentRate)
			deviceInfo.SetERLTokenCapacity(newCapacity)
			currentTokens := rebalanceTokenBucket(deviceInfo, nowSecs, es.currentRate, newCapacity, targetUtil, es.smoothedUtil)

			klog.Infof(
				"ERL [%s] nvml=%.0f%% smooth=%.0f%% target=%.0f%% err=%.1f%% "+
					"rate=%.0f cap=%.0f tokens=%.1f int=%.3f",
				stateKey,
				nvmlUtil*100, es.smoothedUtil*100, targetUtil*100,
				(targetUtil-es.smoothedUtil)*100,
				es.currentRate, newCapacity, currentTokens, es.integralErr,
			)
		}
	}
}
