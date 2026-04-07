package computing

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"k8s.io/klog/v2"
)

const (
	erlUpdateInterval = 500 * time.Millisecond

	// Rate estimation parameters (BBR-inspired)
	burstWindow = 0.5      // capacity = rate × burstWindow (seconds)
	rateMin     = 10.0     // minimum refill rate (tokens/s)
	rateMax     = 200000.0 // maximum refill rate
	capacityMin = 200.0    // minimum token capacity
	capacityMax = 200000.0

	// Smoothing: EMA filter to reduce NVML noise and prevent oscillation
	utilAlpha = 0.3 // utilization smoothing (lower = smoother)
	rateAlpha = 0.2 // rate adjustment smoothing (lower = more stable)
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
	initialized  bool
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
	}
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

	c.running = true
	c.stopCh = make(chan struct{})

	go c.runERLUpdateLoop()
	klog.Info("Soft quota limiter started (AIMD congestion control)")
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
			currentRate:  100.0, // Start low — forces throttling so NVML shows real util
			smoothedUtil: 0,
			initialized:  false,
		}
		c.erlStates[key] = st
	}
	return st
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
				es.smoothedUtil = utilAlpha*nvmlUtil + (1-utilAlpha)*es.smoothedUtil
			}

			// Rate control via NVML utilization feedback.
			// Token rate controls how many kernels/s the workload can issue.
			// More rate → more kernels → higher GPU util.
			// Less rate → fewer kernels → lower GPU util.
			//
			// idealRate = currentRate × (targetUtil / actualUtil)
			// If actual=100% and target=70%: rate reduces to 70% of current.
			// If actual=50% and target=70%: rate increases to 140% of current.
			if es.smoothedUtil > 0.05 {
				idealRate := es.currentRate * (targetUtil / es.smoothedUtil)
				es.currentRate = rateAlpha*idealRate + (1-rateAlpha)*es.currentRate
			} else {
				// Near-idle or no measurement: double rate quickly
				es.currentRate = math.Min(es.currentRate*2.0, rateMax)
			}

			// Clamp rate
			es.currentRate = math.Max(es.currentRate, rateMin)
			es.currentRate = math.Min(es.currentRate, rateMax)

			// Dynamic capacity
			newCapacity := math.Max(es.currentRate*burstWindow, capacityMin)
			newCapacity = math.Min(newCapacity, capacityMax)

			// Write to shm
			deviceInfo.SetERLTokenRefillRate(es.currentRate)
			deviceInfo.SetERLTokenCapacity(newCapacity)

			// Refill tokens
			_, lastUpdate := deviceInfo.LoadERLTokenState()
			if lastUpdate > 0 {
				refillElapsed := nowSecs - lastUpdate
				if refillElapsed > 0 && refillElapsed < 5.0 {
					deviceInfo.FetchAddERLTokens(es.currentRate * refillElapsed)
				}
			}
			deviceInfo.SetERLLastTokenUpdate(nowSecs)

			// Clamp tokens to capacity
			currentTokens, _ := deviceInfo.LoadERLTokenState()
			if currentTokens > newCapacity {
				deviceInfo.SetERLCurrentTokens(newCapacity)
			}

			klog.Infof("ERL [%s] nvml=%.0f%% smooth=%.0f%% target=%.0f%% rate=%.0f cap=%.0f tokens=%.1f",
				stateKey, nvmlUtil*100, es.smoothedUtil*100, targetUtil*100, es.currentRate, newCapacity, currentTokens)
		}
	}
}
