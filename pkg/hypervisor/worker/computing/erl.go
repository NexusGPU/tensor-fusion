package computing

import (
	"errors"
	"fmt"
	"math"
)

var (
	ErrInvalidConfig = errors.New("invalid configuration")
)

// DeviceBackend defines the interface for device token/quota operations
type DeviceBackend interface {
	ReadTokenState(device int) (*TokenState, error)
	WriteTokenState(device int, state *TokenState) error
	ReadQuota(device int) (*DeviceQuota, error)
	WriteRefillRate(device int, refillRate float64) error
	WriteCapacity(device int, capacity float64) error
	FetchSubTokens(device int, cost float64) (float64, error)
	FetchAddTokens(device int, amount float64) (float64, error)
}

// TokenState represents the current token bucket state
type TokenState struct {
	Tokens     float64
	LastUpdate float64
}

// DeviceQuota represents device quota configuration
type DeviceQuota struct {
	Capacity   float64
	RefillRate float64
}

// DeviceControllerConfig holds configuration for the PID-based device controller
type DeviceControllerConfig struct {
	// Target GPU utilization (0.0 to 1.0, e.g., 0.5 = 50%)
	TargetUtilization float64

	// Minimum refill rate (tokens/second) - prevents rate from dropping to zero
	RateMin float64

	// Maximum refill rate (tokens/second)
	RateMax float64

	// PID proportional gain - how aggressively to respond to error
	Kp float64

	// PID integral gain - how quickly to eliminate steady-state error
	Ki float64

	// PID derivative gain - how much to dampen oscillations
	Kd float64

	// Low-pass filter coefficient for smoothing utilization (0.0 to 1.0)
	// Higher values = less filtering (more responsive, more noise)
	FilterAlpha float64

	// Burst window in seconds - capacity = refill_rate × burst_window
	BurstWindow float64

	// Minimum capacity (tokens)
	CapacityMin float64

	// Maximum capacity (tokens) - prevents unbounded growth
	CapacityMax float64

	// Minimum time between updates (seconds)
	MinDeltaTime float64

	// Integral decay factor (0.0 to 1.0) for exponential decay of integral term
	// Higher values (closer to 1.0) = slower decay, retains more history
	// Lower values = faster decay, responds more quickly to changes
	// Default 0.95 means ~20 update cycles for integral to decay to ~35.8% of original value
	IntegralDecayFactor float64
}

// DefaultDeviceControllerConfig returns a default configuration
func DefaultDeviceControllerConfig() DeviceControllerConfig {
	return DeviceControllerConfig{
		TargetUtilization:   0.5,
		RateMin:             10.0,
		RateMax:             100_000.0,
		Kp:                  0.5,
		Ki:                  0.1,
		Kd:                  0.05,
		FilterAlpha:         0.3,
		BurstWindow:         2.0,
		CapacityMin:         100.0,
		CapacityMax:         200_000.0,
		MinDeltaTime:        0.05,
		IntegralDecayFactor: 0.95,
	}
}

// DeviceControllerState is a snapshot of controller state after an update
type DeviceControllerState struct {
	TargetUtilization   float64
	SmoothedUtilization float64
	CurrentRate         float64
	CurrentCapacity     float64
	TokenDrainRate      float64
}

// DeviceController is a PID-based controller that dynamically adjusts token refill rates
type DeviceController struct {
	backend DeviceBackend
	device  int
	cfg     DeviceControllerConfig

	// PID state
	integral  float64
	lastError float64

	// Filtering state
	smoothedUtil *float64

	// Rate tracking
	currentRate float64

	// Drain rate estimation
	lastTokenLevel float64
	lastTimestamp  *float64
}

// NewDeviceController creates a new device controller
func NewDeviceController(backend DeviceBackend, device int, cfg DeviceControllerConfig) (*DeviceController, error) {
	// Validate configuration
	if cfg.TargetUtilization < 0.0 || cfg.TargetUtilization > 1.0 {
		return nil, fmt.Errorf("%w: target_utilization must be in [0, 1]", ErrInvalidConfig)
	}
	if cfg.RateMin <= 0.0 || cfg.RateMax <= cfg.RateMin {
		return nil, fmt.Errorf("%w: rate_max must be greater than rate_min > 0", ErrInvalidConfig)
	}
	if cfg.FilterAlpha < 0.0 || cfg.FilterAlpha > 1.0 {
		return nil, fmt.Errorf("%w: filter_alpha must be in [0, 1]", ErrInvalidConfig)
	}
	if cfg.IntegralDecayFactor < 0.0 || cfg.IntegralDecayFactor > 1.0 {
		return nil, fmt.Errorf("%w: integral_decay_factor must be in [0, 1]", ErrInvalidConfig)
	}

	// Initialize with a conservative starting rate
	startRate := math.Min(100.0, cfg.RateMax)
	startRate = math.Max(startRate, cfg.RateMin)
	initialCapacity := math.Max(cfg.CapacityMin, math.Min(cfg.CapacityMax, startRate*cfg.BurstWindow))

	// Initialize backend
	if err := backend.WriteCapacity(device, initialCapacity); err != nil {
		return nil, err
	}
	if err := backend.WriteRefillRate(device, startRate); err != nil {
		return nil, err
	}

	tokenState, err := backend.ReadTokenState(device)
	if err != nil {
		return nil, err
	}
	tokenState.Tokens = initialCapacity
	if err := backend.WriteTokenState(device, tokenState); err != nil {
		return nil, err
	}

	return &DeviceController{
		backend:        backend,
		device:         device,
		cfg:            cfg,
		integral:       0.0,
		lastError:      0.0,
		smoothedUtil:   nil,
		currentRate:    startRate,
		lastTokenLevel: initialCapacity,
		lastTimestamp:  nil,
	}, nil
}

// State returns the current controller state
func (dc *DeviceController) State() DeviceControllerState {
	capacity := math.Max(dc.cfg.CapacityMin, math.Min(dc.cfg.CapacityMax, dc.currentRate*dc.cfg.BurstWindow))
	smoothedUtil := 0.0
	if dc.smoothedUtil != nil {
		smoothedUtil = *dc.smoothedUtil
	}
	return DeviceControllerState{
		TargetUtilization:   dc.cfg.TargetUtilization,
		SmoothedUtilization: smoothedUtil,
		CurrentRate:         dc.currentRate,
		CurrentCapacity:     capacity,
		TokenDrainRate:      0.0, // Will be updated during next cycle
	}
}

// Update updates controller with new utilization measurement and explicit delta time
func (dc *DeviceController) Update(utilization float64, deltaTime float64) (*DeviceControllerState, error) {
	if deltaTime < dc.cfg.MinDeltaTime {
		state := dc.State()
		return &state, nil
	}
	return dc.updateInternal(utilization, deltaTime)
}

// UpdateWithTimestamp updates controller with timestamp (calculates delta automatically)
func (dc *DeviceController) UpdateWithTimestamp(
	utilization float64,
	timestampMicros uint64,
) (*DeviceControllerState, error) {
	seconds := float64(timestampMicros) / 1_000_000.0
	var delta float64
	if dc.lastTimestamp != nil {
		rawDelta := seconds - *dc.lastTimestamp
		if rawDelta < dc.cfg.MinDeltaTime {
			state := dc.State()
			return &state, nil
		}
		delta = rawDelta
	} else {
		delta = dc.cfg.MinDeltaTime
	}
	dc.lastTimestamp = &seconds
	return dc.updateInternal(utilization, delta)
}

// updateInternal performs the core update logic
func (dc *DeviceController) updateInternal(measuredUtil float64, deltaTime float64) (*DeviceControllerState, error) {
	// Clamp measured utilization
	measured := math.Max(0.0, math.Min(1.0, measuredUtil))

	// Step 1: Low-pass filter to smooth NVML noise
	smoothed := dc.smoothUtilization(measured)

	// Step 2: Estimate token drain rate
	drainRate, err := dc.estimateDrainRate(deltaTime)
	if err != nil {
		return nil, err
	}

	// Step 3: Calculate base rate from drain rate and target
	baseRate := dc.calculateBaseRate(smoothed, drainRate)

	// Step 4: Compute PID correction
	error := dc.cfg.TargetUtilization - smoothed
	correction := dc.computePIDCorrection(error, deltaTime)

	// Step 5: Apply correction to base rate
	newRate := math.Max(dc.cfg.RateMin, math.Min(dc.cfg.RateMax, baseRate*(1.0+correction)))
	dc.currentRate = newRate

	// Step 6: Calculate capacity (bounded)
	newCapacity := math.Max(dc.cfg.CapacityMin, math.Min(dc.cfg.CapacityMax, newRate*dc.cfg.BurstWindow))

	// Step 7: Refill tokens
	refillAmount := newRate * deltaTime
	if _, err := dc.backend.FetchAddTokens(dc.device, refillAmount); err != nil {
		return nil, err
	}

	// Step 8: Update backend (capacity must be updated before clamping)
	if err := dc.backend.WriteRefillRate(dc.device, newRate); err != nil {
		return nil, err
	}
	if err := dc.backend.WriteCapacity(dc.device, newCapacity); err != nil {
		return nil, err
	}

	// Step 9: Clamp tokens to capacity (after capacity update, tokens may exceed new capacity)
	// Optimization: only read and write if clamping is needed
	state, err := dc.backend.ReadTokenState(dc.device)
	if err != nil {
		return nil, err
	}
	if state.Tokens > newCapacity {
		state.Tokens = newCapacity
		if err := dc.backend.WriteTokenState(dc.device, state); err != nil {
			return nil, err
		}
	}

	return &DeviceControllerState{
		TargetUtilization:   dc.cfg.TargetUtilization,
		SmoothedUtilization: smoothed,
		CurrentRate:         newRate,
		CurrentCapacity:     newCapacity,
		TokenDrainRate:      drainRate,
	}, nil
}

// smoothUtilization applies exponential moving average to smooth utilization measurements
func (dc *DeviceController) smoothUtilization(measured float64) float64 {
	alpha := dc.cfg.FilterAlpha
	var smoothed float64
	if dc.smoothedUtil != nil {
		smoothed = alpha*measured + (1.0-alpha)**dc.smoothedUtil
	} else {
		smoothed = measured
	}
	dc.smoothedUtil = &smoothed
	return smoothed
}

// estimateDrainRate estimates token drain rate from bucket level changes
func (dc *DeviceController) estimateDrainRate(deltaTime float64) (float64, error) {
	currentState, err := dc.backend.ReadTokenState(dc.device)
	if err != nil {
		return 0, err
	}
	currentTokens := currentState.Tokens

	// Expected tokens = last level + refill during delta_time
	expectedTokens := dc.lastTokenLevel + dc.currentRate*deltaTime

	// Actual drain = expected - actual
	drainRate := math.Max(0.0, (expectedTokens-currentTokens)/deltaTime)

	dc.lastTokenLevel = currentTokens
	return drainRate, nil
}

// calculateBaseRate calculates base refill rate from current utilization and drain rate
// The idea: if we're at `actual_util` with `drain_rate`, then to reach
// `target_util` we need: `base_rate = drain_rate × (target / actual)`
func (dc *DeviceController) calculateBaseRate(smoothedUtil float64, drainRate float64) float64 {
	if smoothedUtil > 0.01 {
		// Theoretical base rate to reach target
		theoretical := drainRate * (dc.cfg.TargetUtilization / smoothedUtil)
		return math.Max(dc.cfg.RateMin, math.Min(dc.cfg.RateMax, theoretical))
	}
	// Very low utilization - maintain current rate or use minimum
	return math.Max(dc.currentRate, dc.cfg.RateMin)
}

// computePIDCorrection computes PID correction term
// Returns a correction factor in the range [-0.5, 0.5] to apply to base_rate
func (dc *DeviceController) computePIDCorrection(error float64, deltaTime float64) float64 {
	// Proportional term
	p := dc.cfg.Kp * error

	// Integral term with exponential decay and anti-windup
	// Apply decay factor to forget old errors gradually
	dc.integral *= dc.cfg.IntegralDecayFactor
	// Add new error contribution
	dc.integral += error * deltaTime
	// Clamp to prevent windup
	dc.integral = math.Max(-1.0, math.Min(1.0, dc.integral))
	i := dc.cfg.Ki * dc.integral

	// Derivative term
	derivative := (error - dc.lastError) / deltaTime
	d := dc.cfg.Kd * derivative

	dc.lastError = error

	// Total correction, clamped to avoid over-reaction
	return math.Max(-0.5, math.Min(0.5, p+i+d))
}
