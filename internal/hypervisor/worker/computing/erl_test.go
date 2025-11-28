package computing

import (
	"math"
	"sync"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestERL(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ERL Controller Suite")
}

var _ = Describe("DeviceController", func() {
	var (
		backend *MockBackend
		device  int
		cfg     DeviceControllerConfig
	)

	BeforeEach(func() {
		device = 0
		cfg = DefaultDeviceControllerConfig()
		cfg.RateMax = 50000.0
		cfg.CapacityMax = 100_000.0
	})

	Describe("Initialization", func() {
		It("should initialize correctly with valid config", func() {
			backend = NewMockBackend(0.0, 0.0, 0.0)
			cfg.TargetUtilization = 0.7

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())
			Expect(ctrl).NotTo(BeNil())
			Expect(ctrl.cfg.TargetUtilization).To(Equal(0.7))
			Expect(ctrl.currentRate).To(BeNumerically(">=", ctrl.cfg.RateMin))
			Expect(ctrl.currentRate).To(BeNumerically("<=", ctrl.cfg.RateMax))
		})

		It("should reject invalid target_utilization", func() {
			backend = NewMockBackend(0.0, 0.0, 0.0)
			cfg.TargetUtilization = 1.5

			_, err := NewDeviceController(backend, device, cfg)
			Expect(err).To(HaveOccurred())
			Expect(err).To(MatchError(ContainSubstring("target_utilization must be in [0, 1]")))
		})

		It("should reject invalid rate_min/rate_max", func() {
			backend = NewMockBackend(0.0, 0.0, 0.0)
			cfg.RateMin = 100.0
			cfg.RateMax = 50.0

			_, err := NewDeviceController(backend, device, cfg)
			Expect(err).To(HaveOccurred())
			Expect(err).To(MatchError(ContainSubstring("rate_max must be greater than rate_min")))
		})

		It("should reject invalid filter_alpha", func() {
			backend = NewMockBackend(0.0, 0.0, 0.0)
			cfg.FilterAlpha = 1.5

			_, err := NewDeviceController(backend, device, cfg)
			Expect(err).To(HaveOccurred())
			Expect(err).To(MatchError(ContainSubstring("filter_alpha must be in [0, 1]")))
		})

		It("should reject invalid integral_decay_factor", func() {
			backend = NewMockBackend(0.0, 0.0, 0.0)
			cfg.IntegralDecayFactor = 1.5

			_, err := NewDeviceController(backend, device, cfg)
			Expect(err).To(HaveOccurred())
			Expect(err).To(MatchError(ContainSubstring("integral_decay_factor must be in [0, 1]")))
		})
	})

	Describe("Rate Adjustment", func() {
		It("should increase rate when utilization is below target", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.7

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			rateBefore := ctrl.currentRate

			// Utilization 20% when target is 70% -> should increase rate
			_, err = ctrl.Update(0.2, 0.1)
			Expect(err).NotTo(HaveOccurred())

			rateAfter := ctrl.currentRate
			Expect(rateAfter).To(BeNumerically(">", rateBefore), "Rate should increase when utilization is below target")
		})

		It("should decrease rate when utilization is above target", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			// First establish a higher rate
			_, err = ctrl.Update(0.3, 0.1)
			Expect(err).NotTo(HaveOccurred())
			_, err = ctrl.Update(0.3, 0.1)
			Expect(err).NotTo(HaveOccurred())

			rateBefore := ctrl.currentRate

			// Now push utilization above target
			_, err = ctrl.Update(0.95, 0.1)
			Expect(err).NotTo(HaveOccurred())

			rateAfter := ctrl.currentRate
			Expect(rateAfter).To(BeNumerically("<", rateBefore), "Rate should decrease when utilization is above target")
		})

		It("should respect rate limits", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5
			cfg.RateMin = 50.0
			cfg.RateMax = 500.0
			cfg.CapacityMax = 1000.0

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			// Try to push rate very low
			for i := 0; i < 10; i++ {
				_, err = ctrl.Update(0.99, 0.1)
				Expect(err).NotTo(HaveOccurred())
			}
			Expect(ctrl.currentRate).To(BeNumerically(">=", 50.0), "Rate should not go below rate_min")

			// Try to push rate very high
			for i := 0; i < 10; i++ {
				_, err = ctrl.Update(0.01, 0.1)
				Expect(err).NotTo(HaveOccurred())
			}
			Expect(ctrl.currentRate).To(BeNumerically("<=", 500.0), "Rate should not exceed rate_max")
		})
	})

	Describe("Utilization Smoothing", func() {
		It("should smooth utilization measurements", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5
			cfg.FilterAlpha = 0.3

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			// Feed alternating utilization values
			_, err = ctrl.Update(0.8, 0.1)
			Expect(err).NotTo(HaveOccurred())
			_, err = ctrl.Update(0.2, 0.1)
			Expect(err).NotTo(HaveOccurred())

			state := ctrl.State()
			// Smoothed value should be between the extremes
			Expect(state.SmoothedUtilization).To(BeNumerically(">", 0.2))
			Expect(state.SmoothedUtilization).To(BeNumerically("<", 0.8))
		})
	})

	Describe("Edge Cases", func() {
		It("should handle zero utilization", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			// Feed zero utilization repeatedly
			for i := 0; i < 5; i++ {
				_, err = ctrl.Update(0.0, 0.1)
				Expect(err).NotTo(HaveOccurred())
			}

			// Rate should still be above minimum
			Expect(ctrl.currentRate).To(BeNumerically(">=", ctrl.cfg.RateMin), "Rate should never drop below rate_min")
		})

		It("should handle very small delta_time", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			rateBefore := ctrl.currentRate

			// Update with delta_time smaller than min_delta_time
			_, err = ctrl.Update(0.3, 0.001)
			Expect(err).NotTo(HaveOccurred())

			// Rate should not change
			Expect(ctrl.currentRate).To(Equal(rateBefore))
		})
	})

	Describe("Capacity Scaling", func() {
		It("should scale capacity with rate", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			_, err = ctrl.Update(0.2, 0.1)
			Expect(err).NotTo(HaveOccurred())
			state1 := ctrl.State()

			// Continue to increase rate
			for i := 0; i < 5; i++ {
				_, err = ctrl.Update(0.2, 0.1)
				Expect(err).NotTo(HaveOccurred())
			}

			state2 := ctrl.State()
			if state2.CurrentRate > state1.CurrentRate {
				Expect(state2.CurrentCapacity).To(BeNumerically(">=", state1.CurrentCapacity), "Capacity should scale with rate")
			}
		})
	})

	Describe("Timestamp-based Updates", func() {
		It("should handle timestamp-based updates", func() {
			backend = NewMockBackend(1000.0, 100.0, 500.0)
			cfg.TargetUtilization = 0.5

			ctrl, err := NewDeviceController(backend, device, cfg)
			Expect(err).NotTo(HaveOccurred())

			// Update with timestamps (in microseconds)
			t1 := uint64(1_000_000) // 1 second
			t2 := uint64(1_200_000) // 1.2 seconds (0.2s delta)

			_, err = ctrl.UpdateWithTimestamp(0.3, t1)
			Expect(err).NotTo(HaveOccurred())

			_, err = ctrl.UpdateWithTimestamp(0.4, t2)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

// MockBackend is a mock implementation of DeviceBackend for testing
type MockBackend struct {
	mu              sync.RWMutex
	quotaCapacity   float64
	quotaRefillRate float64
	tokens          float64
	lastUpdate      float64
}

func NewMockBackend(capacity, refillRate, tokens float64) *MockBackend {
	return &MockBackend{
		quotaCapacity:   capacity,
		quotaRefillRate: refillRate,
		tokens:          tokens,
		lastUpdate:      0,
	}
}

func (m *MockBackend) ReadTokenState(device int) (*TokenState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return &TokenState{
		Tokens:     m.tokens,
		LastUpdate: m.lastUpdate,
	}, nil
}

func (m *MockBackend) WriteTokenState(device int, state *TokenState) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokens = state.Tokens
	m.lastUpdate = state.LastUpdate
	return nil
}

func (m *MockBackend) ReadQuota(device int) (*DeviceQuota, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return &DeviceQuota{
		Capacity:   m.quotaCapacity,
		RefillRate: m.quotaRefillRate,
	}, nil
}

func (m *MockBackend) WriteRefillRate(device int, refillRate float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.quotaRefillRate = refillRate
	return nil
}

func (m *MockBackend) WriteCapacity(device int, capacity float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.quotaCapacity = capacity
	return nil
}

func (m *MockBackend) FetchSubTokens(device int, cost float64) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	current := m.tokens
	if current < cost {
		return current, nil
	}

	capacity := m.quotaCapacity
	newTokens := math.Max(0.0, math.Min(capacity, current-cost))
	m.tokens = newTokens
	return current, nil
}

func (m *MockBackend) FetchAddTokens(device int, amount float64) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	current := m.tokens
	capacity := m.quotaCapacity
	newTokens := math.Max(0.0, math.Min(capacity, current+amount))
	m.tokens = newTokens
	return current, nil
}
