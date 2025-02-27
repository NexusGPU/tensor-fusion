package scheduler

import (
	"fmt"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
)

// NaiveScheduler implements a simple scheduling strategy
type NaiveScheduler struct {
	sync.RWMutex
	gpus map[string]*tfv1.GPU
}

// NewNaiveScheduler creates a new NaiveScheduler
func NewNaiveScheduler() *NaiveScheduler {
	return &NaiveScheduler{
		gpus: make(map[string]*tfv1.GPU),
	}
}

// Schedule implements Scheduler interface
func (s *NaiveScheduler) Schedule(request tfv1.Resource) (*tfv1.GPU, error) {
	s.Lock()
	defer s.Unlock()

	// Simple strategy: return the first gpu that has enough resources
	for _, gpu := range s.gpus {
		if gpu.Status.Available.Tflops.Cmp(request.Tflops) >= 0 &&
			gpu.Status.Available.Vram.Cmp(request.Vram) >= 0 {
			// Update the gpu's available resources
			gpu.Status.Available.Tflops.Sub(request.Tflops)
			gpu.Status.Available.Vram.Sub(request.Vram)
			return gpu, nil
		}
	}

	// TODO: requirement can not be fulfilled by single GPU, group scheduling

	return nil, fmt.Errorf("no suitable gpu found for request: %v", request)
}

// OnAdd implements Scheduler interface
func (s *NaiveScheduler) OnAdd(gpu *tfv1.GPU) {
	s.Lock()
	defer s.Unlock()
	s.gpus[gpu.Name] = gpu
}

// OnUpdate implements Scheduler interface
func (s *NaiveScheduler) OnUpdate(oldgpu, newgpu *tfv1.GPU) {
	s.Lock()
	defer s.Unlock()
	s.gpus[newgpu.Name] = newgpu
}

// OnDelete implements Scheduler interface
func (s *NaiveScheduler) OnDelete(gpu *tfv1.GPU) {
	s.Lock()
	defer s.Unlock()
	delete(s.gpus, gpu.Name)
}

// Release implements Scheduler interface
func (s *NaiveScheduler) Release(request tfv1.Resource, gpu *tfv1.GPU) error {
	s.Lock()
	defer s.Unlock()

	existinggpu, ok := s.gpus[gpu.Name]
	if !ok {
		return fmt.Errorf("gpu %s not found", gpu.Name)
	}

	// Add back the released resources
	existinggpu.Status.Available.Tflops.Add(request.Tflops)
	existinggpu.Status.Available.Vram.Add(request.Vram)
	// output the updated gpu
	gpu.Status.Available = existinggpu.Status.Available
	return nil
}
