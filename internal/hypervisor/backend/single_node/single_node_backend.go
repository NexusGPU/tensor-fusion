package single_node

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
)

type SingleNodeBackend struct {
	ctx              context.Context
	deviceController framework.DeviceController
}

func NewSingleNodeBackend(ctx context.Context, deviceController framework.DeviceController) *SingleNodeBackend {
	return &SingleNodeBackend{ctx: ctx, deviceController: deviceController}
}

func (b *SingleNodeBackend) Start() error {
	return nil
}

func (b *SingleNodeBackend) Stop() error {
	return nil
}

func (b *SingleNodeBackend) ListAndWatchWorkers(ctx context.Context, stopCh <-chan struct{}) ([]string, error) {
	return []string{}, nil
}

func (b *SingleNodeBackend) GetWorkerToProcessMap(ctx context.Context) (map[string][]string, error) {
	return make(map[string][]string), nil
}

func (b *SingleNodeBackend) StartWorker(ctx context.Context, workerUID string) error {
	return nil
}

func (b *SingleNodeBackend) StopWorker(ctx context.Context, workerUID string) error {
	return nil
}

func (b *SingleNodeBackend) ReconcileDevices(ctx context.Context, devices []string) error {
	return nil
}
