package scheduler

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedulerserverconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const Name = "tf-scheduler"

var _ framework.PreFilterPlugin = &TensorFusionScheduling{}
var _ framework.ReservePlugin = &TensorFusionScheduling{}

type TensorFusionScheduling struct {
	logger    *klog.Logger
	fh        framework.Handle
	podLister cache.Indexer
	pdbLister cache.Indexer
	client    *rest.RESTClient
}

func New(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	lh := klog.FromContext(ctx).WithValues("plugin", Name)
	c := &TensorFusionScheduling{
		logger: &lh,
		fh:     handle,
	}
	return c, nil
}

func (s *TensorFusionScheduling) Run(ctx context.Context, cc *schedulerserverconfig.CompletedConfig, sched *scheduler.Scheduler) {

}

func (s *TensorFusionScheduling) Name() string {
	return Name
}

func (s *TensorFusionScheduling) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	return nil, nil
}

func (s *TensorFusionScheduling) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func (s *TensorFusionScheduling) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	return nil
}

func (s *TensorFusionScheduling) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
}
