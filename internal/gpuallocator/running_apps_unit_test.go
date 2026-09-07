package gpuallocator

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/require"
)

func TestRemoveRunningAppRemovesOnlyMatchingWorkload(t *testing.T) {
	gpu := &tfv1.GPU{
		Status: tfv1.GPUStatus{
			RunningApps: []*tfv1.RunningAppDetail{
				{Name: "app-a", Namespace: "ns-a", Count: 1},
				{Name: "app-b", Namespace: "ns-a", Count: 1},
				{Name: "app-a", Namespace: "ns-b", Count: 1},
				{Name: "app-b", Namespace: "ns-b", Count: 1},
			},
		},
	}

	removeRunningApp(context.Background(), gpu, &tfv1.AllocRequest{
		WorkloadNameNamespace: tfv1.NameNamespace{Name: "app-a", Namespace: "ns-a"},
	})

	remaining := make([]string, 0, len(gpu.Status.RunningApps))
	for _, app := range gpu.Status.RunningApps {
		remaining = append(remaining, app.Namespace+"/"+app.Name)
	}
	require.ElementsMatch(t, []string{"ns-a/app-b", "ns-b/app-a", "ns-b/app-b"}, remaining)
}
