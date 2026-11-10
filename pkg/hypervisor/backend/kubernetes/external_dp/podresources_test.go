package external_dp

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/grpc"
	podresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

type fakePodResourcesLister struct {
	podresourcesv1.UnimplementedPodResourcesListerServer
}

func (fakePodResourcesLister) List(
	context.Context, *podresourcesv1.ListPodResourcesRequest,
) (*podresourcesv1.ListPodResourcesResponse, error) {
	return &podresourcesv1.ListPodResourcesResponse{PodResources: []*podresourcesv1.PodResources{{
		Name: "native", Namespace: "default",
		Containers: []*podresourcesv1.ContainerResources{{Name: "c", Devices: []*podresourcesv1.ContainerDevices{
			{ResourceName: "nvidia.com/gpu", DeviceIds: []string{"GPU-AAA", "GPU-BBB::1"}},
		}}},
	}}}, nil
}

// The response must be decoded with the kubelet's generated types: a struct
// without protobuf field tags decodes to an empty response without an error,
// which made every GPU used by a native Pod look free to TensorFusion.
func TestGetAllocatedDevicesDecodesKubeletResponse(t *testing.T) {
	// unix socket paths are limited to ~108 bytes; t.TempDir() can exceed that
	dir, err := os.MkdirTemp("", "podres")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	socket := filepath.Join(dir, "kubelet.sock")
	lis, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	podresourcesv1.RegisterPodResourcesListerServer(server, fakePodResourcesLister{})
	go func() { _ = server.Serve(lis) }()
	defer server.Stop()

	d := &DevicePluginDetector{ctx: context.Background()}
	got, err := d.getAllocatedDevices(socket)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"gpu-aaa": "nvidia.com/gpu", "gpu-bbb": "nvidia.com/gpu"}
	if len(got) != len(want) || got["gpu-aaa"] != want["gpu-aaa"] || got["gpu-bbb"] != want["gpu-bbb"] {
		t.Fatalf("allocated devices = %v, want %v", got, want)
	}
}
