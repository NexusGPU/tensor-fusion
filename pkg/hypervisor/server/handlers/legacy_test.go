package handlers

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	hyperapi "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/gin-gonic/gin"
	"k8s.io/apimachinery/pkg/api/resource"
)

type fakeWorkerController struct {
	workers []*hyperapi.WorkerInfo
	err     error
}

func (f *fakeWorkerController) Start() error { return nil }

func (f *fakeWorkerController) Stop() error { return nil }

func (f *fakeWorkerController) ListWorkers() ([]*hyperapi.WorkerInfo, error) {
	return f.workers, f.err
}

func (f *fakeWorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*hyperapi.WorkerMetrics, error) {
	return nil, nil
}

type fakeAllocationController struct {
	allocations map[string]*hyperapi.WorkerAllocation
}

func (f *fakeAllocationController) AllocateWorkerDevices(
	request *hyperapi.WorkerInfo,
) (*hyperapi.WorkerAllocation, error) {
	return nil, nil
}

func (f *fakeAllocationController) DeallocateWorker(workerUID string) error { return nil }

func (f *fakeAllocationController) RecoverPartitionedWorker(request *hyperapi.WorkerInfo, partitionUUIDs string) {
}

func (f *fakeAllocationController) GetWorkerAllocation(workerUID string) (*hyperapi.WorkerAllocation, bool) {
	allocation, exists := f.allocations[workerUID]
	return allocation, exists
}

func (f *fakeAllocationController) GetDeviceAllocations() map[string][]*hyperapi.WorkerAllocation {
	return nil
}

type fakeBackend struct{}

func (f *fakeBackend) Start() error { return nil }

func (f *fakeBackend) Stop() error { return nil }

func (f *fakeBackend) RegisterWorkerUpdateHandler(handler framework.WorkerChangeHandler) error {
	return nil
}

func (f *fakeBackend) StartWorker(worker *hyperapi.WorkerInfo) error { return nil }

func (f *fakeBackend) StopWorker(workerUID string) error { return nil }

func (f *fakeBackend) GetProcessMappingInfo(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	return nil, nil
}

func (f *fakeBackend) GetDeviceChangeHandler() framework.DeviceChangeHandler {
	return framework.DeviceChangeHandler{}
}

func (f *fakeBackend) ListWorkers() []*hyperapi.WorkerInfo { return nil }

func TestHandleGetPodsModernResponse(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	worker := newTestWorker("worker-uid", "tensor-fusion-sys", "worker-pod")
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice("gpu-1234"),
		},
	}

	handler := NewLegacyHandler(
		&fakeWorkerController{workers: []*hyperapi.WorkerInfo{worker}},
		&fakeAllocationController{allocations: map[string]*hyperapi.WorkerAllocation{worker.WorkerUID: allocation}},
		nil, nil,
	)
	handler.shmBasePath = t.TempDir()

	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/pod?container_name=tensorfusion-worker", nil)
	req.Header.Set("Authorization", "Bearer "+createTestJWT("tensor-fusion-sys", "worker-pod"))
	ctx.Request = req

	handler.HandleGetPods(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("unexpected status code: got %d", recorder.Code)
	}

	var response hyperapi.PodInfoResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if !response.Success {
		t.Fatalf("expected success response, got message %q", response.Message)
	}
	if response.Data == nil {
		t.Fatalf("expected pod info data")
	}
	if response.Data.PodName != "worker-pod" {
		t.Fatalf("unexpected pod name: %s", response.Data.PodName)
	}
	if response.Data.Namespace != "tensor-fusion-sys" {
		t.Fatalf("unexpected namespace: %s", response.Data.Namespace)
	}
	if len(response.Data.GPUIDs) != 1 || response.Data.GPUIDs[0] != "GPU-1234" {
		t.Fatalf("unexpected gpu uuids: %#v", response.Data.GPUIDs)
	}
	if response.Data.QoSLevel == nil || *response.Data.QoSLevel != "Low" {
		t.Fatalf("unexpected qos level: %#v", response.Data.QoSLevel)
	}
	if response.Data.Isolation != string(tfv1.IsolationModeShared) {
		t.Fatalf("unexpected isolation mode: %s", response.Data.Isolation)
	}
	if response.Data.ComputeShard {
		t.Fatalf("compute_shard should default to false")
	}

	// Note: shared memory file creation requires liblimiter.so which is not loaded in tests.
	// The API response is validated above; shared memory creation is tested in limiter_test.cc.
}

func TestHandleGetPodsLegacyResponse(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	worker := newTestWorker("worker-uid", "tensor-fusion-sys", "worker-pod")
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice("gpu-1234"),
		},
	}

	handler := NewLegacyHandler(
		&fakeWorkerController{workers: []*hyperapi.WorkerInfo{worker}},
		&fakeAllocationController{allocations: map[string]*hyperapi.WorkerAllocation{worker.WorkerUID: allocation}},
		&fakeBackend{}, nil,
	)

	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/v1/pod", nil)

	handler.HandleGetPods(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("unexpected status code: got %d", recorder.Code)
	}

	var response hyperapi.ListPodsResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal legacy response: %v", err)
	}
	if len(response.Pods) != 1 {
		t.Fatalf("expected one pod in legacy response, got %d", len(response.Pods))
	}
}

func TestHandleInitProcess(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	worker := newTestWorker("worker-uid", "tensor-fusion-sys", "worker-pod")
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice("gpu-1234"),
		},
	}

	handler := NewLegacyHandler(
		&fakeWorkerController{workers: []*hyperapi.WorkerInfo{worker}},
		&fakeAllocationController{allocations: map[string]*hyperapi.WorkerAllocation{worker.WorkerUID: allocation}},
		&fakeBackend{}, nil,
	)
	handler.shmBasePath = t.TempDir()
	handler.listHostPIDsFunc = func() ([]uint32, error) {
		return []uint32{11, 22}, nil
	}
	handler.processMappingFunc = func(hostPID uint32) (*framework.ProcessMappingInfo, error) {
		if hostPID != 22 {
			return nil, nil
		}
		return &framework.ProcessMappingInfo{
			Namespace:     "tensor-fusion-sys",
			PodName:       "worker-pod",
			ContainerName: "tensorfusion-worker",
			GuestPID:      1,
			HostPID:       22,
		}, nil
	}

	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/process?container_name=tensorfusion-worker&container_pid=1", nil)
	req.Header.Set("Authorization", "Bearer "+createTestJWT("tensor-fusion-sys", "worker-pod"))
	ctx.Request = req

	handler.HandleInitProcess(ctx)

	if recorder.Code != http.StatusOK {
		t.Fatalf("unexpected status code: got %d", recorder.Code)
	}

	var response hyperapi.ProcessInitResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if !response.Success {
		t.Fatalf("expected success response, got message %q", response.Message)
	}
	if response.Data == nil {
		t.Fatalf("expected process data")
	}
	if response.Data.HostPID != 22 {
		t.Fatalf("unexpected host pid: %d", response.Data.HostPID)
	}
	if response.Data.ContainerPID != 1 {
		t.Fatalf("unexpected container pid: %d", response.Data.ContainerPID)
	}

	// Shared memory file creation requires liblimiter.so (tested in limiter_test.cc).
}

func newTestWorker(workerUID, namespace, podName string) *hyperapi.WorkerInfo {
	return &hyperapi.WorkerInfo{
		WorkerUID:     workerUID,
		Namespace:     namespace,
		WorkerName:    podName,
		QoS:           tfv1.QoSLow,
		IsolationMode: hyperapi.IsolationMode(tfv1.IsolationModeShared),
		Limits: tfv1.Resource{
			Tflops: resource.MustParse("12"),
			Vram:   resource.MustParse("8Gi"),
		},
		AllocatedDevices: []string{"gpu-1234"},
	}
}

func newTestDevice(uuid string) *hyperapi.DeviceInfo {
	return &hyperapi.DeviceInfo{
		UUID:             uuid,
		Index:            0,
		TotalMemoryBytes: 24 << 30,
		MaxTflops:        60,
		Properties: map[string]string{
			"computeCapability": "8.6",
			"totalComputeUnits": "82",
		},
	}
}

func createTestJWT(namespace, podName string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	payloadJSON := `{"kubernetes.io":{"namespace":"` + namespace + `","pod":{"name":"` + podName + `"}}}`
	payload := base64.RawURLEncoding.EncodeToString([]byte(payloadJSON))
	return header + "." + payload + ".signature"
}
