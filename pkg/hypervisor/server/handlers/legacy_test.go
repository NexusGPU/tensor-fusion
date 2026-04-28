package handlers

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	hyperapi "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
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

	worker := newTestWorker()
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice(),
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

	worker := newTestWorker()
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice(),
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

	worker := newTestWorker()
	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo: worker,
		DeviceInfos: []*hyperapi.DeviceInfo{
			newTestDevice(),
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

func newTestWorker() *hyperapi.WorkerInfo {
	return &hyperapi.WorkerInfo{
		WorkerUID:     "worker-uid",
		Namespace:     "tensor-fusion-sys",
		WorkerName:    "worker-pod",
		QoS:           tfv1.QoSLow,
		IsolationMode: hyperapi.IsolationMode(tfv1.IsolationModeShared),
		Limits: tfv1.Resource{
			Tflops: resource.MustParse("12"),
			Vram:   resource.MustParse("8Gi"),
		},
		AllocatedDevices: []string{"gpu-1234"},
	}
}

func newTestDevice() *hyperapi.DeviceInfo {
	return &hyperapi.DeviceInfo{
		UUID:             "gpu-1234",
		Index:            0,
		TotalMemoryBytes: 24 << 30,
		MaxTflops:        60,
		Properties: map[string]string{
			"computeCapability": "8.6",
			"totalComputeUnits": "82",
		},
	}
}

// TestEnsureWorkerSharedMemory_PreservesExistingState exercises the
// /process re-init path. Calling /process N times for the same pod must NOT
// truncate the shm — that would zero live ERL state (used tokens, mem
// counters) every retry. We assert that bytes mutated between two calls to
// ensureWorkerSharedMemory survive across the second call.
func TestEnsureWorkerSharedMemory_PreservesExistingState(t *testing.T) {
	t.Parallel()

	const namespace = "tensor-fusion-sys"
	const podName = "worker-pod"

	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo:  newTestWorker(),
		DeviceInfos: []*hyperapi.DeviceInfo{newTestDevice()},
	}

	handler := NewLegacyHandler(nil, nil, nil, nil)
	handler.shmBasePath = t.TempDir()

	if err := handler.ensureWorkerSharedMemory(namespace, podName, allocation); err != nil {
		t.Fatalf("first ensureWorkerSharedMemory failed: %v", err)
	}

	shmPath := filepath.Join(handler.shmBasePath, namespace, podName, workerstate.ShmPathSuffix)

	before, err := os.ReadFile(shmPath)
	if err != nil {
		t.Fatalf("failed to read shm file after create: %v", err)
	}
	if len(before) < 64 {
		t.Fatalf("shm file unexpectedly small: %d bytes", len(before))
	}

	// Mutate a sentinel byte well past the V2 enum discriminant (offset 0..3)
	// to simulate live ERL state mutation by the worker. If the second call
	// goes through CreateSharedMemoryHandle, O_TRUNC + Truncate will rewrite
	// the layout and our sentinel will be gone.
	const sentinelOffset = 64
	const sentinel = byte(0xAB)
	mutated := make([]byte, len(before))
	copy(mutated, before)
	mutated[sentinelOffset] = sentinel
	if err := os.WriteFile(shmPath, mutated, 0o600); err != nil {
		t.Fatalf("failed to write sentinel into shm file: %v", err)
	}

	if err := handler.ensureWorkerSharedMemory(namespace, podName, allocation); err != nil {
		t.Fatalf("second ensureWorkerSharedMemory failed: %v", err)
	}

	after, err := os.ReadFile(shmPath)
	if err != nil {
		t.Fatalf("failed to read shm file after second ensure: %v", err)
	}

	if !bytes.Equal(after, mutated) {
		t.Fatalf("shm bytes changed across re-init — Create() truncated live state.\n"+
			"sentinel byte at offset %d: want 0x%02X, got 0x%02X",
			sentinelOffset, sentinel, after[sentinelOffset])
	}
}

// TestEnsureWorkerSharedMemory_OpenErrorDoesNotRecreate covers the safety
// gate: when OpenSharedMemoryHandle fails for a reason OTHER than ENOENT
// (legacy layout, wrong size, discriminant mismatch, permission, transient
// IO), we MUST surface the error instead of falling back to Create — which
// uses O_TRUNC and would silently destroy live shm state.
func TestEnsureWorkerSharedMemory_OpenErrorDoesNotRecreate(t *testing.T) {
	t.Parallel()

	const namespace = "tensor-fusion-sys"
	const podName = "worker-pod"

	allocation := &hyperapi.WorkerAllocation{
		WorkerInfo:  newTestWorker(),
		DeviceInfos: []*hyperapi.DeviceInfo{newTestDevice()},
	}

	handler := NewLegacyHandler(nil, nil, nil, nil)
	handler.shmBasePath = t.TempDir()

	// Stage a corrupt-on-disk shm: file exists, wrong size — Open returns
	// "unexpected shared memory size" (NOT os.ErrNotExist). The fix must
	// surface that error instead of recreating.
	shmPath := filepath.Join(handler.shmBasePath, namespace, podName, workerstate.ShmPathSuffix)
	if err := os.MkdirAll(filepath.Dir(shmPath), 0o755); err != nil {
		t.Fatalf("failed to create shm dir: %v", err)
	}
	corrupt := bytes.Repeat([]byte{0xCD}, 16)
	if err := os.WriteFile(shmPath, corrupt, 0o600); err != nil {
		t.Fatalf("failed to write corrupt shm: %v", err)
	}

	err := handler.ensureWorkerSharedMemory(namespace, podName, allocation)
	if err == nil {
		t.Fatalf("expected ensureWorkerSharedMemory to surface non-NotExist Open error, got nil")
	}

	after, readErr := os.ReadFile(shmPath)
	if readErr != nil {
		t.Fatalf("failed to read shm file after ensure: %v", readErr)
	}
	if !bytes.Equal(after, corrupt) {
		t.Fatalf("shm file was rewritten despite non-NotExist Open error: len=%d (expected %d unchanged bytes)",
			len(after), len(corrupt))
	}
}

func createTestJWT(namespace, podName string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	payloadJSON := `{"kubernetes.io":{"namespace":"` + namespace + `","pod":{"name":"` + podName + `"}}}`
	payload := base64.RawURLEncoding.EncodeToString([]byte(payloadJSON))
	return header + "." + payload + ".signature"
}
