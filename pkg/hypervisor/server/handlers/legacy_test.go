package handlers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	hyperapi "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/gin-gonic/gin"
	"k8s.io/apimachinery/pkg/api/resource"
)

type fakeWorkerController struct {
	workers  []*hyperapi.WorkerInfo
	err      error
	shmCalls []string
	shmErr   error
}

func (f *fakeWorkerController) Start() error { return nil }

func (f *fakeWorkerController) Stop() error { return nil }

func (f *fakeWorkerController) ListWorkers() ([]*hyperapi.WorkerInfo, error) {
	return f.workers, f.err
}

func (f *fakeWorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*hyperapi.WorkerMetrics, error) {
	return nil, nil
}

func (f *fakeWorkerController) WithWorkerSharedMemory(uid string, fn func(*workerstate.SharedDeviceState)) error {
	f.shmCalls = append(f.shmCalls, uid)
	if f.shmErr != nil {
		return f.shmErr
	}
	if fn != nil {
		state, err := workerstate.NewSharedDeviceState(nil)
		if err != nil {
			return err
		}
		fn(state)
	}
	return nil
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

func (f *fakeAllocationController) RecoverPartitionedWorker(request *hyperapi.WorkerInfo, partitionUUIDs string) error {
	return nil
}

func (f *fakeAllocationController) GetWorkerAllocation(workerUID string) (*hyperapi.WorkerAllocation, bool) {
	allocation, exists := f.allocations[workerUID]
	return allocation, exists
}

func (f *fakeAllocationController) GetDeviceAllocations() map[string][]*hyperapi.WorkerAllocation {
	return nil
}

type fakeBackend struct {
	podUID  string
	authErr error
}

func (f *fakeBackend) AuthenticatePodToken(context.Context, string) (string, error) {
	return f.podUID, f.authErr
}

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

func TestNewLegacyHandlerLoadsAutoFreezeConfig(t *testing.T) {
	t.Setenv(constants.HypervisorSchedulingConfigEnv, `{
		"autoFreezeAndResume": {
			"autoFreeze": [{
				"qos": "low",
				"enable": true,
				"freezeToMemTTL": "1m",
				"freezeToDiskTTL": "1h"
			}]
		}
	}`)

	handler := NewLegacyHandler(nil, nil, nil, nil)
	if len(handler.autoFreeze) != 1 {
		t.Fatalf("expected one auto-freeze rule, got %d", len(handler.autoFreeze))
	}
	if handler.autoFreeze[0].Qos != tfv1.QoSLow || handler.autoFreeze[0].Enable == nil || !*handler.autoFreeze[0].Enable {
		t.Fatalf("unexpected auto-freeze rule: %#v", handler.autoFreeze[0])
	}
}

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
		&fakeBackend{podUID: worker.WorkerUID}, nil,
	)
	enableAutoFreeze := true
	handler.autoFreeze = []tfv1.AutoFreeze{{
		Qos:             tfv1.QoSLow,
		FreezeToMemTTL:  "1m",
		FreezeToDiskTTL: "1h",
		Enable:          &enableAutoFreeze,
	}}

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
	if response.Data.AutoFreeze == nil || !response.Data.AutoFreeze.Enable {
		t.Fatal("expected enabled auto-freeze config for low QoS")
	}
	if response.Data.AutoFreeze.FreezeToMemTTL == nil || *response.Data.AutoFreeze.FreezeToMemTTL != "1m" {
		t.Fatalf("unexpected freeze-to-memory TTL: %#v", response.Data.AutoFreeze.FreezeToMemTTL)
	}
	if response.Data.AutoFreeze.FreezeToDiskTTL == nil || *response.Data.AutoFreeze.FreezeToDiskTTL != "1h" {
		t.Fatalf("unexpected freeze-to-disk TTL: %#v", response.Data.AutoFreeze.FreezeToDiskTTL)
	}

	calls := handler.workerController.(*fakeWorkerController).shmCalls
	if len(calls) != 1 || calls[0] != worker.WorkerUID {
		t.Fatalf("shared memory was not prepared for authenticated UID: %v", calls)
	}
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
		&fakeBackend{podUID: worker.WorkerUID}, nil,
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
		&fakeBackend{podUID: worker.WorkerUID}, nil,
	)
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
			PodUID:        worker.WorkerUID,
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

func createTestJWT(namespace, podName string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","typ":"JWT"}`))
	payloadJSON := `{"kubernetes.io":{"namespace":"` + namespace + `","pod":{"name":"` + podName + `"}}}`
	payload := base64.RawURLEncoding.EncodeToString([]byte(payloadJSON))
	return header + "." + payload + ".signature"
}

func TestLegacyPodAPIsRequireVerifiedPodIdentity(t *testing.T) {
	for _, endpoint := range []string{
		"/api/v1/pod?container_name=worker",
		"/api/v1/process?container_name=worker&container_pid=1",
	} {
		for _, test := range []struct {
			name, uid string
			err       error
		}{
			{name: "forged token", err: errors.New("invalid signature")},
			{name: "old UID with reused Pod name", uid: "deleted-pod-uid"},
		} {
			t.Run(endpoint+"/"+test.name, func(t *testing.T) {
				worker := newTestWorker()
				handler := NewLegacyHandler(
					&fakeWorkerController{workers: []*hyperapi.WorkerInfo{worker}},
					&fakeAllocationController{allocations: map[string]*hyperapi.WorkerAllocation{
						worker.WorkerUID: {WorkerInfo: worker, DeviceInfos: []*hyperapi.DeviceInfo{newTestDevice()}},
					}},
					&fakeBackend{podUID: test.uid, authErr: test.err}, nil,
				)
				method := http.MethodGet
				if strings.Contains(endpoint, "/process") {
					method = http.MethodPost
				}
				recorder := httptest.NewRecorder()
				ctx, _ := gin.CreateTestContext(recorder)
				ctx.Request = httptest.NewRequest(method, endpoint, nil)
				// Arbitrary unsigned claims target a real worker by name.
				ctx.Request.Header.Set("Authorization", "Bearer "+createTestJWT(worker.Namespace, worker.WorkerName))
				if method == http.MethodGet {
					handler.HandleGetPods(ctx)
				} else {
					handler.HandleInitProcess(ctx)
				}
				if test.err != nil && recorder.Code != http.StatusUnauthorized {
					t.Fatalf("invalid credential accepted: %d %s", recorder.Code, recorder.Body.String())
				}
				body := recorder.Body.String()
				if strings.Contains(body, `"success":true`) || strings.Contains(body, "GPU-1234") {
					t.Fatalf("unauthorized Pod identity accessed allocation: %s", recorder.Body.String())
				}
				calls := handler.workerController.(*fakeWorkerController).shmCalls
				if len(calls) != 0 {
					t.Fatalf("unauthorized request initialized shared memory for %v", calls)
				}
			})
		}
	}
}

func TestFindHostPIDRequiresCgroupPodUID(t *testing.T) {
	h := NewLegacyHandler(nil, nil, nil, nil)
	h.listHostPIDsFunc = func() ([]uint32, error) { return []uint32{10, 20, 30}, nil }
	h.processMappingFunc = func(pid uint32) (*framework.ProcessMappingInfo, error) {
		uid := "current-uid"
		if pid == 10 {
			uid = "previous-uid"
		}
		if pid == 20 {
			uid = ""
		}
		return &framework.ProcessMappingInfo{
			HostPID: pid, GuestPID: 1, PodUID: uid, Namespace: "ns", PodName: "pod", ContainerName: "worker",
		}, nil
	}
	pid, err := h.findHostPID("current-uid", "ns", "pod", "worker", 1)
	if err != nil || pid != 30 {
		t.Fatalf("expected UID-matching process 30, got %d, %v", pid, err)
	}
}

func TestLegacyPodAPIRejectsMissingAuthenticator(t *testing.T) {
	h := NewLegacyHandler(nil, nil, nil, nil)
	response := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(response)
	ctx.Request = httptest.NewRequest("GET", "/api/v1/pod?container_name=worker", nil)
	ctx.Request.Header.Set("Authorization", "Bearer "+createTestJWT("ns", "pod"))
	h.HandleGetPods(ctx)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unexpected status %d", response.Code)
	}
}
