//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testShmBasePath = "/tmp/test_shm"
	testDeviceIdx   = uint32(0)
	testTotalCores  = uint32(1024)
	testUpLimit     = uint32(80)
	testMemLimit    = uint64(1024 * 1024 * 1024) // 1GB
)

func createTestConfigs() []DeviceConfig {
	return []DeviceConfig{
		{
			DeviceIdx:      testDeviceIdx,
			DeviceUUID:     "test-device-uuid",
			UpLimit:        testUpLimit,
			MemLimit:       testMemLimit,
			TotalCudaCores: testTotalCores,
			SMCount:        10,
			MaxThreadPerSM: 1024,
		},
	}
}

func TestDeviceEntryBasicOperations(t *testing.T) {
	entry := NewDeviceEntryV2()

	// Test UUID operations
	entry.SetUUID("test-uuid-123")
	assert.Equal(t, "test-uuid-123", entry.GetUUID())

	// Test active status
	assert.False(t, entry.IsActive())
	entry.SetActive(true)
	assert.True(t, entry.IsActive())
	entry.SetActive(false)
	assert.False(t, entry.IsActive())

	// Test very long UUID handling
	longUUID := strings.Repeat("a", MaxUUIDLen+10)
	entry.SetUUID(longUUID)
	storedUUID := entry.GetUUID()
	assert.Less(t, len(storedUUID), MaxUUIDLen)
	assert.Contains(t, storedUUID, "a")
}

func TestSharedDeviceStateCreationAndBasicOps(t *testing.T) {
	configs := createTestConfigs()
	state, err := NewSharedDeviceState(configs)
	require.NoError(t, err)

	// Test initial state (V2 by default)
	assert.Equal(t, uint32(2), state.Version())
	assert.Equal(t, 1, state.DeviceCount())

	// Test that heartbeat is initialized to current time (should be non-zero and recent)
	heartbeat := state.GetLastHeartbeat()
	assert.Greater(t, heartbeat, uint64(0))
	now := uint64(time.Now().Unix())
	assert.Less(t, now-heartbeat, uint64(2)) // Should be within 2 seconds

	// Should be healthy since heartbeat was just set
	assert.True(t, state.IsHealthy(30*time.Second))

	// Test device exists by index
	deviceIdx := int(configs[0].DeviceIdx)
	assert.True(t, state.HasDevice(deviceIdx))
}

func TestSharedDeviceStateHeartbeatFunctionality(t *testing.T) {
	state, err := NewSharedDeviceState([]DeviceConfig{})
	require.NoError(t, err)

	// Test initial healthy state (heartbeat is initialized to current time)
	assert.True(t, state.IsHealthy(30*time.Second))

	// Test setting heartbeat to a specific time
	now := uint64(time.Now().Unix())
	state.UpdateHeartbeat(now)
	assert.Equal(t, now, state.GetLastHeartbeat())
	assert.True(t, state.IsHealthy(30*time.Second))

	// Test old heartbeat (should be unhealthy)
	state.UpdateHeartbeat(now - 60)
	assert.False(t, state.IsHealthy(30*time.Second))
}

func TestSharedDeviceInfoAtomicOperations(t *testing.T) {
	// Test V1 device info (has available_cores)
	deviceInfoV1 := NewSharedDeviceInfoV1(testTotalCores, testUpLimit, testMemLimit)

	// Test available cores operations (V1 only)
	deviceInfoV1.AvailableCudaCores = 512
	assert.Equal(t, int32(512), deviceInfoV1.AvailableCudaCores)

	deviceInfoV1.AvailableCudaCores = 600
	assert.Equal(t, int32(600), deviceInfoV1.AvailableCudaCores)

	// Test negative values
	deviceInfoV1.AvailableCudaCores = -50
	assert.Equal(t, int32(-50), deviceInfoV1.AvailableCudaCores)

	// Test other fields
	deviceInfoV1.UpLimit = 90
	assert.Equal(t, uint32(90), deviceInfoV1.UpLimit)

	deviceInfoV1.MemLimit = 2 * 1024 * 1024 * 1024
	assert.Equal(t, uint64(2*1024*1024*1024), deviceInfoV1.MemLimit)

	// Test V2 device info (has ERL fields)
	deviceInfoV2 := NewSharedDeviceInfoV2(testTotalCores, testUpLimit, testMemLimit)
	// Test ERL fields - refill rate is now the control parameter
	deviceInfoV2.SetERLTokenRefillRate(15.0)
	assert.Equal(t, 15.0, deviceInfoV2.GetERLTokenRefillRate())

	deviceInfoV2.SetERLTokenCapacity(100.0)
	assert.Equal(t, 100.0, deviceInfoV2.GetERLTokenCapacity())

	deviceInfoV2.PodMemoryUsed = 512 * 1024 * 1024
	assert.Equal(t, uint64(512*1024*1024), deviceInfoV2.PodMemoryUsed)
}

func TestERLTokenBucketPreservesTokensWhenInsufficient(t *testing.T) {
	deviceInfo := NewSharedDeviceInfoV2(testTotalCores, testUpLimit, testMemLimit)

	deviceInfo.SetERLCurrentTokens(1.5)
	before := deviceInfo.FetchSubERLTokens(2.0)
	assert.Equal(t, 1.5, before)
	assert.Equal(t, 1.5, deviceInfo.GetERLCurrentTokens())

	deviceInfo.SetERLCurrentTokens(5.0)
	beforeSuccess := deviceInfo.FetchSubERLTokens(2.0)
	assert.Equal(t, 5.0, beforeSuccess)
	assert.Equal(t, 3.0, deviceInfo.GetERLCurrentTokens())
}

func TestSharedMemoryHandleCreateAndOpen(t *testing.T) {
	configs := createTestConfigs()
	identifier := NewPodIdentifier("handle_create_open", "test")

	podPath := identifier.ToPath(testShmBasePath)
	defer func() {
		_ = os.RemoveAll(podPath)
	}()

	// Create shared memory
	handle1, err := CreateSharedMemoryHandle(podPath, configs)
	require.NoError(t, err)
	defer func() {
		_ = handle1.Close()
	}()

	state1 := handle1.GetState()
	assert.Equal(t, uint32(2), state1.Version())
	assert.Equal(t, 1, state1.DeviceCount())

	// Verify shared memory file exists after creation
	assert.True(t, fileExists(filepath.Join(podPath, ShmPathSuffix)))

	// Open existing shared memory
	handle2, err := OpenSharedMemoryHandle(podPath)
	require.NoError(t, err)
	defer func() {
		_ = handle2.Close()
	}()

	state2 := handle2.GetState()
	assert.Equal(t, uint32(2), state2.Version())
	assert.Equal(t, 1, state2.DeviceCount())

	// Verify they access the same memory
	deviceIdx := int(configs[0].DeviceIdx)
	state1.SetPodMemoryUsed(deviceIdx, 42)
	memory := state2.GetPodMemoryUsed(deviceIdx)
	assert.Equal(t, uint64(42), memory)
}

func TestSharedMemoryHandleErrorHandling(t *testing.T) {
	_, err := OpenSharedMemoryHandle("non_existent_memory")
	assert.Error(t, err)
}

func TestConcurrentDeviceAccess(t *testing.T) {
	configs := createTestConfigs()
	identifier := NewPodIdentifier("concurrent_access", "test")
	podPath := identifier.ToPath(testShmBasePath)
	defer func() {
		_ = os.RemoveAll(podPath)
	}()

	handle, err := CreateSharedMemoryHandle(podPath, configs)
	require.NoError(t, err)
	defer func() {
		_ = handle.Close()
	}()

	deviceIdx := int(configs[0].DeviceIdx)
	var wg sync.WaitGroup
	numGoroutines := 5
	iterations := 20

	// Spawn multiple goroutines doing concurrent access
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			state := handle.GetState()

			for j := 0; j < iterations; j++ {
				value := uint64(id*iterations + j)
				state.SetPodMemoryUsed(deviceIdx, value)

				time.Sleep(time.Millisecond)

				readValue := state.GetPodMemoryUsed(deviceIdx)
				// Value should be valid (set by some goroutine)
				assert.GreaterOrEqual(t, readValue, uint64(0))
				assert.Less(t, readValue, uint64(100))
			}
		}(i)
	}

	wg.Wait()
}

func TestDeviceIterationMethods(t *testing.T) {
	// Create multiple device configurations
	configs := []DeviceConfig{
		{
			DeviceIdx:      0,
			DeviceUUID:     "device-0",
			UpLimit:        80,
			MemLimit:       1024 * 1024 * 1024,
			TotalCudaCores: 1024,
			SMCount:        10,
			MaxThreadPerSM: 1024,
		},
		{
			DeviceIdx:      2,
			DeviceUUID:     "device-2",
			UpLimit:        70,
			MemLimit:       2 * 1024 * 1024 * 1024,
			TotalCudaCores: 2048,
			SMCount:        20,
			MaxThreadPerSM: 1024,
		},
	}

	state, err := NewSharedDeviceState(configs)
	require.NoError(t, err)

	// Test iterating over active devices
	activeCount := 0
	for i := 0; i < MaxDevices; i++ {
		if state.HasDevice(i) {
			activeCount++
		}
	}
	assert.Equal(t, 2, activeCount)

	// Check that indices match the device_idx from configs
	assert.True(t, state.HasDevice(0))
	assert.True(t, state.HasDevice(2))

	// Test deactivating a device and checking
	if state.V2 != nil {
		state.V2.Devices[2].SetActive(false)
		assert.False(t, state.HasDevice(2))
		assert.True(t, state.HasDevice(0))
	}
}

func TestPIDSetDeduplicatesOnAdd(t *testing.T) {
	state, err := NewSharedDeviceState([]DeviceConfig{})
	require.NoError(t, err)

	// Add the same pid multiple times
	state.AddPID(1234)
	state.AddPID(1234)
	state.AddPID(1234)

	pids := state.GetAllPIDs()
	assert.Equal(t, 1, len(pids), "should contain only one PID after duplicate adds")
	if len(pids) > 0 {
		assert.Equal(t, 1234, pids[0])
	}
}

func TestPIDRemoveByValueWorks(t *testing.T) {
	state, err := NewSharedDeviceState([]DeviceConfig{})
	require.NoError(t, err)

	state.AddPID(111)
	state.AddPID(222)
	state.AddPID(333)

	state.RemovePID(222)

	pids := state.GetAllPIDs()
	assert.Equal(t, 2, len(pids), "should remove the specified PID")
	assert.Contains(t, pids, 111)
	assert.Contains(t, pids, 333)
	assert.NotContains(t, pids, 222)
}

func TestPIDSetCapacityAndDuplicateBehavior(t *testing.T) {
	state, err := NewSharedDeviceState([]DeviceConfig{})
	require.NoError(t, err)

	// Fill to capacity with unique PIDs
	for pid := 0; pid < MaxProcesses; pid++ {
		state.AddPID(pid)
	}

	pids := state.GetAllPIDs()
	assert.Equal(t, MaxProcesses, len(pids), "should reach max capacity with unique PIDs")

	// Adding an existing PID should not change the count
	state.AddPID(0)
	pidsAfterDup := state.GetAllPIDs()
	assert.Equal(t, MaxProcesses, len(pidsAfterDup), "should remain at capacity when inserting duplicate")
}

func TestCleanupEmptyParentDirectories(t *testing.T) {
	// Create a temporary directory structure
	tempDir, err := os.MkdirTemp("", "test_cleanup_*")
	require.NoError(t, err)
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	// Create nested directory structure: base/namespace/podname/
	namespaceDir := filepath.Join(tempDir, "test-namespace")
	podDir := filepath.Join(namespaceDir, "test-pod")
	err = os.MkdirAll(podDir, 0755)
	require.NoError(t, err)

	// Create a file in the pod directory
	testFile := filepath.Join(podDir, ShmPathSuffix)
	err = os.WriteFile(testFile, []byte("test data"), 0644)
	require.NoError(t, err)

	// Verify structure exists
	assert.True(t, fileExists(testFile))
	assert.True(t, fileExists(podDir))
	assert.True(t, fileExists(namespaceDir))

	// Remove the file
	err = os.Remove(testFile)
	require.NoError(t, err)

	// Test cleanup without stop_at_path (should remove all empty dirs)
	err = CleanupEmptyParentDirectories(testFile, nil)
	assert.NoError(t, err)

	// Pod directory should be removed
	assert.False(t, fileExists(podDir))
	// Namespace directory should be removed
	assert.False(t, fileExists(namespaceDir))
}

func TestCleanupEmptyParentDirectoriesWithStopAtPath(t *testing.T) {
	// Create a temporary directory structure
	tempDir, err := os.MkdirTemp("", "test_cleanup_*")
	require.NoError(t, err)
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	// Create nested directory structure: base/namespace/podname/
	namespaceDir := filepath.Join(tempDir, "test-namespace")
	podDir := filepath.Join(namespaceDir, "test-pod")
	err = os.MkdirAll(podDir, 0755)
	require.NoError(t, err)

	// Create a file in the pod directory
	testFile := filepath.Join(podDir, ShmPathSuffix)
	err = os.WriteFile(testFile, []byte("test data"), 0644)
	require.NoError(t, err)

	// Remove the file
	err = os.Remove(testFile)
	require.NoError(t, err)

	// Test cleanup with stop_at_path set to base_path
	stopAtPath := tempDir
	err = CleanupEmptyParentDirectories(testFile, &stopAtPath)
	assert.NoError(t, err)

	// Pod directory should be removed
	assert.False(t, fileExists(podDir))
	// Namespace directory should be removed
	assert.False(t, fileExists(namespaceDir))
	// Base directory should remain (it's the stop_at_path)
	assert.True(t, fileExists(tempDir))
}

func TestCleanupEmptyParentDirectoriesStopsAtNonEmptyDir(t *testing.T) {
	// Create a temporary directory structure
	tempDir, err := os.MkdirTemp("", "test_cleanup_*")
	require.NoError(t, err)
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	// Create nested directory structure: base/namespace/podname/
	namespaceDir := filepath.Join(tempDir, "test-namespace")
	podDir := filepath.Join(namespaceDir, "test-pod")
	err = os.MkdirAll(podDir, 0755)
	require.NoError(t, err)

	// Create two files in the pod directory
	testFile1 := filepath.Join(podDir, ShmPathSuffix)
	testFile2 := filepath.Join(podDir, "other_file")
	err = os.WriteFile(testFile1, []byte("test data"), 0644)
	require.NoError(t, err)
	err = os.WriteFile(testFile2, []byte("other data"), 0644)
	require.NoError(t, err)

	// Remove only one file
	err = os.Remove(testFile1)
	require.NoError(t, err)

	// Test cleanup - should not remove pod directory since it's not empty
	stopAtPath := tempDir
	err = CleanupEmptyParentDirectories(testFile1, &stopAtPath)
	assert.NoError(t, err)

	// Pod directory should still exist (not empty)
	assert.True(t, fileExists(podDir))
	assert.True(t, fileExists(namespaceDir))
	assert.True(t, fileExists(testFile2))
}

func TestPodIdentifierFromShmFilePath(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		expectError  bool
		expectedNS   string
		expectedName string
	}{
		{
			name:         "valid path",
			path:         "/base/namespace/podname/shm",
			expectError:  false,
			expectedNS:   "namespace",
			expectedName: "podname",
		},
		{
			name:        "invalid path - too short",
			path:        "/base/shm",
			expectError: true,
		},
		{
			name:        "invalid path - only two components",
			path:        "/namespace/shm",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pid, err := FromShmFilePath(tt.path)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, pid)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, pid)
				assert.Equal(t, tt.expectedNS, pid.Namespace)
				assert.Equal(t, tt.expectedName, pid.Name)
			}
		})
	}
}

func TestPodIdentifierToPath(t *testing.T) {
	pid := NewPodIdentifier("test-namespace", "test-pod")
	path := pid.ToPath("/base")
	expected := filepath.Join("/base", "test-namespace", "test-pod")
	assert.Equal(t, expected, path)
}

func TestSharedDeviceStateSetPodMemoryUsed(t *testing.T) {
	configs := createTestConfigs()
	state, err := NewSharedDeviceState(configs)
	require.NoError(t, err)

	deviceIdx := int(configs[0].DeviceIdx)

	// Test setting memory
	success := state.SetPodMemoryUsed(deviceIdx, 1024*1024*1024)
	assert.True(t, success)

	// Test setting memory for non-existent device
	success = state.SetPodMemoryUsed(999, 1024)
	assert.False(t, success)
}

func TestERLTokenOperations(t *testing.T) {
	deviceInfo := NewSharedDeviceInfoV2(testTotalCores, testUpLimit, testMemLimit)

	// Test initial values
	assert.Equal(t, 10.0, deviceInfo.GetERLTokenRefillRate())
	assert.Equal(t, 100.0, deviceInfo.GetERLTokenCapacity())
	assert.Equal(t, 100.0, deviceInfo.GetERLCurrentTokens())

	// Test setting values
	deviceInfo.SetERLTokenRefillRate(50.0)
	deviceInfo.SetERLTokenCapacity(200.0)
	deviceInfo.SetERLCurrentTokens(150.0)

	assert.Equal(t, 50.0, deviceInfo.GetERLTokenRefillRate())
	assert.Equal(t, 200.0, deviceInfo.GetERLTokenCapacity())
	assert.Equal(t, 150.0, deviceInfo.GetERLCurrentTokens())

	// Test LoadERLTokenState
	tokens, timestamp := deviceInfo.LoadERLTokenState()
	assert.Equal(t, 150.0, tokens)
	assert.Equal(t, 0.0, timestamp) // Initial timestamp is 0.0

	// Test StoreERLTokenState
	deviceInfo.StoreERLTokenState(175.0, 12345.0)
	tokens, timestamp = deviceInfo.LoadERLTokenState()
	assert.Equal(t, 175.0, tokens)
	assert.Equal(t, 12345.0, timestamp)

	// Test LoadERLQuota
	capacity, rate := deviceInfo.LoadERLQuota()
	assert.Equal(t, 200.0, capacity)
	assert.Equal(t, 50.0, rate)
}

func TestFetchAddERLTokens(t *testing.T) {
	deviceInfo := NewSharedDeviceInfoV2(testTotalCores, testUpLimit, testMemLimit)
	deviceInfo.SetERLTokenCapacity(100.0)
	deviceInfo.SetERLCurrentTokens(50.0)

	// Add tokens
	before := deviceInfo.FetchAddERLTokens(30.0)
	assert.Equal(t, 50.0, before)
	assert.Equal(t, 80.0, deviceInfo.GetERLCurrentTokens())

	// Add tokens that would exceed capacity
	before = deviceInfo.FetchAddERLTokens(50.0)
	assert.Equal(t, 80.0, before)
	assert.Equal(t, 100.0, deviceInfo.GetERLCurrentTokens()) // Capped at capacity
}

func TestSharedDeviceStateV1Operations(t *testing.T) {
	configs := createTestConfigs()
	state, err := NewSharedDeviceStateV1(configs)
	require.NoError(t, err)

	assert.Equal(t, 1, state.DeviceCount())
	assert.True(t, state.HasDevice(0))
	assert.False(t, state.HasDevice(1))

	// Test heartbeat
	now := uint64(time.Now().Unix())
	state.UpdateHeartbeat(now)
	assert.Equal(t, now, state.GetLastHeartbeat())
	assert.True(t, state.IsHealthy(30*time.Second))
}

func TestSharedDeviceStateV2Operations(t *testing.T) {
	configs := createTestConfigs()
	state, err := NewSharedDeviceStateV2(configs)
	require.NoError(t, err)

	assert.Equal(t, 1, state.DeviceCount())
	assert.True(t, state.HasDevice(0))
	assert.False(t, state.HasDevice(1))

	// Test heartbeat
	now := uint64(time.Now().Unix())
	state.UpdateHeartbeat(now)
	assert.Equal(t, now, state.GetLastHeartbeat())
	assert.True(t, state.IsHealthy(30*time.Second))
}

func TestDeviceEntryV1Operations(t *testing.T) {
	entry := NewDeviceEntryV1()

	entry.SetUUID("v1-uuid-test")
	assert.Equal(t, "v1-uuid-test", entry.GetUUID())

	assert.False(t, entry.IsActive())
	entry.SetActive(true)
	assert.True(t, entry.IsActive())
}

func TestSharedMemoryHandleCleanup(t *testing.T) {
	configs := createTestConfigs()
	identifier := NewPodIdentifier("cleanup_test", "test")
	podPath := identifier.ToPath(testShmBasePath)
	defer func() {
		_ = os.RemoveAll(testShmBasePath)
	}()

	handle, err := CreateSharedMemoryHandle(podPath, configs)
	require.NoError(t, err)

	shmPath := filepath.Join(podPath, ShmPathSuffix)
	assert.True(t, fileExists(shmPath))

	// Cleanup
	stopAtPath := testShmBasePath
	err = handle.Cleanup(&stopAtPath)
	assert.NoError(t, err)

	// File should be removed
	assert.False(t, fileExists(shmPath))
}

// Helper function to check if file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

// Helper function to get pod memory used (needed for tests)
func (s *SharedDeviceState) GetPodMemoryUsed(index int) uint64 {
	if s.V1 != nil {
		if index >= MaxDevices || !s.V1.Devices[index].IsActive() {
			return 0
		}
		return atomic.LoadUint64(&s.V1.Devices[index].DeviceInfo.PodMemoryUsed)
	}
	if index >= MaxDevices || !s.V2.Devices[index].IsActive() {
		return 0
	}
	return atomic.LoadUint64(&s.V2.Devices[index].DeviceInfo.PodMemoryUsed)
}
