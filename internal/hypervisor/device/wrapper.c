/*
 * Copyright 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../../../provider/accelerator.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>
#include <dlfcn.h>

// Forward declaration of Go Log function
extern void GoLog(const char* level, const char* message);

// Function pointer types for dynamic loading
typedef Result (*GetDeviceCountFunc)(size_t*);
typedef Result (*GetAllDevicesFunc)(ExtendedDeviceInfo*, size_t, size_t*);
typedef Result (*GetPartitionTemplatesFunc)(int32_t, PartitionTemplate*, size_t, size_t*);
typedef bool (*AssignPartitionFunc)(PartitionAssignment*);
typedef bool (*RemovePartitionFunc)(const char*, const char*);
typedef Result (*SetMemHardLimitFunc)(const char*, const char*, uint64_t);
typedef Result (*SetComputeUnitHardLimitFunc)(const char*, const char*, uint32_t);
typedef Result (*GetProcessComputeUtilizationFunc)(ComputeUtilization*, size_t, size_t*);
typedef Result (*GetProcessMemoryUtilizationFunc)(MemoryUtilization*, size_t, size_t*);
typedef Result (*LogFunc)(const char*, const char*);

// Global handle for the loaded library
static void* libHandle = NULL;

// Function pointers
static GetDeviceCountFunc getDeviceCountFunc = NULL;
static GetAllDevicesFunc getAllDevicesFunc = NULL;
static GetPartitionTemplatesFunc getPartitionTemplatesFunc = NULL;
static AssignPartitionFunc assignPartitionFunc = NULL;
static RemovePartitionFunc removePartitionFunc = NULL;
static SetMemHardLimitFunc setMemHardLimitFunc = NULL;
static SetComputeUnitHardLimitFunc setComputeUnitHardLimitFunc = NULL;
static GetProcessComputeUtilizationFunc getProcessComputeUtilizationFunc = NULL;
static GetProcessMemoryUtilizationFunc getProcessMemoryUtilizationFunc = NULL;
static LogFunc logFunc = NULL;

// Load library dynamically
int loadAcceleratorLibrary(const char* libPath) {
    if (libHandle != NULL) {
        dlclose(libHandle);
    }
    
    libHandle = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (libHandle == NULL) {
        return -1; // Failed to load
    }
    
    // Load function symbols
    getDeviceCountFunc = (GetDeviceCountFunc)dlsym(libHandle, "GetDeviceCount");
    getAllDevicesFunc = (GetAllDevicesFunc)dlsym(libHandle, "GetAllDevices");
    getPartitionTemplatesFunc = (GetPartitionTemplatesFunc)dlsym(libHandle, "GetPartitionTemplates");
    assignPartitionFunc = (AssignPartitionFunc)dlsym(libHandle, "AssignPartition");
    removePartitionFunc = (RemovePartitionFunc)dlsym(libHandle, "RemovePartition");
    setMemHardLimitFunc = (SetMemHardLimitFunc)dlsym(libHandle, "SetMemHardLimit");
    setComputeUnitHardLimitFunc = (SetComputeUnitHardLimitFunc)dlsym(libHandle, "SetComputeUnitHardLimit");
    getProcessComputeUtilizationFunc = (GetProcessComputeUtilizationFunc)dlsym(libHandle, "GetProcessComputeUtilization");
    getProcessMemoryUtilizationFunc = (GetProcessMemoryUtilizationFunc)dlsym(libHandle, "GetProcessMemoryUtilization");
    logFunc = (LogFunc)dlsym(libHandle, "Log");
    
    // Check if all required functions are loaded (Log is optional)
    if (!getDeviceCountFunc || !getAllDevicesFunc || !getPartitionTemplatesFunc ||
        !assignPartitionFunc || !removePartitionFunc || !setMemHardLimitFunc ||
        !setComputeUnitHardLimitFunc || !getProcessComputeUtilizationFunc ||
        !getProcessMemoryUtilizationFunc) {
        dlclose(libHandle);
        libHandle = NULL;
        return -2; // Missing symbols
    }
    
    // If the library has a Log function, we can't directly replace it,
    // but we provide our own Log function that the library can use.
    // The library's internal Log calls will use its own implementation,
    // but if the library is designed to call Log via function pointer or
    // if it doesn't have its own Log, it will use our implementation.
    
    return 0; // Success
}

// Unload library
void unloadAcceleratorLibrary(void) {
    if (libHandle != NULL) {
        dlclose(libHandle);
        libHandle = NULL;
        getDeviceCountFunc = NULL;
        getAllDevicesFunc = NULL;
        getPartitionTemplatesFunc = NULL;
        assignPartitionFunc = NULL;
        removePartitionFunc = NULL;
        setMemHardLimitFunc = NULL;
        setComputeUnitHardLimitFunc = NULL;
        getProcessComputeUtilizationFunc = NULL;
        getProcessMemoryUtilizationFunc = NULL;
        logFunc = NULL;
    }
}

// Wrapper functions that call the dynamically loaded functions
Result GetDeviceCountWrapper(size_t* deviceCount) {
    if (getDeviceCountFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return getDeviceCountFunc(deviceCount);
}

Result GetAllDevicesWrapper(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (getAllDevicesFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return getAllDevicesFunc(devices, maxCount, deviceCount);
}

Result GetPartitionTemplatesWrapper(int32_t deviceIndex, PartitionTemplate* templates, size_t maxCount, size_t* templateCount) {
    if (getPartitionTemplatesFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return getPartitionTemplatesFunc(deviceIndex, templates, maxCount, templateCount);
}

bool AssignPartitionWrapper(PartitionAssignment* assignment) {
    if (assignPartitionFunc == NULL) {
        return false;
    }
    return assignPartitionFunc(assignment);
}

bool RemovePartitionWrapper(const char* templateId, const char* deviceUUID) {
    if (removePartitionFunc == NULL) {
        return false;
    }
    return removePartitionFunc(templateId, deviceUUID);
}

Result SetMemHardLimitWrapper(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes) {
    if (setMemHardLimitFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return setMemHardLimitFunc(workerId, deviceUUID, memoryLimitBytes);
}

Result SetComputeUnitHardLimitWrapper(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit) {
    if (setComputeUnitHardLimitFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return setComputeUnitHardLimitFunc(workerId, deviceUUID, computeUnitLimit);
}

Result GetProcessComputeUtilizationWrapper(ComputeUtilization* utilizations, size_t maxCount, size_t* utilizationCount) {
    if (getProcessComputeUtilizationFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return getProcessComputeUtilizationFunc(utilizations, maxCount, utilizationCount);
}

Result GetProcessMemoryUtilizationWrapper(MemoryUtilization* utilizations, size_t maxCount, size_t* utilizationCount) {
    if (getProcessMemoryUtilizationFunc == NULL) {
        return RESULT_ERROR_INTERNAL;
    }
    return getProcessMemoryUtilizationFunc(utilizations, maxCount, utilizationCount);
}

// Get error message from dlopen
const char* getDlError(void) {
    return dlerror();
}

// Log wrapper that calls Go's Log function
// This function provides a Log implementation that the dynamically loaded library can use
// When the library calls Log(), it will call this function which forwards to Go's klog
Result LogWrapper(const char* level, const char* message) {
    if (level == NULL || message == NULL) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // Call Go's Log function
    GoLog(level, message);
    
    return RESULT_SUCCESS;
}

// Provide a Log function that can be called by the dynamically loaded library
// This is the Log function that accelerator.h defines - we provide an implementation
// that forwards to Go's klog via GoLog
Result Log(const char* level, const char* message) {
    return LogWrapper(level, message);
}

