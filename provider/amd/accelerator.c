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

// Feature test macros for POSIX functions
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "../accelerator.h"
#include <amd_smi/amdsmi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>

// ============================================================================
// Global State
// ============================================================================

static LogCallbackFunc g_logCallback = NULL;
static bool g_initialized = false;
static pthread_mutex_t g_initMutex = PTHREAD_MUTEX_INITIALIZER;

// ============================================================================
// Helper Functions
// ============================================================================

static void logMessage(const char* level, const char* message) {
    if (g_logCallback != NULL) {
        g_logCallback(level, message);
    }
}

static void logError(const char* func, amdsmi_status_t status) {
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%s failed with status: %d", func, status);
    logMessage("ERROR", buffer);
}

// Convert PCI bus info to UUID-like string
static void generateUUID(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function, char* uuid, size_t size) {
    snprintf(uuid, size, "AMD-GPU-%04x:%02x:%02x.%x", domain, bus, device, function);
}

// Find renderD device path for a given PCI slot address
// Returns 0 on success with renderD path in output buffer, -1 on failure
static int findRenderDevice(const char* pci_slot, char* output, size_t output_size) {
    DIR* dir = opendir("/sys/class/drm");
    if (!dir) {
        return -1;
    }
    
    struct dirent* entry;
    int found = -1;
    
    while ((entry = readdir(dir)) != NULL) {
        // Look for card* entries (not renderD* or controlD*)
        if (strncmp(entry->d_name, "card", 4) != 0) {
            continue;
        }
        
        // Skip if not a plain card entry (e.g., card0-VGA-1)
        char* dash = strchr(entry->d_name + 4, '-');
        if (dash != NULL) {
            continue;
        }
        
        // Read the device symlink to get PCI address
        char path[512];
        char target[512];
        snprintf(path, sizeof(path), "/sys/class/drm/%s/device", entry->d_name);
        
        ssize_t len = readlink(path, target, sizeof(target) - 1);
        if (len > 0) {
            target[len] = '\0';
            
            // Extract PCI address from symlink path
            // Example: ../../devices/.../0000:85:00.0/drm/card41
            // We need to find the component matching the PCI slot pattern
            char target_copy[512];
            strncpy(target_copy, target, sizeof(target_copy) - 1);
            target_copy[sizeof(target_copy) - 1] = '\0';
            
            char* token = strtok(target_copy, "/");
            int match_found = 0;
            while (token != NULL) {
                // Check if token matches PCI address pattern (contains ':' and '.')
                if (strchr(token, ':') && strchr(token, '.') && strcmp(token, pci_slot) == 0) {
                    match_found = 1;
                    break;
                }
                token = strtok(NULL, "/");
            }
            
            if (match_found) {
                // Found matching card - now look for renderD device in its drm subdirectory
                char drm_path[512];
                snprintf(drm_path, sizeof(drm_path), "/sys/class/drm/%s/device/drm", entry->d_name);
                
                DIR* drm_dir = opendir(drm_path);
                if (drm_dir) {
                    struct dirent* drm_entry;
                    while ((drm_entry = readdir(drm_dir)) != NULL) {
                        if (strncmp(drm_entry->d_name, "renderD", 7) == 0) {
                            snprintf(output, output_size, "/dev/dri/%s", drm_entry->d_name);
                            found = 0;
                            break;
                        }
                    }
                    closedir(drm_dir);
                }
                break;
            }
        }
    }
    
    closedir(dir);
    return found;
}

// Helper to add a property to the properties array
static void addProperty(DeviceProperties* props, const char* key, const char* value) {
    if (props->count >= MAX_DEVICE_PROPERTIES) {
        return; // No room for more properties
    }
    
    size_t key_len = strlen(key);
    size_t val_len = strlen(value);
    
    if (key_len >= sizeof(props->properties[0].key) || val_len >= sizeof(props->properties[0].value)) {
        return; // Key or value too long
    }
    
    strncpy(props->properties[props->count].key, key, sizeof(props->properties[0].key) - 1);
    strncpy(props->properties[props->count].value, value, sizeof(props->properties[0].value) - 1);
    props->properties[props->count].key[sizeof(props->properties[0].key) - 1] = '\0';
    props->properties[props->count].value[sizeof(props->properties[0].value) - 1] = '\0';
    props->count++;
}

// ============================================================================
// Initialization
// ============================================================================

Result VirtualGPUInit(void) {
    pthread_mutex_lock(&g_initMutex);
    
    if (g_initialized) {
        pthread_mutex_unlock(&g_initMutex);
        return RESULT_SUCCESS;
    }
    
    amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (status != AMDSMI_STATUS_SUCCESS) {
        logError("amdsmi_init", status);
        pthread_mutex_unlock(&g_initMutex);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    g_initialized = true;
    logMessage("INFO", "AMD SMI initialized successfully");
    pthread_mutex_unlock(&g_initMutex);
    return RESULT_SUCCESS;
}

// ============================================================================
// Device Info APIs
// ============================================================================

Result GetDeviceCount(size_t* deviceCount) {
    if (!deviceCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    if (!g_initialized) {
        Result initResult = VirtualGPUInit();
        if (initResult != RESULT_SUCCESS) {
            return initResult;
        }
    }
    
    uint32_t socket_count = 0;
    amdsmi_status_t status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        logError("amdsmi_get_socket_handles", status);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    amdsmi_socket_handle* sockets = malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) {
        return RESULT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        logError("amdsmi_get_socket_handles", status);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    *deviceCount = 0;
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles(sockets[i], &processor_count, NULL);
        if (status == AMDSMI_STATUS_SUCCESS) {
            *deviceCount += processor_count;
        }
    }
    
    free(sockets);
    return RESULT_SUCCESS;
}

Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (!devices || !deviceCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    if (!g_initialized) {
        Result initResult = VirtualGPUInit();
        if (initResult != RESULT_SUCCESS) {
            return initResult;
        }
    }
    
    uint32_t socket_count = 0;
    amdsmi_status_t status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        logError("amdsmi_get_socket_handles", status);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    amdsmi_socket_handle* sockets = malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) {
        return RESULT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        logError("amdsmi_get_socket_handles", status);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    size_t totalDevices = 0;
    
    for (uint32_t socketIdx = 0; socketIdx < socket_count && totalDevices < maxCount; socketIdx++) {
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles(sockets[socketIdx], &processor_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        
        amdsmi_processor_handle* processors = malloc(processor_count * sizeof(amdsmi_processor_handle));
        if (!processors) {
            continue;
        }
        
        status = amdsmi_get_processor_handles(sockets[socketIdx], &processor_count, processors);
        if (status != AMDSMI_STATUS_SUCCESS) {
            free(processors);
            continue;
        }
        
        for (uint32_t procIdx = 0; procIdx < processor_count && totalDevices < maxCount; procIdx++) {
            ExtendedDeviceInfo* info = &devices[totalDevices];
            memset(info, 0, sizeof(ExtendedDeviceInfo));
            // Initialize properties before adding any key-values
            info->props.count = 0;
            
            amdsmi_processor_handle processor = processors[procIdx];
            
            // Get BDF (Bus/Device/Function) for UUID
            uint64_t bdfid = 0;
            char renderDev[64] = {0};
            status = amdsmi_get_gpu_bdf_id(processor, &bdfid);
            if (status == AMDSMI_STATUS_SUCCESS) {
                uint32_t domain = (bdfid >> 32) & 0xFFFF;
                uint32_t bus = (bdfid >> 8) & 0xFF;
                uint32_t device = (bdfid >> 3) & 0x1F;
                uint32_t function = bdfid & 0x7;
                generateUUID(domain, bus, device, function, info->basic.uuid, sizeof(info->basic.uuid));
                
                // Look up renderD device from /sys/class/drm
                char pci_slot[64];
                snprintf(pci_slot, sizeof(pci_slot), "%04x:%02x:%02x.%d", domain, bus, device, function);
                findRenderDevice(pci_slot, renderDev, sizeof(renderDev));
            } else {
                snprintf(info->basic.uuid, sizeof(info->basic.uuid), "AMD-GPU-%zu", totalDevices);
            }
            
            // Store render device path in properties if found
            if (renderDev[0] != '\0') {
                addProperty(&info->props, "renderDevice", renderDev);
            }
            
            // Set vendor
            snprintf(info->basic.vendor, sizeof(info->basic.vendor), "AMD");
            
            // Get ASIC info for model name
            amdsmi_asic_info_t asic_info;
            memset(&asic_info, 0, sizeof(asic_info));
            status = amdsmi_get_gpu_asic_info(processor, &asic_info);
            if (status == AMDSMI_STATUS_SUCCESS) {
                if (asic_info.market_name[0] != '\0') {
                    snprintf(info->basic.model, sizeof(info->basic.model), "%s", asic_info.market_name);
                } else {
                    snprintf(info->basic.model, sizeof(info->basic.model), "AMD GPU");
                }
                // Use num_of_compute_units from asic_info
                info->basic.totalComputeUnits = asic_info.num_of_compute_units;
            } else {
                snprintf(info->basic.model, sizeof(info->basic.model), "AMD GPU");
            }
            
            // Get VRAM info from AMD SMI
            // AMD SMI returns vram_size in MEGABYTES, convert to bytes
            amdsmi_vram_info_t vram_info;
            memset(&vram_info, 0, sizeof(vram_info));
            status = amdsmi_get_gpu_vram_info(processor, &vram_info);
            if (status == AMDSMI_STATUS_SUCCESS) {
                info->basic.totalMemoryBytes = vram_info.vram_size * 1024ULL * 1024ULL;
            }
            
            // Get driver version
            amdsmi_driver_info_t driver_info;
            memset(&driver_info, 0, sizeof(driver_info));
            status = amdsmi_get_gpu_driver_info(processor, &driver_info);
            if (status == AMDSMI_STATUS_SUCCESS) {
                snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "%s", driver_info.driver_version);
            }
            
            // Get firmware version (VBIOS)
            amdsmi_vbios_info_t vbios_info;
            memset(&vbios_info, 0, sizeof(vbios_info));
            status = amdsmi_get_gpu_vbios_info(processor, &vbios_info);
            if (status == AMDSMI_STATUS_SUCCESS) {
                snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "%s", vbios_info.name);
            }
            
            // Get PCIe info - for MI325X, use known values or query from alternate sources
            // Note: TheRock 7.11 may not expose PCIe gen/width directly in board_info
            // MI325X specifications: PCIe Gen5 x16
            info->basic.pcieGen = 5; 
            info->basic.pcieWidth = 16;
            
            // Set device index
            info->basic.index = totalDevices;
            
            // Get NUMA node
            uint32_t numa_node = 0;
            status = amdsmi_topo_get_numa_node_number(processor, &numa_node);
            if (status == AMDSMI_STATUS_SUCCESS) {
                info->basic.numaNode = numa_node;
            } else {
                info->basic.numaNode = -1;
            }
            
            // Set TFLOPS based on GPU model
            // These are FP16 peak TFLOPS from official AMD specs
            // Model name from AMD SMI is like "AMD Instinct MI325X"
            if (strstr(info->basic.model, "MI325X") != NULL) {
                info->basic.maxTflops = 1307.0; // AMD Instinct MI325X FP16
            } else {
                // Fallback: rough estimate based on compute units
                // Assumes ~4 TFLOPS per CU (very rough approximation)
                info->basic.maxTflops = info->basic.totalComputeUnits * 4.0;
            }
            
            // Set virtualization capabilities
            info->virtualizationCapabilities.supportsPartitioning = false;
            info->virtualizationCapabilities.supportsSoftIsolation = true;
            info->virtualizationCapabilities.supportsHardIsolation = true;
            info->virtualizationCapabilities.supportsSnapshot = false;
            info->virtualizationCapabilities.supportsMetrics = true;
            info->virtualizationCapabilities.supportsRemoting = true;
            info->virtualizationCapabilities.maxPartitions = 0;
            info->virtualizationCapabilities.maxWorkersPerDevice = 32;
            
            totalDevices++;
        }
        
        free(processors);
    }
    
    free(sockets);
    *deviceCount = totalDevices;
    
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "Discovered %zu AMD GPU(s)", totalDevices);
    logMessage("INFO", buffer);
    
    return RESULT_SUCCESS;
}

Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology) {
    if (!deviceIndexArray || !topology || deviceCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    if (!g_initialized) {
        Result initResult = VirtualGPUInit();
        if (initResult != RESULT_SUCCESS) {
            return initResult;
        }
    }
    
    // Get all devices first to map indices to processors
    ExtendedDeviceInfo devices[256];
    size_t totalDevices = 0;
    Result result = GetAllDevices(devices, 256, &totalDevices);
    if (result != RESULT_SUCCESS) {
        return result;
    }
    
    topology->deviceCount = 0;
    snprintf(topology->topologyType, sizeof(topology->topologyType), "PCIe");
    
    for (size_t i = 0; i < deviceCount && i < MAX_TOPOLOGY_DEVICES && i < totalDevices; i++) {
        int32_t idx = deviceIndexArray[i];
        if (idx >= 0 && (size_t)idx < totalDevices) {
            snprintf(topology->devices[topology->deviceCount].deviceUUID, 
                    sizeof(topology->devices[topology->deviceCount].deviceUUID),
                    "%s", devices[idx].basic.uuid);
            topology->devices[topology->deviceCount].numaNode = devices[idx].basic.numaNode;
            topology->deviceCount++;
        }
    }
    
    return RESULT_SUCCESS;
}

// ============================================================================
// Virtualization APIs - Not Supported for AMD (No MIG equivalent)
// ============================================================================

bool AssignPartition(PartitionAssignment* assignment) {
    (void)assignment;
    logMessage("WARN", "AssignPartition not supported for AMD GPUs");
    return false;
}

bool RemovePartition(const char* templateId, const char* deviceUUID) {
    (void)templateId;
    (void)deviceUUID;
    logMessage("WARN", "RemovePartition not supported for AMD GPUs");
    return false;
}

Result SetMemHardLimit(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes) {
    (void)workerId;
    (void)deviceUUID;
    (void)memoryLimitBytes;
    logMessage("WARN", "SetMemHardLimit not yet implemented for AMD GPUs");
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit) {
    (void)workerId;
    (void)deviceUUID;
    (void)computeUnitLimit;
    logMessage("WARN", "SetComputeUnitHardLimit not yet implemented for AMD GPUs");
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result Snapshot(ProcessArray* processes) {
    (void)processes;
    logMessage("WARN", "Snapshot not supported for AMD GPUs");
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result Resume(ProcessArray* processes) {
    (void)processes;
    logMessage("WARN", "Resume not supported for AMD GPUs");
    return RESULT_ERROR_NOT_SUPPORTED;
}

// ============================================================================
// Metrics APIs
// ============================================================================

Result GetProcessInformation(ProcessInformation* processInfos, size_t maxCount, size_t* processInfoCount) {
    if (!processInfos || !processInfoCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    if (!g_initialized) {
        Result initResult = VirtualGPUInit();
        if (initResult != RESULT_SUCCESS) {
            return initResult;
        }
    }
    
    *processInfoCount = 0;
    
    // Get all devices
    uint32_t socket_count = 0;
    amdsmi_status_t status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    amdsmi_socket_handle* sockets = malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) {
        return RESULT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    for (uint32_t socketIdx = 0; socketIdx < socket_count; socketIdx++) {
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles(sockets[socketIdx], &processor_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        
        amdsmi_processor_handle* processors = malloc(processor_count * sizeof(amdsmi_processor_handle));
        if (!processors) {
            continue;
        }
        
        status = amdsmi_get_processor_handles(sockets[socketIdx], &processor_count, processors);
        if (status != AMDSMI_STATUS_SUCCESS) {
            free(processors);
            continue;
        }
        
        for (uint32_t procIdx = 0; procIdx < processor_count; procIdx++) {
            amdsmi_processor_handle processor = processors[procIdx];
            
            // Generate device UUID
            char deviceUUID[64];
            uint64_t bdfid = 0;
            status = amdsmi_get_gpu_bdf_id(processor, &bdfid);
            if (status == AMDSMI_STATUS_SUCCESS) {
                uint32_t domain = (bdfid >> 32) & 0xFFFF;
                uint32_t bus = (bdfid >> 8) & 0xFF;
                uint32_t device = (bdfid >> 3) & 0x1F;
                uint32_t function = bdfid & 0x7;
                generateUUID(domain, bus, device, function, deviceUUID, sizeof(deviceUUID));
            } else {
                snprintf(deviceUUID, sizeof(deviceUUID), "AMD-GPU-%u", procIdx);
            }
            
            // Get process info - API returns list directly
            uint32_t num_processes = 0;
            
            // First call to get count
            status = amdsmi_get_gpu_process_list(processor, &num_processes, NULL);
            if (status == AMDSMI_STATUS_SUCCESS && num_processes > 0) {
                amdsmi_proc_info_t* process_list = malloc(num_processes * sizeof(amdsmi_proc_info_t));
                if (process_list) {
                    // Second call to get actual process list
                    status = amdsmi_get_gpu_process_list(processor, &num_processes, process_list);
                    if (status == AMDSMI_STATUS_SUCCESS) {
                        for (uint32_t i = 0; i < num_processes && *processInfoCount < maxCount; i++) {
                            ProcessInformation* info = &processInfos[*processInfoCount];
                            memset(info, 0, sizeof(ProcessInformation));
                            
                            snprintf(info->processId, sizeof(info->processId), "%u", process_list[i].pid);
                            snprintf(info->deviceUUID, sizeof(info->deviceUUID), "%s", deviceUUID);
                            
                            // Memory information (in bytes)
                            info->memoryUsedBytes = process_list[i].memory_usage.vram_mem;
                            
                            // AMD SMI doesn't provide per-process compute utilization
                            info->computeUtilizationPercent = 0.0;
                            info->activeSMs = 0;
                            info->totalSMs = 0;
                            
                            (*processInfoCount)++;
                        }
                    }
                    free(process_list);
                }
            }
        }
        
        free(processors);
    }
    
    free(sockets);
    return RESULT_SUCCESS;
}

Result GetDeviceMetrics(const char** deviceUUIDs, size_t deviceCount, DeviceMetrics* metrics) {
    if (!deviceUUIDs || !metrics || deviceCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    if (!g_initialized) {
        Result initResult = VirtualGPUInit();
        if (initResult != RESULT_SUCCESS) {
            return initResult;
        }
    }
    
    // Get all devices to map UUIDs to processors
    ExtendedDeviceInfo devices[256];
    size_t totalDevices = 0;
    Result result = GetAllDevices(devices, 256, &totalDevices);
    if (result != RESULT_SUCCESS) {
        return result;
    }
    
    // Get socket and processor handles
    uint32_t socket_count = 0;
    amdsmi_status_t status = amdsmi_get_socket_handles(&socket_count, NULL);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    amdsmi_socket_handle* sockets = malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) {
        return RESULT_ERROR_RESOURCE_EXHAUSTED;
    }
    
    status = amdsmi_get_socket_handles(&socket_count, sockets);
    if (status != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        return RESULT_ERROR_OPERATION_FAILED;
    }
    
    // Build list of all processors
    amdsmi_processor_handle processors[256];
    size_t processorCount = 0;
    
    for (uint32_t socketIdx = 0; socketIdx < socket_count; socketIdx++) {
        uint32_t proc_count = 0;
        status = amdsmi_get_processor_handles(sockets[socketIdx], &proc_count, NULL);
        if (status != AMDSMI_STATUS_SUCCESS) {
            continue;
        }
        
        amdsmi_processor_handle* socket_procs = malloc(proc_count * sizeof(amdsmi_processor_handle));
        if (!socket_procs) {
            continue;
        }
        
        status = amdsmi_get_processor_handles(sockets[socketIdx], &proc_count, socket_procs);
        if (status == AMDSMI_STATUS_SUCCESS) {
            for (uint32_t i = 0; i < proc_count && processorCount < 256; i++) {
                processors[processorCount++] = socket_procs[i];
            }
        }
        
        free(socket_procs);
    }
    
    // Collect metrics for requested devices
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceMetrics* metric = &metrics[i];
        memset(metric, 0, sizeof(DeviceMetrics));
        snprintf(metric->deviceUUID, sizeof(metric->deviceUUID), "%s", deviceUUIDs[i]);
        
        // Find matching device
        size_t deviceIdx = 0;
        for (deviceIdx = 0; deviceIdx < totalDevices; deviceIdx++) {
            if (strcmp(devices[deviceIdx].basic.uuid, deviceUUIDs[i]) == 0) {
                break;
            }
        }
        
        if (deviceIdx >= totalDevices || deviceIdx >= processorCount) {
            continue;
        }
        
        amdsmi_processor_handle processor = processors[deviceIdx];
        
        // Get GPU activity (utilization)
        amdsmi_engine_usage_t usage;
        memset(&usage, 0, sizeof(usage));
        status = amdsmi_get_gpu_activity(processor, &usage);
        if (status == AMDSMI_STATUS_SUCCESS) {
            // gfx_activity is in percentage (0-100)
            metric->utilizationPercent = (uint32_t)usage.gfx_activity;
        }
        
        // Get memory usage
        uint64_t memory_used = 0;
        status = amdsmi_get_gpu_memory_usage(processor, AMDSMI_MEM_TYPE_VRAM, &memory_used);
        if (status == AMDSMI_STATUS_SUCCESS) {
            metric->memoryUsedBytes = memory_used;
        }
        
        // Get power
        amdsmi_power_info_t power_info;
        memset(&power_info, 0, sizeof(power_info));
        status = amdsmi_get_power_info(processor, &power_info);
        if (status == AMDSMI_STATUS_SUCCESS) {
            metric->powerUsageWatts = power_info.current_socket_power / 1000000.0; // Convert from microwatts
        }
        
        // Get temperature
        int64_t temp = 0;
        status = amdsmi_get_temp_metric(processor, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temp);
        if (status == AMDSMI_STATUS_SUCCESS) {
            metric->temperatureCelsius = temp / 1000.0; // Convert from millidegrees
        }
        
        // Get PCIe throughput
        uint64_t sent = 0, received = 0, max_pkt_sz = 0;
        status = amdsmi_get_gpu_pci_throughput(processor, &sent, &received, &max_pkt_sz);
        if (status == AMDSMI_STATUS_SUCCESS) {
            metric->pcieRxBytes = received;
            metric->pcieTxBytes = sent;
        }
        
        metric->extraMetricsCount = 0;
    }
    
    free(sockets);
    return RESULT_SUCCESS;
}

Result GetVendorMountLibs(Mount* mounts, size_t maxCount, size_t* mountCount) {
    if (!mounts || !mountCount || maxCount < 2) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // TheRock installs to /opt/rocm (via symlink) - same as standard ROCm
    snprintf(mounts[0].hostPath, MAX_MOUNT_PATH, "/opt/rocm/lib");
    snprintf(mounts[0].guestPath, MAX_MOUNT_PATH, "/usr/local/rocm/lib");
    
    snprintf(mounts[1].hostPath, MAX_MOUNT_PATH, "/opt/rocm/bin");
    snprintf(mounts[1].guestPath, MAX_MOUNT_PATH, "/usr/local/rocm/bin");
    
    *mountCount = 2;
    return RESULT_SUCCESS;
}

// ============================================================================
// Utility APIs
// ============================================================================

Result RegisterLogCallback(LogCallbackFunc callback) {
    g_logCallback = callback;
    return RESULT_SUCCESS;
}
