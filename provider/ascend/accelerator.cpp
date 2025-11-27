/*
 * Ascend accelerator provider backed by libdcmi.
 *
 * Implements the accelerator.h ABI using the Ascend DCMI interface to
 * enumerate devices and create/destroy vNPU partitions (virXX templates).
 * This path is intended for real hardware. Hard/snapshot remain
 * unimplemented stubs.
 */

#include "../accelerator.h"
#include "dcmi_wrapper.h"
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct TemplateSpec {
    const char* id;
    uint64_t memoryBytes;
    uint64_t computeUnits;
    const char* description;
    bool isDefault;
};

// vir templates from ascend_split.md
const TemplateSpec kTemplates[] = {
    {"vir01", 3ULL * 1024 * 1024 * 1024, 1, "1 AICore, 3GB", true},
    {"vir02", 6ULL * 1024 * 1024 * 1024, 2, "2 AICore, 6GB", false},
    {"vir02_1c", 6ULL * 1024 * 1024 * 1024, 2, "2 AICore, 6GB 1c", false},
    {"vir04", 12ULL * 1024 * 1024 * 1024, 4, "4 AICore, 12GB", false},
    {"vir04_3c", 12ULL * 1024 * 1024 * 1024, 4, "4 AICore, 12GB 3c", false},
    {"vir04_3c_ndvpp", 12ULL * 1024 * 1024 * 1024, 4, "4 AICore, 12GB NDVPP", false},
    {"vir04_4c_dvpp", 12ULL * 1024 * 1024 * 1024, 4, "4 AICore, 12GB DVPP", false},
};

struct PartitionRecord {
    std::string deviceUUID;
    std::string templateId;
    unsigned int vdevId;
    std::string partitionUUID;
};

struct DeviceIdMapping {
    std::string uuid;
    int cardId;
    int deviceId;
};

std::vector<PartitionRecord> gPartitions;
std::vector<DeviceIdMapping> gDeviceMappings;
std::mutex gPartitionMutex;
std::mutex gMappingMutex;
std::atomic<uint64_t> gUUIDSeed{1};

// ---------------------------------------------------------------------------
// libdcmi dynamic loader (avoid link-time arch conflicts)
// ---------------------------------------------------------------------------

void* gDcmiHandle = nullptr;
bool gDcmiLoaded = false;
bool gDcmiTried = false;
std::mutex gDcmiMutex;

using dcmi_get_all_device_count_fn = int (*)(int*);
using dcmi_get_card_id_device_id_from_logicid_fn = int (*)(int*, int*, unsigned int);
using dcmi_get_device_chip_info_v2_fn = int (*)(int, int, struct dcmi_chip_info_v2*);
using dcmi_get_device_memory_info_v3_fn = int (*)(int, int, struct dcmi_get_memory_info_stru*);
using dcmi_get_memory_info_fn = int (*)(int, int, struct dcmi_memory_info_stru*);
using dcmi_create_vdevice_fn = int (*)(int, int, struct dcmi_create_vdev_res_stru*, struct dcmi_create_vdev_out*);
using dcmi_set_destroy_vdevice_fn = int (*)(int, int, unsigned int);
using dcmi_init_fn = int (*)(void);
using dcmi_get_device_power_info_fn = int (*)(int, int, int*);
using dcmi_get_device_temperature_fn = int (*)(int, int, int*);
using dcmi_get_device_utilization_rate_fn = int (*)(int, int, int, unsigned int*);
using dcmi_get_device_pcie_info_fn = int (*)(int, int, void*);
using dcmi_get_topo_info_by_device_id_fn = int (*)(int, int, int, int, int*);
using dcmi_get_hccs_link_bandwidth_info_fn = int (*)(int, int, void*);

dcmi_get_all_device_count_fn p_dcmi_get_all_device_count = nullptr;
dcmi_get_card_id_device_id_from_logicid_fn p_dcmi_get_card_id_device_id_from_logicid = nullptr;
dcmi_get_device_chip_info_v2_fn p_dcmi_get_device_chip_info_v2 = nullptr;
dcmi_get_device_memory_info_v3_fn p_dcmi_get_device_memory_info_v3 = nullptr;
dcmi_get_memory_info_fn p_dcmi_get_memory_info = nullptr;
dcmi_create_vdevice_fn p_dcmi_create_vdevice = nullptr;
dcmi_set_destroy_vdevice_fn p_dcmi_set_destroy_vdevice = nullptr;
dcmi_init_fn p_dcmi_init = nullptr;
dcmi_get_device_power_info_fn p_dcmi_get_device_power_info = nullptr;
dcmi_get_device_temperature_fn p_dcmi_get_device_temperature = nullptr;
dcmi_get_device_utilization_rate_fn p_dcmi_get_device_utilization_rate = nullptr;
dcmi_get_device_pcie_info_fn p_dcmi_get_device_pcie_info = nullptr;
dcmi_get_topo_info_by_device_id_fn p_dcmi_get_topo_info_by_device_id = nullptr;
dcmi_get_hccs_link_bandwidth_info_fn p_dcmi_get_hccs_link_bandwidth_info = nullptr;

template <typename T>
bool loadSym(void* handle, const char* name, T& fn) {
    fn = reinterpret_cast<T>(dlsym(handle, name));
    return fn != nullptr;
}

void parseIdList(const char* env, std::vector<int>& out) {
    if (!env) {
        return;
    }
    const char* ptr = env;
    while (*ptr) {
        while (*ptr && std::isspace(static_cast<unsigned char>(*ptr))) {
            ptr++;
        }
        char* end = nullptr;
        long v = std::strtol(ptr, &end, 10);
        if (end == ptr) {
            break;
        }
        out.push_back(static_cast<int>(v));
        ptr = end;
        if (*ptr == ',') {
            ptr++;
        }
    }
}

// Force reload DCMI (for testing/debugging when environment changes)
void resetDcmiLoader() {
    std::lock_guard<std::mutex> lock(gDcmiMutex);
    if (gDcmiHandle) {
        dlclose(gDcmiHandle);
        gDcmiHandle = nullptr;
    }
    gDcmiTried = false;
    gDcmiLoaded = false;
}

bool ensureDcmiLoaded() {
    std::lock_guard<std::mutex> lock(gDcmiMutex);
    if (gDcmiTried) {
        return gDcmiLoaded;
    }
    gDcmiTried = true;

    const char* envPath = std::getenv("DCMI_LIB_PATH");
    const char* candidates[] = {
        envPath,
        "/usr/local/dcmi/libdcmi.so",
        "/usr/local/Ascend/driver/lib64/libdcmi.so",
        "/usr/lib64/libdcmi.so",
        nullptr,
    };

    std::fprintf(stderr, "[ascend] attempting to load dcmi\n");
    std::fprintf(stderr, "[ascend] DCMI_LIB_PATH env: %s\n", envPath ? envPath : "(not set)");
    std::fflush(stderr);  // Force flush to see output immediately

    for (size_t i = 0; candidates[i] != nullptr; ++i) {
        const char* libPath = candidates[i];
        if (!libPath) {
            std::fprintf(stderr, "[ascend] candidate[%zu] is null, skipping\n", i);
            continue;
        }
        std::fprintf(stderr, "[ascend] trying dlopen(%s)...\n", libPath);
        std::fflush(stderr);

        gDcmiHandle = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
        if (gDcmiHandle) {
            std::fprintf(stderr, "[ascend] loaded dcmi from %s\n", libPath);
            break;
        } else {
            const char* err = dlerror();
            std::fprintf(stderr, "[ascend] dlopen failed for %s: %s\n", libPath, err ? err : "unknown");
            std::fflush(stderr);
        }
    }

    if (!gDcmiHandle) {
        std::fprintf(stderr, "[ascend] dcmi handle remains null after dlopen attempts\n");
        return false;
    }

    // Load required symbols for device discovery and metrics
    // Don't fail if partition-related symbols (create/destroy_vdevice) are missing
    bool ok = true;
    ok &= loadSym(gDcmiHandle, "dcmi_get_all_device_count", p_dcmi_get_all_device_count);
    ok &= loadSym(gDcmiHandle, "dcmi_get_card_id_device_id_from_logicid", p_dcmi_get_card_id_device_id_from_logicid);
    ok &= loadSym(gDcmiHandle, "dcmi_get_device_chip_info_v2", p_dcmi_get_device_chip_info_v2);
    ok &= loadSym(gDcmiHandle, "dcmi_get_device_memory_info_v3", p_dcmi_get_device_memory_info_v3);
    ok &= loadSym(gDcmiHandle, "dcmi_get_memory_info", p_dcmi_get_memory_info);
    ok &= loadSym(gDcmiHandle, "dcmi_init", p_dcmi_init);

    // Optional partition management APIs (don't fail if missing)
    loadSym(gDcmiHandle, "dcmi_create_vdevice", p_dcmi_create_vdevice);
    loadSym(gDcmiHandle, "dcmi_set_destroy_vdevice", p_dcmi_set_destroy_vdevice);

    // Optional metrics APIs (don't fail if not available)
    loadSym(gDcmiHandle, "dcmi_get_device_power_info", p_dcmi_get_device_power_info);
    loadSym(gDcmiHandle, "dcmi_get_device_temperature", p_dcmi_get_device_temperature);
    loadSym(gDcmiHandle, "dcmi_get_device_utilization_rate", p_dcmi_get_device_utilization_rate);
    loadSym(gDcmiHandle, "dcmi_get_device_pcie_info", p_dcmi_get_device_pcie_info);

    // Optional topology APIs (don't fail if not available)
    loadSym(gDcmiHandle, "dcmi_get_topo_info_by_device_id", p_dcmi_get_topo_info_by_device_id);
    loadSym(gDcmiHandle, "dcmi_get_hccs_link_bandwidth_info", p_dcmi_get_hccs_link_bandwidth_info);

    if (ok && p_dcmi_init) {
        int initRet = p_dcmi_init();
        if (initRet != 0) {
            std::fprintf(stderr, "[ascend] dcmi_init failed ret=%d\n", initRet);
            ok = false;
            dlclose(gDcmiHandle);
            gDcmiHandle = nullptr;
        }
    }

    gDcmiLoaded = ok;
    if (!ok) {
        dlclose(gDcmiHandle);
        gDcmiHandle = nullptr;
    }
    return gDcmiLoaded;
}

const TemplateSpec* findTemplate(const char* id) {
    if (!id) {
        return nullptr;
    }
    for (const auto& t : kTemplates) {
        if (std::strncmp(t.id, id, 63) == 0) {
            return &t;
        }
    }
    return nullptr;
}

std::string makePartitionUUID(const std::string& deviceUUID, unsigned int vdevId) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s-vnpu-%u", deviceUUID.c_str(), vdevId);
    return std::string(buf);
}

void fillTemplate(const TemplateSpec& spec, PartitionTemplate* out) {
    std::snprintf(out->templateId, sizeof(out->templateId), "%s", spec.id);
    std::snprintf(out->name, sizeof(out->name), "%s", spec.id);
    out->memoryBytes = spec.memoryBytes;
    out->computeUnits = spec.computeUnits;
    out->tflops = 0.0;
    out->sliceCount = 1;
    out->isDefault = spec.isDefault;
    std::snprintf(out->description, sizeof(out->description), "%s", spec.description);
}

} // namespace

extern "C" {

Result GetDeviceCount(size_t* deviceCount) {
    if (!deviceCount) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    if (!ensureDcmiLoaded()) {
        std::fprintf(stderr, "[ascend] dcmi not loaded, returning 0 devices\n");
        *deviceCount = 0;
        return RESULT_SUCCESS;
    }
    int count = 0;
    int ret = p_dcmi_get_all_device_count(&count);
    std::fprintf(stderr, "[ascend] dcmi_get_all_device_count ret=%d count=%d\n", ret, count);
    if (ret != 0) {
        std::fprintf(stderr, "[ascend] dcmi_get_all_device_count ret=%d\n", ret);
        *deviceCount = 0;
        return RESULT_ERROR_OPERATION_FAILED;
    }
    if (count < 0) {
        count = 0;
    }
    *deviceCount = static_cast<size_t>(count);
    if (*deviceCount == 0) {
        std::fprintf(stderr, "[ascend] no devices discovered via dcmi logic-id mapping\n");
    }
    return RESULT_SUCCESS;
}

Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (!devices || !deviceCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // Reset deviceCount to avoid caller's stale value causing issues
    *deviceCount = 0;

    // Clear stale device mappings to avoid hot-plug/re-enumeration issues
    {
        std::lock_guard<std::mutex> lock(gMappingMutex);
        gDeviceMappings.clear();
    }

    size_t total = 0;
    Result rc = GetDeviceCount(&total);
    if (rc != RESULT_SUCCESS) {
        return rc;
    }
    if (!ensureDcmiLoaded()) {
        std::fprintf(stderr, "[ascend] dcmi not loaded in GetAllDevices\n");
        return RESULT_SUCCESS;
    }

    const int maxLogic = 256;
    size_t filled = 0;
    std::vector<int> logicIds;
    parseIdList(std::getenv("ASCEND_LOGIC_IDS"), logicIds);
    if (logicIds.empty()) {
        // Default: probe more logic IDs to handle sparse/non-contiguous IDs
        // Use total*16 to be safe, or max 64 to limit noise
        int probeLimit = std::min(maxLogic, std::max(static_cast<int>(total * 16), 64));
        for (int logic = 0; logic < probeLimit; ++logic) {
            logicIds.push_back(logic);
        }
    }

    for (size_t idx = 0; idx < logicIds.size() && filled < maxCount; ++idx) {
        int logic = logicIds[idx];
        int card = 0;
        int dev = 0;
        int mapRet = p_dcmi_get_card_id_device_id_from_logicid(&card, &dev, static_cast<unsigned int>(logic));
        if (mapRet != 0) {
            // Only log first few failures to reduce noise
            if (idx < 5 || (logicIds.size() > 0 && idx == logicIds.size() - 1)) {
                std::fprintf(stderr, "[ascend] map logic=%d failed ret=%d\n", logic, mapRet);
            }
            continue;
        }

        ExtendedDeviceInfo* info = &devices[*deviceCount];
        std::memset(info, 0, sizeof(ExtendedDeviceInfo));

        dcmi_chip_info_v2 chipInfo{};
        if (p_dcmi_get_device_chip_info_v2(card, dev, &chipInfo) != 0) {
            std::fprintf(stderr, "[ascend] skip card=%d device=%d chip_info failed\n", card, dev);
            continue;
        }

        dcmi_get_memory_info_stru memInfo{};
        uint64_t totalMem = 0;
        if (p_dcmi_get_device_memory_info_v3(card, dev, &memInfo) == 0) {
            totalMem = memInfo.memory_size * 1024ULL * 1024ULL; // MB -> bytes
        } else if (p_dcmi_get_memory_info(card, dev, reinterpret_cast<dcmi_memory_info_stru*>(&memInfo)) == 0) {
            totalMem = memInfo.memory_size * 1024ULL * 1024ULL;
        } else {
            std::fprintf(stderr, "[ascend] card=%d dev=%d mem_info failed\n", card, dev);
        }

        std::snprintf(info->basic.uuid, sizeof(info->basic.uuid), "npu-%zu-chip-%d", filled, dev);

        // Store UUID to card/device mapping for metrics queries
        {
            std::lock_guard<std::mutex> lock(gMappingMutex);
            DeviceIdMapping mapping;
            mapping.uuid = info->basic.uuid;
            mapping.cardId = card;
            mapping.deviceId = dev;
            gDeviceMappings.push_back(mapping);
        }

        std::snprintf(info->basic.vendor, sizeof(info->basic.vendor), "Huawei");
        std::snprintf(info->basic.model, sizeof(info->basic.model), "%s", chipInfo.npu_name);
        std::snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "dcmi");
        std::snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "unknown");
        info->basic.index = static_cast<int32_t>(filled);
        info->basic.numaNode = -1;
        info->basic.totalMemoryBytes = totalMem;
        info->basic.totalComputeUnits = chipInfo.aicore_cnt;
        info->basic.maxTflops = 0.0;

        // Try to get PCIe info
        info->basic.pcieGen = 0;
        info->basic.pcieWidth = 0;
        if (p_dcmi_get_device_pcie_info) {
            struct {
                unsigned int deviceid;
                unsigned int venderid;
                unsigned int subvenderid;
                unsigned int subdeviceid;
                unsigned int bdf_deviceid;
                unsigned int bdf_busid;
                unsigned int bdf_funcid;
            } pcieInfo{};
            if (p_dcmi_get_device_pcie_info(card, dev, &pcieInfo) == 0) {
                // PCIe gen/width not directly available from basic pcie_info
                // These would need additional DCMI calls or parsing
                info->basic.pcieGen = 0;  // TODO: get from dcmi_get_device_pcie_info_v2
                info->basic.pcieWidth = 0;
            }
        }

        info->props.clockGraphics = 0;
        info->props.clockSM = 0;
        info->props.clockMem = 0;
        info->props.clockAI = 0;
        info->props.powerLimit = 0;
        info->props.temperatureThreshold = 0;
        info->props.eccEnabled = false;
        info->props.persistenceModeEnabled = false;
        std::snprintf(info->props.computeCapability, sizeof(info->props.computeCapability), "%.15s", chipInfo.chip_name);
        std::snprintf(info->props.chipType, sizeof(info->props.chipType), "Ascend");

        info->capabilities.supportsPartitioning = true;
        info->capabilities.supportsSoftIsolation = false;
        info->capabilities.supportsHardIsolation = false;
        info->capabilities.supportsSnapshot = false;
        info->capabilities.supportsMetrics = true;
        info->capabilities.maxPartitions = 32;
        info->capabilities.maxWorkersPerDevice = 32;

        info->relatedDevices = NULL;
        info->relatedDeviceCount = 0;

        (*deviceCount)++;
        filled++;
    }

    if (*deviceCount == 0) {
        // Fallback: scan cards/devices directly if mapping failed.
        std::vector<int> cardIds;
        parseIdList(std::getenv("ASCEND_CARD_IDS"), cardIds);
        if (cardIds.empty()) {
            for (int c = 0; c < 8; ++c) {
                cardIds.push_back(c);
            }
        }
        for (size_t ci = 0; ci < cardIds.size() && filled < maxCount; ++ci) {
            int card = cardIds[ci];
            for (int dev = 0; dev < 4 && filled < maxCount; ++dev) {
                dcmi_chip_info_v2 chipInfo{};
                int chipRc = p_dcmi_get_device_chip_info_v2(card, dev, &chipInfo);
                if (chipRc != 0) {
                    std::fprintf(stderr, "[ascend] fallback chip_info card=%d dev=%d rc=%d\n", card, dev, chipRc);
                    continue;
                }

                dcmi_get_memory_info_stru memInfo{};
                uint64_t totalMem = 0;
                if (p_dcmi_get_device_memory_info_v3(card, dev, &memInfo) == 0) {
                    totalMem = memInfo.memory_size * 1024ULL * 1024ULL; // MB -> bytes
                } else if (p_dcmi_get_memory_info(card, dev, reinterpret_cast<dcmi_memory_info_stru*>(&memInfo)) == 0) {
                    totalMem = memInfo.memory_size * 1024ULL * 1024ULL;
                } else {
                    std::fprintf(stderr, "[ascend] fallback mem_info card=%d dev=%d failed\n", card, dev);
                }

                ExtendedDeviceInfo* info = &devices[*deviceCount];
                std::memset(info, 0, sizeof(ExtendedDeviceInfo));
                std::snprintf(info->basic.uuid, sizeof(info->basic.uuid), "npu-%zu-card-%d-dev-%d", filled, card, dev);

                // Store UUID to card/device mapping for metrics queries
                {
                    std::lock_guard<std::mutex> lock(gMappingMutex);
                    DeviceIdMapping mapping;
                    mapping.uuid = info->basic.uuid;
                    mapping.cardId = card;
                    mapping.deviceId = dev;
                    gDeviceMappings.push_back(mapping);
                }

                std::snprintf(info->basic.vendor, sizeof(info->basic.vendor), "Huawei");
                std::snprintf(info->basic.model, sizeof(info->basic.model), "%s", chipInfo.npu_name);
                std::snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "dcmi");
                std::snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "unknown");
                info->basic.index = static_cast<int32_t>(filled);
                info->basic.numaNode = -1;
                info->basic.totalMemoryBytes = totalMem;
                info->basic.totalComputeUnits = chipInfo.aicore_cnt;
                info->basic.maxTflops = 0.0;
                info->basic.pcieGen = 0;
                info->basic.pcieWidth = 0;

                info->props.clockGraphics = 0;
                info->props.clockSM = 0;
                info->props.clockMem = 0;
                info->props.clockAI = 0;
                info->props.powerLimit = 0;
                info->props.temperatureThreshold = 0;
                info->props.eccEnabled = false;
                info->props.persistenceModeEnabled = false;
                std::snprintf(info->props.computeCapability, sizeof(info->props.computeCapability), "%.15s", chipInfo.chip_name);
                std::snprintf(info->props.chipType, sizeof(info->props.chipType), "Ascend");

                info->capabilities.supportsPartitioning = true;
                info->capabilities.supportsSoftIsolation = false;
                info->capabilities.supportsHardIsolation = false;
                info->capabilities.supportsSnapshot = false;
                info->capabilities.supportsMetrics = true;
                info->capabilities.maxPartitions = 32;
                info->capabilities.maxWorkersPerDevice = 32;

                info->relatedDevices = NULL;
                info->relatedDeviceCount = 0;

                (*deviceCount)++;
                filled++;
            }
        }
    }

    if (*deviceCount == 0) {
        std::fprintf(stderr, "[ascend] no devices discovered via dcmi logic-id mapping\n");
    }
    return RESULT_SUCCESS;
}

Result GetPartitionTemplates(int32_t deviceIndex, PartitionTemplate* templates, size_t maxCount, size_t* templateCount) {
    if (!templates || !templateCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    // deviceIndex is used for future chip-specific filtering; currently all templates returned.
    (void)deviceIndex;
    size_t available = sizeof(kTemplates) / sizeof(kTemplates[0]);
    size_t count = std::min(maxCount, available);
    for (size_t i = 0; i < count; ++i) {
        fillTemplate(kTemplates[i], &templates[i]);
    }
    *templateCount = count;
    return RESULT_SUCCESS;
}

Result GetDeviceTopology(int32_t* deviceIndexArray, size_t deviceCount, ExtendedDeviceTopology* topology, size_t maxConnectionsPerDevice) {
    if (!deviceIndexArray || deviceCount == 0 || !topology || maxConnectionsPerDevice == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    if (!topology->devices) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    if (!ensureDcmiLoaded()) {
        return RESULT_ERROR_INTERNAL;
    }

    // Retrieve UUIDs and card/device mappings
    std::lock_guard<std::mutex> lock(gMappingMutex);

    topology->deviceCount = deviceCount;
    topology->nvlinkBandwidthMBps = 0;
    topology->ibNicCount = 0;
    std::snprintf(topology->topologyType, sizeof(topology->topologyType), "HCCS");

    // For each device, populate UUID and detect connections
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceTopology* dt = &topology->devices[i];
        int32_t devIdx = deviceIndexArray[i];

        // Set UUID to align with GetAllDevices
        if (static_cast<size_t>(devIdx) < gDeviceMappings.size()) {
            std::snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "%s",
                         gDeviceMappings[devIdx].uuid.c_str());
        } else {
            std::snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "npu-%d-chip-0", devIdx);
        }

        dt->numaNode = -1;
        dt->connectionCount = 0;
        dt->connections = nullptr;

        // If topology API unavailable, skip connection detection
        if (!p_dcmi_get_topo_info_by_device_id) {
            continue;
        }

        // Get card_id and device_id for this device
        if (static_cast<size_t>(devIdx) >= gDeviceMappings.size()) {
            continue;
        }
        int card1 = gDeviceMappings[devIdx].cardId;
        int dev1 = gDeviceMappings[devIdx].deviceId;

        // Probe connections to all other devices
        std::vector<RelatedDevice> connections;
        for (size_t j = 0; j < deviceCount; j++) {
            if (i == j) continue;  // Skip self

            int32_t otherIdx = deviceIndexArray[j];
            if (static_cast<size_t>(otherIdx) >= gDeviceMappings.size()) {
                continue;
            }
            int card2 = gDeviceMappings[otherIdx].cardId;
            int dev2 = gDeviceMappings[otherIdx].deviceId;

            // Query topology relationship
            int topoType = 0;
            int ret = p_dcmi_get_topo_info_by_device_id(card1, dev1, card2, dev2, &topoType);
            if (ret != 0) {
                continue;  // No connection or query failed
            }

            // Only include HCCS connections (type 3 and 7)
            // Type 3: DCMI_TOPO_TYPE_HCCS (direct HCCS link)
            // Type 7: DCMI_TOPO_TYPE_HCCS_SW (HCCS via switch)
            if (topoType != 3 && topoType != 7) {
                continue;
            }

            // Create connection entry
            RelatedDevice conn;
            if (static_cast<size_t>(otherIdx) < gDeviceMappings.size()) {
                std::snprintf(conn.deviceUUID, sizeof(conn.deviceUUID), "%s",
                             gDeviceMappings[otherIdx].uuid.c_str());
            } else {
                std::snprintf(conn.deviceUUID, sizeof(conn.deviceUUID), "npu-%d-chip-0", otherIdx);
            }

            if (topoType == 7) {
                std::snprintf(conn.connectionType, sizeof(conn.connectionType), "HCCS-SW");
            } else {
                std::snprintf(conn.connectionType, sizeof(conn.connectionType), "HCCS");
            }

            // TODO: Get actual bandwidth using dcmi_get_hccs_link_bandwidth_info
            // For now, use default HCCS bandwidth (assume 400 Gbps = 50000 MB/s)
            conn.bandwidthMBps = 50000;
            conn.latencyNs = 0;  // Unknown

            connections.push_back(conn);

            if (connections.size() >= maxConnectionsPerDevice) {
                break;  // Respect caller's buffer limit
            }
        }

        // Allocate and populate connections array if any found
        if (!connections.empty()) {
            dt->connections = new RelatedDevice[connections.size()];
            dt->connectionCount = connections.size();
            std::memcpy(dt->connections, connections.data(), connections.size() * sizeof(RelatedDevice));
        }
    }

    return RESULT_SUCCESS;
}

bool AssignPartition(PartitionAssignment* assignment) {
    if (!assignment || assignment->templateId[0] == '\0' || assignment->deviceUUID[0] == '\0') {
        return false;
    }
    if (assignment->partitionUUID[0] != '\0') {
        // Caller provided partition ID; honor it and short-circuit.
        assignment->partitionOverheadBytes = 0;
        return true;
    }
    const TemplateSpec* tmpl = findTemplate(assignment->templateId);
    if (!tmpl) {
        return false;
    }

    // Software-only tracking: no dcmi create_vdevice call. Partition UUID is synthetic.
    {
        std::lock_guard<std::mutex> lock(gPartitionMutex);
        for (const auto& rec : gPartitions) {
            if (rec.deviceUUID == assignment->deviceUUID && rec.templateId == assignment->templateId) {
                std::snprintf(assignment->partitionUUID, sizeof(assignment->partitionUUID), "%s", rec.partitionUUID.c_str());
                assignment->partitionOverheadBytes = 0;
                return true;
            }
        }
    }

    unsigned int syntheticId = static_cast<unsigned int>(gUUIDSeed.fetch_add(1));

    PartitionRecord rec;
    rec.deviceUUID = assignment->deviceUUID;
    rec.templateId = assignment->templateId;
    rec.vdevId = syntheticId;
    rec.partitionUUID = makePartitionUUID(rec.deviceUUID, syntheticId);

    {
        std::lock_guard<std::mutex> lock(gPartitionMutex);
        gPartitions.push_back(rec);
    }

    std::snprintf(assignment->partitionUUID, sizeof(assignment->partitionUUID), "%s", rec.partitionUUID.c_str());
    assignment->partitionOverheadBytes = 0;
    return true;
}

bool RemovePartition(const char* templateId, const char* deviceUUID) {
    if (!templateId || !deviceUUID) {
        return false;
    }
    // Pure software tracking: remove bookkeeping only.
    std::lock_guard<std::mutex> lock(gPartitionMutex);
    for (auto it = gPartitions.begin(); it != gPartitions.end(); ++it) {
        if (it->deviceUUID == deviceUUID && it->templateId == templateId) {
            gPartitions.erase(it);
            return true;
        }
    }
    return true;
}

Result SetMemHardLimit(const char* workerId, const char* deviceUUID, uint64_t memoryLimitBytes) {
    (void)workerId;
    (void)deviceUUID;
    (void)memoryLimitBytes;
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result SetComputeUnitHardLimit(const char* workerId, const char* deviceUUID, uint32_t computeUnitLimit) {
    (void)workerId;
    (void)deviceUUID;
    (void)computeUnitLimit;
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result Snapshot(ProcessArray* processes) {
    if (!processes || !processes->processIds || processes->processCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result Resume(ProcessArray* processes) {
    if (!processes || !processes->processIds || processes->processCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    return RESULT_ERROR_NOT_SUPPORTED;
}

Result GetProcessComputeUtilization(ComputeUtilization* utilizations, size_t maxCount, size_t* utilizationCount) {
    if (!utilizations || !utilizationCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    *utilizationCount = 0;
    return RESULT_SUCCESS;
}

Result GetProcessMemoryUtilization(MemoryUtilization* utilizations, size_t maxCount, size_t* utilizationCount) {
    if (!utilizations || !utilizationCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    *utilizationCount = 0;
    return RESULT_SUCCESS;
}

Result GetDeviceMetrics(const char** deviceUUIDArray, size_t deviceCount, DeviceMetrics* metrics) {
    if (!deviceUUIDArray || deviceCount == 0 || !metrics) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    if (!ensureDcmiLoaded()) {
        // Return zeroed metrics if DCMI not available
        for (size_t i = 0; i < deviceCount; i++) {
            std::snprintf(metrics[i].deviceUUID, sizeof(metrics[i].deviceUUID), "%s", deviceUUIDArray[i]);
            metrics[i].powerUsageWatts = 0;
            metrics[i].temperatureCelsius = 0;
            metrics[i].pcieRxBytes = 0;
            metrics[i].pcieTxBytes = 0;
            metrics[i].smActivePercent = 0;
            metrics[i].tensorCoreUsagePercent = 0;
            metrics[i].memoryUsedBytes = 0;
            metrics[i].memoryTotalBytes = 0;
        }
        return RESULT_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(gMappingMutex);

    for (size_t i = 0; i < deviceCount; i++) {
        std::snprintf(metrics[i].deviceUUID, sizeof(metrics[i].deviceUUID), "%s", deviceUUIDArray[i]);

        // Find card_id and device_id from UUID
        int card = -1;
        int dev = -1;
        for (const auto& mapping : gDeviceMappings) {
            if (mapping.uuid == deviceUUIDArray[i]) {
                card = mapping.cardId;
                dev = mapping.deviceId;
                break;
            }
        }

        if (card == -1 || dev == -1) {
            // UUID not found, return zeros
            metrics[i].powerUsageWatts = 0;
            metrics[i].temperatureCelsius = 0;
            metrics[i].pcieRxBytes = 0;
            metrics[i].pcieTxBytes = 0;
            metrics[i].smActivePercent = 0;
            metrics[i].tensorCoreUsagePercent = 0;
            metrics[i].memoryUsedBytes = 0;
            metrics[i].memoryTotalBytes = 0;
            continue;
        }

        // Get power info (in mW, convert to W)
        int powerMw = 0;
        if (p_dcmi_get_device_power_info && p_dcmi_get_device_power_info(card, dev, &powerMw) == 0) {
            metrics[i].powerUsageWatts = powerMw / 1000.0;
        } else {
            metrics[i].powerUsageWatts = 0;
        }

        // Get temperature (in °C)
        int tempC = 0;
        if (p_dcmi_get_device_temperature && p_dcmi_get_device_temperature(card, dev, &tempC) == 0) {
            metrics[i].temperatureCelsius = static_cast<double>(tempC);
        } else {
            metrics[i].temperatureCelsius = 0;
        }

        // Get utilization rate
        // Type 0 (AICore) may not be supported on some models (e.g., 310P3)
        // Try type 1 (AICpu) as fallback, then type 3 (VectorCore)
        unsigned int util = 0;
        if (p_dcmi_get_device_utilization_rate) {
            // Try AICore first (type 0)
            if (p_dcmi_get_device_utilization_rate(card, dev, 0, &util) == 0) {
                metrics[i].smActivePercent = util;
                metrics[i].tensorCoreUsagePercent = util;
            }
            // Fallback to AICpu (type 1)
            else if (p_dcmi_get_device_utilization_rate(card, dev, 1, &util) == 0) {
                metrics[i].smActivePercent = util;
                metrics[i].tensorCoreUsagePercent = util;
            }
            // Last resort: VectorCore (type 3)
            else if (p_dcmi_get_device_utilization_rate(card, dev, 3, &util) == 0) {
                metrics[i].smActivePercent = util;
                metrics[i].tensorCoreUsagePercent = util;
            } else {
                metrics[i].smActivePercent = 0;
                metrics[i].tensorCoreUsagePercent = 0;
            }
        } else {
            metrics[i].smActivePercent = 0;
            metrics[i].tensorCoreUsagePercent = 0;
        }

        // Get memory info
        dcmi_memory_info_stru memInfo{};
        if (p_dcmi_get_memory_info && p_dcmi_get_memory_info(card, dev, &memInfo) == 0) {
            metrics[i].memoryTotalBytes = memInfo.memory_size * 1024ULL * 1024ULL;  // MB to bytes
            // memory utilization is a percentage (0-100)
            metrics[i].memoryUsedBytes = (metrics[i].memoryTotalBytes * memInfo.utiliza) / 100;
        } else {
            metrics[i].memoryUsedBytes = 0;
            metrics[i].memoryTotalBytes = 0;
        }

        // PCIe RX/TX bytes not directly available from DCMI
        metrics[i].pcieRxBytes = 0;
        metrics[i].pcieTxBytes = 0;
    }

    return RESULT_SUCCESS;
}

Result GetExtendedDeviceMetrics(const char** deviceUUIDArray, size_t deviceCount, ExtendedDeviceMetrics* metrics, size_t maxNvlinkPerDevice, size_t maxIbNicPerDevice, size_t maxPciePerDevice) {
    if (!deviceUUIDArray || deviceCount == 0 || !metrics || maxNvlinkPerDevice == 0 || maxIbNicPerDevice == 0 || maxPciePerDevice == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < deviceCount; i++) {
        std::snprintf(metrics[i].deviceUUID, sizeof(metrics[i].deviceUUID), "%s", deviceUUIDArray[i]);
        metrics[i].nvlinkCount = 0;
        metrics[i].nvlinkBandwidthMBps = nullptr;
        metrics[i].ibNicCount = 0;
        metrics[i].ibNicBandwidthMBps = nullptr;
        metrics[i].pcieLinkCount = 0;
        metrics[i].pcieBandwidthMBps = nullptr;
    }
    return RESULT_SUCCESS;
}

Result Log(const char* level, const char* message) {
    if (!level || !message) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    std::fprintf(stderr, "[%s] %s\n", level, message);
    return RESULT_SUCCESS;
}

} // extern "C"
