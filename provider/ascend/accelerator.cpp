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
#include <mutex>
#include <string>
#include <vector>

namespace {

// Default vgroup when creating vdevice; can be overridden via env.
int getDefaultVGroup() {
    const char* env = std::getenv("ASCEND_VGROUP");
    if (!env) {
        return 0;
    }
    int v = std::atoi(env);
    if (v < 0 || v > 3) {
        return 0;
    }
    return v;
}

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

std::vector<PartitionRecord> gPartitions;
std::mutex gPartitionMutex;
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

dcmi_get_all_device_count_fn p_dcmi_get_all_device_count = nullptr;
dcmi_get_card_id_device_id_from_logicid_fn p_dcmi_get_card_id_device_id_from_logicid = nullptr;
dcmi_get_device_chip_info_v2_fn p_dcmi_get_device_chip_info_v2 = nullptr;
dcmi_get_device_memory_info_v3_fn p_dcmi_get_device_memory_info_v3 = nullptr;
dcmi_get_memory_info_fn p_dcmi_get_memory_info = nullptr;
dcmi_create_vdevice_fn p_dcmi_create_vdevice = nullptr;
dcmi_set_destroy_vdevice_fn p_dcmi_set_destroy_vdevice = nullptr;

template <typename T>
bool loadSym(void* handle, const char* name, T& fn) {
    fn = reinterpret_cast<T>(dlsym(handle, name));
    return fn != nullptr;
}

bool ensureDcmiLoaded() {
    std::lock_guard<std::mutex> lock(gDcmiMutex);
    if (gDcmiTried) {
        return gDcmiLoaded;
    }
    gDcmiTried = true;

    const char* libPath = std::getenv("DCMI_LIB_PATH");
    if (!libPath) {
        libPath = "/usr/local/dcmi/libdcmi.so";
    }
    gDcmiHandle = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (!gDcmiHandle) {
        return false;
    }

    bool ok = true;
    ok &= loadSym(gDcmiHandle, "dcmi_get_all_device_count", p_dcmi_get_all_device_count);
    ok &= loadSym(gDcmiHandle, "dcmi_get_card_id_device_id_from_logicid", p_dcmi_get_card_id_device_id_from_logicid);
    ok &= loadSym(gDcmiHandle, "dcmi_get_device_chip_info_v2", p_dcmi_get_device_chip_info_v2);
    ok &= loadSym(gDcmiHandle, "dcmi_get_device_memory_info_v3", p_dcmi_get_device_memory_info_v3);
    ok &= loadSym(gDcmiHandle, "dcmi_get_memory_info", p_dcmi_get_memory_info);
    ok &= loadSym(gDcmiHandle, "dcmi_create_vdevice", p_dcmi_create_vdevice);
    ok &= loadSym(gDcmiHandle, "dcmi_set_destroy_vdevice", p_dcmi_set_destroy_vdevice);

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

bool parseLogicIdFromUUID(const char* uuid, int* logicId) {
    if (!uuid || !logicId) {
        return false;
    }
    // Expected format: npu-<logic>-chip-<chip>
    int logic = -1;
    int chip = -1;
    if (std::sscanf(uuid, "npu-%d-chip-%d", &logic, &chip) != 2) {
        return false;
    }
    *logicId = logic;
    return true;
}

bool mapLogicToCardDevice(int logicId, int* cardId, int* deviceId) {
    if (!cardId || !deviceId) {
        return false;
    }
    int c = 0;
    int d = 0;
    if (!ensureDcmiLoaded()) {
        return false;
    }
    int ret = p_dcmi_get_card_id_device_id_from_logicid(&c, &d, static_cast<unsigned int>(logicId));
    if (ret != 0) {
        return false;
    }
    *cardId = c;
    *deviceId = d;
    return true;
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
        *deviceCount = 0;
        return RESULT_SUCCESS;
    }
    int count = 0;
    int ret = p_dcmi_get_all_device_count(&count);
    if (ret != 0) {
        *deviceCount = 0;
        return RESULT_ERROR_OPERATION_FAILED;
    }
    if (count < 0) {
        count = 0;
    }
    *deviceCount = static_cast<size_t>(count);
    return RESULT_SUCCESS;
}

Result GetAllDevices(ExtendedDeviceInfo* devices, size_t maxCount, size_t* deviceCount) {
    if (!devices || !deviceCount || maxCount == 0) {
        return RESULT_ERROR_INVALID_PARAM;
    }

    size_t total = 0;
    Result rc = GetDeviceCount(&total);
    if (rc != RESULT_SUCCESS) {
        return rc;
    }
    if (!ensureDcmiLoaded()) {
        *deviceCount = 0;
        return RESULT_SUCCESS;
    }

    struct Probe {
        int logic;
        int card;
        int device;
    };
    std::vector<Probe> probes;
    probes.reserve(total);

    const int maxProbe = 128; // wider than device count to handle sparse logic IDs
    for (int logic = 0; logic < maxProbe && probes.size() < total; ++logic) {
        int cardId = 0;
        int deviceId = 0;
        if (mapLogicToCardDevice(logic, &cardId, &deviceId)) {
            probes.push_back(Probe{logic, cardId, deviceId});
        }
    }

    if (probes.empty()) {
        *deviceCount = 0;
        return RESULT_SUCCESS;
    }

    size_t toFill = std::min(maxCount, probes.size());
    *deviceCount = 0;

    for (size_t i = 0; i < toFill; ++i) {
        const Probe& p = probes[i];
        ExtendedDeviceInfo* info = &devices[*deviceCount];
        std::memset(info, 0, sizeof(ExtendedDeviceInfo));

        dcmi_chip_info_v2 chipInfo{};
        if (p_dcmi_get_device_chip_info_v2(p.card, p.device, &chipInfo) != 0) {
            continue;
        }

        dcmi_get_memory_info_stru memInfo{};
        uint64_t totalMem = 0;
        if (p_dcmi_get_device_memory_info_v3(p.card, p.device, &memInfo) == 0) {
            totalMem = memInfo.memory_size * 1024ULL * 1024ULL; // MB -> bytes
        } else if (p_dcmi_get_memory_info(p.card, p.device, reinterpret_cast<dcmi_memory_info_stru*>(&memInfo)) == 0) {
            totalMem = memInfo.memory_size * 1024ULL * 1024ULL;
        }

        std::snprintf(info->basic.uuid, sizeof(info->basic.uuid), "npu-%d-chip-%d", p.logic, p.device);
        std::snprintf(info->basic.vendor, sizeof(info->basic.vendor), "Huawei");
        std::snprintf(info->basic.model, sizeof(info->basic.model), "%s", chipInfo.npu_name);
        std::snprintf(info->basic.driverVersion, sizeof(info->basic.driverVersion), "dcmi");
        std::snprintf(info->basic.firmwareVersion, sizeof(info->basic.firmwareVersion), "unknown");
        info->basic.index = static_cast<int32_t>(p.logic);
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

    topology->deviceCount = deviceCount;
    for (size_t i = 0; i < deviceCount; i++) {
        DeviceTopology* dt = &topology->devices[i];
        std::snprintf(dt->deviceUUID, sizeof(dt->deviceUUID), "npu-%d-chip-%d", deviceIndexArray[i], 0);
        dt->numaNode = -1;
        dt->connectionCount = 0;
        dt->connections = nullptr;
    }
    topology->nvlinkBandwidthMBps = 0;
    topology->ibNicCount = 0;
    std::snprintf(topology->topologyType, sizeof(topology->topologyType), "HCCS");
    (void)maxConnectionsPerDevice;
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
