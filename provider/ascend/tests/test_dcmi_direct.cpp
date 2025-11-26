/*
 * Direct DCMI API test to debug power and utilization queries
 */

#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>

// Function pointer types
typedef int (*dcmi_init_fn)(void);
typedef int (*dcmi_get_device_power_info_fn)(int, int, int*);
typedef int (*dcmi_get_device_temperature_fn)(int, int, int*);
typedef int (*dcmi_get_device_utilization_rate_fn)(int, int, int, unsigned int*);

int main() {
    const char* libPath = std::getenv("DCMI_LIB_PATH");
    if (!libPath) {
        libPath = "/usr/local/dcmi/libdcmi.so";
    }

    std::fprintf(stderr, "Loading DCMI from: %s\n", libPath);
    void* handle = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "Failed to load DCMI: %s\n", dlerror());
        return 1;
    }

    // Load symbols
    auto p_dcmi_init = reinterpret_cast<dcmi_init_fn>(dlsym(handle, "dcmi_init"));
    auto p_dcmi_get_device_power_info = reinterpret_cast<dcmi_get_device_power_info_fn>(
        dlsym(handle, "dcmi_get_device_power_info"));
    auto p_dcmi_get_device_temperature = reinterpret_cast<dcmi_get_device_temperature_fn>(
        dlsym(handle, "dcmi_get_device_temperature"));
    auto p_dcmi_get_device_utilization_rate = reinterpret_cast<dcmi_get_device_utilization_rate_fn>(
        dlsym(handle, "dcmi_get_device_utilization_rate"));

    if (!p_dcmi_init) {
        std::fprintf(stderr, "Failed to load dcmi_init\n");
        dlclose(handle);
        return 1;
    }

    // Initialize DCMI
    int initRet = p_dcmi_init();
    std::fprintf(stderr, "dcmi_init returned: %d\n\n", initRet);
    if (initRet != 0) {
        std::fprintf(stderr, "DCMI initialization failed\n");
        dlclose(handle);
        return 1;
    }

    // Test with card=2, device=0 (from npu-smi info)
    int card = 2;
    int dev = 0;

    std::fprintf(stderr, "=== Testing DCMI APIs for card=%d, device=%d ===\n\n", card, dev);

    // Test power info
    if (p_dcmi_get_device_power_info) {
        int power = -1;
        int ret = p_dcmi_get_device_power_info(card, dev, &power);
        std::fprintf(stderr, "dcmi_get_device_power_info:\n");
        std::fprintf(stderr, "  Return code: %d\n", ret);
        std::fprintf(stderr, "  Power value: %d mW (%.2f W)\n", power, power / 1000.0);
        if (ret != 0) {
            std::fprintf(stderr, "  ⚠️  FAILED - API returned error %d\n", ret);
        }
        std::fprintf(stderr, "\n");
    } else {
        std::fprintf(stderr, "dcmi_get_device_power_info: Symbol not found\n\n");
    }

    // Test temperature
    if (p_dcmi_get_device_temperature) {
        int temp = -1;
        int ret = p_dcmi_get_device_temperature(card, dev, &temp);
        std::fprintf(stderr, "dcmi_get_device_temperature:\n");
        std::fprintf(stderr, "  Return code: %d\n", ret);
        std::fprintf(stderr, "  Temperature: %d °C\n", temp);
        if (ret != 0) {
            std::fprintf(stderr, "  ⚠️  FAILED - API returned error %d\n", ret);
        }
        std::fprintf(stderr, "\n");
    } else {
        std::fprintf(stderr, "dcmi_get_device_temperature: Symbol not found\n\n");
    }

    // Test utilization rate for different types
    if (p_dcmi_get_device_utilization_rate) {
        const char* types[] = {"AICore", "AICpu", "CtrlCPU", "VectorCore"};
        for (int type = 0; type < 4; type++) {
            unsigned int util = 0;
            int ret = p_dcmi_get_device_utilization_rate(card, dev, type, &util);
            std::fprintf(stderr, "dcmi_get_device_utilization_rate (type=%d - %s):\n", type, types[type]);
            std::fprintf(stderr, "  Return code: %d\n", ret);
            std::fprintf(stderr, "  Utilization: %u%%\n", util);
            if (ret != 0) {
                std::fprintf(stderr, "  ⚠️  FAILED - API returned error %d\n", ret);
            }
            std::fprintf(stderr, "\n");
        }
    } else {
        std::fprintf(stderr, "dcmi_get_device_utilization_rate: Symbol not found\n\n");
    }

    dlclose(handle);
    return 0;
}
