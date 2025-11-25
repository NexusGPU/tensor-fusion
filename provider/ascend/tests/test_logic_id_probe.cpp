#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>

// Function pointer types
typedef int (*dcmi_init_fn)(void);
typedef int (*dcmi_get_all_device_count_fn)(int*);
typedef int (*dcmi_get_card_id_device_id_from_logicid_fn)(int*, int*, unsigned int);

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
    auto p_dcmi_get_all_device_count = reinterpret_cast<dcmi_get_all_device_count_fn>(
        dlsym(handle, "dcmi_get_all_device_count"));
    auto p_dcmi_get_card_id_device_id_from_logicid =
        reinterpret_cast<dcmi_get_card_id_device_id_from_logicid_fn>(
            dlsym(handle, "dcmi_get_card_id_device_id_from_logicid"));

    if (!p_dcmi_init || !p_dcmi_get_all_device_count || !p_dcmi_get_card_id_device_id_from_logicid) {
        std::fprintf(stderr, "Failed to load DCMI symbols\n");
        dlclose(handle);
        return 1;
    }

    // Initialize DCMI
    int initRet = p_dcmi_init();
    std::fprintf(stderr, "dcmi_init returned: %d\n", initRet);
    if (initRet != 0) {
        std::fprintf(stderr, "DCMI initialization failed\n");
        dlclose(handle);
        return 1;
    }

    // Get device count
    int count = 0;
    int ret = p_dcmi_get_all_device_count(&count);
    std::fprintf(stderr, "dcmi_get_all_device_count ret=%d, count=%d\n", ret, count);

    // Probe logic IDs from 0 to 10
    std::fprintf(stderr, "\n=== Probing Logic IDs ===\n");
    for (unsigned int logicId = 0; logicId < 10; ++logicId) {
        int card = -1;
        int dev = -1;
        int mapRet = p_dcmi_get_card_id_device_id_from_logicid(&card, &dev, logicId);

        if (mapRet == 0) {
            std::fprintf(stderr, "[SUCCESS] logic_id=%u -> card_id=%d, device_id=%d\n",
                        logicId, card, dev);
        } else {
            std::fprintf(stderr, "[FAILED]  logic_id=%u -> ret=%d (card=%d, dev=%d)\n",
                        logicId, mapRet, card, dev);
        }
    }

    // Also try the known NPU ID as logic ID
    std::fprintf(stderr, "\n=== Trying NPU ID as Logic ID ===\n");
    unsigned int knownNpuIds[] = {2, 5};  // From your npu-smi output and Bus-Id
    for (unsigned int npuId : knownNpuIds) {
        int card = -1;
        int dev = -1;
        int mapRet = p_dcmi_get_card_id_device_id_from_logicid(&card, &dev, npuId);

        if (mapRet == 0) {
            std::fprintf(stderr, "[SUCCESS] npu_id=%u -> card_id=%d, device_id=%d\n",
                        npuId, card, dev);
        } else {
            std::fprintf(stderr, "[FAILED]  npu_id=%u -> ret=%d\n", npuId, mapRet);
        }
    }

    dlclose(handle);
    return 0;
}
