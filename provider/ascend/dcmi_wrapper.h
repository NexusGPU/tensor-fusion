#pragma once

// Prefer the real DCMI header when available; otherwise provide a minimal stub
// so editors can parse the file without installing Ascend SDK locally.
#if __has_include(<dcmi_interface_api.h>)
#include <dcmi_interface_api.h>
#elif __has_include("/usr/local/dcmi/dcmi_interface_api.h")
#include "/usr/local/dcmi/dcmi_interface_api.h"
#else
#define MAX_CHIP_NAME_LEN 32

struct dcmi_chip_info_v2 {
    unsigned char chip_type[MAX_CHIP_NAME_LEN];
    unsigned char chip_name[MAX_CHIP_NAME_LEN];
    unsigned char chip_ver[MAX_CHIP_NAME_LEN];
    unsigned int aicore_cnt;
    unsigned char npu_name[MAX_CHIP_NAME_LEN];
};

struct dcmi_get_memory_info_stru {
    unsigned long long memory_size; // MB
};

struct dcmi_memory_info_stru {
    unsigned long long memory_size; // MB
    unsigned int freq;
    unsigned int utiliza;
};

struct dcmi_create_vdev_res_stru {
    unsigned int vdev_id;
    unsigned int vfg_id;
    char template_name[32];
    unsigned char reserved[64];
};

struct dcmi_create_vdev_out {
    unsigned int vdev_id;
    unsigned int pcie_bus;
    unsigned int pcie_device;
    unsigned int pcie_func;
    unsigned int vfg_id;
    unsigned char reserved[64];
};

#endif // __has_include
