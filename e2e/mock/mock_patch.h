/*
 * Test-only shim, force-included (gcc -include) when e2e/terraform/scripts/mock-gpu.sh
 * builds provider/example/accelerator.c for the rehearsal mode.
 *
 * The stock example provider hard-codes 4 devices with the UUIDs
 * example-device-0..3. GPU custom resources are named after the device UUID and
 * are cluster-scoped, so two mock nodes would fight over the same four objects.
 * The build script rewrites three lines of accelerator.c to call these helpers:
 * the UUID gets the node's short hostname (the hypervisor runs with hostNetwork)
 * and the device count comes from TF_MOCK_GPU_COUNT (1-4, default 2).
 */
#ifndef TF_E2E_MOCK_PATCH_H
#define TF_E2E_MOCK_PATCH_H

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char* tf_mock_host(void) {
    static char host[40];
    if (!host[0]) {
        char buf[256] = {0};
        if (gethostname(buf, sizeof(buf) - 1) != 0 || !buf[0]) {
            strcpy(buf, "mock");
        }
        char* dot = strchr(buf, '.');
        if (dot) {
            *dot = 0;
        }
        strncpy(host, buf, sizeof(host) - 1);
    }
    return host;
}

static size_t tf_mock_count(void) {
    const char* e = getenv("TF_MOCK_GPU_COUNT");
    int n = e ? atoi(e) : 2;
    if (n < 1) n = 1;
    if (n > 4) n = 4; /* the example provider sizes several arrays for 4 devices */
    return (size_t)n;
}

#endif
