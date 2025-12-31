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

#define _POSIX_C_SOURCE 200809L

#include "driver_mock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

int main(void) {
    printf("=== Testing Driver Mock Metrics APIs ===\n\n");
    
    // Initialize
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipInit failed\n");
        return 1;
    }
    printf("✓ hipInit() succeeded\n");
    
    // Allocate some VRAM
    hipDevicePtr_t ptr1, ptr2;
    size_t size1 = 100 * 1024 * 1024;  // 100MB
    size_t size2 = 50 * 1024 * 1024;    // 50MB
    
    err = hipMalloc(&ptr1, size1);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipMalloc failed\n");
        return 1;
    }
    printf("✓ Allocated %zu bytes (ptr: %p)\n", size1, ptr1);
    
    err = hipMalloc(&ptr2, size2);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipMalloc failed\n");
        return 1;
    }
    printf("✓ Allocated %zu bytes (ptr: %p)\n", size2, ptr2);
    
    // Launch some kernels
    printf("\nLaunching kernels...\n");
    for (int i = 0; i < 50; i++) {
        err = hipLaunchKernel(NULL, 256, 1, 1, 64, 1, 1, 0, NULL, NULL, NULL);
        if (err != hipSuccess) {
            fprintf(stderr, "Error: hipLaunchKernel failed at iteration %d\n", i);
            break;
        }
        struct timespec delay = {0, 10000000};  // 10ms
        nanosleep(&delay, NULL);
    }
    printf("✓ Launched 50 kernels\n");
    
    // Test Mock AMD SMI Process List (includes both VRAM and utilization info)
    printf("\n--- Mock AMD SMI Process List ---\n");
    amdsmi_proc_info_t procInfos[10];
    uint32_t maxProcs = 10;
    amdsmi_status_t status = amdsmi_get_gpu_process_list(0, &maxProcs, procInfos);
    if (status == AMDSMI_STATUS_SUCCESS || status == AMDSMI_STATUS_OUT_OF_RESOURCES) {
        printf("Found %u process(es):\n", maxProcs);
        for (uint32_t i = 0; i < maxProcs && i < 10; i++) {
            printf("  Process: %s (PID %d)\n", procInfos[i].name, (int)procInfos[i].pid);
            printf("    Memory: VRAM=%lu bytes (%.2f MB), GTT=%lu bytes, CPU=%lu bytes\n",
                   procInfos[i].memory_usage.vram_mem,
                   procInfos[i].memory_usage.vram_mem / (1024.0 * 1024.0),
                   procInfos[i].memory_usage.gtt_mem,
                   procInfos[i].memory_usage.cpu_mem);
            printf("    Engine Usage: GFX=%lu ns, ENC=%lu ns\n",
                   procInfos[i].engine_usage.gfx,
                   procInfos[i].engine_usage.enc);
            printf("    CU Occupancy: %u CUs\n", procInfos[i].cu_occupancy);
            if (procInfos[i].cu_occupancy > 0) {
                double utilPercent = ((double)procInfos[i].cu_occupancy / 108.0) * 100.0;
                printf("    Estimated GPU Utilization: %.2f%%\n", utilPercent);
            }
        }
    } else {
        printf("✗ amdsmi_get_gpu_process_list failed (status: %d)\n", status);
    }
    
    // Test Device VRAM Usage (AMD SMI API)
    printf("\n--- Device VRAM Usage (AMD SMI) ---\n");
    amdsmi_vram_usage_t vramUsage;
    amdsmi_status_t statusVRAM = amdsmi_get_gpu_vram_usage(NULL, &vramUsage);
    if (statusVRAM == AMDSMI_STATUS_SUCCESS) {
        // Note: Real API returns values in MB (uint32_t)
        uint64_t totalBytes = (uint64_t)vramUsage.vram_total * 1024ULL * 1024ULL;
        uint64_t usedBytes = (uint64_t)vramUsage.vram_used * 1024ULL * 1024ULL;
        uint64_t freeBytes = totalBytes - usedBytes;
        printf("  Total: %u MB (%.2f GB)\n",
               vramUsage.vram_total,
               totalBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Used: %u MB (%.2f GB)\n",
               vramUsage.vram_used,
               usedBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Free: %lu bytes (%.2f GB)\n",
               freeBytes,
               freeBytes / (1024.0 * 1024.0 * 1024.0));
        if (vramUsage.vram_total > 0) {
            double utilPercent = ((double)vramUsage.vram_used / (double)vramUsage.vram_total) * 100.0;
            printf("  Utilization: %.2f%%\n", utilPercent);
        }
    } else {
        printf("✗ amdsmi_get_gpu_vram_usage failed (status: %d)\n", status);
    }
    
    // Test Device GPU Utilization (AMD SMI API)
    printf("\n--- Device GPU Utilization (AMD SMI) ---\n");
    amdsmi_engine_usage_t gpuActivity;
    amdsmi_status_t statusActivity = amdsmi_get_gpu_activity(NULL, &gpuActivity);
    if (statusActivity == AMDSMI_STATUS_SUCCESS) {
        printf("  GFX Utilization: %u%%\n", gpuActivity.gfx_activity);
        printf("  UMC Utilization: %u%%\n", gpuActivity.umc_activity);
        printf("  MM Utilization: %u%%\n", gpuActivity.mm_activity);
    } else {
        printf("✗ amdsmi_get_gpu_activity failed (status: %d)\n", statusActivity);
    }
    
    // Free memory
    printf("\nFreeing memory...\n");
    err = hipFree(ptr1);
    if (err != hipSuccess) {
        fprintf(stderr, "Warning: hipFree(ptr1) failed\n");
    } else {
        printf("✓ Freed ptr1\n");
    }
    
    err = hipFree(ptr2);
    if (err != hipSuccess) {
        fprintf(stderr, "Warning: hipFree(ptr2) failed\n");
    } else {
        printf("✓ Freed ptr2\n");
    }
    
    // Check VRAM after free
    printf("\n--- Device VRAM Usage After Free (AMD SMI) ---\n");
    statusVRAM = amdsmi_get_gpu_vram_usage(NULL, &vramUsage);
    if (statusVRAM == AMDSMI_STATUS_SUCCESS) {
        // Note: Real API returns values in MB (uint32_t)
        uint64_t usedBytes = (uint64_t)vramUsage.vram_used * 1024ULL * 1024ULL;
        printf("  Used: %u MB (%.2f MB)\n",
               vramUsage.vram_used,
               usedBytes / (1024.0 * 1024.0));
        if (vramUsage.vram_total > 0) {
            double utilPercent = ((double)vramUsage.vram_used / (double)vramUsage.vram_total) * 100.0;
            printf("  Utilization: %.2f%%\n", utilPercent);
        }
    }
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

