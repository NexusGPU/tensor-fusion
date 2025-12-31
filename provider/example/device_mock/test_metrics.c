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
    
    // Test Process VRAM Usage
    printf("\n--- Process VRAM Usage ---\n");
    ProcessVRAMUsage vramUsages[10];
    size_t vramCount = 0;
    err = hipGetProcessVRAMUsage(vramUsages, 10, &vramCount);
    if (err == hipSuccess) {
        printf("Found %zu process(es):\n", vramCount);
        for (size_t i = 0; i < vramCount; i++) {
            printf("  PID %d: %lu bytes (%.2f MB), Reserved: %lu bytes\n",
                   vramUsages[i].processId,
                   vramUsages[i].usedBytes,
                   vramUsages[i].usedBytes / (1024.0 * 1024.0),
                   vramUsages[i].reservedBytes);
        }
    } else {
        printf("✗ hipGetProcessVRAMUsage failed\n");
    }
    
    // Test Process Utilization
    printf("\n--- Process GPU Utilization ---\n");
    ProcessUtilization utilizations[10];
    size_t utilCount = 0;
    err = hipGetProcessUtilization(utilizations, 10, &utilCount);
    if (err == hipSuccess) {
        printf("Found %zu process(es):\n", utilCount);
        for (size_t i = 0; i < utilCount; i++) {
            printf("  PID %d: %.2f%% GPU utilization\n",
                   utilizations[i].processId,
                   utilizations[i].utilizationPercent);
        }
    } else {
        printf("✗ hipGetProcessUtilization failed\n");
    }
    
    // Test Device VRAM Usage
    printf("\n--- Device VRAM Usage ---\n");
    DeviceVRAMUsage deviceVRAM;
    err = hipGetDeviceVRAMUsage(&deviceVRAM);
    if (err == hipSuccess) {
        printf("Device: %s\n", deviceVRAM.deviceUUID);
        printf("  Total: %lu bytes (%.2f GB)\n",
               deviceVRAM.totalBytes,
               deviceVRAM.totalBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Used: %lu bytes (%.2f GB)\n",
               deviceVRAM.usedBytes,
               deviceVRAM.usedBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Free: %lu bytes (%.2f GB)\n",
               deviceVRAM.freeBytes,
               deviceVRAM.freeBytes / (1024.0 * 1024.0 * 1024.0));
        printf("  Utilization: %.2f%%\n", deviceVRAM.utilizationPercent);
    } else {
        printf("✗ hipGetDeviceVRAMUsage failed\n");
    }
    
    // Test Device Utilization
    printf("\n--- Device GPU Utilization ---\n");
    DeviceUtilization deviceUtil;
    err = hipGetDeviceUtilization(&deviceUtil);
    if (err == hipSuccess) {
        printf("Device: %s\n", deviceUtil.deviceUUID);
        printf("  GPU Utilization: %.2f%%\n", deviceUtil.utilizationPercent);
    } else {
        printf("✗ hipGetDeviceUtilization failed\n");
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
    printf("\n--- Device VRAM Usage After Free ---\n");
    err = hipGetDeviceVRAMUsage(&deviceVRAM);
    if (err == hipSuccess) {
        printf("  Used: %lu bytes (%.2f MB)\n",
               deviceVRAM.usedBytes,
               deviceVRAM.usedBytes / (1024.0 * 1024.0));
        printf("  Utilization: %.2f%%\n", deviceVRAM.utilizationPercent);
    }
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

