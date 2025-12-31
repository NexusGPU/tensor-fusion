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
#include <time.h>

int main(void) {
    printf("=== Testing Rate Limiting (100 launches/sec = 100%% utilization) ===\n\n");
    
    // Initialize
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipInit failed\n");
        return 1;
    }
    
    DeviceUtilization deviceUtil;
    
    printf("Testing rate limiting:\n");
    printf("- Each kernel launch = 1%% GPU utilization\n");
    printf("- Max 100 launches/second before hitting 100%% and blocking\n\n");
    
    // Launch kernels rapidly to test rate limiting
    int successCount = 0;
    int blockedCount = 0;
    struct timespec start, current;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Launching kernels as fast as possible for 2 seconds...\n");
    for (int i = 0; i < 500; i++) {  // Try to launch 500 kernels
        err = hipLaunchKernel(NULL, 256, 1, 1, 64, 1, 1, 0, NULL, NULL, NULL);
        if (err == hipSuccess) {
            successCount++;
        } else {
            blockedCount++;
        }
        
        // Check time
        clock_gettime(CLOCK_MONOTONIC, &current);
        double elapsed = (double)(current.tv_sec - start.tv_sec) +
                        (double)(current.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed >= 2.0) {
            break;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &current);
    double elapsed = (double)(current.tv_sec - start.tv_sec) +
                    (double)(current.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("\nResults:\n");
    printf("  Time elapsed: %.2f seconds\n", elapsed);
    printf("  Successful launches: %d\n", successCount);
    printf("  Blocked launches: %d\n", blockedCount);
    printf("  Launches per second: %.2f\n", successCount / elapsed);
    
    // Check device utilization
    err = hipGetDeviceUtilization(&deviceUtil);
    if (err == hipSuccess) {
        printf("\nDevice GPU Utilization: %.2f%%\n", deviceUtil.utilizationPercent);
        if (deviceUtil.utilizationPercent >= 100.0) {
            printf("✓ Correctly reached 100%% utilization\n");
        } else {
            printf("  Note: Utilization is %.2f%%, expected 100%% if rate limit was hit\n",
                   deviceUtil.utilizationPercent);
        }
    }
    
    // Wait a bit and check again (should reset)
    printf("\nWaiting 1.5 seconds for rate limit window to reset...\n");
    struct timespec delay = {1, 500000000};  // 1.5 seconds
    nanosleep(&delay, NULL);
    
    err = hipGetDeviceUtilization(&deviceUtil);
    if (err == hipSuccess) {
        printf("Device GPU Utilization after reset: %.2f%%\n", deviceUtil.utilizationPercent);
    }
    
    // Try launching again (should work now)
    printf("\nTrying to launch a kernel after reset...\n");
    err = hipLaunchKernel(NULL, 256, 1, 1, 64, 1, 1, 0, NULL, NULL, NULL);
    if (err == hipSuccess) {
        printf("✓ Kernel launch succeeded after reset\n");
    } else {
        printf("✗ Kernel launch still blocked\n");
    }
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

