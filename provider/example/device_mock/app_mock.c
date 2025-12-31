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
#include <signal.h>
#include <time.h>

static volatile int g_running = 1;
static hipDevicePtr_t g_devicePtr = NULL;
static size_t g_allocatedSize = 0;

// Signal handler for cleanup
static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

// Mock kernel function (does nothing, just for demonstration)
static void mock_kernel(void) {
    // Simulate some computation
    volatile int dummy = 0;
    for (int i = 0; i < 1000; i++) {
        dummy += i;
    }
    (void)dummy;
}

// Print usage information
static void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m <size>     Memory size to allocate in MB (default: 100)\n");
    fprintf(stderr, "  -k <size>     Kernel launch grid size (default: 1024)\n");
    fprintf(stderr, "  -f <freq>     Kernel launch frequency in Hz (default: 10)\n");
    fprintf(stderr, "  -d <seconds> Duration to run in seconds (default: 60, 0 for infinite)\n");
    fprintf(stderr, "  -h            Show this help message\n");
    fprintf(stderr, "\nExample: %s -m 512 -k 2048 -f 20 -d 30\n", prog_name);
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t memorySizeMB = 100;
    uint32_t kernelGridSize = 1024;
    double launchFrequencyHz = 10.0;
    int durationSeconds = 60;

    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "m:k:f:d:h")) != -1) {
        switch (opt) {
            case 'm':
                memorySizeMB = (size_t)atoi(optarg);
                break;
            case 'k':
                kernelGridSize = (uint32_t)atoi(optarg);
                break;
            case 'f':
                launchFrequencyHz = atof(optarg);
                break;
            case 'd':
                durationSeconds = atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    printf("=== GPU AI App Mock ===\n");
    printf("Memory size: %zu MB\n", memorySizeMB);
    printf("Kernel grid size: %u\n", kernelGridSize);
    printf("Launch frequency: %.2f Hz\n", launchFrequencyHz);
    printf("Duration: %d seconds (0 = infinite)\n", durationSeconds);
    printf("PID: %d\n", getpid());
    printf("\n");

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize HIP (this will register the process in shared memory)
    printf("Calling hipInit()...\n");
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipInit failed with code %d\n", err);
        return 1;
    }
    printf("hipInit() succeeded\n");

    // Get device count
    int deviceCount = 0;
    err = hipDeviceGetCount(&deviceCount);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipDeviceGetCount failed\n");
        return 1;
    }
    printf("Found %d device(s)\n", deviceCount);

    // Get device
    hipDevice_t device = 0;
    err = hipDeviceGet(&device, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipDeviceGet failed\n");
        return 1;
    }
    printf("Using device %d\n", device);

    // Allocate memory
    size_t memorySizeBytes = memorySizeMB * 1024 * 1024;
    printf("Allocating %zu bytes (%.2f MB)...\n", memorySizeBytes, (double)memorySizeMB);
    err = hipMalloc(&g_devicePtr, memorySizeBytes);
    if (err != hipSuccess) {
        fprintf(stderr, "Error: hipMalloc failed with code %d\n", err);
        return 1;
    }
    g_allocatedSize = memorySizeBytes;
    printf("Memory allocated successfully at %p\n", g_devicePtr);

    // Calculate sleep interval based on frequency
    struct timespec sleepInterval;
    if (launchFrequencyHz > 0.0) {
        double intervalSeconds = 1.0 / launchFrequencyHz;
        sleepInterval.tv_sec = (time_t)intervalSeconds;
        sleepInterval.tv_nsec = (long)((intervalSeconds - (double)sleepInterval.tv_sec) * 1e9);
    } else {
        sleepInterval.tv_sec = 1;
        sleepInterval.tv_nsec = 0;
    }

    // Launch kernels at specified frequency
    printf("\nStarting kernel launches...\n");
    printf("Press Ctrl+C to stop\n\n");

    struct timespec startTime, currentTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    uint64_t launchCount = 0;

    while (g_running) {
        // Check duration limit
        if (durationSeconds > 0) {
            clock_gettime(CLOCK_MONOTONIC, &currentTime);
            double elapsed = (double)(currentTime.tv_sec - startTime.tv_sec) +
                           (double)(currentTime.tv_nsec - startTime.tv_nsec) / 1e9;
            if (elapsed >= durationSeconds) {
                printf("\nDuration limit reached, stopping...\n");
                break;
            }
        }

        // Launch kernel
        err = hipLaunchKernel((const void*)mock_kernel,
                              kernelGridSize, 1, 1,  // gridDim
                              256, 1, 1,            // blockDim
                              0,                     // sharedMemBytes
                              NULL,                  // stream
                              NULL,                  // kernelParams
                              NULL);                 // extra

        if (err != hipSuccess) {
            fprintf(stderr, "Error: hipLaunchKernel failed with code %d\n", err);
            break;
        }

        launchCount++;
        if (launchCount % 100 == 0) {
            printf("Launched %lu kernels...\n", launchCount);
        }

        // Sleep for the calculated interval
        nanosleep(&sleepInterval, NULL);
    }

    printf("\n=== Summary ===\n");
    printf("Total kernels launched: %lu\n", launchCount);
    printf("Memory allocated: %zu bytes (%.2f MB)\n", g_allocatedSize, g_allocatedSize / (1024.0 * 1024.0));

    // Cleanup: free memory
    if (g_devicePtr) {
        printf("Freeing memory...\n");
        err = hipFree(g_devicePtr);
        if (err != hipSuccess) {
            fprintf(stderr, "Warning: hipFree failed with code %d\n", err);
        }
        g_devicePtr = NULL;
    }

    printf("App mock completed successfully\n");
    return 0;
}

