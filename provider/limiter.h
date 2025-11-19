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

#ifndef LIMITER_H
#define LIMITER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Limiter Types
// ============================================================================

// Memory operation record
typedef struct {
    char deviceUUID[64];             // Device UUID
    int64_t bytesDiff;                // Bytes difference (positive = allocation, negative = deallocation)
    bool shouldBlock;                 // Output: whether this operation should be blocked
    uint64_t availableBytes;          // Output: available bytes after this operation
} MemoryOpRecord;

// Compute operation record
typedef struct {
    char deviceUUID[64];             // Device UUID
    uint64_t computeTokens;          // Compute tokens consumed (e.g., SM-cycles)
    bool shouldBlock;                 // Output: whether this operation should be blocked
    uint64_t availableTokens;         // Output: available tokens after this operation
} ComputeOpRecord;

// Worker freeze state
typedef struct {
    char workerId[64];               // Worker identifier
    bool isFrozen;                    // Current freeze state
    uint64_t freezeTimeMs;            // Time frozen in milliseconds
} WorkerFreezeState;

// ============================================================================
// Limiter APIs (Implemented by limiter.so, NOT by vendor accelerator.so)
// ============================================================================

/**
 * Check and record memory operations for soft isolation.
 * This API is called from hooks in CUDA runtime (via dlsym replacement).
 * 
 * @param processId Process identifier
 * @param deviceUUID Device UUID
 * @param bytesDiff Bytes difference (positive = allocation, negative = deallocation)
 * @param record Output parameter for operation record
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result CheckAndRecordMemoryOps(const char* processId, const char* deviceUUID, int64_t bytesDiff, MemoryOpRecord* record);

/**
 * Check and record compute operations for soft isolation.
 * This API is called from hooks in CUDA runtime (via dlsym replacement).
 * 
 * @param processId Process identifier
 * @param deviceUUID Device UUID
 * @param computeTokens Compute tokens consumed (e.g., SM-cycles)
 * @param record Output parameter for operation record
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result CheckAndRecordComputeOps(const char* processId, const char* deviceUUID, uint64_t computeTokens, ComputeOpRecord* record);

/**
 * Freeze a worker process (pause execution when resource limit reached).
 * This API is called automatically when resources are exhausted.
 * 
 * @param workerId Worker identifier
 * @param state Output parameter for freeze state
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result FreezeWorker(const char* workerId, WorkerFreezeState* state);

/**
 * Resume a worker process (resume execution when resources become available).
 * This API is called automatically when resources become available.
 * 
 * @param workerId Worker identifier
 * @param state Output parameter for freeze state
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result ResumeWorker(const char* workerId, WorkerFreezeState* state);

/**
 * Auto-freeze hook: called when resource limit is reached.
 * This triggers automatic freezing of the worker.
 * 
 * @param workerId Worker identifier
 * @param deviceUUID Device UUID
 * @param resourceType Resource type ("memory" or "compute")
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result AutoFreeze(const char* workerId, const char* deviceUUID, const char* resourceType);

/**
 * Auto-resume hook: called when resources become available.
 * This triggers automatic resuming of the worker.
 * 
 * @param workerId Worker identifier
 * @param deviceUUID Device UUID
 * @param resourceType Resource type ("memory" or "compute")
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result AutoResume(const char* workerId, const char* deviceUUID, const char* resourceType);

/**
 * Add a worker process to the limiter tracking.
 * This API is called when a process starts using a device.
 * 
 * @param deviceUUID Device UUID
 * @param processId Process identifier (as string)
 * @return RESULT_SUCCESS on success, error code otherwise
 */
Result AddWorkerProcess(const char* deviceUUID, const char* processId);

#ifdef __cplusplus
}
#endif

#endif // LIMITER_H

