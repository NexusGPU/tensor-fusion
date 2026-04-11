/*
 * Soft isolation limiter interface.
 *
 * This header defines two API sets:
 *
 * 1. Worker-facing APIs (called from cuda_hook / LD_PRELOAD inside Client Pod):
 *    - CheckAndRecordMemoryOps, CheckAndRecordComputeOps
 *    - FreezeWorker, ResumeWorker, AutoFreeze, AutoResume
 *    - AddWorkerProcess
 *
 * 2. Hypervisor-facing APIs (called by Go hypervisor via purego):
 *    - LimiterInit, LimiterShutdown
 *    - LimiterCreateWorker, LimiterRemoveWorker
 *    - LimiterRegisterPID
 *    - LimiterUpdateERL, LimiterUpdateHeartbeat, LimiterSetPodMemoryUsed
 */

#ifndef LIMITER_H
#define LIMITER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "accelerator.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Limiter Types
// ============================================================================

// Memory operation record
typedef struct {
    char deviceUUID[64];
    int64_t bytesDiff;
    bool shouldBlock;
    uint64_t availableBytes;
} MemoryOpRecord;

// Compute operation record
typedef struct {
    char deviceUUID[64];
    uint64_t computeTokens;
    bool shouldBlock;
    uint64_t availableTokens;
} ComputeOpRecord;

// Worker freeze state
typedef struct {
    char workerId[64];
    bool isFrozen;
    uint64_t freezeTimeMs;
} WorkerFreezeState;

// Device configuration for shared memory initialization
typedef struct {
    uint32_t deviceIdx;
    char deviceUUID[64];
    uint32_t upLimit;        // compute limit percentage (0-100)
    uint64_t memLimit;       // memory limit in bytes
    uint32_t totalCudaCores;
} LimiterDeviceConfig;

// ============================================================================
// Worker-facing APIs (called from cuda_hook inside worker/client processes)
// ============================================================================

AccelResult CheckAndRecordMemoryOps(const char* processId, const char* deviceUUID,
    int64_t bytesDiff, MemoryOpRecord* record);

AccelResult CheckAndRecordComputeOps(const char* processId, const char* deviceUUID,
    uint64_t computeTokens, ComputeOpRecord* record);

AccelResult FreezeWorker(const char* workerId, WorkerFreezeState* state);
AccelResult ResumeWorker(const char* workerId, WorkerFreezeState* state);

AccelResult AutoFreeze(const char* workerId, const char* deviceUUID, const char* resourceType);
AccelResult AutoResume(const char* workerId, const char* deviceUUID, const char* resourceType);

AccelResult AddWorkerProcess(const char* deviceUUID, const char* processId);

// ============================================================================
// Hypervisor-facing APIs (called by Go hypervisor to manage shared memory & ERL)
// ============================================================================

AccelResult LimiterInit(const char* shmBasePath);
AccelResult LimiterShutdown(void);

AccelResult LimiterCreateWorker(const char* namespace_, const char* podName,
    const LimiterDeviceConfig* configs, size_t configCount);
AccelResult LimiterRemoveWorker(const char* namespace_, const char* podName);

AccelResult LimiterRegisterPID(const char* namespace_, const char* podName, uint32_t hostPID);

AccelResult LimiterUpdateERL(const char* namespace_, const char* podName,
    uint32_t deviceIdx, uint32_t upLimit,
    double utilizationPercent, uint64_t timestampMicros);

AccelResult LimiterUpdateHeartbeat(const char* namespace_, const char* podName,
    uint64_t timestampSecs);

AccelResult LimiterSetPodMemoryUsed(const char* namespace_, const char* podName,
    uint32_t deviceIdx, uint64_t memoryUsed);

#ifdef __cplusplus
}
#endif

#endif // LIMITER_H
