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

/**
 * @file hip_remote_protocol.h
 * @brief Minimal remote protocol for HIP API forwarding (demo subset)
 *
 * This header intentionally only contains the operations needed for:
 * - torch.cuda.is_available()
 * - torch.cuda.device_count()
 */

#ifndef HIP_REMOTE_PROTOCOL_H
#define HIP_REMOTE_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Protocol constants
// ============================================================================

#define HIP_REMOTE_MAGIC 0x48495052  // 'HIPR' in ASCII
#define HIP_REMOTE_VERSION 0x0002
#define HIP_REMOTE_DEFAULT_PORT 8000
#define HIP_REMOTE_MAX_PAYLOAD_SIZE (4u * 1024u * 1024u)  // 4MB

// ============================================================================
// Operation codes
// ============================================================================

typedef enum {
    HIP_OP_INIT = 0x0001,
    HIP_OP_SHUTDOWN = 0x0002,

    HIP_OP_GET_DEVICE_COUNT = 0x0100,
    HIP_OP_SET_DEVICE = 0x0101,
    HIP_OP_GET_DEVICE = 0x0102,

    HIP_OP_GET_LAST_ERROR = 0x0600,
    HIP_OP_PEEK_AT_LAST_ERROR = 0x0601,
} HipRemoteOpCode;

// ============================================================================
// Protocol header
// ============================================================================

typedef struct __attribute__((packed)) {
    uint32_t magic;           // HIP_REMOTE_MAGIC
    uint16_t version;         // HIP_REMOTE_VERSION
    uint16_t op_code;         // HipRemoteOpCode
    uint32_t request_id;      // Correlation id
    uint32_t payload_length;  // Payload length following header
    uint32_t flags;           // Reserved
} HipRemoteHeader;

// ============================================================================
// Requests
// ============================================================================

typedef struct __attribute__((packed)) {
    int32_t device_id;
} HipRemoteDeviceRequest;

// ============================================================================
// Responses
// ============================================================================

typedef struct __attribute__((packed)) {
    int32_t error_code;  // hipError_t
} HipRemoteResponseHeader;

typedef struct __attribute__((packed)) {
    HipRemoteResponseHeader header;
    int32_t count;
} HipRemoteDeviceCountResponse;

typedef struct __attribute__((packed)) {
    HipRemoteResponseHeader header;
    int32_t device_id;
} HipRemoteGetDeviceResponse;

// ============================================================================
// Utilities
// ============================================================================

static inline void hip_remote_init_header(
    HipRemoteHeader* header,
    HipRemoteOpCode op_code,
    uint32_t request_id,
    uint32_t payload_length
) {
    header->magic = HIP_REMOTE_MAGIC;
    header->version = HIP_REMOTE_VERSION;
    header->op_code = (uint16_t)op_code;
    header->request_id = request_id;
    header->payload_length = payload_length;
    header->flags = 0;
}

static inline int hip_remote_validate_header(const HipRemoteHeader* header) {
    if (header->magic != HIP_REMOTE_MAGIC) {
        return -1;
    }
    if (header->version != HIP_REMOTE_VERSION) {
        return -2;
    }
    if (header->payload_length > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        return -3;
    }
    return 0;
}

static inline const char* hip_remote_op_name(HipRemoteOpCode op_code) {
    switch (op_code) {
        case HIP_OP_INIT: return "hipInit(remote)";
        case HIP_OP_SHUTDOWN: return "hipShutdown(remote)";
        case HIP_OP_GET_DEVICE_COUNT: return "hipGetDeviceCount";
        case HIP_OP_SET_DEVICE: return "hipSetDevice";
        case HIP_OP_GET_DEVICE: return "hipGetDevice";
        case HIP_OP_GET_LAST_ERROR: return "hipGetLastError";
        case HIP_OP_PEEK_AT_LAST_ERROR: return "hipPeekAtLastError";
        default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif  // HIP_REMOTE_PROTOCOL_H

