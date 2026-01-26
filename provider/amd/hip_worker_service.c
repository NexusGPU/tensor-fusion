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
 * @file hip_worker_service.c
 * @brief Minimal worker service executing HIP calls for remote clients (demo subset)
 */

#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "hip_remote_protocol.h"

static int g_listen_port = HIP_REMOTE_DEFAULT_PORT;
static int g_device_id = 0;
static bool g_debug_enabled = false;
static volatile bool g_running = true;
static int g_server_fd = -1;

#define TF_LOG(fmt, ...) do { \
    if (g_debug_enabled) { \
        fprintf(stderr, "[TF-Worker] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

#define TF_ERROR(fmt, ...) \
    fprintf(stderr, "[TF-Worker ERROR] " fmt "\n", ##__VA_ARGS__)

#define TF_INFO(fmt, ...) \
    fprintf(stdout, "[TF-Worker] " fmt "\n", ##__VA_ARGS__)

#if defined(MSG_NOSIGNAL)
#define TF_SEND_FLAGS MSG_NOSIGNAL
#else
#define TF_SEND_FLAGS 0
#endif

static int send_all(int fd, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
        ssize_t n = send(fd, p, len, TF_SEND_FLAGS);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = EPIPE;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void* data, size_t len) {
    uint8_t* p = (uint8_t*)data;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            errno = ECONNRESET;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int send_response(int fd, HipRemoteOpCode op_code, uint32_t request_id, const void* payload, size_t payload_size) {
    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code, request_id, (uint32_t)payload_size);
    if (send_all(fd, &header, sizeof(header)) != 0) return -1;
    if (payload && payload_size > 0) {
        if (send_all(fd, payload, payload_size) != 0) return -1;
    }
    return 0;
}

static int send_simple_response(int fd, HipRemoteOpCode op_code, uint32_t request_id, hipError_t err) {
    HipRemoteResponseHeader resp = { .error_code = (int32_t)err };
    return send_response(fd, op_code, request_id, &resp, sizeof(resp));
}

static void handle_init(int fd, uint32_t request_id) {
    hipError_t err = hipSetDevice(g_device_id);
    (void)send_simple_response(fd, HIP_OP_INIT, request_id, err);
}

static void handle_shutdown(int fd, uint32_t request_id) {
    (void)send_simple_response(fd, HIP_OP_SHUTDOWN, request_id, hipSuccess);
}

static void handle_get_device_count(int fd, uint32_t request_id) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    HipRemoteDeviceCountResponse resp = {
        .header = { .error_code = (int32_t)err },
        .count = count,
    };
    (void)send_response(fd, HIP_OP_GET_DEVICE_COUNT, request_id, &resp, sizeof(resp));
}

static void handle_set_device(int fd, uint32_t request_id, const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceRequest)) {
        (void)send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, hipErrorInvalidValue);
        return;
    }
    const HipRemoteDeviceRequest* req = (const HipRemoteDeviceRequest*)payload;
    hipError_t err = hipSetDevice(req->device_id);
    (void)send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, err);
}

static void handle_get_device(int fd, uint32_t request_id) {
    int device = 0;
    hipError_t err = hipGetDevice(&device);
    HipRemoteGetDeviceResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_id = device,
    };
    (void)send_response(fd, HIP_OP_GET_DEVICE, request_id, &resp, sizeof(resp));
}

static void handle_get_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipGetLastError();
    (void)send_simple_response(fd, HIP_OP_GET_LAST_ERROR, request_id, err);
}

static void handle_peek_at_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipPeekAtLastError();
    (void)send_simple_response(fd, HIP_OP_PEEK_AT_LAST_ERROR, request_id, err);
}

static void handle_client(int client_fd) {
    while (g_running) {
        HipRemoteHeader header;
        if (recv_all(client_fd, &header, sizeof(header)) != 0) break;
        if (hip_remote_validate_header(&header) != 0) break;

        void* payload = NULL;
        if (header.payload_length > 0) {
            payload = malloc(header.payload_length);
            if (!payload) break;
            if (recv_all(client_fd, payload, header.payload_length) != 0) {
                free(payload);
                break;
            }
        }

        switch ((HipRemoteOpCode)header.op_code) {
            case HIP_OP_INIT:
                handle_init(client_fd, header.request_id);
                break;
            case HIP_OP_SHUTDOWN:
                handle_shutdown(client_fd, header.request_id);
                free(payload);
                goto done;
            case HIP_OP_GET_DEVICE_COUNT:
                handle_get_device_count(client_fd, header.request_id);
                break;
            case HIP_OP_SET_DEVICE:
                handle_set_device(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GET_DEVICE:
                handle_get_device(client_fd, header.request_id);
                break;
            case HIP_OP_GET_LAST_ERROR:
                handle_get_last_error(client_fd, header.request_id);
                break;
            case HIP_OP_PEEK_AT_LAST_ERROR:
                handle_peek_at_last_error(client_fd, header.request_id);
                break;
            default:
                (void)send_simple_response(client_fd, (HipRemoteOpCode)header.op_code, header.request_id, hipErrorInvalidValue);
                break;
        }

        free(payload);
    }

done:
    close(client_fd);
}

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    if (g_server_fd >= 0) {
        shutdown(g_server_fd, SHUT_RDWR);
        close(g_server_fd);
        g_server_fd = -1;
    }
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str) g_listen_port = atoi(port_str);
    const char* debug = getenv("TF_DEBUG");
    if (debug && strcmp(debug, "1") == 0) g_debug_enabled = true;
    const char* device_str = getenv("TF_DEVICE_ID");
    if (device_str) g_device_id = atoi(device_str);

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    TF_INFO("Initializing HIP on device %d...", g_device_id);
    hipError_t err = hipSetDevice(g_device_id);
    if (err != hipSuccess) {
        TF_ERROR("Failed to set device: %s", hipGetErrorString(err));
        return 1;
    }

    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) return 1;

    int opt = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)g_listen_port);
    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) return 1;
    if (listen(g_server_fd, 5) < 0) return 1;

    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        struct timeval io_timeout = {.tv_sec = 60, .tv_usec = 0};
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &io_timeout, sizeof(io_timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &io_timeout, sizeof(io_timeout));

        handle_client(client_fd);
    }

    if (g_server_fd >= 0) close(g_server_fd);
    return 0;
}

