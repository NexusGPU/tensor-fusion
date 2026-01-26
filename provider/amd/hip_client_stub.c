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
 * @file hip_client_stub.c
 * @brief HIP API interception stub for remote GPU execution
 *
 * Supported (demo subset):
 * - hipGetDeviceCount
 * - hipSetDevice / hipGetDevice
 * - hipGetLastError / hipPeekAtLastError
 * - hipGetErrorString / hipGetErrorName (local mapping)
 */

 #ifndef _GNU_SOURCE
 #define _GNU_SOURCE
 #endif
 
 #include <arpa/inet.h>
 #include <errno.h>
 #include <netdb.h>
 #include <netinet/in.h>
 #include <netinet/tcp.h>
 #include <pthread.h>
 #include <stdbool.h>
 #include <stdint.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/socket.h>
 #include <unistd.h>
 
 #include "hip_remote_protocol.h"
 
 typedef int hipError_t;
 
 #define hipSuccess 0
 #define hipErrorInvalidValue 1
 #define hipErrorOutOfMemory 2
 #define hipErrorNotInitialized 3
 #define hipErrorInvalidDevice 101
 #define hipErrorInvalidResourceHandle 400
 #define hipErrorNotReady 600
 
 typedef struct {
     int socket_fd;
     pthread_mutex_t lock;
     uint32_t next_request_id;
     bool connected;
     bool debug_enabled;
     char worker_host[256];
     int worker_port;
 } HipClientState;
 
 static HipClientState g_state = {
     .socket_fd = -1,
     .lock = PTHREAD_MUTEX_INITIALIZER,
     .next_request_id = 1,
     .connected = false,
     .debug_enabled = false,
     .worker_host = "localhost",
     .worker_port = HIP_REMOTE_DEFAULT_PORT
 };
 
 #define TF_LOG(fmt, ...) do { \
     if (g_state.debug_enabled) { \
         fprintf(stderr, "[TF-HIP] " fmt "\n", ##__VA_ARGS__); \
     } \
 } while (0)
 
 #define TF_ERROR(fmt, ...) \
     fprintf(stderr, "[TF-HIP ERROR] " fmt "\n", ##__VA_ARGS__)
 
 #if defined(MSG_NOSIGNAL)
 #define TF_SEND_FLAGS MSG_NOSIGNAL
 #else
 #define TF_SEND_FLAGS 0
 #endif
 
 static int tf_send_all(int fd, const void* buf, size_t len) {
     const uint8_t* p = (const uint8_t*)buf;
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
 
 static int tf_recv_all(int fd, void* buf, size_t len) {
     uint8_t* p = (uint8_t*)buf;
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
 
 static int tf_drain_bytes(int fd, size_t len) {
     uint8_t drain[256];
     while (len > 0) {
         size_t chunk = (len < sizeof(drain)) ? len : sizeof(drain);
         if (tf_recv_all(fd, drain, chunk) != 0) return -1;
         len -= chunk;
     }
     return 0;
 }
 
 static void mark_disconnected_locked(const char* reason) {
     if (reason) {
         TF_LOG("Marking disconnected: %s (errno=%d: %s)", reason, errno, strerror(errno));
     }
     if (g_state.socket_fd >= 0) close(g_state.socket_fd);
     g_state.socket_fd = -1;
     g_state.connected = false;
 }
 
 #define TF_SA_TOKEN_PATH "/var/run/secrets/kubernetes.io/serviceaccount/token"
 
 static int read_serviceaccount_token(char* out_token, size_t out_size) {
     if (!out_token || out_size == 0) return -1;
     out_token[0] = '\0';
 
     FILE* f = fopen(TF_SA_TOKEN_PATH, "r");
     if (!f) return -1;
     size_t n = fread(out_token, 1, out_size - 1, f);
     fclose(f);
     if (n == 0) return -1;
     out_token[n] = '\0';
 
     while (n > 0 && (out_token[n - 1] == '\n' || out_token[n - 1] == '\r' || out_token[n - 1] == ' ' || out_token[n - 1] == '\t')) {
         out_token[n - 1] = '\0';
         n--;
     }
     return (n > 0) ? 0 : -1;
 }
 
 static int fetch_connection_url(const char* url, char* out_buffer, size_t buffer_size) {
     char host[256] = {0};
     int port = 80;
     char path[512] = "/";
 
     if (!url || !out_buffer || buffer_size == 0) return -1;
     out_buffer[0] = '\0';
 
     if (strncmp(url, "http://", 7) == 0) url += 7;
 
     const char* port_start = strchr(url, ':');
     const char* path_start = strchr(url, '/');
 
     if (port_start && (!path_start || port_start < path_start)) {
         size_t host_len = (size_t)(port_start - url);
         if (host_len >= sizeof(host)) host_len = sizeof(host) - 1;
         strncpy(host, url, host_len);
         port = atoi(port_start + 1);
     } else if (path_start) {
         size_t host_len = (size_t)(path_start - url);
         if (host_len >= sizeof(host)) host_len = sizeof(host) - 1;
         strncpy(host, url, host_len);
     } else {
         strncpy(host, url, sizeof(host) - 1);
     }
     if (path_start) strncpy(path, path_start, sizeof(path) - 1);
 
     int sock = socket(AF_INET, SOCK_STREAM, 0);
     if (sock < 0) return -1;
 
     struct timeval timeout = {.tv_sec = 30, .tv_usec = 0};
     setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
     setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
 
     struct hostent* server = gethostbyname(host);
     if (!server) { close(sock); return -1; }
 
     struct sockaddr_in server_addr;
     memset(&server_addr, 0, sizeof(server_addr));
     server_addr.sin_family = AF_INET;
     memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
     server_addr.sin_port = htons((uint16_t)port);
 
     if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
         close(sock);
         return -1;
     }
 
     char token[4096];
     int has_token = (read_serviceaccount_token(token, sizeof(token)) == 0);
 
     char request[2048];
     int req_len;
     if (has_token) {
         req_len = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Authorization: Bearer %s\r\n"
             "Connection: close\r\n"
             "\r\n", path, host, port, token);
     } else {
         req_len = snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Connection: close\r\n"
             "\r\n", path, host, port);
     }
 
     if (req_len < 0 || req_len >= (int)sizeof(request)) { close(sock); return -1; }
     if (tf_send_all(sock, request, (size_t)req_len) != 0) { close(sock); return -1; }
 
     char response[4096];
     int total_read = 0;
     int bytes_read = 0;
     while ((bytes_read = (int)recv(sock, response + total_read, sizeof(response) - (size_t)total_read - 1, 0)) > 0) {
         total_read += bytes_read;
         if (total_read >= (int)sizeof(response) - 1) break;
     }
     response[total_read] = '\0';
     close(sock);
     if (total_read >= (int)sizeof(response) - 1 && bytes_read > 0) return -1;
 
     int status_code = 0;
     if (sscanf(response, "HTTP/%*s %d", &status_code) != 1) status_code = 0;
     if (status_code != 200) return -1;
 
     char* body = strstr(response, "\r\n\r\n");
     if (!body) return -1;
     body += 4;
     while (*body && (*body == ' ' || *body == '\n' || *body == '\r')) body++;
     size_t body_len = strlen(body);
     if (body_len == 0) return -1;
     char* end = body + body_len - 1;
     while (end > body && (*end == ' ' || *end == '\n' || *end == '\r')) *end-- = '\0';
 
     strncpy(out_buffer, body, buffer_size - 1);
     out_buffer[buffer_size - 1] = '\0';
     return 0;
 }
 
 static int parse_connection_url(const char* url, char* out_host, size_t host_size, int* out_port) {
     if (!url || !out_host || host_size == 0 || !out_port) return -1;
     char* url_copy = strdup(url);
     if (!url_copy) return -1;
 
     char* token = strtok(url_copy, "+");  // protocol
     if (!token) { free(url_copy); return -1; }
     token = strtok(NULL, "+");  // host
     if (!token) { free(url_copy); return -1; }
     strncpy(out_host, token, host_size - 1);
     out_host[host_size - 1] = '\0';
     token = strtok(NULL, "+");  // port
     if (!token) { free(url_copy); return -1; }
     *out_port = atoi(token);
     free(url_copy);
     return 0;
 }
 
 static int connect_to_worker(void) {
     if (g_state.connected) return 0;
 
     const char* debug = getenv("TF_DEBUG");
     if (debug && strcmp(debug, "1") == 0) g_state.debug_enabled = true;
 
     const char* connection_url_api = getenv("TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL");
     if (connection_url_api && strlen(connection_url_api) > 0) {
         char connection_url[512];
         if (fetch_connection_url(connection_url_api, connection_url, sizeof(connection_url)) != 0) return -1;
         if (parse_connection_url(connection_url, g_state.worker_host, sizeof(g_state.worker_host), &g_state.worker_port) != 0) return -1;
     } else {
         const char* host = getenv("TF_WORKER_HOST");
         if (host) {
             strncpy(g_state.worker_host, host, sizeof(g_state.worker_host) - 1);
             g_state.worker_host[sizeof(g_state.worker_host) - 1] = '\0';
         }
         const char* port_str = getenv("TF_WORKER_PORT");
         if (port_str) g_state.worker_port = atoi(port_str);
     }
 
     g_state.socket_fd = socket(AF_INET, SOCK_STREAM, 0);
     if (g_state.socket_fd < 0) return -1;
 
     int nodelay = 1;
     setsockopt(g_state.socket_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
     struct timeval io_timeout = {.tv_sec = 60, .tv_usec = 0};
     setsockopt(g_state.socket_fd, SOL_SOCKET, SO_RCVTIMEO, &io_timeout, sizeof(io_timeout));
     setsockopt(g_state.socket_fd, SOL_SOCKET, SO_SNDTIMEO, &io_timeout, sizeof(io_timeout));
 
     struct hostent* server = gethostbyname(g_state.worker_host);
     if (!server) { mark_disconnected_locked("resolve"); return -1; }
 
     struct sockaddr_in server_addr;
     memset(&server_addr, 0, sizeof(server_addr));
     server_addr.sin_family = AF_INET;
     memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
     server_addr.sin_port = htons((uint16_t)g_state.worker_port);
     if (connect(g_state.socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
         mark_disconnected_locked("connect");
         return -1;
     }
 
     g_state.connected = true;
 
     HipRemoteHeader header;
     hip_remote_init_header(&header, HIP_OP_INIT, g_state.next_request_id++, 0);
     if (tf_send_all(g_state.socket_fd, &header, sizeof(header)) != 0) { mark_disconnected_locked("send init"); return -1; }
     HipRemoteHeader resp_header;
     if (tf_recv_all(g_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) { mark_disconnected_locked("recv init hdr"); return -1; }
     if (hip_remote_validate_header(&resp_header) != 0 || resp_header.payload_length != sizeof(HipRemoteResponseHeader)) {
         mark_disconnected_locked("bad init hdr");
         return -1;
     }
     HipRemoteResponseHeader resp;
     if (tf_recv_all(g_state.socket_fd, &resp, sizeof(resp)) != 0) { mark_disconnected_locked("recv init body"); return -1; }
     if (resp.error_code != hipSuccess) { mark_disconnected_locked("init err"); return -1; }
 
     return 0;
 }
 
 static void disconnect_from_worker(void) {
     pthread_mutex_lock(&g_state.lock);
     if (!g_state.connected) { pthread_mutex_unlock(&g_state.lock); return; }
     HipRemoteHeader header;
     hip_remote_init_header(&header, HIP_OP_SHUTDOWN, g_state.next_request_id++, 0);
     (void)tf_send_all(g_state.socket_fd, &header, sizeof(header));
     mark_disconnected_locked("shutdown");
     pthread_mutex_unlock(&g_state.lock);
 }
 
 static int send_request(HipRemoteOpCode op_code, const void* payload, size_t payload_size) {
     pthread_mutex_lock(&g_state.lock);
     if (connect_to_worker() < 0) { pthread_mutex_unlock(&g_state.lock); return -1; }
 
     HipRemoteHeader header;
     hip_remote_init_header(&header, op_code, g_state.next_request_id++, (uint32_t)payload_size);
     if (tf_send_all(g_state.socket_fd, &header, sizeof(header)) != 0) { mark_disconnected_locked("send hdr"); pthread_mutex_unlock(&g_state.lock); return -1; }
     if (payload && payload_size > 0) {
         if (tf_send_all(g_state.socket_fd, payload, payload_size) != 0) { mark_disconnected_locked("send payload"); pthread_mutex_unlock(&g_state.lock); return -1; }
     }
     return 0;  // lock held
 }
 
 static int receive_response(void* response, size_t response_size) {
     HipRemoteHeader resp_header;
     if (tf_recv_all(g_state.socket_fd, &resp_header, sizeof(resp_header)) != 0) { mark_disconnected_locked("recv hdr"); pthread_mutex_unlock(&g_state.lock); return -1; }
     if (hip_remote_validate_header(&resp_header) != 0) { mark_disconnected_locked("bad hdr"); pthread_mutex_unlock(&g_state.lock); return -1; }
 
     if (response && response_size > 0) {
         size_t to_read = (resp_header.payload_length < response_size) ? resp_header.payload_length : response_size;
         if (tf_recv_all(g_state.socket_fd, response, to_read) != 0) { mark_disconnected_locked("recv body"); pthread_mutex_unlock(&g_state.lock); return -1; }
         if (resp_header.payload_length > response_size) (void)tf_drain_bytes(g_state.socket_fd, resp_header.payload_length - response_size);
     }
 
     pthread_mutex_unlock(&g_state.lock);
     return 0;
 }
 
 static hipError_t send_simple_request(HipRemoteOpCode op_code, const void* payload, size_t payload_size) {
     if (send_request(op_code, payload, payload_size) < 0) return hipErrorNotInitialized;
     HipRemoteResponseHeader resp;
     if (receive_response(&resp, sizeof(resp)) < 0) return hipErrorNotInitialized;
     return resp.error_code;
 }
 
 hipError_t hipGetDeviceCount(int* count) {
     if (!count) return hipErrorInvalidValue;
     if (send_request(HIP_OP_GET_DEVICE_COUNT, NULL, 0) < 0) return hipErrorNotInitialized;
     HipRemoteDeviceCountResponse resp;
     if (receive_response(&resp, sizeof(resp)) < 0) return hipErrorNotInitialized;
     *count = resp.count;
     return resp.header.error_code;
 }
 
 hipError_t hipSetDevice(int deviceId) {
     HipRemoteDeviceRequest req = { .device_id = deviceId };
     return send_simple_request(HIP_OP_SET_DEVICE, &req, sizeof(req));
 }
 
 hipError_t hipGetDevice(int* deviceId) {
     if (!deviceId) return hipErrorInvalidValue;
     if (send_request(HIP_OP_GET_DEVICE, NULL, 0) < 0) return hipErrorNotInitialized;
     HipRemoteGetDeviceResponse resp;
     if (receive_response(&resp, sizeof(resp)) < 0) return hipErrorNotInitialized;
     *deviceId = resp.device_id;
     return resp.header.error_code;
 }
 
 hipError_t hipGetLastError(void) {
     return send_simple_request(HIP_OP_GET_LAST_ERROR, NULL, 0);
 }
 
 hipError_t hipPeekAtLastError(void) {
     return send_simple_request(HIP_OP_PEEK_AT_LAST_ERROR, NULL, 0);
 }
 
 const char* hipGetErrorString(hipError_t error) {
     switch (error) {
         case hipSuccess: return "no error";
         case hipErrorInvalidValue: return "invalid argument";
         case hipErrorOutOfMemory: return "out of memory";
         case hipErrorNotInitialized: return "driver not initialized";
         case hipErrorInvalidDevice: return "invalid device ordinal";
         case hipErrorInvalidResourceHandle: return "invalid resource handle";
         case hipErrorNotReady: return "device not ready";
         default: return "unknown error";
     }
 }
 
 const char* hipGetErrorName(hipError_t error) {
     switch (error) {
         case hipSuccess: return "hipSuccess";
         case hipErrorInvalidValue: return "hipErrorInvalidValue";
         case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
         case hipErrorNotInitialized: return "hipErrorNotInitialized";
         case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
         case hipErrorInvalidResourceHandle: return "hipErrorInvalidResourceHandle";
         case hipErrorNotReady: return "hipErrorNotReady";
         default: return "hipErrorUnknown";
     }
 }
 
 __attribute__((constructor))
 static void hip_client_stub_init(void) {
     const char* debug = getenv("TF_DEBUG");
     if (debug && strcmp(debug, "1") == 0) g_state.debug_enabled = true;
     TF_LOG("TensorFusion HIP client stub loaded");
 }
 
 __attribute__((destructor))
 static void hip_client_stub_cleanup(void) {
     disconnect_from_worker();
 }
