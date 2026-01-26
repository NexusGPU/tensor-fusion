# Makefile for AMD HIP remote GPU components
#
# Builds:
#   - libhip_client_stub.so: LD_PRELOAD library for HIP API interception
#   - hip_worker_service: Worker daemon that executes HIP calls
#
# Usage:
#   make            - Build all components
#   make client     - Build client stub only
#   make worker     - Build worker service only
#   make clean      - Remove build artifacts
#   make test       - Run basic tests

CC ?= gcc
CFLAGS := -Wall -Wextra -Werror -O2 -fPIC -D_GNU_SOURCE
LDFLAGS := -shared
LIBS := -lpthread

# ROCm/HIP paths (for worker service)
# Modern ROCm (6.x+) puts HIP at /opt/rocm/include/hip
# Older versions used /opt/rocm/hip/include
ROCM_PATH ?= /opt/rocm
HIP_INCLUDE := $(ROCM_PATH)/include

# Check if HIP is available for worker build
HIP_AVAILABLE := $(shell test -f $(HIP_INCLUDE)/hip/hip_runtime.h && echo yes || echo no)

.PHONY: all client worker clean test

all: client worker

# Client stub - no HIP dependency, pure C
client: libhip_client_stub.so

libhip_client_stub.so: hip_client_stub.c hip_remote_protocol.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo "Built $@"

# Worker service - requires HIP runtime
worker: hip_worker_service

ifeq ($(HIP_AVAILABLE),yes)
# HIP requires __HIP_PLATFORM_AMD__ to be defined for AMD GPUs
HIP_CFLAGS := -D__HIP_PLATFORM_AMD__ -I$(HIP_INCLUDE)

hip_worker_service: hip_worker_service.c hip_remote_protocol.h
	$(CC) $(CFLAGS) $(HIP_CFLAGS) -o $@ $< -L$(ROCM_PATH)/lib -lamdhip64 $(LIBS)
	@echo "Built $@"
else
hip_worker_service:
	@echo "Warning: HIP not found at $(HIP_INCLUDE)/hip/hip_runtime.h"
	@echo "Set ROCM_PATH to your ROCm installation to build the worker service"
endif

clean:
	rm -f libhip_client_stub.so hip_worker_service *.o

# Basic test - just verify client stub loads
test: client
	@echo "Testing client stub load..."
	@LD_PRELOAD=./libhip_client_stub.so TF_DEBUG=1 /bin/true && echo "Client stub loads successfully"

# Install to standard locations
install: all
	install -d $(DESTDIR)/usr/lib/tensor-fusion
	install -m 755 libhip_client_stub.so $(DESTDIR)/usr/lib/tensor-fusion/
ifeq ($(HIP_AVAILABLE),yes)
	install -m 755 hip_worker_service $(DESTDIR)/usr/lib/tensor-fusion/
endif

# Debug build
debug: CFLAGS += -g -O0 -DDEBUG
debug: all
