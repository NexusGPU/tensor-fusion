# Makefile for building accelerator libraries
# Supports both example and vendor-specific implementations (NVIDIA, Ascend, etc.)

CC ?= gcc
CFLAGS ?= -Wall -Wextra -std=c11 -fPIC -O2
LDFLAGS ?= -shared

# Directories
PROVIDER_DIR := $(shell pwd)
STUB_DIR := $(PROVIDER_DIR)/example
ASCEND_DIR := $(PROVIDER_DIR)/ascend
AMD_DIR := $(PROVIDER_DIR)/amd
BUILD_DIR := $(PROVIDER_DIR)/build
TEST_DIR := $(PROVIDER_DIR)/test
MOCK_DIR := $(STUB_DIR)/device_mock

# Output libraries
STUB_LIB := $(BUILD_DIR)/libaccelerator_example.so
ASCEND_LIB := $(BUILD_DIR)/libaccelerator_ascend.so
AMD_LIB := $(BUILD_DIR)/libaccelerator_amd.so

# Source files
STUB_SRC := $(STUB_DIR)/accelerator.c
ASCEND_SRC := $(ASCEND_DIR)/accelerator.c
AMD_SRC := $(AMD_DIR)/accelerator.c

# Object files
STUB_OBJ := $(BUILD_DIR)/accelerator_example.o
ASCEND_OBJ := $(BUILD_DIR)/accelerator_ascend.o
AMD_OBJ := $(BUILD_DIR)/accelerator_amd.o

# Test executables
TEST_BIN := $(BUILD_DIR)/test_accelerator

# Device mock files
MOCK_DRIVER_SRC := $(MOCK_DIR)/driver_mock.c
MOCK_DRIVER_OBJ := $(BUILD_DIR)/driver_mock.o
MOCK_DRIVER_LIB := $(BUILD_DIR)/libdriver_mock.a
MOCK_APP_SRC := $(MOCK_DIR)/app_mock.c
MOCK_APP_BIN := $(BUILD_DIR)/app_mock

# ROCm paths (TheRock installs to /opt/rocm via symlink)
ROCM_PATH ?= /opt/rocm
AMD_CFLAGS := -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__
AMD_LDFLAGS := -L$(ROCM_PATH)/lib -lamd_smi -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

.PHONY: all clean example ascend amd test test-amd-run install mock mock-driver mock-app mock-run mock-test mock-test-run mock-rate-test

all: example

# Build example implementation
example: mock-driver $(STUB_LIB)

$(STUB_LIB): $(STUB_OBJ) $(MOCK_DRIVER_LIB) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $< -L$(BUILD_DIR) -ldriver_mock -lpthread

$(STUB_OBJ): $(STUB_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(PROVIDER_DIR) -I$(MOCK_DIR) -c -o $@ $<

# Build Ascend implementation (requires Ascend CANN SDK)
ascend: $(ASCEND_LIB)

$(ASCEND_LIB): $(ASCEND_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $< $(ASCEND_LDFLAGS)

$(ASCEND_OBJ): $(ASCEND_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(PROVIDER_DIR) $(ASCEND_CFLAGS) -c -o $@ $<

# Build AMD implementation (requires ROCm/TheRock)
amd: $(AMD_LIB)

$(AMD_LIB): $(AMD_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $< $(AMD_LDFLAGS)

$(AMD_OBJ): $(AMD_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(AMD_CFLAGS) -I$(PROVIDER_DIR) -c -o $@ $<

# Build test executable
test: $(TEST_BIN)

$(TEST_BIN): $(TEST_DIR)/test_accelerator.c $(STUB_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(PROVIDER_DIR) -o $@ $(TEST_DIR)/test_accelerator.c -L$(BUILD_DIR) -laccelerator_example -Wl,-rpath,$(BUILD_DIR)

# Run tests
test-run: test
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_BIN)

# Test AMD provider
test-amd-run: amd test
	LD_LIBRARY_PATH=$(ROCM_PATH)/lib:$(BUILD_DIR):$$LD_LIBRARY_PATH \
	$(TEST_BIN) $(AMD_LIB)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build device mock driver library
mock-driver: $(MOCK_DRIVER_LIB)

$(MOCK_DRIVER_LIB): $(MOCK_DRIVER_OBJ) | $(BUILD_DIR)
	ar rcs $@ $<

$(MOCK_DRIVER_OBJ): $(MOCK_DRIVER_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(MOCK_DIR) -pthread -c -o $@ $<

# Build device mock app
mock-app: $(MOCK_APP_BIN)

$(MOCK_APP_BIN): $(MOCK_APP_SRC) $(MOCK_DRIVER_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(MOCK_DIR) -pthread -o $@ $(MOCK_APP_SRC) -L$(BUILD_DIR) -ldriver_mock -lpthread

# Test metrics
MOCK_TEST_SRC := $(MOCK_DIR)/test_metrics.c
MOCK_TEST_BIN := $(BUILD_DIR)/test_metrics
MOCK_RATE_TEST_SRC := $(MOCK_DIR)/test_rate_limit.c
MOCK_RATE_TEST_BIN := $(BUILD_DIR)/test_rate_limit

mock-test: $(MOCK_TEST_BIN)

$(MOCK_TEST_BIN): $(MOCK_TEST_SRC) $(MOCK_DRIVER_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(MOCK_DIR) -pthread -o $@ $(MOCK_TEST_SRC) -L$(BUILD_DIR) -ldriver_mock -lpthread

mock-rate-test: $(MOCK_RATE_TEST_BIN)

$(MOCK_RATE_TEST_BIN): $(MOCK_RATE_TEST_SRC) $(MOCK_DRIVER_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(MOCK_DIR) -pthread -o $@ $(MOCK_RATE_TEST_SRC) -L$(BUILD_DIR) -ldriver_mock -lpthread

# Build both mock driver and app
mock: mock-driver mock-app

# Run mock app (with default parameters)
mock-run: mock-app
	cd $(BUILD_DIR) && ./app_mock

# Run metrics test
mock-test-run: mock-test
	cd $(BUILD_DIR) && ./test_metrics

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MOCK_DIR)/tmp.example_accelerator.bin

# Install libraries to system path (optional)
install: $(STUB_LIB)
	install -d /usr/local/lib/tensor-fusion
	install -m 755 $(STUB_LIB) /usr/local/lib/tensor-fusion/
	install -d /usr/local/include/tensor-fusion
	install -m 644 $(PROVIDER_DIR)/accelerator.h /usr/local/include/tensor-fusion/
	install -m 644 $(PROVIDER_DIR)/limiter.h /usr/local/include/tensor-fusion/

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build example implementation (default)"
	@echo "  example      - Build example accelerator library"
	@echo "  ascend       - Build Ascend accelerator library (requires CANN SDK)"
	@echo "  amd          - Build AMD accelerator library (requires ROCm/TheRock)"
	@echo "  test         - Build test executable"
	@echo "  test-run     - Build and run tests"
	@echo "  test-amd-run - Build and test AMD provider"
	@echo "  mock         - Build device mock driver and app"
	@echo "  mock-driver  - Build device mock driver library"
	@echo "  mock-app     - Build device mock app"
	@echo "  mock-run     - Build and run mock app with default parameters"
	@echo "  mock-test    - Build metrics test"
	@echo "  mock-test-run - Build and run metrics test"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install libraries to system path"
	@echo "  help         - Show this help message"

