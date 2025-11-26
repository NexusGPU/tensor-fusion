# Makefile for building accelerator libraries
# Supports both stub and vendor-specific implementations (NVIDIA, Ascend, etc.)

CC ?= gcc
CXX ?= g++
CFLAGS ?= -Wall -Wextra -std=c11 -fPIC -O2
CXXFLAGS ?= -Wall -Wextra -std=c++17 -fPIC -O2
LDFLAGS ?= -shared

# Directories
PROVIDER_DIR := $(shell pwd)
STUB_DIR := $(PROVIDER_DIR)/stub
ASCEND_DIR := $(PROVIDER_DIR)/ascend
BUILD_DIR := $(PROVIDER_DIR)/build
TEST_DIR := $(PROVIDER_DIR)/test

# Extra flags for Ascend implementation
ASCEND_CFLAGS += -pthread -I/usr/local/dcmi
ASCEND_LDFLAGS += -pthread -ldl

# Test binary for Ascend fake validation
TEST_ASCEND_BIN := $(BUILD_DIR)/test_ascend_fake

# Output libraries
STUB_LIB := $(BUILD_DIR)/libaccelerator_stub.so
ASCEND_LIB := $(BUILD_DIR)/libaccelerator_ascend.so

# Source files
STUB_SRC := $(STUB_DIR)/accelerator.c
ASCEND_SRC := $(ASCEND_DIR)/accelerator.cpp

# Object files
STUB_OBJ := $(BUILD_DIR)/accelerator_stub.o
ASCEND_OBJ := $(BUILD_DIR)/accelerator_ascend.o

# Test executables
TEST_BIN := $(BUILD_DIR)/test_accelerator
TEST_ASCEND_BIN := $(BUILD_DIR)/test_ascend_fake
TEST_ASCEND_COMPREHENSIVE_BIN := $(BUILD_DIR)/test_ascend_comprehensive

.PHONY: all clean stub ascend test test-ascend test-comprehensive install help

all: stub

# Build stub implementation
stub: $(STUB_LIB)

$(STUB_LIB): $(STUB_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $<

$(STUB_OBJ): $(STUB_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(PROVIDER_DIR) -c -o $@ $<

# Build Ascend implementation (requires Ascend CANN SDK)
ascend: $(ASCEND_LIB)

$(ASCEND_LIB): $(ASCEND_OBJ) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $@ $< $(ASCEND_LDFLAGS)

$(ASCEND_OBJ): $(ASCEND_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROVIDER_DIR) $(ASCEND_CFLAGS) -c -o $@ $<

# Build stub test executable
test: $(TEST_BIN)

$(TEST_BIN): $(TEST_DIR)/test_accelerator.c $(STUB_LIB) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(PROVIDER_DIR) -o $@ $(TEST_DIR)/test_accelerator.c -L$(BUILD_DIR) -laccelerator_stub -Wl,-rpath,$(BUILD_DIR)

test-run: test
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_BIN)

# Build Ascend fake partition test
test-ascend: $(TEST_ASCEND_BIN)

$(TEST_ASCEND_BIN): $(TEST_DIR)/ascend_partition_fake.c $(ASCEND_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROVIDER_DIR) -o $@ $(TEST_DIR)/ascend_partition_fake.c -L$(BUILD_DIR) -laccelerator_ascend $(ASCEND_LDFLAGS) -Wl,-rpath,$(BUILD_DIR)

test-ascend-run: test-ascend
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ASCEND_BIN)

# Build comprehensive Ascend test tool (all-in-one test suite)
test-comprehensive: $(TEST_ASCEND_COMPREHENSIVE_BIN)

$(TEST_ASCEND_COMPREHENSIVE_BIN): $(ASCEND_DIR)/tests/test_ascend_comprehensive.cpp $(ASCEND_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(PROVIDER_DIR) $(ASCEND_CFLAGS) -o $@ $< -L$(BUILD_DIR) -laccelerator_ascend $(ASCEND_LDFLAGS) -Wl,-rpath,$(BUILD_DIR)

test-comprehensive-run: test-comprehensive
	@echo "Note: You may need to set DCMI_LIB_PATH and run with sudo -E"
	@echo "Example: sudo -E DCMI_LIB_PATH=/usr/local/dcmi/libdcmi.so make test-comprehensive-run"
	LD_LIBRARY_PATH=$(BUILD_DIR):$$LD_LIBRARY_PATH $(TEST_ASCEND_COMPREHENSIVE_BIN)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

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
	@echo "  all                  - Build stub implementation (default)"
	@echo "  stub                 - Build stub accelerator library"
	@echo "  ascend               - Build Ascend accelerator library (requires CANN SDK)"
	@echo ""
	@echo "Test targets:"
	@echo "  test                 - Build stub test executable"
	@echo "  test-run             - Build and run stub tests"
	@echo "  test-ascend          - Build Ascend fake partition test"
	@echo "  test-ascend-run      - Build and run Ascend fake partition test"
	@echo "  test-comprehensive   - Build comprehensive Ascend test suite (recommended)"
	@echo "  test-comprehensive-run - Build and run comprehensive test"
	@echo ""
	@echo "Other targets:"
	@echo "  clean                - Remove build artifacts"
	@echo "  install              - Install libraries to system path"
	@echo "  help                 - Show this help message"
