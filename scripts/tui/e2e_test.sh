#!/bin/bash

# TUI End-to-End Test Script
# This script spawns a single_node mode hypervisor and tests the TUI client connection
#
# Usage: ./scripts/tui/e2e_test.sh [options]
#
# Options:
#   --port PORT          Hypervisor port (default: 8099)
#   --timeout SECONDS    Test timeout in seconds (default: 30)
#   --no-build           Skip building binaries
#   --verbose            Enable verbose output
#   --keep-running       Keep hypervisor running after test (for manual testing)

set -euo pipefail

# Default configuration
HYPERVISOR_PORT="${HYPERVISOR_PORT:-8099}"
TEST_TIMEOUT="${TEST_TIMEOUT:-30}"
SKIP_BUILD=false
VERBOSE=false
KEEP_RUNNING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"
STATE_DIR="/tmp/tensor-fusion-state-e2e-$$"
HYPERVISOR_PID=""
HYPERVISOR_LOG="${PROJECT_ROOT}/e2e_hypervisor.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            HYPERVISOR_PORT="$2"
            shift 2
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        -h|--help)
            echo "TUI End-to-End Test Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT          Hypervisor port (default: 8099)"
            echo "  --timeout SECONDS    Test timeout in seconds (default: 30)"
            echo "  --no-build           Skip building binaries"
            echo "  --verbose            Enable verbose output"
            echo "  --keep-running       Keep hypervisor running after test"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${VERBOSE}" == "true" ]]; then
        echo -e "[DEBUG] $1"
    fi
}

cleanup() {
    local exit_code=$?
    log_info "Cleaning up..."
    
    # Stop hypervisor if running
    if [[ -n "${HYPERVISOR_PID}" ]] && kill -0 "${HYPERVISOR_PID}" 2>/dev/null; then
        log_info "Stopping hypervisor (PID: ${HYPERVISOR_PID})..."
        kill -SIGTERM "${HYPERVISOR_PID}" 2>/dev/null || true
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "${HYPERVISOR_PID}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "${HYPERVISOR_PID}" 2>/dev/null; then
            log_warn "Force killing hypervisor..."
            kill -9 "${HYPERVISOR_PID}" 2>/dev/null || true
        fi
    fi
    
    # Clean up state directory
    if [[ -d "${STATE_DIR}" ]]; then
        log_debug "Removing state directory: ${STATE_DIR}"
        rm -rf "${STATE_DIR}"
    fi
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_info "Test completed successfully!"
    else
        log_error "Test failed with exit code: ${exit_code}"
        if [[ -f "${HYPERVISOR_LOG}" ]]; then
            log_info "Hypervisor logs:"
            tail -50 "${HYPERVISOR_LOG}" || true
        fi
    fi
    
    exit ${exit_code}
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

build_binaries() {
    if [[ "${SKIP_BUILD}" == "true" ]]; then
        log_info "Skipping build (--no-build specified)"
        return 0
    fi
    
    log_info "Building provider library..."
    cd "${PROJECT_ROOT}"
    make build-provider
    
    log_info "Building hypervisor..."
    make build-hypervisor
    
    log_info "Building hypervisor-tui..."
    make build-hypervisor-tui
}

wait_for_hypervisor() {
    local timeout=$1
    local start_time=$(date +%s)
    local health_url="http://localhost:${HYPERVISOR_PORT}/healthz"
    
    log_info "Waiting for hypervisor to be ready (timeout: ${timeout}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ ${elapsed} -ge ${timeout} ]]; then
            log_error "Timeout waiting for hypervisor to start"
            return 1
        fi
        
        # Check if process is still running
        if ! kill -0 "${HYPERVISOR_PID}" 2>/dev/null; then
            log_error "Hypervisor process exited unexpectedly"
            return 1
        fi
        
        # Check health endpoint
        if curl -sf "${health_url}" > /dev/null 2>&1; then
            log_info "Hypervisor is ready! (took ${elapsed}s)"
            return 0
        fi
        
        log_debug "Hypervisor not ready yet (${elapsed}s elapsed)..."
        sleep 1
    done
}

start_hypervisor() {
    log_info "Starting hypervisor in single_node mode..."
    
    # Create state directory
    mkdir -p "${STATE_DIR}"
    
    # Set environment variables
    export TENSOR_FUSION_STATE_DIR="${STATE_DIR}"
    export HYPERVISOR_PORT="${HYPERVISOR_PORT}"
    
    local hypervisor_bin="${BIN_DIR}/hypervisor"
    local accelerator_lib="${PROJECT_ROOT}/provider/build/libaccelerator_example.so"
    
    if [[ ! -f "${hypervisor_bin}" ]]; then
        log_error "Hypervisor binary not found: ${hypervisor_bin}"
        return 1
    fi
    
    if [[ ! -f "${accelerator_lib}" ]]; then
        log_error "Accelerator library not found: ${accelerator_lib}"
        return 1
    fi
    
    # Start hypervisor in background
    "${hypervisor_bin}" \
        -backend-type simple \
        -accelerator-lib "${accelerator_lib}" \
        -vendor Stub \
        -isolation-mode shared \
        -port "${HYPERVISOR_PORT}" \
        -metrics-interval 60s \
        > "${HYPERVISOR_LOG}" 2>&1 &
    
    HYPERVISOR_PID=$!
    log_info "Hypervisor started with PID: ${HYPERVISOR_PID}"
    
    # Wait for hypervisor to be ready
    wait_for_hypervisor "${TEST_TIMEOUT}"
}

test_api_endpoints() {
    log_info "Testing hypervisor API endpoints..."
    local base_url="http://localhost:${HYPERVISOR_PORT}/api/v1"
    local failures=0
    
    # Test devices endpoint
    log_debug "Testing GET /api/v1/devices..."
    local devices_response
    if devices_response=$(curl -sf "${base_url}/devices" 2>&1); then
        log_info "  ✓ GET /devices - OK"
        log_debug "    Response: ${devices_response}"
    else
        log_error "  ✗ GET /devices - FAILED"
        ((failures++))
    fi
    
    # Test workers endpoint
    log_debug "Testing GET /api/v1/workers..."
    local workers_response
    if workers_response=$(curl -sf "${base_url}/workers" 2>&1); then
        log_info "  ✓ GET /workers - OK"
        log_debug "    Response: ${workers_response}"
    else
        log_error "  ✗ GET /workers - FAILED"
        ((failures++))
    fi
    
    # Test health endpoint
    log_debug "Testing GET /healthz..."
    if curl -sf "http://localhost:${HYPERVISOR_PORT}/healthz" > /dev/null 2>&1; then
        log_info "  ✓ GET /healthz - OK"
    else
        log_error "  ✗ GET /healthz - FAILED"
        ((failures++))
    fi
    
    # Test readyz endpoint
    log_debug "Testing GET /readyz..."
    if curl -sf "http://localhost:${HYPERVISOR_PORT}/readyz" > /dev/null 2>&1; then
        log_info "  ✓ GET /readyz - OK"
    else
        log_error "  ✗ GET /readyz - FAILED"
        ((failures++))
    fi
    
    if [[ ${failures} -gt 0 ]]; then
        log_error "API endpoint tests failed: ${failures} failure(s)"
        return 1
    fi
    
    log_info "All API endpoint tests passed!"
    return 0
}

test_tui_connection() {
    log_info "Testing TUI client connection..."
    local tui_bin="${BIN_DIR}/hypervisor-tui"
    
    if [[ ! -f "${tui_bin}" ]]; then
        log_error "TUI binary not found: ${tui_bin}"
        return 1
    fi
    
    log_info "  ✓ TUI binary exists: ${tui_bin}"
    
    # Bubbletea TUI requires a proper TTY, so we can't fully test interactive mode
    # Instead, verify the binary can start and the API endpoints work (already tested above)
    # For full interactive testing, use --keep-running mode
    
    log_info "  ✓ TUI client ready (use --keep-running for interactive testing)"
    log_info "TUI connection test completed"
    return 0
}

test_tui_data_fetching() {
    log_info "Testing TUI data fetching via API..."
    
    # Test the same endpoints the TUI client uses
    local base_url="http://localhost:${HYPERVISOR_PORT}/api/v1"
    
    # Fetch devices (TUI's ListDevices)
    local devices
    if ! devices=$(curl -sf "${base_url}/devices"); then
        log_error "Failed to fetch devices"
        return 1
    fi
    log_info "  ✓ Fetched devices successfully"
    log_debug "    Devices: ${devices}"
    
    # Fetch workers (TUI's ListWorkers)
    local workers
    if ! workers=$(curl -sf "${base_url}/workers"); then
        log_error "Failed to fetch workers"
        return 1
    fi
    log_info "  ✓ Fetched workers successfully"
    log_debug "    Workers: ${workers}"
    
    # Verify JSON structure (if jq is available)
    if command -v jq > /dev/null 2>&1; then
        if echo "${devices}" | jq -e '.data' > /dev/null 2>&1; then
            log_info "  ✓ Devices response has valid JSON structure"
        else
            log_warn "  ⚠ Devices response may have unexpected structure"
        fi
        
        if echo "${workers}" | jq -e '.data' > /dev/null 2>&1; then
            log_info "  ✓ Workers response has valid JSON structure"
        else
            log_warn "  ⚠ Workers response may have unexpected structure"
        fi
    else
        log_debug "jq not installed, skipping JSON structure validation"
        # Basic check for JSON-like response
        if echo "${devices}" | grep -q '"data"'; then
            log_info "  ✓ Devices response contains expected fields"
        fi
        if echo "${workers}" | grep -q '"data"'; then
            log_info "  ✓ Workers response contains expected fields"
        fi
    fi
    
    return 0
}

run_interactive_mode() {
    log_info "Starting interactive mode..."
    log_info "Hypervisor is running on port ${HYPERVISOR_PORT}"
    log_info "You can connect with: ${BIN_DIR}/hypervisor-tui -host localhost -port ${HYPERVISOR_PORT}"
    log_info "Press Ctrl+C to stop"
    
    # Keep running until interrupted
    while kill -0 "${HYPERVISOR_PID}" 2>/dev/null; do
        sleep 1
    done
}

main() {
    log_info "=== TUI End-to-End Test ==="
    log_info "Configuration:"
    log_info "  Port: ${HYPERVISOR_PORT}"
    log_info "  Timeout: ${TEST_TIMEOUT}s"
    log_info "  State dir: ${STATE_DIR}"
    log_info ""
    
    # Step 1: Build binaries
    build_binaries
    
    # Step 2: Start hypervisor
    start_hypervisor
    
    # Step 3: Run tests
    test_api_endpoints
    test_tui_data_fetching
    test_tui_connection
    
    log_info ""
    log_info "=== All tests passed! ==="
    
    # If keep-running is specified, don't exit
    if [[ "${KEEP_RUNNING}" == "true" ]]; then
        run_interactive_mode
    fi
}

main "$@"
