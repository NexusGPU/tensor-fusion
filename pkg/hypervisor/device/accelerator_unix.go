//go:build darwin || linux || freebsd || netbsd

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

package device

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
	"k8s.io/klog/v2"
)

// Load loads the accelerator library dynamically using purego on Unix systems
func (a *AcceleratorInterface) Load() error {
	if a.libPath == "" {
		return fmt.Errorf("library path is empty")
	}

	handle, err := purego.Dlopen(a.libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("failed to open library: %w", err)
	}
	libHandle = handle

	// Register all required functions - names must match C header exactly
	purego.RegisterLibFunc(&accelInit, handle, "AccelInit")
	purego.RegisterLibFunc(&accelShutdown, handle, "AccelShutdown")
	purego.RegisterLibFunc(&getDeviceCount, handle, "AccelGetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, handle, "AccelGetAllDevices")
	purego.RegisterLibFunc(&getAllDevicesTopology, handle, "AccelGetAllDevicesTopology")
	purego.RegisterLibFunc(&assignPartition, handle, "AccelAssignPartition")
	purego.RegisterLibFunc(&removePartition, handle, "AccelRemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, handle, "AccelSetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, handle, "AccelSetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, handle, "AccelSnapshot")
	purego.RegisterLibFunc(&resume, handle, "AccelResume")
	purego.RegisterLibFunc(&getProcessInformation, handle, "AccelGetProcessInformation")
	purego.RegisterLibFunc(&getDeviceMetrics, handle, "AccelGetDeviceMetrics")
	purego.RegisterLibFunc(&getVendorMountLibs, handle, "AccelGetVendorMountLibs")

	// Register log callback only on non-macOS platforms
	// purego callback has issues on macOS ARM64, causing bus errors when C code calls back into Go
	if runtime.GOOS != "darwin" {
		purego.RegisterLibFunc(&registerLogCallback, handle, "AccelRegisterLogCallback")
		callback := purego.NewCallback(goLogCallback)
		if result := registerLogCallback(callback); result != ResultSuccess {
			klog.Warningf("Failed to register log callback: %d", result)
		}
	}

	result := accelInit()
	if result != ResultSuccess {
		return fmt.Errorf("failed to initialize accelerator: %d", result)
	}

	a.loaded = true
	return nil
}

// goLogCallback is the Go callback function called by C library for logging
// Note: Only used on Unix platforms. Disabled on macOS due to purego callback issues on ARM64.
func goLogCallback(level *byte, message *byte) {
	var levelStr, messageStr string
	if level != nil {
		levelStr = cStringToGoString(level)
	}
	if message != nil {
		messageStr = cStringToGoString(message)
	}

	// Map C log levels to klog levels
	switch levelStr {
	case "DEBUG", "debug":
		klog.V(4).Info(messageStr)
	case "INFO", "info":
		klog.Info(messageStr)
	case "WARN", "warn", "WARNING", "warning":
		klog.Warning(messageStr)
	case "ERROR", "error":
		klog.Error(messageStr)
	case "FATAL", "fatal":
		klog.Fatal(messageStr)
	default:
		klog.Info(messageStr)
	}
}

// cStringToGoString converts a C string (null-terminated byte array) to Go string
func cStringToGoString(cstr *byte) string {
	if cstr == nil {
		return ""
	}
	ptr := unsafe.Pointer(cstr)
	length := 0
	for *(*byte)(unsafe.Add(ptr, uintptr(length))) != 0 {
		length++
	}
	return string(unsafe.Slice(cstr, length))
}
