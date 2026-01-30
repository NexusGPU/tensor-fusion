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
	purego.RegisterLibFunc(&vgpuInit, handle, "VGPUInit")
	purego.RegisterLibFunc(&vgpuShutdown, handle, "VGPUShutdown")
	purego.RegisterLibFunc(&getDeviceCount, handle, "GetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, handle, "GetAllDevices")
	purego.RegisterLibFunc(&getAllDevicesTopology, handle, "GetAllDevicesTopology")
	purego.RegisterLibFunc(&assignPartition, handle, "AssignPartition")
	purego.RegisterLibFunc(&removePartition, handle, "RemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, handle, "SetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, handle, "SetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, handle, "Snapshot")
	purego.RegisterLibFunc(&resume, handle, "Resume")
	purego.RegisterLibFunc(&getProcessInformation, handle, "GetProcessInformation")
	purego.RegisterLibFunc(&getDeviceMetrics, handle, "GetDeviceMetrics")
	purego.RegisterLibFunc(&getVendorMountLibs, handle, "GetVendorMountLibs")

	// Register log callback only on non-macOS platforms
	// purego callback has issues on macOS ARM64, causing bus errors when C code calls back into Go
	if runtime.GOOS != "darwin" {
		purego.RegisterLibFunc(&registerLogCallback, handle, "RegisterLogCallback")
		callback := purego.NewCallback(goLogCallback)
		if result := registerLogCallback(callback); result != ResultSuccess {
			klog.Warningf("Failed to register log callback: %d", result)
		}
	}

	result := vgpuInit()
	if result != ResultSuccess {
		return fmt.Errorf("failed to initialize VGPU: %d", result)
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
