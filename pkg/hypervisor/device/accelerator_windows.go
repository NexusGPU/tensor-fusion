//go:build windows

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
	"syscall"

	"github.com/ebitengine/purego"
)

// Load loads the accelerator library dynamically using syscall.LoadDLL on Windows
func (a *AcceleratorInterface) Load() error {
	if a.libPath == "" {
		return fmt.Errorf("library path is empty")
	}

	// Windows: use syscall.LoadDLL instead of purego.Dlopen
	dll, err := syscall.LoadDLL(a.libPath)
	if err != nil {
		return fmt.Errorf("failed to open library: %w", err)
	}
	libHandle = uintptr(dll.Handle)

	// purego.RegisterLibFunc works with Windows DLL handles - names must match C header exactly
	purego.RegisterLibFunc(&vgpuInit, libHandle, "VGPUInit")
	purego.RegisterLibFunc(&vgpuShutdown, libHandle, "VGPUShutdown")
	purego.RegisterLibFunc(&getDeviceCount, libHandle, "GetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, libHandle, "GetAllDevices")
	purego.RegisterLibFunc(&getAllDevicesTopology, libHandle, "GetAllDevicesTopology")
	purego.RegisterLibFunc(&assignPartition, libHandle, "AssignPartition")
	purego.RegisterLibFunc(&removePartition, libHandle, "RemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, libHandle, "SetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, libHandle, "SetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, libHandle, "Snapshot")
	purego.RegisterLibFunc(&resume, libHandle, "Resume")
	purego.RegisterLibFunc(&getProcessInformation, libHandle, "GetProcessInformation")
	purego.RegisterLibFunc(&getDeviceMetrics, libHandle, "GetDeviceMetrics")
	purego.RegisterLibFunc(&getVendorMountLibs, libHandle, "GetVendorMountLibs")

	// Note: Log callback is not registered on Windows due to calling convention differences
	// Windows uses stdcall/cdecl while Unix uses System V ABI

	result := vgpuInit()
	if result != ResultSuccess {
		return fmt.Errorf("failed to initialize VGPU: %d", result)
	}

	a.loaded = true
	return nil
}
