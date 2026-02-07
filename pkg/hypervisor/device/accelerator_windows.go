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
	purego.RegisterLibFunc(&accelInit, libHandle, "AccelInit")
	purego.RegisterLibFunc(&accelShutdown, libHandle, "AccelShutdown")
	purego.RegisterLibFunc(&getDeviceCount, libHandle, "AccelGetDeviceCount")
	purego.RegisterLibFunc(&getAllDevices, libHandle, "AccelGetAllDevices")
	purego.RegisterLibFunc(&getAllDevicesTopology, libHandle, "AccelGetAllDevicesTopology")
	purego.RegisterLibFunc(&assignPartition, libHandle, "AccelAssignPartition")
	purego.RegisterLibFunc(&removePartition, libHandle, "AccelRemovePartition")
	purego.RegisterLibFunc(&setMemHardLimit, libHandle, "AccelSetMemHardLimit")
	purego.RegisterLibFunc(&setComputeUnitHardLimit, libHandle, "AccelSetComputeUnitHardLimit")
	purego.RegisterLibFunc(&snapshot, libHandle, "AccelSnapshot")
	purego.RegisterLibFunc(&resume, libHandle, "AccelResume")
	purego.RegisterLibFunc(&getProcessInformation, libHandle, "AccelGetProcessInformation")
	purego.RegisterLibFunc(&getDeviceMetrics, libHandle, "AccelGetDeviceMetrics")
	purego.RegisterLibFunc(&getVendorMountLibs, libHandle, "AccelGetVendorMountLibs")

	// Note: Log callback is not registered on Windows due to calling convention differences
	// Windows uses stdcall/cdecl while Unix uses System V ABI

	result := accelInit()
	if result != ResultSuccess {
		return fmt.Errorf("failed to initialize accelerator: %d", result)
	}

	a.loaded = true
	return nil
}
