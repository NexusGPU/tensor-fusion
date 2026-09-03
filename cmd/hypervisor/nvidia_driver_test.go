package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

func TestConfigureNvidiaDriverLibraries(t *testing.T) {
	driverRoot := t.TempDir()
	archDir := map[string]string{
		"amd64": "x86_64-linux-gnu",
		"arm64": "aarch64-linux-gnu",
	}[runtime.GOARCH]
	if archDir == "" {
		t.Skipf("unsupported test architecture %s", runtime.GOARCH)
	}
	libraryDir := filepath.Join(driverRoot, "usr", "lib", archDir)
	if err := os.MkdirAll(libraryDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cudaPath := filepath.Join(libraryDir, nvidiaCudaLibraryName)
	nvmlPath := filepath.Join(libraryDir, nvidiaNvmlLibraryName)
	for _, path := range []string{cudaPath, nvmlPath} {
		if err := os.WriteFile(path, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	contractPath := filepath.Join(t.TempDir(), "driver-ready")
	contract := "IS_HOST_DRIVER=false\n" + driverRootCtrPathEnv + "=" + driverRoot + "\n"
	if err := os.WriteFile(contractPath, []byte(contract), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(constants.TFCudaLibPathEnv, "")
	t.Setenv(constants.TFNvmlLibPathEnv, "")

	configureNvidiaDriverLibraries(constants.AcceleratorVendorNvidia, contractPath)

	if got := os.Getenv(constants.TFCudaLibPathEnv); got != cudaPath {
		t.Fatalf("%s = %q, want %q", constants.TFCudaLibPathEnv, got, cudaPath)
	}
	if got := os.Getenv(constants.TFNvmlLibPathEnv); got != nvmlPath {
		t.Fatalf("%s = %q, want %q", constants.TFNvmlLibPathEnv, got, nvmlPath)
	}
}

func TestConfigureNvidiaDriverLibrariesPreservesExplicitPaths(t *testing.T) {
	t.Setenv(constants.TFCudaLibPathEnv, "/custom/libcuda.so.1")
	t.Setenv(constants.TFNvmlLibPathEnv, "/custom/libnvidia-ml.so.1")

	configureNvidiaDriverLibraries(constants.AcceleratorVendorNvidia, "/missing/driver-ready")

	if got := os.Getenv(constants.TFCudaLibPathEnv); got != "/custom/libcuda.so.1" {
		t.Fatalf("explicit CUDA path was changed to %q", got)
	}
	if got := os.Getenv(constants.TFNvmlLibPathEnv); got != "/custom/libnvidia-ml.so.1" {
		t.Fatalf("explicit NVML path was changed to %q", got)
	}
}

func TestConfigureNvidiaDriverLibrariesFillsOnlyMissingPath(t *testing.T) {
	driverRoot := t.TempDir()
	libraryDir := filepath.Join(driverRoot, "usr", "lib64")
	if err := os.MkdirAll(libraryDir, 0o755); err != nil {
		t.Fatal(err)
	}
	nvmlPath := filepath.Join(libraryDir, nvidiaNvmlLibraryName)
	if err := os.WriteFile(nvmlPath, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	contractPath := filepath.Join(t.TempDir(), "driver-ready")
	if err := os.WriteFile(contractPath, []byte(driverRootCtrPathEnv+"="+driverRoot+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(constants.TFCudaLibPathEnv, "/custom/libcuda.so.1")
	t.Setenv(constants.TFNvmlLibPathEnv, "")

	configureNvidiaDriverLibraries(constants.AcceleratorVendorNvidia, contractPath)

	if got := os.Getenv(constants.TFCudaLibPathEnv); got != "/custom/libcuda.so.1" {
		t.Fatalf("explicit CUDA path was changed to %q", got)
	}
	if got := os.Getenv(constants.TFNvmlLibPathEnv); got != nvmlPath {
		t.Fatalf("%s = %q, want %q", constants.TFNvmlLibPathEnv, got, nvmlPath)
	}
}

func TestConfigureNvidiaDriverLibrariesIgnoresOtherVendors(t *testing.T) {
	t.Setenv(constants.TFCudaLibPathEnv, "")
	t.Setenv(constants.TFNvmlLibPathEnv, "")

	configureNvidiaDriverLibraries("Ascend", "/missing/driver-ready")

	if got := os.Getenv(constants.TFCudaLibPathEnv); got != "" {
		t.Fatalf("unexpected CUDA path %q", got)
	}
	if got := os.Getenv(constants.TFNvmlLibPathEnv); got != "" {
		t.Fatalf("unexpected NVML path %q", got)
	}
}

func TestNvidiaDriverRootFromContractFallbacks(t *testing.T) {
	tests := []struct {
		name     string
		contract string
		want     string
	}{
		{
			name:     "legacy container driver contract",
			contract: driverRootCtrPathEnv + "=" + constants.NvidiaDriverRootHostPath + "\n",
			want:     constants.NvidiaDriverRootMountPath,
		},
		{
			name:     "host driver",
			contract: nvidiaDriverRootEnv + "=/\n",
			want:     constants.NvidiaHostRootMountPath,
		},
		{
			name:     "containerized driver",
			contract: nvidiaDriverRootEnv + "=" + constants.NvidiaDriverRootHostPath + "\n",
			want:     constants.NvidiaDriverRootMountPath,
		},
		{
			name:     "custom host driver root",
			contract: nvidiaDriverRootEnv + "=/opt/nvidia/driver\n",
			want:     "/host/opt/nvidia/driver",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "driver-ready")
			if err := os.WriteFile(path, []byte(tt.contract), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := nvidiaDriverRootFromContract(path)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("root = %q, want %q", got, tt.want)
			}
		})
	}
}
