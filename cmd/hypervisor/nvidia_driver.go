package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/klog/v2"
)

const (
	nvidiaDriverRootEnv   = "NVIDIA_DRIVER_ROOT"
	driverRootCtrPathEnv  = "DRIVER_ROOT_CTR_PATH"
	nvidiaCudaLibraryName = "libcuda.so.1"
	nvidiaNvmlLibraryName = "libnvidia-ml.so.1"
)

func configureNvidiaDriverLibraries(vendor, contractPath string) {
	if !strings.EqualFold(strings.TrimSpace(vendor), constants.AcceleratorVendorNvidia) {
		return
	}
	if os.Getenv(constants.TFCudaLibPathEnv) != "" && os.Getenv(constants.TFNvmlLibPathEnv) != "" {
		return
	}

	driverRoot, err := nvidiaDriverRootFromContract(contractPath)
	if err != nil {
		if !os.IsNotExist(err) {
			klog.Warningf("Failed to read NVIDIA driver contract %s: %v", contractPath, err)
		}
		return
	}

	setNvidiaLibraryEnvIfMissing(constants.TFCudaLibPathEnv, driverRoot, nvidiaCudaLibraryName)
	setNvidiaLibraryEnvIfMissing(constants.TFNvmlLibPathEnv, driverRoot, nvidiaNvmlLibraryName)
}

func setNvidiaLibraryEnvIfMissing(envName, driverRoot, libraryName string) {
	if os.Getenv(envName) != "" {
		return
	}
	libraryPath, err := findNvidiaDriverLibrary(driverRoot, libraryName)
	if err != nil {
		klog.Warningf("NVIDIA driver contract root %s does not contain %s: %v", driverRoot, libraryName, err)
		return
	}
	if err := os.Setenv(envName, libraryPath); err != nil {
		klog.Warningf("Failed to set %s: %v", envName, err)
		return
	}
	klog.Infof("Using NVIDIA driver library from %s=%s", envName, libraryPath)
}

func nvidiaDriverRootFromContract(contractPath string) (string, error) {
	file, err := os.Open(contractPath)
	if err != nil {
		return "", err
	}
	defer file.Close() //nolint:errcheck

	values := make(map[string]string, 2)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		key = strings.TrimSpace(key)
		if key == driverRootCtrPathEnv || key == nvidiaDriverRootEnv {
			values[key] = strings.Trim(strings.TrimSpace(value), `"'`)
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("scan driver contract: %w", err)
	}

	if root := validDriverRoot(values[driverRootCtrPathEnv]); root != "" {
		switch root {
		case "/", constants.NvidiaHostRootMountPath:
			return constants.NvidiaHostRootMountPath, nil
		case constants.NvidiaDriverRootHostPath, constants.NvidiaDriverRootMountPath:
			return constants.NvidiaDriverRootMountPath, nil
		default:
			return root, nil
		}
	}
	switch root := filepath.Clean(values[nvidiaDriverRootEnv]); root {
	case "/":
		return constants.NvidiaHostRootMountPath, nil
	case constants.NvidiaDriverRootHostPath:
		return constants.NvidiaDriverRootMountPath, nil
	default:
		if root := validDriverRoot(root); root != "" {
			return filepath.Join(constants.NvidiaHostRootMountPath, root), nil
		}
	}
	return "", fmt.Errorf("driver contract does not contain a valid driver root")
}

func validDriverRoot(root string) string {
	root = filepath.Clean(strings.TrimSpace(root))
	if root == "." || !filepath.IsAbs(root) {
		return ""
	}
	return root
}

func findNvidiaDriverLibrary(driverRoot, libraryName string) (string, error) {
	archDir := map[string]string{
		"amd64": "x86_64-linux-gnu",
		"arm64": "aarch64-linux-gnu",
	}[runtime.GOARCH]
	searchDirs := []string{"", "usr/lib64", "lib64"}
	if archDir != "" {
		searchDirs = append(searchDirs, filepath.Join("usr/lib", archDir), filepath.Join("lib", archDir))
	}
	searchDirs = append(searchDirs, "usr/lib", "lib")

	for _, dir := range searchDirs {
		candidate := filepath.Join(driverRoot, dir, libraryName)
		info, err := os.Stat(candidate)
		if err == nil && !info.IsDir() {
			// Keep the path in the container namespace. EvalSymlinks can
			// incorrectly resolve an absolute target outside a /host mount.
			return candidate, nil
		}
	}
	return "", fmt.Errorf("could not locate %s", libraryName)
}
