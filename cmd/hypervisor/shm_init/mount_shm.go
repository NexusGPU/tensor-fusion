package shm_init

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"

	"k8s.io/klog/v2"
)

// runMountShm handles the "mount-shm" subcommand
func RunMountShm() {
	// Create a new flag set for mount-shm subcommand
	mountShmFlags := flag.NewFlagSet("mount-shm", flag.ExitOnError)
	mountPoint := mountShmFlags.String("mount-point", "", "Mount point directory path (required)")
	sizeMB := mountShmFlags.Int("size", 0, "Size in MB (required)")

	klog.InitFlags(nil)
	mountShmFlags.Parse(os.Args[2:])

	if *mountPoint == "" {
		klog.Fatalf("mount-point is required")
	}
	if *sizeMB <= 0 {
		klog.Fatalf("size must be greater than 0")
	}

	klog.Infof("mount point: %s", *mountPoint)
	klog.Infof("size: %d MB", *sizeMB)

	// Create mount point directory if it doesn't exist
	if _, err := os.Stat(*mountPoint); os.IsNotExist(err) {
		klog.Infof("create mount point directory: %s", *mountPoint)
		if err := os.MkdirAll(*mountPoint, 0755); err != nil {
			klog.Fatalf("create mount point directory failed: %v", err)
		}
	}

	// Check if tmpfs is already mounted
	mountCmd := exec.Command("mount")
	mountOutput, err := mountCmd.Output()
	if err != nil {
		klog.Fatalf("execute mount command failed: %v", err)
	}

	mountInfo := string(mountOutput)
	mountPointAbs, err := filepath.Abs(*mountPoint)
	if err != nil {
		klog.Fatalf("get absolute path failed: %v", err)
	}

	expectedMountStr := fmt.Sprintf("on %s type tmpfs", mountPointAbs)
	if strings.Contains(mountInfo, expectedMountStr) {
		klog.Infof("tmpfs is already mounted on %s", *mountPoint)
	} else {
		// Mount tmpfs
		klog.Infof("mount tmpfs on %s", *mountPoint)
		sizeArg := fmt.Sprintf("size=%dM", *sizeMB)

		mountTmpfsCmd := exec.Command("mount",
			"-t", "tmpfs",
			"-o", fmt.Sprintf("rw,nosuid,nodev,%s", sizeArg),
			"tmpfs",
			mountPointAbs,
		)

		if err := mountTmpfsCmd.Run(); err != nil {
			klog.Fatalf("mount tmpfs failed: %v", err)
		}

		klog.Info("mount tmpfs successfully")
	}

	// Set directory permissions to 0777
	// Save old umask
	oldUmask := syscall.Umask(0)
	defer syscall.Umask(oldUmask)

	// Set permissions
	if err := os.Chmod(*mountPoint, 0777); err != nil {
		klog.Fatalf("set permissions failed: %v", err)
	}

	klog.Info("mount-shm completed successfully")
}
