//go:build windows

package shm_init

import "k8s.io/klog/v2"

func RunMountShm() {
	klog.Fatalf("mount-shm is not supported on Windows")
}
