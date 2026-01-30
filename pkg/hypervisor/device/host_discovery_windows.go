//go:build windows

package device

import (
	"fmt"

	"github.com/shirou/gopsutil/mem"
)

// GetDiskInfo returns 0 on Windows to unblock cross-platform builds.
// Windows-specific disk space discovery is not implemented.
func GetDiskInfo(path string) (total int64) {
	return 0
}

func GetTotalHostRAMBytes() int64 {
	v, err := mem.VirtualMemory()
	if err != nil {
		fmt.Printf("[warning] getting memory info failed: %v\n", err)
		return 0
	}
	return int64(v.Total)
}
