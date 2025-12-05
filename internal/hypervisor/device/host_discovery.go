package device

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/shirou/gopsutil/mem"
)

func GetTotalHostRAMBytes() int64 {
	v, err := mem.VirtualMemory()
	if err != nil {
		fmt.Printf("[warning] getting memory info failed: %v\n", err)
		return 0
	}
	return int64(v.Total)
}

func GetDiskInfo(path string) (total int64) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		fmt.Printf("[warning] getting disk path failed: %v\n", err)
		return 0
	}

	var stat syscall.Statfs_t
	err = syscall.Statfs(absPath, &stat)
	if err != nil {
		if errors.Is(err, syscall.ENOENT) {
			err = os.MkdirAll(absPath, 0o755)
			if err != nil {
				fmt.Printf("[warning] creating folder to discover disk space failed: %s, err: %v\n", absPath, err)
				return 0
			}
			err = syscall.Statfs(absPath, &stat)
			if err != nil {
				fmt.Printf("[warning] getting disk stats after creation failed: %v\n", err)
				return 0
			}
		} else {
			fmt.Printf("[warning] getting disk stats failed: %v\n", err)
			return 0
		}
	}

	total = int64(stat.Blocks * uint64(stat.Bsize))
	return total
}
