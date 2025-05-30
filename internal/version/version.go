package version

import (
	"fmt"
	"runtime/debug"
	"time"
)

var (
	BuildVersion string
)

func Version() string {
	return BuildVersion
}

func Time() string {
	bi, ok := debug.ReadBuildInfo()
	if ok {
		for _, setting := range bi.Settings {
			//   - vcs.time: the modification time associated with vcs.revision, in RFC3339 format
			if setting.Key == "vcs.time" {
				return setting.Value
			}
		}
	}
	return time.Now().Format(time.RFC3339)
}

func Hash() string {
	var (
		hash string
	)
	bi, ok := debug.ReadBuildInfo()
	if ok {
		for _, setting := range bi.Settings {
			//   - vcs.revision: the revision identifier for the current commit or checkout
			if setting.Key == "vcs.revision" {
				hash = setting.Value
			}
			if setting.Key == "vcs.modified" {
				hash = fmt.Sprintf("%s+dirty", hash)
			}
		}
	}
	return hash
}

func VersionInfo() string {
	return fmt.Sprintf("version: %s, hash: %s, time: %s", Version(), Hash(), Time())
}
