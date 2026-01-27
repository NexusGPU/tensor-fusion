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

/*
#cgo CFLAGS: -I../../../provider
#include <stdlib.h>
*/
import "C"
import (
	"k8s.io/klog/v2"
)

// GoLog is exported to C code via //export directive
// This function is called by C code (wrapper.c) to log messages using klog
//
//export GoLog
func GoLog(level *C.char, message *C.char) {
	if level == nil || message == nil {
		return
	}

	levelStr := C.GoString(level)
	messageStr := C.GoString(message)

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
		// Default to Info level for unknown levels
		klog.Info(messageStr)
	}
}
