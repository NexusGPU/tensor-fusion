package constants

import "strings"

const (
	// GPGPU vendors - Global
	AcceleratorVendorNvidia = "NVIDIA"
	AcceleratorVendorAMD    = "AMD"
	AcceleratorVendorIntel  = "Intel"

	// DSA vendors - Global
	AcceleratorVendorQualcomm  = "Qualcomm"
	AcceleratorVendorAWSNeuron = "AWSNeuron"
	AcceleratorVendorGoogleTPU = "Google"
	AcceleratorVendorCerebras  = "Cerebras"

	// GPGPU vendors - CN
	AcceleratorVendorHygon        = "Hygon"
	AcceleratorVendorMetaX        = "MetaX"
	AcceleratorVendorMThreads     = "MooreThreads"
	AcceleratorVendorBiren        = "Biren"
	AcceleratorVendorAlibabaTHead = "THead"

	// DSA vendors - CN
	AcceleratorVendorHuaweiAscendNPU = "Ascend"
	AcceleratorVendorCambricon       = "Cambricon"
	AcceleratorVendorEnflame         = "Enflame"
	AcceleratorVendorKunlunX         = "KunlunXin"
	AcceleratorVendorHorizon         = "Horizon"
	AcceleratorVendorMoore           = "Moore"

	// The mock vendor for example implementation
	AcceleratorVendorExample = "Example"

	AcceleratorVendorUnknown = "Unknown"
)

// L1 Virtualization means simple device partitioning, such as dynamic MIG/VirtualizationTemplates
// Can not auto-scale dynamically, limited partition count, coarse-grained isolation
var L1VirtualizationSupportedVendors = []map[string]bool{
	{
		AcceleratorVendorNvidia: false,

		// WIP
		AcceleratorVendorHuaweiAscendNPU: false,
		AcceleratorVendorAMD:             false,
		AcceleratorVendorHygon:           false,
		AcceleratorVendorMetaX:           false,
		// MUSA backend implements AccelAssignPartition via env vars (MTHREADS_VISIBLE_DEVICES,
		// MUSA_VGPU_TEMPLATE, ...) — see vgpu-provider-internal/musa/accelerator.cpp:836-876.
		AcceleratorVendorMThreads: true,
	},
}

// L2 Virtualization means dynamic user-space soft isolation, best performance and scalability
// Can auto-scale dynamically, accurate resource isolation, but can not used in untrusted environment
//
// A vendor flipping to `true` here is what gates the webhook from injecting
// the LD_PRELOAD soft-limiter for that vendor — see SupportsSoftIsolation
// below. Flip a vendor on only after its `libXXX_limiter.so` ships in the
// per-vendor middleware image (`/build/libXXX_limiter.so`).
var L2VirtualizationSupportedVendors = []map[string]bool{
	{
		AcceleratorVendorNvidia: true,

		// Soft-isolation limiters shipped from vgpu-provider-internal
		// (libascend_limiter.so / libmusa_limiter.so), shm V2 protocol.
		AcceleratorVendorHuaweiAscendNPU: true,
		AcceleratorVendorMThreads:        true,

		// WIP
		AcceleratorVendorAMD:   false,
		AcceleratorVendorHygon: false,
		AcceleratorVendorMetaX: false,
	},
}

// SupportsSoftIsolation reports whether `vendor` has a soft-isolation
// limiter shipped today. Empty / unknown vendor returns true so we don't
// regress test setups and stubs that never declare a vendor.
//
// Match style follows applyProviderRemoteWorkerConfigToContainerIndex:
// case-insensitive after trimming whitespace.
func SupportsSoftIsolation(vendor string) bool {
	v := strings.TrimSpace(vendor)
	if v == "" {
		return true
	}
	for _, m := range L2VirtualizationSupportedVendors {
		for k, supported := range m {
			if strings.EqualFold(k, v) {
				return supported
			}
		}
	}
	// Vendor isn't enumerated in the L2 map at all — don't block, just
	// pass through. Hard-coding a deny list here would surprise operators
	// using vendors we haven't added yet.
	return true
}

// L3 Virtualization means full-featured soft and hard isolated virtualization, including API remoting
// support live-migration, can be used in both VM and container environments for untrusted workloads
var L3VirtualizationSupportedVendors = []map[string]bool{
	{
		AcceleratorVendorNvidia: true,

		// WIP
		AcceleratorVendorAMD:             false,
		AcceleratorVendorHygon:           false,
		AcceleratorVendorMetaX:           false,
		AcceleratorVendorHuaweiAscendNPU: false,
	},
}

// GetAcceleratorLibPath returns the accelerator library path based on vendor
// Vendor string should match constants from internal/constants/vendors.go
func GetAcceleratorLibPath(vendor string) string {
	switch vendor {
	case AcceleratorVendorNvidia:
		return "libaccelerator_nvidia.so"
	case AcceleratorVendorAMD:
		return "libaccelerator_amd.so"
	case AcceleratorVendorHuaweiAscendNPU:
		return "libaccelerator_ascend.so"
	case AcceleratorVendorMThreads:
		return "libaccelerator_musa.so"
	default:
		// Default to stub library for unknown vendors
		return "libaccelerator_example.so"
	}
}

// GetSoftLimiterLibName returns the LD_PRELOAD .so basename for a vendor's
// soft-isolation limiter. The middleware image must ship the matching file
// under `/build/` so the soft-limiter init container can `cp /build/*` into
// the limiter emptyDir. NVIDIA is the default for backward compatibility
// with single-vendor deployments that don't propagate GPUVendor.
func GetSoftLimiterLibName(vendor string) string {
	switch strings.TrimSpace(vendor) {
	case AcceleratorVendorHuaweiAscendNPU:
		return "libascend_limiter.so"
	case AcceleratorVendorMThreads:
		return "libmusa_limiter.so"
	default:
		return "libcuda_limiter.so"
	}
}
