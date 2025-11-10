package constants

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
	AcceleratorVendorMThreads     = "MThreads"
	AcceleratorVendorBiren        = "Biren"
	AcceleratorVendorAlibabaTHead = "THead"

	// DSA vendors - CN
	AcceleratorVendorHuaweiAscendNPU = "Ascend"
	AcceleratorVendorCambricon       = "Cambricon"
	AcceleratorVendorEnflame         = "Enflame"
	AcceleratorVendorKunlunX         = "KunlunXin"

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
		AcceleratorVendorMThreads:        false,
	},
}

// L2 Virtualization means dynamic user-space soft isolation, best performance and scalability
// Can auto-scale dynamically, accurate resource isolation, but can not used in untrusted environment
var L2VirtualizationSupportedVendors = []map[string]bool{
	{
		AcceleratorVendorNvidia: true,

		// WIP
		AcceleratorVendorAMD:             false,
		AcceleratorVendorHuaweiAscendNPU: false,
		AcceleratorVendorHygon:           false,
		AcceleratorVendorMetaX:           false,
	},
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
