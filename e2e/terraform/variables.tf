# ---------------------------------------------------------------------------
# Identity / placement
# ---------------------------------------------------------------------------
variable "name" {
  description = "Stack name. Used as EKS cluster name, resource name prefix and the value of the Project tag (the sweep, the dead-man switch and `make cost` all key on it)."
  type        = string
  default     = "tf-e2e"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name))
    error_message = "name must be lowercase alphanumeric/hyphen, 2-21 chars."
  }
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-west-2"
}

variable "aws_profile" {
  description = "Named AWS CLI profile used by Terraform and by the helper scripts. null = use the ambient credential chain (env vars, AWS_PROFILE...)."
  type        = string
  default     = null
}

variable "azs" {
  description = "Exactly two AZ names. Empty = auto-pick the two AZs that offer the most of var.gpu_instance_types."
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.azs) == 0 || length(var.azs) == 2
    error_message = "azs must be empty or contain exactly two AZs (EKS requires subnets in two AZs)."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR. Two public /20 subnets are carved out of it."
  type        = string
  default     = "10.42.0.0/16"
}

variable "api_public_access_cidrs" {
  description = "CIDRs allowed to reach the public EKS API endpoint. Narrow this to your own IP/32 if you care."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "extra_tags" {
  description = "Extra tags applied to every resource (e.g. Owner)."
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------------------
# EKS
# ---------------------------------------------------------------------------
variable "kubernetes_version" {
  description = "EKS Kubernetes version. Must be in STANDARD support, >= the kube-scheduler the operator embeds (k8s.io/kubernetes v1.36.x) and supported by var.karpenter_version. See README 版本选择."
  type        = string
  default     = "1.36"
}

variable "system_instance_types" {
  description = "Instance types of the system managed node group (CoreDNS, Karpenter, TensorFusion controller, GreptimeDB, alertmanager). 2 vCPU / 8 GiB is enough; the account's standard on-demand vCPU quota is shared with the Karpenter CPU NodePool."
  type        = list(string)
  default     = ["t3.large"]
}

variable "system_node_count" {
  description = "Number of system nodes. Everything on them is single-replica, so 1 is the honest size for a throwaway stack."
  type        = number
  default     = 1
}

variable "system_disk_gb" {
  description = "Root volume size (GiB) of system nodes. Also holds the GreptimeDB hostPath volume."
  type        = number
  default     = 40
}

# ---------------------------------------------------------------------------
# GPU capacity (all of it comes from Karpenter)
# ---------------------------------------------------------------------------
variable "gpu_instance_types" {
  description = "Single-GPU instance types the Karpenter GPU NodePool may launch. Karpenter picks the cheapest that fits; tests pin a type with a nodeSelector when they need a specific GPU (T4 / A10G / L4)."
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g6.xlarge"]
}

variable "gpu_capacity_types" {
  description = "Capacity types of the Karpenter GPU NodePool. With both, Karpenter prefers spot and falls back to on-demand."
  type        = list(string)
  default     = ["spot", "on-demand"]
}

variable "gpu_node_limit" {
  description = "Hard cap on nvidia.com/gpu across the Karpenter GPU NodePool (= max GPU nodes, all types are single-GPU). Does NOT cap NodeClaims created directly by TensorFusion in provisioningMode=Karpenter; those are capped by the GPUPool maxResources and by the dead-man switch."
  type        = number
  default     = 4
}

variable "gpu_consolidate_after" {
  description = "How long an empty Karpenter GPU (or mock-GPU) node lives before Karpenter removes it. A node that TensorFusion's expander adds is empty in Karpenter's eyes until the pending pod is bound to it, and with operator <= 2.17.23 that can take the scheduler's full 5-minute unschedulable flush (case KRP-A16). With 1m and with 5m the rehearsal lost freshly added nodes before their pod landed; 10m is the smallest round value that is safely above 5 min + node start-up. Costs up to 10 idle minutes per scale-down."
  type        = string
  default     = "10m"
}

variable "gpu_disk_gb" {
  description = "Root volume size (GiB) of GPU nodes. CUDA / LLM images are large."
  type        = number
  default     = 80
}

variable "gpu_static_count" {
  description = "GPU nodes in an EKS managed node group. Keep 0: TensorFusion's expander only clones nodes owned by a Karpenter NodeClaim, so managed-node-group GPU nodes are useful for exactly one case (NEG-10). Needs on-demand G quota."
  type        = number
  default     = 0
}

variable "ami_alias" {
  description = "EC2NodeClass AMI alias. `al2023@latest` resolves to the EKS-optimized AL2023 AMI, and to its NVIDIA variant (driver + nvidia-container-toolkit) for GPU instance types. Pin (al2023@v2026xxxx) for strict repeatability."
  type        = string
  default     = "al2023@latest"
}

# ---------------------------------------------------------------------------
# General-purpose capacity (Karpenter)
# ---------------------------------------------------------------------------
variable "cpu_instance_types" {
  description = "Instance types of the general-purpose Karpenter NodePool (non-GPU test pods, remote-mode clients, load generators)."
  type        = list(string)
  default     = ["t3.large", "t3a.large", "m5.large", "m5a.large", "m6i.large", "c5.large", "c6i.large"]
}

variable "cpu_capacity_types" {
  description = "Capacity types of the general-purpose NodePool."
  type        = list(string)
  default     = ["spot", "on-demand"]
}

variable "cpu_limit" {
  description = "Hard cap on vCPUs of the general-purpose NodePool. system node (2) + this must stay within the standard on-demand vCPU quota (8 on a fresh account)."
  type        = number
  default     = 4
}

# ---------------------------------------------------------------------------
# Mock-GPU capacity (rehearsal mode, Karpenter)
# ---------------------------------------------------------------------------
variable "mock_instance_types" {
  description = "CPU instance types of the mock-GPU NodePool. Nodes of this pool carry e2e.tensor-fusion.ai/mock-gpu=true and are turned into TensorFusion GPU nodes with mock devices by `make mock-install` (rehearsal mode, no GPU quota needed)."
  type        = list(string)
  default     = ["t3.medium", "t3a.medium", "t3.large", "t3a.large", "m5.large", "c5.large"]
}

variable "mock_capacity_types" {
  description = "Capacity types of the mock-GPU NodePool."
  type        = list(string)
  default     = ["spot", "on-demand"]
}

variable "mock_cpu_limit" {
  description = "Hard cap on vCPUs of the mock-GPU NodePool (2 vCPU per node)."
  type        = number
  default     = 6
}

# ---------------------------------------------------------------------------
# In-cluster components (stage 2, scripts/install.sh; TensorFusion itself is
# stage 3, scripts/install-tf.sh, driven by `make tf-install`)
# ---------------------------------------------------------------------------
variable "run_install" {
  description = "Run scripts/install.sh (Karpenter, EC2NodeClass/NodePools, NVIDIA device plugin, GreptimeDB volume) at the end of terraform apply."
  type        = bool
  default     = true
}

variable "karpenter_version" {
  description = "Karpenter Helm chart version (oci://public.ecr.aws/karpenter/karpenter). Must satisfy Karpenter's compatibility matrix for var.kubernetes_version (1.36 needs >= 1.13) and serve karpenter.sh/v1 NodeClaim + v1alpha1 NodeOverlay, which is all the operator's sigs.k8s.io/karpenter v1.8.2 types need."
  type        = string
  default     = "1.14.1"
}

variable "enable_spot_interruption_queue" {
  description = "Create the SQS queue + EventBridge rules for Karpenter spot interruption handling. Needed by KRP-A11 (FIS spot interruption); off by default: 5 fewer resources."
  type        = bool
  default     = false
}

variable "install_nvidia_device_plugin" {
  description = "Install the NVIDIA k8s-device-plugin. Needed so nodes advertise nvidia.com/gpu, which Karpenter waits for before a NodeClaim that requests it (what TensorFusion creates in provisioningMode=Karpenter) becomes Initialized, and by the native-GPU coexistence / auto-migration cases."
  type        = bool
  default     = true
}

variable "nvidia_device_plugin_version" {
  description = "nvidia-device-plugin Helm chart version."
  type        = string
  default     = "0.20.1"
}

# ---------------------------------------------------------------------------
# Cost guard
# ---------------------------------------------------------------------------
variable "deadman_ttl" {
  description = "Lifetime of the stack's compute, counted from the first apply (Go duration: 90m, 3h). When it expires an AWS-side Lambda scales every node group to zero and terminates every instance of the stack, with no local process involved. `make extend TTL=2h` moves the deadline."
  type        = string
  default     = "3h"

  validation {
    condition     = can(regex("^[0-9]+(m|h)$", var.deadman_ttl))
    error_message = "deadman_ttl must look like 45m or 3h."
  }
}

variable "deadman_cluster_grace" {
  description = "How long after the deadline the dead-man switch also deletes the node groups and the EKS control plane ($0.10/h). Empty string = never delete the control plane."
  type        = string
  default     = "6h"

  validation {
    condition     = var.deadman_cluster_grace == "" || can(regex("^[0-9]+(m|h)$", var.deadman_cluster_grace))
    error_message = "deadman_cluster_grace must be empty or look like 30m or 6h."
  }
}

variable "budget_alert_email" {
  description = "If set, create a daily AWS Budget scoped to the Project tag that mails this address. The Project tag must be activated as a cost allocation tag for the filter to match. Never commit a real address: pass it via TF_VAR_budget_alert_email."
  type        = string
  default     = null
}

variable "budget_daily_usd" {
  description = "Daily budget in USD for the alert."
  type        = number
  default     = 25
}
