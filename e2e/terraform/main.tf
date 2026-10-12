data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# One (instance type, AZ) pair per offering of the GPU types. Used to put the
# two subnets into the AZs where the most GPU types can actually be launched.
data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = var.gpu_instance_types
  }
}

locals {
  cluster_name = var.name

  tags = merge(
    {
      Project   = var.name
      ManagedBy = "terraform"
      Purpose   = "tensor-fusion-e2e"
      Stack     = "e2e/terraform"
    },
    var.extra_tags,
  )

  # AZ => number of GPU instance types offered there; ranked best first, name as tie-break.
  az_score = {
    for az in data.aws_availability_zones.available.names :
    az => length([for l in data.aws_ec2_instance_type_offerings.gpu.locations : l if l == az])
  }
  az_ranked = [
    for s in sort([for az, n in local.az_score : format("%02d/%s", 99 - n, az) if n > 0]) :
    split("/", s)[1]
  ]
  azs = length(var.azs) == 2 ? var.azs : try(slice(local.az_ranked, 0, 2), local.az_ranked)

  # Node labels.
  #  - e2e.tensor-fusion.ai/role=system|cpu : system node group / general-purpose NodePool.
  #  - nvidia.com/gpu.present=true          : matched by the chart's initialGpuNodeLabelSelector
  #                                           (normally set by GPU operator / GFD, which we do not run).
  #  - e2e.tensor-fusion.ai/shared-pool=true: selector of the default AutoSelect GPUPool, so nodes
  #                                           TensorFusion provisions itself (provisioningMode=Karpenter,
  #                                           which only carry the first label) stay out of it.
  role_label_key        = "e2e.tensor-fusion.ai/role"
  shared_pool_label_key = "e2e.tensor-fusion.ai/shared-pool"
  gpu_present_label_key = "nvidia.com/gpu.present"
  gpu_nodeclass         = "tf-gpu"
  gpu_nodepool          = "tf-gpu"
  cpu_nodeclass         = "tf-cpu"
  cpu_nodepool          = "tf-cpu"
  mock_nodepool         = "tf-mock"
  mock_label_key        = "e2e.tensor-fusion.ai/mock-gpu"

  scripts_dir             = abspath("${path.module}/scripts")
  kubeconfig_path         = abspath("${path.module}/.kube/config")
  aws_profile             = var.aws_profile == null ? "" : var.aws_profile
  interruption_queue_name = var.enable_spot_interruption_queue ? module.karpenter.queue_name : ""
}
