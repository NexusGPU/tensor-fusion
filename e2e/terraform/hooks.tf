# Stage 2 (everything that lives INSIDE the cluster) is driven by shell scripts
# instead of the helm/kubernetes providers:
#
#  * no provider that needs a live cluster endpoint at plan time, so
#    `terraform destroy` never dies on "cluster unreachable" / stale tokens;
#  * the teardown order is explicit and works even when the Kubernetes API is
#    already broken, because the important part talks to the EC2 API.
#
# Order on apply   : deadman -> VPC -> cluster/nodes/addons/IAM -> cluster_cleanup -> install
# Order on destroy : install (no-op) -> cluster_cleanup (HOOK) -> nodes -> cluster
#                    -> vpc_cleanup (HOOK, see vpc.tf) -> subnets -> VPC -> deadman
#
# TensorFusion itself is NOT installed here: it is stage 3 (`make tf-install`,
# scripts/install-tf.sh), because chart source and image tags change per run.

# Destroy-only hook. Runs while the cluster, the node groups and Karpenter's IAM
# are all still alive: stops the controllers that create AWS resources, lets
# them delete load balancers, then terminates every Karpenter-launched instance
# and its launch templates straight through the EC2 API.
#
# `input` holds only values that never change for the life of the stack, so
# this resource is never replaced (a replacement would run the hook!).
resource "terraform_data" "cluster_cleanup" {
  input = {
    cluster     = aws_eks_cluster.this.name
    region      = var.region
    aws_profile = local.aws_profile
    scripts_dir = local.scripts_dir
    kubeconfig  = local.kubeconfig_path
  }

  provisioner "local-exec" {
    when    = destroy
    command = "\"${self.input.scripts_dir}/pre-destroy.sh\""
    environment = {
      TF_E2E_CLUSTER     = self.input.cluster
      TF_E2E_REGION      = self.input.region
      TF_E2E_AWS_PROFILE = self.input.aws_profile
      TF_E2E_KUBECONFIG  = self.input.kubeconfig
    }
  }

  depends_on = [
    aws_eks_node_group.this,
    aws_eks_addon.pod_identity,
    module.karpenter,
  ]
}

locals {
  install_env = {
    TF_E2E_CLUSTER               = aws_eks_cluster.this.name
    TF_E2E_REGION                = var.region
    TF_E2E_AWS_PROFILE           = local.aws_profile
    TF_E2E_KUBECONFIG            = local.kubeconfig_path
    TF_E2E_TAGS_JSON             = jsonencode(local.tags)
    TF_E2E_ROLE_LABEL_KEY        = local.role_label_key
    TF_E2E_SHARED_POOL_LABEL_KEY = local.shared_pool_label_key
    TF_E2E_GPU_PRESENT_LABEL_KEY = local.gpu_present_label_key
    TF_E2E_MOCK_LABEL_KEY        = local.mock_label_key
    TF_E2E_CLUSTER_SG_ID         = aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
    TF_E2E_KARPENTER_VERSION     = var.karpenter_version
    TF_E2E_KARPENTER_QUEUE       = local.interruption_queue_name
    TF_E2E_KARPENTER_PROFILE     = module.karpenter.instance_profile_name
    TF_E2E_AMI_ALIAS             = var.ami_alias
    TF_E2E_GPU_NODECLASS         = local.gpu_nodeclass
    TF_E2E_GPU_NODEPOOL          = local.gpu_nodepool
    TF_E2E_GPU_INSTANCE_JSON     = jsonencode(var.gpu_instance_types)
    TF_E2E_GPU_CAPACITY_JSON     = jsonencode(var.gpu_capacity_types)
    TF_E2E_GPU_NODE_LIMIT        = tostring(var.gpu_node_limit)
    TF_E2E_GPU_CONSOLIDATE       = var.gpu_consolidate_after
    TF_E2E_GPU_DISK_GB           = tostring(var.gpu_disk_gb)
    TF_E2E_CPU_NODECLASS         = local.cpu_nodeclass
    TF_E2E_CPU_NODEPOOL          = local.cpu_nodepool
    TF_E2E_CPU_INSTANCE_JSON     = jsonencode(var.cpu_instance_types)
    TF_E2E_CPU_CAPACITY_JSON     = jsonencode(var.cpu_capacity_types)
    TF_E2E_CPU_LIMIT             = tostring(var.cpu_limit)
    TF_E2E_MOCK_NODEPOOL         = local.mock_nodepool
    TF_E2E_MOCK_INSTANCE_JSON    = jsonencode(var.mock_instance_types)
    TF_E2E_MOCK_CAPACITY_JSON    = jsonencode(var.mock_capacity_types)
    TF_E2E_MOCK_CPU_LIMIT        = tostring(var.mock_cpu_limit)
    TF_E2E_NVDP_ENABLED          = tostring(var.install_nvidia_device_plugin)
    TF_E2E_NVDP_VERSION          = var.nvidia_device_plugin_version
  }
}

# Create-only hook: Karpenter + EC2NodeClasses/NodePools + NVIDIA device plugin
# + GreptimeDB volume. Idempotent (`helm upgrade --install`, `kubectl apply`);
# re-runs whenever one of its inputs changes, and by hand with `make install`.
resource "terraform_data" "install" {
  count = var.run_install ? 1 : 0

  triggers_replace = [
    sha256(jsonencode(local.install_env)),
    filesha256("${path.module}/scripts/install.sh"),
    filesha256("${path.module}/scripts/lib.sh"),
  ]

  provisioner "local-exec" {
    command     = "\"${local.scripts_dir}/install.sh\""
    environment = local.install_env
  }

  depends_on = [terraform_data.cluster_cleanup]
}
