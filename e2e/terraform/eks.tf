# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------
# Deliberately NOT using terraform-aws-modules/eks for the cluster itself: that
# module creates a KMS key (7-30 day deletion window) and a CloudWatch log group
# by default, both classic leftovers. Here: no envelope-encryption key, no
# control plane logging, no OIDC provider (Pod Identity instead of IRSA).

data "aws_iam_policy_document" "cluster_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cluster" {
  name               = "${var.name}-cluster"
  assume_role_policy = data.aws_iam_policy_document.cluster_assume.json
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_eks_cluster" "this" {
  name     = local.cluster_name
  version  = var.kubernetes_version
  role_arn = aws_iam_role.cluster.arn

  # Let EKS drop in its default (self-managed) vpc-cni / kube-proxy / coredns.
  # Saves three aws_eks_addon round trips on apply and on destroy.
  bootstrap_self_managed_addons = true
  deletion_protection           = false

  access_config {
    authentication_mode                         = "API"
    bootstrap_cluster_creator_admin_permissions = true
  }

  vpc_config {
    subnet_ids              = aws_subnet.public[*].id
    endpoint_public_access  = true
    endpoint_private_access = true
    public_access_cidrs     = var.api_public_access_cidrs
  }

  # Never silently roll into extended support ($0.60/h instead of $0.10/h).
  upgrade_policy {
    support_type = "STANDARD"
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster,
    terraform_data.vpc_cleanup,
    # The dead-man switch must be armed before anything billable exists, and
    # must outlive it on destroy.
    terraform_data.deadman_armed,
  ]
}

# ---------------------------------------------------------------------------
# Managed node groups
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "node_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "${var.name}-node"
  assume_role_policy = data.aws_iam_policy_document.node_assume.json
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each = toset([
    "AmazonEKSWorkerNodePolicy",
    "AmazonEKS_CNI_Policy",
    "AmazonEC2ContainerRegistryPullOnly",
    "AmazonSSMManagedInstanceCore", # shell access via SSM, no SSH key / port 22
  ])

  role       = aws_iam_role.node.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/${each.value}"
}

# Launch templates exist for one reason: managed node groups do not propagate
# the provider's default_tags to the instances / volumes / ENIs they launch.
# Without the Project tag the sweep, the dead-man switch and `make cost` would
# only find these nodes through the eks:cluster-name tag.
locals {
  node_groups = merge(
    {
      system = {
        ami_type       = "AL2023_x86_64_STANDARD"
        instance_types = var.system_instance_types
        count          = var.system_node_count
        disk_gb        = var.system_disk_gb
        labels         = { (local.role_label_key) = "system" }
      }
    },
    var.gpu_static_count > 0 ? {
      gpu-static = {
        # EKS-optimized AL2023 accelerated AMI: NVIDIA driver + nvidia-container-toolkit.
        ami_type       = "AL2023_x86_64_NVIDIA"
        instance_types = [var.gpu_instance_types[0]]
        count          = var.gpu_static_count
        disk_gb        = var.gpu_disk_gb
        labels = {
          (local.gpu_present_label_key) = "true"
          (local.shared_pool_label_key) = "true"
        }
      }
    } : {},
  )
}

resource "aws_launch_template" "node" {
  for_each = local.node_groups

  name = "${var.name}-${each.key}"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = each.value.disk_gb
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  dynamic "tag_specifications" {
    for_each = toset(["instance", "volume", "network-interface"])

    content {
      resource_type = tag_specifications.value
      tags          = merge(local.tags, { Name = "${var.name}-${each.key}" })
    }
  }
}

resource "aws_eks_node_group" "this" {
  for_each = local.node_groups

  cluster_name    = aws_eks_cluster.this.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.public[*].id

  ami_type       = each.value.ami_type
  capacity_type  = "ON_DEMAND" # a spot reclaim of the system node takes Karpenter and the TF controller down
  instance_types = each.value.instance_types

  launch_template {
    id      = aws_launch_template.node[each.key].id
    version = aws_launch_template.node[each.key].latest_version
  }

  scaling_config {
    desired_size = each.value.count
    min_size     = each.value.count
    max_size     = each.value.count
  }

  labels = each.value.labels

  # The dead-man switch scales node groups to zero out of band; a later apply
  # (make up / make extend) is what brings them back, so do not ignore drift here.

  depends_on = [aws_iam_role_policy_attachment.node]
}

# ---------------------------------------------------------------------------
# Addons
# ---------------------------------------------------------------------------
# Pod Identity agent: lets Karpenter get IAM credentials without an OIDC provider.
#
# No EBS CSI driver: the only PVC in the TensorFusion chart is GreptimeDB's
# 20Gi volumeClaimTemplate, and it has no storageClassName, so a pre-created
# hostPath PersistentVolume on the system node satisfies it (scripts/install.sh).
# That removes an addon, an IAM role, ~1.5 min of apply/destroy time and the
# whole "EBS volume outlived its PVC" leftover class.
resource "aws_eks_addon" "pod_identity" {
  cluster_name = aws_eks_cluster.this.name
  addon_name   = "eks-pod-identity-agent"
}
