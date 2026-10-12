# AWS-side prerequisites of Karpenter. IAM only (plus an optional SQS queue):
# the controller policy is the piece most likely to rot, so it comes from the
# maintained terraform-aws-modules submodule instead of a hand-written copy.
module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 21.29"

  cluster_name = aws_eks_cluster.this.name

  # Controller role, bound to the kube-system/karpenter service account via
  # EKS Pod Identity (no OIDC provider / IRSA).
  iam_role_name                   = "${var.name}-karpenter"
  iam_role_use_name_prefix        = false
  iam_policy_name                 = "${var.name}-karpenter"
  iam_policy_use_name_prefix      = false
  create_pod_identity_association = true
  namespace                       = "kube-system"
  service_account                 = "karpenter"

  # Node role for Karpenter-launched instances + its EKS access entry.
  node_iam_role_name            = "${var.name}-karpenter-node"
  node_iam_role_use_name_prefix = false
  node_iam_role_additional_policies = {
    ssm = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  # Terraform owns the instance profile and the EC2NodeClass references it via
  # spec.instanceProfile. If the EC2NodeClass used spec.role instead, Karpenter
  # would create its own instance profile, and tearing the cluster down without
  # letting Karpenter finalize the EC2NodeClass would orphan it and block the
  # deletion of the node role.
  create_instance_profile = true

  enable_spot_termination = var.enable_spot_interruption_queue
  queue_name              = "${var.name}-karpenter"
  rule_name_prefix        = "${var.name}-"
}
