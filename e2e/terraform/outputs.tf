output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "region" {
  value = var.region
}

output "azs" {
  value = local.azs
}

output "vpc_id" {
  value = aws_vpc.this.id
}

output "kubernetes_version" {
  value = aws_eks_cluster.this.version
}

output "kubeconfig" {
  description = "Dedicated kubeconfig written by scripts/install.sh (your ~/.kube/config is not touched)."
  value       = local.kubeconfig_path
}

output "deadman_parameter" {
  description = "SSM parameter holding the dead-man deadline (RFC 3339, UTC)."
  value       = aws_ssm_parameter.deadman_deadline.name
}

output "install_env" {
  description = "Environment consumed by scripts/*.sh (used by the Makefile to re-run stages 2 and 3 by hand)."
  value       = local.install_env
}
