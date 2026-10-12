# Minimal network: one VPC, two PUBLIC subnets (EKS needs two AZs), one IGW.
# No NAT gateway, no private subnets, no VPC endpoints: nodes get a public IP
# and talk to ECR / Docker Hub / the EKS API directly. Inbound is still closed,
# because nodes only carry the EKS cluster security group (self-referencing).

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.name }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = { Name = var.name }
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  availability_zone       = local.azs[count.index]
  cidr_block              = cidrsubnet(var.vpc_cidr, 4, count.index)
  map_public_ip_on_launch = true

  tags = {
    Name                                          = "${var.name}-public-${local.azs[count.index]}"
    "kubernetes.io/role/elb"                      = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "karpenter.sh/discovery"                      = local.cluster_name
  }

  lifecycle {
    precondition {
      condition     = length(local.azs) == 2
      error_message = "Could not find two AZs offering any of var.gpu_instance_types in ${var.region}; set var.azs explicitly."
    }
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Destroy-time safety net. Because the EKS cluster depends on this resource,
# `terraform destroy` reaches it AFTER the node groups and the cluster are gone
# and BEFORE subnets / IGW / VPC are deleted. Its hook removes what in-cluster
# controllers may have left inside the VPC (leaked VPC-CNI ENIs, ELBs and
# k8s-created security groups) so the VPC deletion cannot hang on
# DependencyViolation.
resource "terraform_data" "vpc_cleanup" {
  input = {
    vpc_id      = aws_vpc.this.id
    cluster     = local.cluster_name
    region      = var.region
    aws_profile = local.aws_profile
    scripts_dir = local.scripts_dir
  }

  provisioner "local-exec" {
    when    = destroy
    command = "\"${self.input.scripts_dir}/vpc-cleanup.sh\""
    environment = {
      TF_E2E_VPC_ID      = self.input.vpc_id
      TF_E2E_CLUSTER     = self.input.cluster
      TF_E2E_REGION      = self.input.region
      TF_E2E_AWS_PROFILE = self.input.aws_profile
    }
  }

  depends_on = [
    aws_route_table_association.public,
    aws_internet_gateway.this,
  ]
}
