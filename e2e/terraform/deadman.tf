# AWS-side dead-man switch. Agents and laptops die mid-run; this does not.
#
#   EventBridge rule (every 5 min) -> Lambda -> reads the deadline from SSM
#     now <  deadline          : nothing
#     now >= deadline          : node groups -> 0, terminate every instance of the
#                                stack (Karpenter / TensorFusion-launched ones
#                                included), delete detached volumes
#     now >= deadline + grace  : delete node groups and the EKS control plane
#
# Level-triggered on purpose (not a one-shot schedule): it keeps firing until
# `make extend` / `make up` moves the deadline, so instances that Karpenter or
# an ASG re-create after the first sweep die on the next tick.
#
# The EKS cluster depends on terraform_data.deadman_armed, so the switch exists
# before the first billable resource and is destroyed after the last one.

locals {
  deadman_name          = "${var.name}-deadman"
  deadman_param_prefix  = "/${var.name}/deadman"
  deadman_grace_seconds = var.deadman_cluster_grace == "" ? -1 : (endswith(var.deadman_cluster_grace, "h") ? 3600 : 60) * tonumber(trimsuffix(trimsuffix(var.deadman_cluster_grace, "h"), "m"))
}

# The deadline. Written once by Terraform (first apply: now + deadman_ttl);
# afterwards owned by scripts/deadman.sh (`make up` re-arms, `make extend`
# moves it), hence ignore_changes.
resource "aws_ssm_parameter" "deadman_deadline" {
  name        = "${local.deadman_param_prefix}/deadline"
  description = "tf-e2e dead-man switch: compute of stack ${var.name} is torn down after this instant (RFC 3339 UTC)"
  type        = "String"
  value       = timeadd(plantimestamp(), var.deadman_ttl)

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "deadman_status" {
  name        = "${local.deadman_param_prefix}/last-action"
  description = "tf-e2e dead-man switch: JSON summary of the last tick that removed something"
  type        = "String"
  value       = "{}"

  lifecycle {
    ignore_changes = [value]
  }
}

data "archive_file" "deadman" {
  type        = "zip"
  source_file = "${path.module}/lambda/deadman.py"
  output_path = "${path.module}/.build/deadman.zip"
}

data "aws_iam_policy_document" "deadman_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "deadman" {
  statement {
    sid       = "Describe"
    actions   = ["ec2:DescribeInstances", "ec2:DescribeVolumes"]
    resources = ["*"]
  }

  # Terminate / delete only what carries one of the stack's tags.
  dynamic "statement" {
    for_each = {
      Project        = { key = "Project", value = var.name }
      NodeGroup      = { key = "eks:cluster-name", value = var.name }
      Karpenter      = { key = "eks:eks-cluster-name", value = var.name }
      KubernetesOwns = { key = "kubernetes.io/cluster/${var.name}", value = "owned" }
    }

    content {
      sid       = "Remove${statement.key}"
      actions   = ["ec2:TerminateInstances", "ec2:DeleteVolume"]
      resources = ["*"]

      condition {
        test     = "StringEquals"
        variable = "aws:ResourceTag/${statement.value.key}"
        values   = [statement.value.value]
      }
    }
  }

  statement {
    sid = "Cluster"
    actions = [
      "eks:ListNodegroups",
      "eks:DescribeNodegroup",
      "eks:UpdateNodegroupConfig",
      "eks:DeleteNodegroup",
      "eks:DeleteCluster",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:eks:${var.region}:${data.aws_caller_identity.current.account_id}:cluster/${var.name}",
      "arn:${data.aws_partition.current.partition}:eks:${var.region}:${data.aws_caller_identity.current.account_id}:nodegroup/${var.name}/*",
    ]
  }

  statement {
    sid       = "Deadline"
    actions   = ["ssm:GetParameter", "ssm:PutParameter"]
    resources = ["arn:${data.aws_partition.current.partition}:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${local.deadman_param_prefix}/*"]
  }

  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.deadman.arn}:*"]
  }
}

resource "aws_iam_role" "deadman" {
  name               = local.deadman_name
  assume_role_policy = data.aws_iam_policy_document.deadman_assume.json
}

resource "aws_iam_role_policy" "deadman" {
  name   = "deadman"
  role   = aws_iam_role.deadman.id
  policy = data.aws_iam_policy_document.deadman.json
}

# Created explicitly so it is destroyed with the stack (a log group that Lambda
# auto-creates would be a leftover).
resource "aws_cloudwatch_log_group" "deadman" {
  name              = "/aws/lambda/${local.deadman_name}"
  retention_in_days = 7
}

resource "aws_lambda_function" "deadman" {
  function_name    = local.deadman_name
  description      = "Tears down the compute of the ${var.name} test stack once its TTL has expired"
  role             = aws_iam_role.deadman.arn
  runtime          = "python3.13"
  handler          = "deadman.handler"
  filename         = data.archive_file.deadman.output_path
  source_code_hash = data.archive_file.deadman.output_base64sha256
  timeout          = 120
  memory_size      = 128

  environment {
    variables = {
      CLUSTER        = var.name
      PROJECT        = var.name
      PARAM_DEADLINE = aws_ssm_parameter.deadman_deadline.name
      PARAM_STATUS   = aws_ssm_parameter.deadman_status.name
      GRACE_SECONDS  = tostring(local.deadman_grace_seconds)
    }
  }

  depends_on = [aws_iam_role_policy.deadman, aws_cloudwatch_log_group.deadman]
}

resource "aws_cloudwatch_event_rule" "deadman" {
  name                = local.deadman_name
  description         = "Ticks the ${var.name} dead-man switch"
  schedule_expression = "rate(5 minutes)"
}

resource "aws_cloudwatch_event_target" "deadman" {
  rule = aws_cloudwatch_event_rule.deadman.name
  arn  = aws_lambda_function.deadman.arn
}

resource "aws_lambda_permission" "deadman" {
  statement_id  = "events"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.deadman.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.deadman.arn
}

# Single handle for "the switch is fully wired".
resource "terraform_data" "deadman_armed" {
  input = aws_ssm_parameter.deadman_deadline.name

  depends_on = [
    aws_cloudwatch_event_target.deadman,
    aws_lambda_permission.deadman,
    aws_ssm_parameter.deadman_status,
  ]
}
