# Optional cost alarm. NOTE: AWS Budgets data lags by 8-24h, so this is a
# "you forgot to destroy it yesterday" alarm, not a real-time brake. The real
# brakes are the dead-man switch (deadman.tf), the NodePool limits, the GPUPool
# maxResources and `make down`.
resource "aws_budgets_budget" "daily" {
  count = var.budget_alert_email == null ? 0 : 1

  name         = "${var.name}-daily"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_daily_usd)
  limit_unit   = "USD"
  time_unit    = "DAILY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Project$${var.name}"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
