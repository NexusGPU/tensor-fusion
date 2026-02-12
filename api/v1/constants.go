package v1

// Domain is the default domain for tensor-fusion.ai API group.
const Domain = "tensor-fusion.ai"

// Phase constants for resource lifecycle states.
const (
	PhaseUnknown    = "Unknown"
	PhasePending    = "Pending"
	PhaseUpdating   = "Updating"
	PhaseScheduling = "Scheduling"
	PhaseMigrating  = "Migrating"
	PhaseDestroying = "Destroying"
	PhaseRunning    = "Running"
	PhaseSucceeded  = "Succeeded"
	PhaseFailed     = "Failed"
)

// Condition status type constants.
const (
	ConditionStatusTypeReady = "Ready"
)
