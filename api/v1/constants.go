package v1

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
