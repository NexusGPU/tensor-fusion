package v1

const (
	// Domain is the default domain for tensor-fusion.ai API group.
	Domain = "tensor-fusion.ai"

	// DomainPrefix is the prefix of the domain for tensor-fusion.ai API group.
	DomainPrefix = "tensor-fusion"
)

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
