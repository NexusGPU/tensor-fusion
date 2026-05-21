package utils

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/robfig/cron/v3"
)

// ParseTimezoneAwareCronSchedule validates the (schedule, timezone) pair and
// returns a cron.Schedule whose Next() is evaluated in the given IANA
// timezone. The timezone is composed in via robfig/cron's CRON_TZ= prefix so
// the schedule never silently falls back to the controller process's
// time.Local.
//
// Embedding TZ=/CRON_TZ= directly in the schedule field is rejected so the
// timezone choice stays in its own typed field and remains greppable in
// GitOps reviews.
//
// Returns a descriptive error suitable for surfacing in events/logs when the
// timezone or schedule is malformed; callers must treat the schedule as
// disabled on error.
func ParseTimezoneAwareCronSchedule(parser cron.Parser, schedule, timezone string) (cron.Schedule, error) {
	tz := strings.TrimSpace(timezone)
	if tz == "" {
		return nil, errors.New("timezone is required (e.g. \"UTC\" or \"Asia/Shanghai\")")
	}
	if _, err := time.LoadLocation(tz); err != nil {
		// "unknown time zone" also fires when the image lacks tzdata
		// and the binary was built without time/tzdata; hint both.
		return nil, fmt.Errorf("invalid timezone %q: %w (check IANA name; image needs tzdata or binary must embed time/tzdata)", tz, err)
	}
	sched := strings.TrimSpace(schedule)
	if sched == "" {
		return nil, errors.New("schedule is required")
	}
	if strings.HasPrefix(sched, "TZ=") || strings.HasPrefix(sched, "CRON_TZ=") {
		return nil, errors.New("schedule must not embed TZ=/CRON_TZ= prefix; use the timezone field instead")
	}
	spec := fmt.Sprintf("CRON_TZ=%s %s", tz, sched)
	parsed, err := parser.Parse(spec)
	if err != nil {
		return nil, fmt.Errorf("parse cron schedule %q in timezone %q: %w", sched, tz, err)
	}
	return parsed, nil
}
