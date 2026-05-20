package utils

import (
	"strings"
	"testing"
	"time"

	"github.com/robfig/cron/v3"
)

// newFiveFieldParser mirrors what GPUPoolCompactionReconciler and
// CronRecommender construct, so the tests exercise the exact parser shape
// the production code uses.
func newFiveFieldParser() cron.Parser {
	return cron.NewParser(cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow)
}

func TestParseTimezoneAwareCronSchedule_Validation(t *testing.T) {
	cases := []struct {
		name        string
		schedule    string
		timezone    string
		wantErrFrag string
	}{
		{
			name:        "empty timezone is rejected",
			schedule:    "0 3 * * *",
			timezone:    "",
			wantErrFrag: "timezone is required",
		},
		{
			name:        "whitespace-only timezone is rejected",
			schedule:    "0 3 * * *",
			timezone:    "   ",
			wantErrFrag: "timezone is required",
		},
		{
			name:        "invalid IANA timezone is rejected",
			schedule:    "0 3 * * *",
			timezone:    "NotAZone",
			wantErrFrag: "invalid timezone",
		},
		{
			name:        "empty schedule is rejected",
			schedule:    "",
			timezone:    "UTC",
			wantErrFrag: "schedule is required",
		},
		{
			name:        "schedule embedding CRON_TZ= prefix is rejected",
			schedule:    "CRON_TZ=Asia/Shanghai 0 3 * * *",
			timezone:    "UTC",
			wantErrFrag: "must not embed TZ=/CRON_TZ= prefix",
		},
		{
			name:        "schedule embedding TZ= prefix is rejected",
			schedule:    "TZ=UTC 0 3 * * *",
			timezone:    "UTC",
			wantErrFrag: "must not embed TZ=/CRON_TZ= prefix",
		},
		{
			name:        "syntactically invalid cron is rejected by parser",
			schedule:    "0 25 * * *",
			timezone:    "UTC",
			wantErrFrag: "parse cron schedule",
		},
	}

	parser := newFiveFieldParser()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sched, err := ParseTimezoneAwareCronSchedule(parser, tc.schedule, tc.timezone)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil (schedule=%#v)", tc.wantErrFrag, sched)
			}
			if !strings.Contains(err.Error(), tc.wantErrFrag) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErrFrag)
			}
		})
	}
}

func TestParseTimezoneAwareCronSchedule_HappyPath(t *testing.T) {
	parser := newFiveFieldParser()

	for _, tc := range []struct {
		name     string
		schedule string
		timezone string
	}{
		{"UTC", "0 3 * * *", "UTC"},
		{"surrounding whitespace tolerated", "  0 3 * * *  ", " Asia/Shanghai "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sched, err := ParseTimezoneAwareCronSchedule(parser, tc.schedule, tc.timezone)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if sched == nil {
				t.Fatalf("expected non-nil schedule")
			}
		})
	}
}

// TestParseTimezoneAwareCronSchedule_NextHonorsTimezone locks in that the
// timezone argument actually changes the wall-clock fire time, i.e. callers
// do not silently fall back to time.Local. The same schedule "0 3 * * *"
// must fire at different absolute UTC instants when interpreted in UTC vs
// Asia/Shanghai (fixed at UTC+8, no DST).
func TestParseTimezoneAwareCronSchedule_NextHonorsTimezone(t *testing.T) {
	parser := newFiveFieldParser()

	schedUTC, err := ParseTimezoneAwareCronSchedule(parser, "0 3 * * *", "UTC")
	if err != nil {
		t.Fatalf("parse UTC schedule: %v", err)
	}
	schedSH, err := ParseTimezoneAwareCronSchedule(parser, "0 3 * * *", "Asia/Shanghai")
	if err != nil {
		t.Fatalf("parse Shanghai schedule: %v", err)
	}

	// Anchor at a deterministic instant so the test does not depend on
	// the runner's wall clock or timezone.
	anchor := time.Date(2026, 1, 15, 0, 0, 0, 0, time.UTC)
	nextUTC := schedUTC.Next(anchor).UTC()
	nextSH := schedSH.Next(anchor).UTC()

	wantUTC := time.Date(2026, 1, 15, 3, 0, 0, 0, time.UTC)
	if !nextUTC.Equal(wantUTC) {
		t.Fatalf("UTC schedule next: got %s want %s", nextUTC, wantUTC)
	}
	// Anchor 2026-01-15 00:00 UTC is 2026-01-15 08:00 in Shanghai (UTC+8),
	// past today's 03:00 fire, so Next() returns 2026-01-16 03:00 CST,
	// which is 2026-01-15 19:00 UTC.
	wantSH := time.Date(2026, 1, 15, 19, 0, 0, 0, time.UTC)
	if !nextSH.Equal(wantSH) {
		t.Fatalf("Shanghai schedule next: got %s want %s", nextSH, wantSH)
	}
	if nextUTC.Equal(nextSH) {
		t.Fatalf("UTC and Shanghai schedules fire at the same instant; timezone has no effect")
	}
}
