package gpuallocator

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

// Reports of over-commit changes are limited per GPU and in total; what is not
// reported is counted and handed back once when the interval ends.
func TestOvercommitReportsAreLimited(t *testing.T) {
	start := time.Unix(1_800_000_000, 0)
	gpu := func(i int) types.NamespacedName { return types.NamespacedName{Name: fmt.Sprintf("gpu-%d", i)} }

	// One GPU crossing its capacity back and forth with every sync.
	var flapping overcommitReportLimiter
	reports := 0
	for i := range 104 {
		report, summary := flapping.allow(gpu(0), i%2 == 0, start.Add(time.Duration(i)*30*time.Millisecond))
		require.Zero(t, summary)
		if report {
			reports++
		}
	}
	require.Equal(t, 2, reports, "one report per direction")
	// The next interval reports again and says what was left out.
	report, summary := flapping.allow(gpu(0), true, start.Add(overcommitReportInterval))
	require.True(t, report)
	require.Equal(t, 102, summary)

	// A hundred GPUs changing at once.
	var mass overcommitReportLimiter
	reports = 0
	for i := range 100 {
		if report, _ := mass.allow(gpu(i), true, start); report {
			reports++
		}
	}
	require.Equal(t, overcommitReportBurst, reports)
	report, summary = mass.allow(gpu(0), false, start.Add(overcommitReportInterval+time.Second))
	require.True(t, report)
	require.Equal(t, 100-overcommitReportBurst, summary)
	// Nothing is remembered for ever.
	_, _ = mass.allow(gpu(0), true, start.Add(3*overcommitReportInterval))
	require.LessOrEqual(t, len(mass.last), 2)

	// A clock that steps back does not silence reports.
	var stepped overcommitReportLimiter
	report, _ = stepped.allow(gpu(0), true, start)
	require.True(t, report)
	report, _ = stepped.allow(gpu(0), true, start.Add(-time.Hour))
	require.True(t, report)
}
