package metrics

import (
	"context"
	"regexp"
	"sync"
	"time"

	"gorm.io/gorm"
)

const (
	// DefaultTimeIndexColumn is the time index of a table GreptimeDB creates on
	// the first line-protocol write, which is how the tf_* tables come to exist.
	// It is the schema the operator supports.
	DefaultTimeIndexColumn = "greptime_timestamp"

	// LegacyTimeIndexColumn is the time index of tables created by the DDL that
	// earlier releases ran at start-up (SetupTables). Such tables keep working:
	// every query asks the database for the table's time index instead of
	// assuming a name.
	LegacyTimeIndexColumn = "ts"

	// A table has exactly one time index and cannot change it without being
	// recreated, so a detected name is kept for a long time. A failed detection
	// (database unreachable, table not created yet) is retried soon.
	timeIndexDetectedTTL = 10 * time.Minute
	timeIndexRetryAfter  = time.Minute
	timeIndexQueryLimit  = 5 * time.Second

	timeIndexQuery = "SELECT column_name FROM information_schema.columns " +
		"WHERE table_schema = database() AND table_name = ? AND semantic_type = 'TIMESTAMP'"

	// timeFilterLayout renders a time for a time index filter. The rows are
	// written with millisecond precision, and the zone is explicit so that the
	// value does not depend on the time zone of the session or of this process.
	timeFilterLayout = "2006-01-02T15:04:05.000Z07:00"
)

var sqlIdentifier = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// TimeFilterArg renders t as the argument of a comparison against a time index
// column. A timestamp string is converted by the database to the precision of
// the column it is compared with; an integer is not: it is read in the
// column's own unit, which is milliseconds for tables created by the legacy
// DDL and nanoseconds for tables created on ingestion.
func TimeFilterArg(t time.Time) string {
	return t.UTC().Format(timeFilterLayout)
}

type timeIndexEntry struct {
	column    string
	checkedAt time.Time
	detected  bool
}

// TimeIndexResolver tells which column is the time index of a table, asking
// the database once per table and remembering the answer.
type TimeIndexResolver struct {
	db  *gorm.DB
	now func() time.Time

	mu      sync.Mutex
	entries map[string]timeIndexEntry
}

func NewTimeIndexResolver(db *gorm.DB) *TimeIndexResolver {
	return &TimeIndexResolver{
		db:      db,
		now:     time.Now,
		entries: map[string]timeIndexEntry{},
	}
}

// Column returns the time index column of the table. When the database cannot
// tell (it is unreachable, or the table does not exist yet and will be created
// by the first write), it returns DefaultTimeIndexColumn and asks again later.
func (r *TimeIndexResolver) Column(ctx context.Context, table string) string {
	if r == nil || r.db == nil {
		return DefaultTimeIndexColumn
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.now()
	if entry, ok := r.entries[table]; ok {
		ttl := timeIndexRetryAfter
		if entry.detected {
			ttl = timeIndexDetectedTTL
		}
		if now.Sub(entry.checkedAt) < ttl {
			return entry.column
		}
	}

	entry := timeIndexEntry{column: DefaultTimeIndexColumn, checkedAt: now}
	column, err := r.detect(ctx, table)
	switch {
	case err != nil:
		// Keep an earlier answer over the default: the table did not change
		// because the database could not be asked.
		if previous, ok := r.entries[table]; ok && previous.detected {
			entry.column = previous.column
		}
		log.Error(err, "can not detect the time index column, assuming a name", "table", table, "column", entry.column)
	case column != "":
		entry.column = column
		entry.detected = true
	}
	r.entries[table] = entry
	return entry.column
}

func (r *TimeIndexResolver) detect(ctx context.Context, table string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	queryCtx, cancel := context.WithTimeout(ctx, timeIndexQueryLimit)
	defer cancel()

	var columns []string
	if err := r.db.WithContext(queryCtx).Raw(timeIndexQuery, table).Scan(&columns).Error; err != nil {
		return "", err
	}
	for _, column := range columns {
		// The name goes into SQL text, so only a plain identifier is accepted.
		if sqlIdentifier.MatchString(column) {
			return column, nil
		}
	}
	return "", nil
}

// TimeIndexColumn returns the time index column of a table of this database,
// see TimeIndexResolver.Column.
func (t *TimeSeriesDB) TimeIndexColumn(ctx context.Context, table string) string {
	db := t.Conn()
	if db == nil {
		return DefaultTimeIndexColumn
	}
	t.timeIndexOnce.Do(func() {
		t.timeIndex = NewTimeIndexResolver(db)
	})
	return t.timeIndex.Column(ctx, table)
}
