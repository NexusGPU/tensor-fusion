package metrics

import (
	"context"
	"errors"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
)

func newMockTSDB(t *testing.T) (*TimeSeriesDB, sqlmock.Sqlmock) {
	t.Helper()
	conn, mock, err := sqlmock.New()
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	db, err := gorm.Open(mysql.New(mysql.Config{Conn: conn, SkipInitializeWithVersion: true}),
		&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	require.NoError(t, err)
	return &TimeSeriesDB{DB: db}, mock
}

func expectTimeIndexQuery(mock sqlmock.Sqlmock, table string) *sqlmock.ExpectedQuery {
	return mock.ExpectQuery(regexp.QuoteMeta(timeIndexQuery)).WithArgs(table)
}

func timeIndexRows(columns ...string) *sqlmock.Rows {
	rows := sqlmock.NewRows([]string{"column_name"})
	for _, column := range columns {
		rows.AddRow(column)
	}
	return rows
}

// The time index of a table is whatever the database says it is: "ts" in a
// database initialized by the legacy DDL, "greptime_timestamp" in a table
// created on ingestion. The database is asked once per table.
func TestTimeIndexResolverDetectsColumnPerTable(t *testing.T) {
	for _, column := range []string{LegacyTimeIndexColumn, DefaultTimeIndexColumn} {
		t.Run(column, func(t *testing.T) {
			tsdb, mock := newMockTSDB(t)
			resolver := NewTimeIndexResolver(tsdb.DB)
			expectTimeIndexQuery(mock, "tf_worker_usage").WillReturnRows(timeIndexRows(column))
			// A database can hold tables of both kinds: tf_pool_metrics was
			// never part of the legacy DDL.
			expectTimeIndexQuery(mock, "tf_pool_metrics").WillReturnRows(timeIndexRows(DefaultTimeIndexColumn))

			ctx := context.Background()
			require.Equal(t, column, resolver.Column(ctx, "tf_worker_usage"))
			require.Equal(t, DefaultTimeIndexColumn, resolver.Column(ctx, "tf_pool_metrics"))
			// Answered from memory: the mock has no further query to give.
			require.Equal(t, column, resolver.Column(ctx, "tf_worker_usage"))
			require.Equal(t, DefaultTimeIndexColumn, resolver.Column(ctx, "tf_pool_metrics"))
			require.NoError(t, mock.ExpectationsWereMet())
		})
	}
}

// The query looks the table up in the database of the connection and asks for
// the column GreptimeDB marks as the time index.
func TestTimeIndexQueryShape(t *testing.T) {
	require.Contains(t, timeIndexQuery, "information_schema.columns")
	require.Contains(t, timeIndexQuery, "table_schema = database()")
	require.Contains(t, timeIndexQuery, "table_name = ?")
	require.Contains(t, timeIndexQuery, "semantic_type = 'TIMESTAMP'")
}

// A table that does not exist yet will be created by the first write, with the
// default time index. The resolver says so and asks again later instead of
// remembering a guess.
func TestTimeIndexResolverRetriesUndetectedTable(t *testing.T) {
	tsdb, mock := newMockTSDB(t)
	resolver := NewTimeIndexResolver(tsdb.DB)
	now := time.Now()
	resolver.now = func() time.Time { return now }
	ctx := context.Background()

	expectTimeIndexQuery(mock, "tf_node_metrics").WillReturnRows(timeIndexRows())
	require.Equal(t, DefaultTimeIndexColumn, resolver.Column(ctx, "tf_node_metrics"))
	// Not asked again right away.
	require.Equal(t, DefaultTimeIndexColumn, resolver.Column(ctx, "tf_node_metrics"))

	now = now.Add(timeIndexRetryAfter)
	expectTimeIndexQuery(mock, "tf_node_metrics").WillReturnRows(timeIndexRows(LegacyTimeIndexColumn))
	require.Equal(t, LegacyTimeIndexColumn, resolver.Column(ctx, "tf_node_metrics"))

	// A detected name is kept well beyond the retry delay.
	now = now.Add(timeIndexDetectedTTL - time.Second)
	require.Equal(t, LegacyTimeIndexColumn, resolver.Column(ctx, "tf_node_metrics"))
	require.NoError(t, mock.ExpectationsWereMet())
}

// When the database cannot be asked, the resolver assumes the default name for
// a table it knows nothing about, and keeps the name it detected earlier for a
// table it does know.
func TestTimeIndexResolverFallsBackOnError(t *testing.T) {
	tsdb, mock := newMockTSDB(t)
	resolver := NewTimeIndexResolver(tsdb.DB)
	now := time.Now()
	resolver.now = func() time.Time { return now }
	ctx := context.Background()
	unreachable := errors.New("connection refused")

	expectTimeIndexQuery(mock, "tf_gpu_usage").WillReturnError(unreachable)
	require.Equal(t, DefaultTimeIndexColumn, resolver.Column(ctx, "tf_gpu_usage"))

	now = now.Add(timeIndexRetryAfter)
	expectTimeIndexQuery(mock, "tf_gpu_usage").WillReturnRows(timeIndexRows(LegacyTimeIndexColumn))
	require.Equal(t, LegacyTimeIndexColumn, resolver.Column(ctx, "tf_gpu_usage"))

	now = now.Add(timeIndexDetectedTTL)
	expectTimeIndexQuery(mock, "tf_gpu_usage").WillReturnError(unreachable)
	require.Equal(t, LegacyTimeIndexColumn, resolver.Column(ctx, "tf_gpu_usage"))
	require.NoError(t, mock.ExpectationsWereMet())
}

// The detected name is put into SQL text, so a name that is not a plain
// identifier is not used.
func TestTimeIndexResolverIgnoresUnusableNames(t *testing.T) {
	tsdb, mock := newMockTSDB(t)
	resolver := NewTimeIndexResolver(tsdb.DB)
	expectTimeIndexQuery(mock, "tf_gpu_usage").WillReturnRows(timeIndexRows("ts; DROP TABLE x", "event time"))
	require.Equal(t, DefaultTimeIndexColumn, resolver.Column(context.Background(), "tf_gpu_usage"))
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestTimeIndexColumnWithoutDatabase(t *testing.T) {
	var missing *TimeSeriesDB
	require.Equal(t, DefaultTimeIndexColumn, missing.TimeIndexColumn(context.Background(), "tf_gpu_usage"))
	require.Equal(t, DefaultTimeIndexColumn, (&TimeSeriesDB{}).TimeIndexColumn(context.Background(), "tf_gpu_usage"))
	var resolver *TimeIndexResolver
	require.Equal(t, DefaultTimeIndexColumn, resolver.Column(context.Background(), "tf_gpu_usage"))
}

// A time bound is a timestamp string with an explicit zone: the database
// converts it to the precision of the column it is compared with, and it means
// the same instant whatever the zone of this process or of the session. An
// integer would be read in the unit of the column, which differs between
// tables created by the legacy DDL (ms) and on ingestion (ns).
func TestTimeFilterArg(t *testing.T) {
	instant := time.Date(2026, 3, 9, 18, 4, 5, 123_456_789, time.FixedZone("UTC+8", 8*3600))
	require.Equal(t, "2026-03-09T10:04:05.123Z", TimeFilterArg(instant))
	require.Equal(t, TimeFilterArg(instant), TimeFilterArg(instant.UTC()))
	parsed, err := time.Parse(time.RFC3339Nano, TimeFilterArg(instant))
	require.NoError(t, err)
	require.True(t, parsed.Equal(instant.Truncate(time.Millisecond)))
}

func TestFindRecentNodeMetricsFiltersOnTimeIndex(t *testing.T) {
	t.Run("table created on ingestion", func(t *testing.T) {
		tsdb, mock := newMockTSDB(t)
		expectTimeIndexQuery(mock, "tf_node_metrics").WillReturnRows(timeIndexRows(DefaultTimeIndexColumn))
		recorded := time.Now().Truncate(time.Second)
		mock.ExpectQuery(regexp.QuoteMeta("SELECT * FROM `tf_node_metrics` WHERE greptime_timestamp >= now() - '1 hour'::INTERVAL")).
			WillReturnRows(sqlmock.NewRows([]string{"node", "greptime_timestamp"}).AddRow("node-a", recorded))

		rows, err := tsdb.FindRecentNodeMetrics()
		require.NoError(t, err)
		require.Len(t, rows, 1)
		require.Equal(t, "node-a", rows[0].NodeName)
		require.True(t, recorded.Equal(rows[0].LastRecordTime))
		require.NoError(t, mock.ExpectationsWereMet())
	})

	t.Run("table created by the legacy DDL", func(t *testing.T) {
		tsdb, mock := newMockTSDB(t)
		expectTimeIndexQuery(mock, "tf_node_metrics").WillReturnRows(timeIndexRows(LegacyTimeIndexColumn))
		recorded := time.Now().Truncate(time.Second)
		mock.ExpectQuery(regexp.QuoteMeta("SELECT *, ts AS greptime_timestamp FROM `tf_node_metrics` WHERE ts >= now() - '1 hour'::INTERVAL")).
			WillReturnRows(sqlmock.NewRows([]string{"node", "ts", "greptime_timestamp"}).AddRow("node-a", recorded, recorded))

		rows, err := tsdb.FindRecentNodeMetrics()
		require.NoError(t, err)
		require.Len(t, rows, 1)
		require.True(t, recorded.Equal(rows[0].LastRecordTime))
		require.NoError(t, mock.ExpectationsWereMet())
	})
}

// modelColumns lists the columns a model declares.
func modelColumns(model any) []string {
	var columns []string
	modelType := reflect.TypeOf(model)
	for i := 0; i < modelType.NumField(); i++ {
		field := modelType.Field(i)
		if !field.IsExported() {
			continue
		}
		for _, part := range strings.Split(field.Tag.Get("gorm"), ";") {
			if column, ok := strings.CutPrefix(part, "column:"); ok {
				columns = append(columns, column)
			}
		}
	}
	return columns
}

// The models and the DDL describe one schema: the one ingestion creates, with
// the default time index, and with every column the recorder writes.
func TestModelsAndDDLDescribeTheSupportedSchema(t *testing.T) {
	ddl := map[string]string{}
	for _, version := range TFVersionMigrationMap {
		for _, statement := range version.AlterSQL {
			name := strings.Fields(strings.TrimPrefix(statement, "CREATE TABLE IF NOT EXISTS "))[0]
			ddl[name] = statement
		}
	}

	models := []schema.Tabler{
		WorkerResourceMetrics{},
		NodeResourceMetrics{},
		PoolResourceMetrics{},
		TensorFusionSystemMetrics{},
		HypervisorWorkerUsageMetrics{},
		HypervisorGPUUsageMetrics{},
		GPUAllocationMetrics{},
		GPUResourceMetrics{},
	}
	for _, model := range models {
		table := model.TableName()
		statement, ok := ddl[table]
		require.True(t, ok, "no DDL for %s", table)
		require.Contains(t, statement, "`"+DefaultTimeIndexColumn+"` Timestamp_ns TIME INDEX", table)
		require.NotContains(t, statement, "`"+LegacyTimeIndexColumn+"` ", table)

		columns := modelColumns(model)
		require.Contains(t, columns, DefaultTimeIndexColumn, table)
		for _, column := range columns {
			require.Contains(t, statement, "`"+column+"` ", "column %s of %s is missing from the DDL", column, table)
		}
		// The DDL the models generate agrees on the time index.
		require.Contains(t, GetInitTableSQL(model, "30d"), "`"+DefaultTimeIndexColumn+"` Timestamp_ns TIME INDEX", table)
	}
}
