package metrics

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	goerrors "errors"
	golog "log"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/go-logr/logr"
	mysqlDriver "github.com/go-sql-driver/mysql"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"gorm.io/gorm/schema"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type GreptimeDBConnection struct {
	Host     string
	Port     string
	User     string
	Password string
	Database string
}

// ErrNotConnected is what a query on a TimeSeriesDB returns while the database
// has not been reached yet.
var ErrNotConnected = goerrors.New("time series database is not connected yet")

// connectDialTimeout bounds one connection attempt, so that an address that
// does not answer at all cannot hold up the next attempt for minutes.
const connectDialTimeout = 10 * time.Second

// TimeSeriesDB is the handle of the metrics database. It may exist before the
// database is reachable: the operator starts while GreptimeDB is still coming
// up, and connects in the background (ConnectWithRetry).
//
// DB is nil until then. Code that can run while a connect is still in progress
// must take the connection from Conn, which is nil-safe and synchronised,
// rather than use DB or the gorm methods promoted from it.
type TimeSeriesDB struct {
	*gorm.DB

	connMu sync.RWMutex
	// open replaces the real connection attempt in tests.
	open func(GreptimeDBConnection) (*gorm.DB, error)

	timeIndexOnce sync.Once
	timeIndex     *TimeIndexResolver
}

// Conn returns the connection, or nil while the database has not been reached.
func (m *TimeSeriesDB) Conn() *gorm.DB {
	if m == nil {
		return nil
	}
	m.connMu.RLock()
	defer m.connMu.RUnlock()
	return m.DB
}

// Connected tells whether the database has been reached.
func (m *TimeSeriesDB) Connected() bool {
	return m.Conn() != nil
}

// SetConn makes db the connection of this handle, for a caller that opened it
// itself. An existing connection is kept.
func (m *TimeSeriesDB) SetConn(db *gorm.DB) {
	m.connMu.Lock()
	defer m.connMu.Unlock()
	if m.DB == nil {
		m.DB = db
	}
}

// Setup connects once and returns the error of that attempt. Start-up code
// must not treat the error as fatal; see ConnectWithRetry.
func (m *TimeSeriesDB) Setup(connection GreptimeDBConnection) error {
	if m.Connected() {
		return nil
	}
	open := m.open
	if open == nil {
		open = openGreptimeDB
	}
	db, err := open(connection)
	if err != nil {
		return err
	}
	m.SetConn(db)
	return nil
}

// ConnectBackoff is the wait between connection attempts: Initial after the
// first failure, doubled after each further one, at most Max. Zero fields take
// the defaults.
type ConnectBackoff struct {
	Initial time.Duration
	Max     time.Duration
	// Logger receives the state changes; the zero value is the package logger.
	Logger logr.Logger
}

const (
	defaultConnectBackoffInitial = 2 * time.Second
	defaultConnectBackoffMax     = 30 * time.Second
)

// ConnectWithRetry connects to the database, trying again with backoff for as
// long as it takes. It returns nil once connected and the context's error if
// ctx ends first; an unreachable database is never a reason for the caller to
// stop. Run it in a goroutine: until it returns, Conn is nil and the consumers
// of this handle answer without data.
func (m *TimeSeriesDB) ConnectWithRetry(ctx context.Context, connection GreptimeDBConnection, backoff ConnectBackoff) error {
	return RetryUntilConnected(ctx, "time series database", backoff, func() error {
		return m.Setup(connection)
	})
}

// RetryUntilConnected calls connect until it succeeds, waiting between attempts
// as backoff says, and returns nil; or returns ctx's error if ctx ends first.
//
// It writes one log line when the state changes - the first failure, a failure
// for a different reason, the success after failures, giving up because ctx
// ended - and none for a repeated failure.
func RetryUntilConnected(ctx context.Context, what string, backoff ConnectBackoff, connect func() error) error {
	stateLog := backoff.Logger
	if stateLog.GetSink() == nil {
		stateLog = log
	}
	wait := backoff.Initial
	if wait <= 0 {
		wait = defaultConnectBackoffInitial
	}
	maxWait := backoff.Max
	if maxWait <= 0 {
		maxWait = defaultConnectBackoffMax
	}
	if maxWait < wait {
		maxWait = wait
	}

	started := time.Now()
	lastFailure := ""
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastFailure != "" {
				stateLog.Info("stopped connecting to "+what+", it was never reachable", "attempts", attempt-1, "lastError", lastFailure)
			}
			return err
		}
		err := connect()
		if err == nil {
			if lastFailure != "" {
				stateLog.Info(what+" is reachable now", "attempts", attempt, "after", time.Since(started).Round(time.Millisecond).String())
			}
			return nil
		}
		if failure := err.Error(); failure != lastFailure {
			lastFailure = failure
			stateLog.Error(err, what+" is not reachable, retrying in the background; "+
				"features that need it stay off until it is", "attempt", attempt, "maxRetryInterval", maxWait.String())
		}

		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
		if wait *= 2; wait > maxWait {
			wait = maxWait
		}
	}
}

func openGreptimeDB(connection GreptimeDBConnection) (*gorm.DB, error) {
	dsn := fmt.Sprintf("tcp(%s:%s)/%s?charset=utf8mb4&parseTime=True&loc=Local&timeout=%s",
		connection.Host, connection.Port, connection.Database, connectDialTimeout)
	if connection.User != "" && connection.Password != "" {
		dsn = fmt.Sprintf("%s:%s@%s", connection.User, connection.Password, dsn)
	}

	logFile, err := os.OpenFile("/tmp/gorm.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Error(err, "failed to create or open gorm log file in /tmp/gorm.log")
		return nil, goerrors.New("init gorm log failed")
	}
	logger := logger.New(golog.New(logFile, "\r\n", golog.LstdFlags), logger.Config{
		SlowThreshold:             200 * time.Millisecond,
		LogLevel:                  logger.Info,
		IgnoreRecordNotFoundError: false,
		Colorful:                  true,
	})
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{
		Logger: logger,
	})
	if err != nil {
		// Every failed attempt opened the log file again.
		_ = logFile.Close()
		return nil, err
	}
	return db, nil
}

// Deprecated Code
// No need migration for new tables, Dynamic created during ingestion
// Dynamic indexed in Greptime Cloud/Enterprise edition
//
// Nothing calls this any more: the tf_* tables are created by GreptimeDB on the
// first line-protocol write, with DefaultTimeIndexColumn as their time index.
// Databases this function initialized in earlier releases have
// LegacyTimeIndexColumn instead; queries ask TimeIndexColumn for the name.
func (t *TimeSeriesDB) SetupTables(client client.Client) error {
	db := t.Conn()
	if db == nil {
		return ErrNotConnected
	}

	// read or create configMap, version: v1
	var versionConfig corev1.ConfigMap
	if err := client.Get(context.Background(), types.NamespacedName{
		Namespace: utils.CurrentNamespace(),
		Name:      constants.TSDBVersionConfigMap,
	}, &versionConfig); err != nil {
		if errors.IsNotFound(err) {
			versionConfig = corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      constants.TSDBVersionConfigMap,
					Namespace: utils.CurrentNamespace(),
				},
				Data: map[string]string{
					"version": "0",
				},
			}
			if err := client.Create(context.Background(), &versionConfig); err != nil {
				return err
			}
		} else {
			return err
		}
	}

	version := versionConfig.Data["version"]

	// if version not match, run alter sql since current DB version
	if version != CurrentAppSQLVersion {
		for _, versionedSql := range TFVersionMigrationMap {
			if versionedSql.Version <= version {
				// skip already migrated version
				continue
			}
			for _, sql := range versionedSql.AlterSQL {
				if err := db.Exec(sql).Error; err != nil {
					return err
				}
			}
			log.Info("migrating time series db version", "version", versionedSql.Version)
		}
		versionConfig.Data["version"] = CurrentAppSQLVersion

		if err := client.Update(context.Background(), &versionConfig); err != nil {
			return err
		}
		log.Info("init/upgrade DB schema done, current time series db version", "version", versionConfig.Data["version"])
	}

	return nil
}

func (t *TimeSeriesDB) SetTableTTL(ttl string) error {
	tables := []schema.Tabler{
		&WorkerResourceMetrics{},
		&NodeResourceMetrics{},
		&TensorFusionSystemMetrics{},
		&TFSystemLog{},
		&HypervisorWorkerUsageMetrics{},
		&HypervisorGPUUsageMetrics{},
		&PoolResourceMetrics{},
		&GPUAllocationMetrics{},
		// tf_gpu_metrics — per-GPU capacity/availability rows, a distinct schema
		// from tf_gpu_usage (HypervisorGPUUsageMetrics). Omitting it from the
		// dynamic TTL update pins its retention to the boot-time default while
		// every other table reacts to runtime changes.
		&GPUResourceMetrics{},
	}
	// Nothing to alter while the database is not connected. Whoever connects
	// it applies the configured TTL afterwards.
	db := t.Conn()
	if db == nil {
		return nil
	}
	for _, table := range tables {
		if err := db.Exec("ALTER TABLE " + table.TableName() + " SET ttl = '" + ttl + "'").Error; err != nil {
			var mysqlErr *mysqlDriver.MySQLError
			if goerrors.As(err, &mysqlErr) {
				if mysqlErr.Number == 1146 {
					continue
				}
			}
			return err
		}
	}
	return nil
}

func (t *TimeSeriesDB) FindRecentNodeMetrics() ([]NodeResourceMetrics, error) {
	var monitors []NodeResourceMetrics
	db := t.Conn()
	if db == nil {
		return nil, ErrNotConnected
	}
	table := NodeResourceMetrics{}.TableName()
	timeColumn := t.TimeIndexColumn(context.Background(), table)
	// The time index is read into LastRecordTime under the column name of the
	// model, whatever it is called in this database.
	query := db.Table(table).Where(timeColumn + " >= now() - '1 hour'::INTERVAL")
	if timeColumn != DefaultTimeIndexColumn {
		query = query.Select("*, " + timeColumn + " AS " + DefaultTimeIndexColumn)
	}
	err := query.Find(&monitors).Error
	return monitors, err
}
