package metrics

import (
	"context"
	"fmt"
	"os"
	"time"

	goerrors "errors"
	golog "log"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
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

type TimeSeriesDB struct {
	*gorm.DB
}

func (m *TimeSeriesDB) Setup(connection GreptimeDBConnection) error {
	if m.DB != nil {
		return nil
	}

	dsn := fmt.Sprintf("tcp(%s:%s)/%s?charset=utf8mb4&parseTime=True&loc=Local",
		connection.Host, connection.Port, connection.Database)
	if connection.User != "" && connection.Password != "" {
		dsn = fmt.Sprintf("%s:%s@%s", connection.User, connection.Password, dsn)
	}

	logFile, err := os.OpenFile("/tmp/gorm.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Error(err, "failed to create or open gorm log file in /tmp/gorm.log")
		return goerrors.New("init gorm log failed")
	}
	logger := logger.New(golog.New(logFile, "\r\n", golog.LstdFlags), logger.Config{
		SlowThreshold:             200 * time.Millisecond,
		LogLevel:                  logger.Warn,
		IgnoreRecordNotFoundError: false,
		Colorful:                  true,
	})
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{
		Logger: logger,
	})
	if err != nil {
		return err
	}

	m.DB = db
	return nil
}

// Deprecated Code
// No need migration for new tables, Dynamic created during ingestion
// Dynamic indexed in Greptime Cloud/Enterprise edition
func (t *TimeSeriesDB) SetupTables(client client.Client) error {

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
				if err := t.DB.Exec(sql).Error; err != nil {
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
	}
	if t == nil || t.DB == nil {
		return nil
	}
	for _, table := range tables {
		if err := t.DB.Exec("ALTER TABLE " + table.TableName() + " SET ttl = '" + ttl + "'").Error; err != nil {
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
	err := t.DB.Find(&monitors, map[string]interface{}{
		"ts": gorm.Expr("now() - interval 1 hour"),
	}).Error
	return monitors, err
}
