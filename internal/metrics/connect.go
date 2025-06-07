package metrics

import (
	"fmt"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
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
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return err
	}

	m.DB = db
	return nil
}

func (t *TimeSeriesDB) SetupTables(ttl string) error {
	// can not use db.Migrator because syntax mismatch
	if err := t.DB.Exec(getInitTableSQL(&NodeResourceMetrics{}, ttl)).Error; err != nil {
		return err
	}
	if err := t.DB.Exec(getInitTableSQL(&WorkerResourceMetrics{}, ttl)).Error; err != nil {
		return err
	}
	if err := t.DB.Exec(getInitTableSQL(&HypervisorWorkerUsageMetrics{}, ttl)).Error; err != nil {
		return err
	}
	if err := t.DB.Exec(getInitTableSQL(&HypervisorGPUUsageMetrics{}, ttl)).Error; err != nil {
		return err
	}
	if err := t.DB.Exec(getInitTableSQL(&TensorFusionSystemMetrics{}, ttl)).Error; err != nil {
		return err
	}
	if err := t.DB.Exec(getInitTableSQL(&TFSystemLog{}, ttl)).Error; err != nil {
		return err
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
