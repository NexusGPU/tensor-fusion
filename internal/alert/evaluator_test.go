package alert

import (
	"context"
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestAlertEvaluator(t *testing.T) {
	ctx := context.Background()
	timeSeriesDB := &metrics.TimeSeriesDB{}
	connection := metrics.GreptimeDBConnection{
		Host:     utils.GetEnvOrDefault("TSDB_MYSQL_HOST", "127.0.0.1"),
		Port:     utils.GetEnvOrDefault("TSDB_MYSQL_PORT", "4002"),
		User:     utils.GetEnvOrDefault("TSDB_MYSQL_USER", "root"),
		Password: utils.GetEnvOrDefault("TSDB_MYSQL_PASSWORD", ""),
		Database: utils.GetEnvOrDefault("TSDB_MYSQL_DATABASE", "public"),
	}
	if err := timeSeriesDB.Setup(connection); err != nil {
		t.Error(err, "unable to setup time series db, features including alert, "+
			"autoScaling, rebalance won't work", "connection", connection.Host, "port",
			connection.Port, "user", connection.User, "database", connection.Database)
		return
	}
	evaluator := NewAlertEvaluator(ctx, timeSeriesDB, nil, "http://alertmanager:9093")

	t.Run("test evaluate rules", func(t *testing.T) {
		if err := evaluator.evaluate(Rule{
			Name: "test_rule",
		}); err != nil {
			t.Error(err)
		}
	})

}
