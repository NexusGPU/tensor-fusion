package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorm.io/gorm/schema"
)

func TestGetInitTableSQL(t *testing.T) {

	t.Run("test get init table sql", func(t *testing.T) {

		tables := []schema.Tabler{
			&WorkerResourceMetrics{},
			&NodeResourceMetrics{},
			&TensorFusionSystemMetrics{},
			&TFSystemLog{},
			&HypervisorWorkerUsageMetrics{},
			&HypervisorGPUUsageMetrics{},
		}

		for idx, initSql := range TFVersionMigrationMap[0].AlterSQL {
			assert.Equal(t, initSql, getInitTableSQL(tables[idx], "30d"), "table structure has been changed, need to sync sql and add ALTER table sql in `migrate.go`")
		}
	})
}
