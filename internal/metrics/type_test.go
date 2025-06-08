package metrics

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetInitTableSQL(t *testing.T) {

	t.Run("test get init table sql for worker resource metrics", func(t *testing.T) {
		sql := getInitTableSQL(&WorkerResourceMetrics{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_worker_resources (\n    `worker_name` String NULL INVERTED INDEX,\n    `workload_name` String NULL INVERTED INDEX,\n    `pool_name` String NULL INVERTED INDEX,\n    `namespace` String NULL INVERTED INDEX,\n    `qos` String NULL,\n    `tflops_request` Double NULL,\n    `tflops_limit` Double NULL,\n    `vram_bytes_request` Double NULL,\n    `vram_bytes_limit` Double NULL,\n    `gpu_count` BigInt NULL,\n    `raw_cost` Double NULL,\n    `ts` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})

	t.Run("test get init table sql for node resource metrics", func(t *testing.T) {
		sql := getInitTableSQL(&NodeResourceMetrics{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_node_resources (\n    `node_name` String NULL INVERTED INDEX,\n    `pool_name` String NULL INVERTED INDEX,\n    `allocated_tflops` Double NULL,\n    `allocated_tflops_percent` Double NULL,\n    `allocated_vram_bytes` Double NULL,\n    `allocated_vram_percent` Double NULL,\n    `allocated_tflops_percent_virtual` Double NULL,\n    `allocated_vram_percent_virtual` Double NULL,\n    `raw_cost` Double NULL,\n    `gpu_count` BigInt NULL,\n    `ts` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})

	t.Run("test get init table sql for system log", func(t *testing.T) {
		sql := getInitTableSQL(&TFSystemLog{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_system_log (\n    `component` String NULL INVERTED INDEX,\n    `container` String NULL INVERTED INDEX,\n    `message` String NULL FULLTEXT INDEX WITH (analyzer = 'English' , case_sensitive = 'false'),\n    `namespace` String NULL INVERTED INDEX,\n    `pod` String NULL,\n    `stream` String NULL,\n    `timestamp` String NULL,\n    `greptime_timestamp` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})

	t.Run("test get init table sql for hypervisor gpu usage metrics", func(t *testing.T) {
		sql := getInitTableSQL(&HypervisorGPUUsageMetrics{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_gpu_usage (\n    `node_name` String NULL INVERTED INDEX,\n    `pool_name` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_percent` Double NULL,\n    `vram_bytes` BigInt UNSIGNED NULL,\n    `compute_tflops` Double NULL,\n    `ts` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})

	t.Run("test get init table sql for hypervisor worker usage metrics", func(t *testing.T) {
		sql := getInitTableSQL(&HypervisorWorkerUsageMetrics{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_worker_usage (\n    `workload_name` String NULL INVERTED INDEX,\n    `worker_name` String NULL,\n    `pool_name` String NULL INVERTED INDEX,\n    `node_name` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_percent` Double NULL,\n    `vram_bytes` BigInt UNSIGNED NULL,\n    `compute_tflops` Double NULL,\n    `ts` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})

	t.Run("test get init table sql for system metrics", func(t *testing.T) {
		sql := getInitTableSQL(&TensorFusionSystemMetrics{}, "30d")
		expect := strings.TrimSpace("CREATE TABLE IF NOT EXISTS tf_system_metrics (\n    `pool_name` String NULL INVERTED INDEX,\n    `total_workers_cnt` BigInt NULL,\n    `total_nodes_cnt` BigInt NULL,\n    `total_allocation_fail_cnt` BigInt NULL,\n    `total_allocation_success_cnt` BigInt NULL,\n    `total_scale_up_cnt` BigInt NULL,\n    `total_scale_down_cnt` BigInt NULL,\n    `ts` Timestamp_ms TIME INDEX\n    )\n    ENGINE=mito WITH( ttl='30d', append_mode = 'true')")
		assert.Equal(t, expect, sql, "SQL migrated, should sync sql here and add ALTER table sql for migrating changed fields")
	})
}
