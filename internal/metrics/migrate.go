package metrics

// When upgrading database, should run alter sql in order for every
// version not lower than current version, until the version to be updated
var TFVersionMigrationMap = []struct {
	Version  string
	AlterSQL []string
}{
	// init version, just run init SQL
	{"1.0", []string{
		"CREATE TABLE IF NOT EXISTS tf_worker_resources (\n    `worker` String NULL SKIPPING INDEX,\n    `workload` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `namespace` String NULL INVERTED INDEX,\n    `qos` String NULL,\n    `tflops_request` Double NULL,\n    `tflops_limit` Double NULL,\n    `vram_bytes_request` Double NULL,\n    `vram_bytes_limit` Double NULL,\n    `gpu_count` BigInt NULL,\n    `raw_cost` Double NULL,\n    `ts` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`worker`, `workload`, `pool`, `namespace`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_node_resources (\n    `node` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `allocated_tflops` Double NULL,\n    `allocated_tflops_percent` Double NULL,\n    `allocated_vram_bytes` Double NULL,\n    `allocated_vram_percent` Double NULL,\n    `allocated_tflops_percent_virtual` Double NULL,\n    `allocated_vram_percent_virtual` Double NULL,\n    `raw_cost` Double NULL,\n    `gpu_count` BigInt NULL,\n    `ts` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`node`, `pool`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_system_metrics (\n    `pool` String NULL INVERTED INDEX,\n    `total_workers_cnt` BigInt NULL,\n    `total_nodes_cnt` BigInt NULL,\n    `total_allocation_fail_cnt` BigInt NULL,\n    `total_allocation_success_cnt` BigInt NULL,\n    `total_scale_up_cnt` BigInt NULL,\n    `total_scale_down_cnt` BigInt NULL,\n    `ts` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`pool`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_system_log (\n    `component` String NULL INVERTED INDEX,\n    `container` String NULL INVERTED INDEX,\n    `message` String NULL FULLTEXT INDEX WITH (analyzer = 'English' , case_sensitive = 'false'),\n    `namespace` String NULL INVERTED INDEX,\n    `pod` String NULL SKIPPING INDEX,\n    `stream` String NULL,\n    `timestamp` String NULL,\n    `greptime_timestamp` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`component`, `container`, `namespace`, `pod`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_worker_usage (\n    `workload` String NULL INVERTED INDEX,\n    `worker` String NULL SKIPPING INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `node` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_percentage` Double NULL,\n    `memory_bytes` BigInt UNSIGNED NULL,\n    `compute_tflops` Double NULL,\n    `compute_throttled_cnt` BigInt NULL,\n    `vram_freezed_cnt` BigInt NULL,\n    `vram_resumed_cnt` BigInt NULL,\n    `ts` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`workload`, `worker`, `pool`, `node`, `uuid`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_gpu_usage (\n    `node` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_percentage` Double NULL,\n    `memory_percentage` Double NULL,\n    `memory_bytes` BigInt UNSIGNED NULL,\n    `compute_tflops` Double NULL,\n    `rx` Double NULL,\n    `tx` Double NULL,\n    `temperature` Double NULL,\n    `ts` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`node`, `pool`, `uuid`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",
	}},

	// add alter SQL in future
	{"1.1", []string{}},
}

const CurrentAppSQLVersion = "1.0"
