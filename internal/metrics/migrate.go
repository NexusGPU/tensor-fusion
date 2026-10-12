package metrics

// When upgrading database, should run alter sql in order for every
// version not lower than current version, until the version to be updated
//
// These statements are not executed any more (see SetupTables): GreptimeDB
// creates each table on its first line-protocol write and adds a column the
// first time a row carries it. They describe that schema, so that creating the
// tables by hand gives the same result as ingestion does: the time index is
// DefaultTimeIndexColumn with nanosecond precision, tags are the primary key.
// Earlier revisions of this file named the time index LegacyTimeIndexColumn
// (millisecond precision); databases created from them are still read, see
// TimeIndexResolver.
var TFVersionMigrationMap = []struct {
	Version  string
	AlterSQL []string
}{
	// init version, just run init SQL
	{"1.0", []string{
		"CREATE TABLE IF NOT EXISTS tf_worker_resources (\n    `worker` String NULL SKIPPING INDEX,\n    `workload` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `namespace` String NULL INVERTED INDEX,\n    `node` String NULL INVERTED INDEX,\n    `currency` String NULL INVERTED INDEX,\n    `qos` String NULL,\n    `tflops_request` Double NULL,\n    `tflops_limit` Double NULL,\n    `vram_bytes_request` Double NULL,\n    `vram_bytes_limit` Double NULL,\n    `gpu_count` BigInt NULL,\n    `raw_cost` Double NULL,\n    `interval_seconds` Double NULL,\n    `ready` Boolean NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`worker`, `workload`, `pool`, `namespace`, `node`, `currency`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_node_metrics (\n    `node` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `phase` String NULL INVERTED INDEX,\n    `capacity_type` String NULL INVERTED INDEX,\n    `instance_type` String NULL INVERTED INDEX,\n    `price_source` String NULL INVERTED INDEX,\n    `currency` String NULL INVERTED INDEX,\n    `allocated_tflops` Double NULL,\n    `allocated_tflops_percent` Double NULL,\n    `allocated_vram_bytes` Double NULL,\n    `allocated_vram_percent` Double NULL,\n    `allocated_tflops_percent_virtual` Double NULL,\n    `allocated_vram_percent_virtual` Double NULL,\n    `limited_tflops` Double NULL,\n    `limited_vram_bytes` Double NULL,\n    `limited_tflops_percent_virtual` Double NULL,\n    `limited_vram_percent_virtual` Double NULL,\n    `raw_cost` Double NULL,\n    `hourly_price` Double NULL,\n    `interval_seconds` Double NULL,\n    `gpu_count` BigInt NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`node`, `pool`, `phase`, `capacity_type`, `instance_type`, `price_source`, `currency`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_system_metrics (\n    `pool` String NULL INVERTED INDEX,\n    `total_workers_cnt` BigInt NULL,\n    `total_nodes_cnt` BigInt NULL,\n    `total_allocation_fail_cnt` BigInt NULL,\n    `total_allocation_success_cnt` BigInt NULL,\n    `total_scale_up_cnt` BigInt NULL,\n    `total_scale_down_cnt` BigInt NULL,\n    `total_topo_satisfied_cnt` BigInt NULL,\n    `total_topo_unsatisfied_cnt` BigInt NULL,\n    `total_topo_fallback_cnt` BigInt NULL,\n    `total_topo_search_degraded_cnt` BigInt NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`pool`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_system_log (\n    `component` String NULL INVERTED INDEX,\n    `container` String NULL INVERTED INDEX,\n    `message` String NULL FULLTEXT INDEX WITH (analyzer = 'English' , case_sensitive = 'false'),\n    `namespace` String NULL INVERTED INDEX,\n    `pod` String NULL SKIPPING INDEX,\n    `stream` String NULL,\n    `timestamp` String NULL,\n    `greptime_timestamp` Timestamp_ms TIME INDEX,\n    PRIMARY KEY (`component`, `container`, `namespace`, `pod`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_worker_usage (\n    `workload` String NULL INVERTED INDEX,\n    `worker` String NULL SKIPPING INDEX,\n    `namespace` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `node` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_tflops` Double NULL,\n    `compute_percentage` Double NULL,\n    `memory_bytes` BigInt UNSIGNED NULL,\n    `memory_percentage` Double NULL,\n    `compute_throttled_cnt` Double NULL,\n    `vram_freezed_cnt` Double NULL,\n    `vram_resumed_cnt` Double NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`workload`, `worker`, `namespace`, `pool`, `node`, `uuid`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_gpu_usage (\n    `node` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `uuid` String NULL INVERTED INDEX,\n    `compute_percentage` Double NULL,\n    `memory_percentage` Double NULL,\n    `memory_bytes` BigInt UNSIGNED NULL,\n    `compute_tflops` Double NULL,\n    `rx` Double NULL,\n    `tx` Double NULL,\n    `temperature` Double NULL,\n    `graphics_clock_mhz` Double NULL,\n    `sm_clock_mhz` Double NULL,\n    `memory_clock_mhz` Double NULL,\n    `video_clock_mhz` Double NULL,\n    `power_usage` BigInt NULL,\n    `nvlink_rx` BigInt NULL,\n    `nvlink_tx` BigInt NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`node`, `pool`, `uuid`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",

		"CREATE TABLE IF NOT EXISTS tf_gpu_allocation (\n    `pod` String NULL SKIPPING INDEX,\n    `namespace` String NULL INVERTED INDEX,\n    `pod_uuid` String NULL INVERTED INDEX,\n    `node` String NULL,\n    `gpu_model_index` String NULL,\n    `gpu_name` String NULL INVERTED INDEX,\n    `gpu_count` BigInt NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`pod`, `namespace`, `pod_uuid`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",
	}},

	{"1.1", []string{
		"CREATE TABLE IF NOT EXISTS tf_gpu_metrics (\n    `gpu` String NULL INVERTED INDEX,\n    `node` String NULL INVERTED INDEX,\n    `pool` String NULL INVERTED INDEX,\n    `gpu_model` String NULL INVERTED INDEX,\n    `phase` String NULL INVERTED INDEX,\n    `capacity_tflops` Double NULL,\n    `available_tflops` Double NULL,\n    `allocated_tflops` Double NULL,\n    `allocated_tflops_percent` Double NULL,\n    `capacity_vram_bytes` Double NULL,\n    `available_vram_bytes` Double NULL,\n    `allocated_vram_bytes` Double NULL,\n    `allocated_vram_percent` Double NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`gpu`, `node`, `pool`, `gpu_model`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",
	}},

	{"1.2", []string{
		"CREATE TABLE IF NOT EXISTS tf_pool_metrics (\n    `pool` String NULL INVERTED INDEX,\n    `phase` String NULL INVERTED INDEX,\n    `allocated_tflops` Double NULL,\n    `allocated_tflops_percent` Double NULL,\n    `allocated_tflops_percent_virtual` Double NULL,\n    `allocated_vram_bytes` Double NULL,\n    `allocated_vram_percent` Double NULL,\n    `allocated_vram_percent_virtual` Double NULL,\n    `limited_tflops` Double NULL,\n    `limited_vram_bytes` Double NULL,\n    `limited_tflops_percent_virtual` Double NULL,\n    `limited_vram_percent_virtual` Double NULL,\n    `gpu_count` BigInt NULL,\n    `greptime_timestamp` Timestamp_ns TIME INDEX,\n    PRIMARY KEY (`pool`, `phase`))\n    ENGINE=mito WITH( ttl='30d', merge_mode = 'last_non_null')",
	}},
}

const CurrentAppSQLVersion = "1.2"
