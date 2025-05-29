package alert

// connect TSDB, eval every minute of all rules to generate alerts in mem, and check existing alert resolved or not
// send alerts API to alertmanager, let alertmanager to do deduplication, notification stuff
