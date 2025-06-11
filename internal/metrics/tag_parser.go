package metrics

import (
	"fmt"
	"reflect"
	"strings"
	"time"

	"gorm.io/gorm/schema"
)

const (
	CREATE_TABLE_OPTION_TPL = "ENGINE=mito WITH( ttl='%s', merge_mode = 'last_non_null')"
)

func getInitTableSQL(model schema.Tabler, ttl string) string {
	// Use reflection to get the struct fields and their gorm tags
	t := reflect.TypeOf(model)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	var fields []string
	var partitionKeys []string
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get the gorm tag
		gormTag := field.Tag.Get("gorm")
		if gormTag == "" {
			continue
		}

		// Parse the gorm tag
		var columnName string
		var indexClass string
		var isIndex bool
		var extraOption string

		// Split by semicolon first
		parts := strings.Split(gormTag, ";")
		for _, part := range parts {
			if part == "" {
				continue
			}

			// Split by colon
			keyValue := strings.Split(part, ",")

			for _, key := range keyValue {
				if strings.HasPrefix(key, "column:") {
					columnName = strings.TrimPrefix(key, "column:")
				} else if strings.HasPrefix(key, "index:") {
					isIndex = true
				} else if strings.HasPrefix(key, "class:") {
					indexClass = strings.TrimPrefix(key, "class:")
				} else if strings.HasPrefix(key, "option:") {
					extraOption = strings.TrimPrefix(key, "option:")
				}
			}
		}

		// If no column name specified, use field name
		if columnName == "" {
			columnName = field.Name
		}

		// Map Go types to GreptimeDB types
		var dbType string
		isNullable := true
		switch field.Type.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			dbType = "BigInt"
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			dbType = "BigInt UNSIGNED"
		case reflect.Float32, reflect.Float64:
			dbType = "Double"
		case reflect.String:
			dbType = "String"
		default:
			// Check if it's time.Time
			if field.Type == reflect.TypeOf(time.Time{}) {
				dbType = "Timestamp_ms"
				isNullable = false
			} else {
				// Default to String for unknown types
				dbType = "String"
			}
		}

		// Build the field definition
		var fieldDef string
		if isNullable {
			fieldDef = fmt.Sprintf("`%s` %s NULL", columnName, dbType)
		} else {
			fieldDef = fmt.Sprintf("`%s` %s", columnName, dbType)
		}

		// Add index if needed
		if isIndex && indexClass != "" {
			if indexClass != "TIME" && indexClass != "FULLTEXT" {
				partitionKeys = append(partitionKeys, columnName)
			}
			if extraOption != "" {
				extraOption = strings.ReplaceAll(extraOption, "$comma$", ",")
				fieldDef += fmt.Sprintf(" %s INDEX %s", indexClass, extraOption)
			} else {
				fieldDef += fmt.Sprintf(" %s INDEX", indexClass)
			}
		}

		fields = append(fields, fieldDef)
	}

	// Join all fields with commas
	fieldsSQL := strings.Join(fields, ",\n    ")

	return fmt.Sprintf(
		"CREATE TABLE IF NOT EXISTS %s (\n    %s,\n    PRIMARY KEY (`%s`))\n    %s",
		model.TableName(),
		fieldsSQL,
		strings.Join(partitionKeys, "`, `"),
		fmt.Sprintf(CREATE_TABLE_OPTION_TPL, ttl),
	)
}
