package utils

import (
	"reflect"
)

// MergeStructFields merges non-empty fields from source into destination.
// It copies only non-zero/non-empty values from src to dst.
// Special handling:
//   - bool fields: copies if src is true
//   - string fields: copies if src is non-empty
//   - numeric fields: copies if src is non-zero
//   - pointer fields: copies if src is non-nil
//
// Both dst and src must be pointers to structs of the same type.
func MergeStructFields(dst, src any) {
	dstVal := reflect.ValueOf(dst)
	srcVal := reflect.ValueOf(src)

	// Ensure both are pointers
	if dstVal.Kind() != reflect.Ptr || srcVal.Kind() != reflect.Ptr {
		return
	}

	dstElem := dstVal.Elem()
	srcElem := srcVal.Elem()

	// Ensure both are structs
	if dstElem.Kind() != reflect.Struct || srcElem.Kind() != reflect.Struct {
		return
	}

	// Ensure same type
	if dstElem.Type() != srcElem.Type() {
		return
	}

	mergeStructFields(dstElem, srcElem)
}

// mergeStructFields is the internal implementation that does the actual merging
func mergeStructFields(dst, src reflect.Value) {
	for i := 0; i < src.NumField(); i++ {
		srcField := src.Field(i)
		dstField := dst.Field(i)

		if !srcField.IsValid() || !dstField.CanSet() {
			continue
		}

		// Skip unexported fields
		if !srcField.CanInterface() {
			continue
		}

		switch srcField.Kind() {
		case reflect.Bool:
			// For bool, copy if src is true
			if srcField.Bool() {
				dstField.SetBool(true)
			}

		case reflect.String:
			// For string, copy if src is non-empty
			if srcField.String() != "" {
				dstField.SetString(srcField.String())
			}

		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			// For integers, copy if src is non-zero
			if srcField.Int() != 0 {
				dstField.SetInt(srcField.Int())
			}

		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			// For unsigned integers, copy if src is non-zero
			if srcField.Uint() != 0 {
				dstField.SetUint(srcField.Uint())
			}

		case reflect.Float32, reflect.Float64:
			// For floats, copy if src is non-zero
			if srcField.Float() != 0 {
				dstField.SetFloat(srcField.Float())
			}

		case reflect.Ptr, reflect.Interface, reflect.Slice, reflect.Map:
			// For pointers, interfaces, slices, maps - copy if src is non-nil
			if !srcField.IsNil() {
				dstField.Set(srcField)
			}

		case reflect.Struct:
			// For nested structs, recursively merge
			mergeStructFields(dstField, srcField)
		}
	}
}
