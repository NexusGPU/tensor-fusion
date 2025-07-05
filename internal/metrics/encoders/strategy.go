package encoders

import "time"

// Strategy defines the interface for different encoding strategies
type Strategy interface {
	StartLine(measurement string)
	AddTag(key, value string)
	AddField(key string, value any)
	EndLine(timestamp time.Time)
	Bytes() []byte
	Err() error
}
