package metrics

import (
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics/encoders"
)

// EncoderType represents the encoder type as an enum for better performance
type EncoderType uint8

const (
	EncoderTypeInflux EncoderType = iota
	EncoderTypeJson
	EncoderTypeOTel
)

// stringToEncoderType maps string format to enum type
var stringToEncoderType = map[string]EncoderType{
	config.MetricsFormatInflux: EncoderTypeInflux,
	config.MetricsFormatJson:   EncoderTypeJson,
	config.MetricsFormatOTel:   EncoderTypeOTel,
}

type Encoder interface {
	StartLine(measurement string)
	AddTag(key, value string)
	AddField(key string, value any)
	EndLine(timestamp time.Time)
	Bytes() []byte
	Err() error
}

type MultiProtocolEncoder struct {
	strategy encoders.Strategy
}

func NewEncoder(encoderType string) Encoder {
	encoderEnum, exists := stringToEncoderType[encoderType]
	if !exists {
		// Default to influx for unknown types
		encoderEnum = EncoderTypeInflux
	}

	var strategy encoders.Strategy
	switch encoderEnum {
	case EncoderTypeInflux:
		strategy = encoders.NewInfluxStrategy()
	case EncoderTypeOTel:
		strategy = encoders.NewOtelStrategy()
	default: // EncoderTypeJson
		strategy = encoders.NewJsonStrategy()
	}

	return &MultiProtocolEncoder{strategy: strategy}
}

func (m *MultiProtocolEncoder) StartLine(measurement string) {
	m.strategy.StartLine(measurement)
}

func (m *MultiProtocolEncoder) AddTag(key, value string) {
	m.strategy.AddTag(key, value)
}

func (m *MultiProtocolEncoder) AddField(key string, value any) {
	m.strategy.AddField(key, value)
}

func (m *MultiProtocolEncoder) EndLine(timestamp time.Time) {
	m.strategy.EndLine(timestamp)
}

func (m *MultiProtocolEncoder) Bytes() []byte {
	return m.strategy.Bytes()
}

func (m *MultiProtocolEncoder) Err() error {
	return m.strategy.Err()
}
