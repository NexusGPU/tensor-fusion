package encoders

import (
	"time"

	metricsProto "github.com/influxdata/line-protocol/v2/lineprotocol"
)

// InfluxStrategy implements InfluxDB line protocol encoding
type InfluxStrategy struct {
	enc *metricsProto.Encoder
}

// NewInfluxStrategy creates a new InfluxDB strategy
func NewInfluxStrategy() *InfluxStrategy {
	enc := metricsProto.Encoder{}
	enc.SetPrecision(metricsProto.Millisecond)
	enc.SetLax(true)
	return &InfluxStrategy{enc: &enc}
}

func (s *InfluxStrategy) StartLine(measurement string) {
	s.enc.StartLine(measurement)
}

func (s *InfluxStrategy) AddTag(key, value string) {
	s.enc.AddTag(key, value)
}

func (s *InfluxStrategy) AddField(key string, value any) {
	s.enc.AddField(key, metricsProto.MustNewValue(value))
}

func (s *InfluxStrategy) EndLine(timestamp time.Time) {
	s.enc.EndLine(timestamp)
}

func (s *InfluxStrategy) Bytes() []byte {
	return s.enc.Bytes()
}

func (s *InfluxStrategy) Err() error {
	return s.enc.Err()
}
