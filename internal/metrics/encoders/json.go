package encoders

import (
	"encoding/json"
	"time"
)

// JsonEncoderMetricsPoint represents a JSON metrics point
type JsonEncoderMetricsPoint struct {
	Measure string            `json:"measure"`
	Tag     map[string]string `json:"tag"`
	Field   map[string]any    `json:"field"`
	Ts      int64             `json:"ts"`
}

// JsonStrategy implements JSON encoding
type JsonStrategy struct {
	point    JsonEncoderMetricsPoint
	tmpBytes []byte
	lastErr  error
}

// NewJsonStrategy creates a new JSON strategy
func NewJsonStrategy() *JsonStrategy {
	return &JsonStrategy{}
}

func (s *JsonStrategy) StartLine(measurement string) {
	s.point = JsonEncoderMetricsPoint{
		Measure: measurement,
		Tag:     make(map[string]string, 4),
		Field:   make(map[string]any, 4),
	}
}

func (s *JsonStrategy) AddTag(key, value string) {
	s.point.Tag[key] = value
}

func (s *JsonStrategy) AddField(key string, value any) {
	s.point.Field[key] = value
}

func (s *JsonStrategy) EndLine(timestamp time.Time) {
	s.point.Ts = timestamp.UnixMilli()
	tmpBytes, err := json.Marshal(s.point)
	if err != nil {
		s.lastErr = err
		return
	}
	s.tmpBytes = append(s.tmpBytes, tmpBytes...)
	s.tmpBytes = append(s.tmpBytes, '\n')
}

func (s *JsonStrategy) Bytes() []byte {
	tmpBytes := s.tmpBytes
	s.tmpBytes = []byte{}
	return tmpBytes
}

func (s *JsonStrategy) Err() error {
	return s.lastErr
}
