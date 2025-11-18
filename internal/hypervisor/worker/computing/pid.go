package worker

import "time"

// PID control algorithm for resource allocation
type PIDController struct {
	Kp         float64
	Ki         float64
	Kd         float64
	integral   float64
	derivative float64
	lastError  float64
	lastTime   time.Time
	sampleTime time.Duration
}

func NewPIDController(Kp, Ki, Kd float64) *PIDController {
	return &PIDController{
		Kp:         Kp,
		Ki:         Ki,
		Kd:         Kd,
		integral:   0,
		derivative: 0,
		lastError:  0,
		lastTime:   time.Now(),
		sampleTime: 1 * time.Second,
	}
}
