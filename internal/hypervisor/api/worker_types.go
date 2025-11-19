package api

type Worker struct {
	WorkerUID string
	AllocatedDevices []string
	Status string
	IsolationMode IsolationMode
}