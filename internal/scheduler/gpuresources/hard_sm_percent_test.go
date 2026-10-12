package gpuresources

import (
	"encoding/json"
	"strconv"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	hvapi "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestEffectiveHardSMPercentUsesGPUCRCapacity(t *testing.T) {
	t.Parallel()

	percent, err := effectiveHardSMPercent(resource.MustParse("10"), []*tfv1.GPU{
		{
			Status: tfv1.GPUStatus{
				Capacity: &tfv1.Resource{Tflops: resource.MustParse("71")},
			},
		},
	})
	if err != nil {
		t.Fatalf("calculate hard SM percent: %v", err)
	}
	if percent != 15 {
		t.Fatalf("hard SM percent = %d, want 15", percent)
	}
}

// schedulerTestDevices serves devices to the hypervisor's allocation controller.
type schedulerTestDevices struct {
	framework.DeviceController
	devices map[string]*hvapi.DeviceInfo
}

func (d *schedulerTestDevices) GetDevice(uuid string) (*hvapi.DeviceInfo, bool) {
	device, ok := d.devices[uuid]
	return device, ok
}

func (d *schedulerTestDevices) GetVendorMountLibs() ([]*hvapi.Mount, error) { return nil, nil }

// The scheduler converts a TFLOPS limit with the capacity it reads from the GPU
// resource; the hypervisor converts it with the capacity on its device, which
// is what it published to that resource. For the same limit on the same GPU
// both must arrive at the same percentage, whatever the model and wherever its
// capacity came from.
func TestSchedulerAndHypervisorDeriveTheSamePercent(t *testing.T) {
	models := []struct {
		name string
		// estimate is what the provider library reports, fp16Tflops the hardware
		// table's figure (zero: not listed).
		estimate, fp16Tflops float64
	}{
		{"RTX 3090 listed", 44.08, 71.2},
		{"RTX 3090 not listed", 44.08, 0},
		{"RTX 4090 listed", 82.58, 165.16},
		{"A100 80GB listed", 19.49184, 312},
		{"A100 80GB not listed", 19.49184, 0},
		{"H100 80GB listed", 66.90816, 989},
		{"L4 listed", 30.28992, 121},
		{"T4 listed", 8.1408, 65},
		{"T4 not listed", 8.1408, 0},
		{"estimate with more digits than a resource keeps", 19.4918399999, 0},
		{"table figure with more digits than a resource keeps", 44.08, 71.2345678},
	}
	for _, model := range models {
		t.Run(model.name, func(t *testing.T) {
			// Hypervisor: the device controller resolves the capacity ...
			device := &hvapi.DeviceInfo{UUID: "gpu-0", Vendor: "NVIDIA", ProviderEstimatedTflops: model.estimate}
			device.ResolveTflopsCapacity(model.fp16Tflops, model.fp16Tflops > 0)
			// ... and the backend publishes it on the GPU resource, which the
			// scheduler reads through the API.
			published := &tfv1.GPU{Status: tfv1.GPUStatus{
				Capacity: &tfv1.Resource{Tflops: hvapi.TflopsQuantity(device.TflopsCapacity)},
			}}
			raw, err := json.Marshal(published)
			if err != nil {
				t.Fatal(err)
			}
			gpu := &tfv1.GPU{}
			if err := json.Unmarshal(raw, gpu); err != nil {
				t.Fatal(err)
			}

			capacity := device.TflopsCapacity
			limits := []string{"0.001", "0.5", "1", "7", "10", "20", "33.3", "40", "64.99", "65", "100", "1000"}
			// Limits at and around every whole percent of the capacity, where a
			// difference in rounding would show.
			for percent := 1; percent <= 100; percent++ {
				exact := capacity * float64(percent) / 100
				for _, tflops := range []float64{exact, exact * (1 - 1e-6), exact * (1 + 1e-6), exact - 0.001, exact + 0.001} {
					// A resource quantity keeps six decimals here; skip what would
					// round to no limit at all.
					if tflops >= 1e-5 {
						limits = append(limits, strconv.FormatFloat(tflops, 'f', 6, 64))
					}
				}
			}
			for i, limit := range limits {
				quantity := resource.MustParse(limit)
				fromScheduler, err := effectiveHardSMPercent(quantity, []*tfv1.GPU{gpu})
				if err != nil {
					t.Fatalf("scheduler: %s TFLOPS: %v", limit, err)
				}

				allocations := worker.NewAllocationController(&schedulerTestDevices{
					devices: map[string]*hvapi.DeviceInfo{device.UUID: device},
				})
				allocation, err := allocations.AllocateWorkerDevices(&hvapi.WorkerInfo{
					WorkerUID: "worker-" + strconv.Itoa(i), AllocatedDevices: []string{device.UUID},
					IsolationMode: tfv1.IsolationModeHard, Limits: tfv1.Resource{Tflops: quantity},
				})
				if err != nil {
					t.Fatalf("hypervisor: %s TFLOPS: %v", limit, err)
				}
				fromHypervisor := allocation.Envs[constants.HardSMLimiterEnv]
				if want := strconv.FormatUint(uint64(fromScheduler), 10); fromHypervisor != want {
					t.Fatalf("%s TFLOPS of %v: scheduler derives %s%%, hypervisor %s%%",
						limit, capacity, want, fromHypervisor)
				}
			}
		})
	}
}
