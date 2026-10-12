package utils

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func workerHashOfTemplate(template string) string {
	return WorkerTemplateHash(&tfv1.WorkerConfig{
		Image:       "registry.example/worker:1.2.3",
		PodTemplate: &runtime.RawExtension{Raw: []byte(template)},
	}, nil)
}

// The worker hash leaves out what this release adds to the worker container
// for the per-pod data path. It must not leave out the same variable names or
// mount expression where the user wrote them: a changed template has to
// replace the worker pods.
func TestWorkerTemplateHashKeepsReservedNamesOfOtherContainers(t *testing.T) {
	useControllerHashInputs(t, "32")
	const worker = `{"name":"tensorfusion-worker","image":"registry.example/worker:1.2.3"}`
	templates := map[string]string{
		"no helper settings": `{"template":{"spec":{"containers":[` + worker + `,{"name":"helper"}]}}}`,
		"helper sets the pod name variable": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","env":[{"name":"TF_SHM_POD_NAME","value":"a"}]}]}}}`,
		"helper sets it to another value": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","env":[{"name":"TF_SHM_POD_NAME","value":"b"}]}]}}}`,
		"helper takes it from the downward API": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","env":[{"name":"TF_SHM_POD_NAME","valueFrom":{"fieldRef":{"fieldPath":"metadata.name"}}}]}]}}}`,
		"helper sets the namespace variable": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","env":[{"name":"TF_SHM_POD_NAMESPACE","value":"a"}]}]}}}`,
		"helper mounts with the per-pod expression": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","volumeMounts":[{"name":"scratch","mountPath":"/scratch",` +
			`"subPathExpr":"shm/$(TF_SHM_POD_NAMESPACE)/$(TF_SHM_POD_NAME)"}]}]}}}`,
		"helper mounts with the former expression": `{"template":{"spec":{"containers":[` + worker +
			`,{"name":"helper","volumeMounts":[{"name":"scratch","mountPath":"/scratch",` +
			`"subPathExpr":"shm/$(POD_NAMESPACE)/$(POD_NAME)"}]}]}}}`,
		"init container sets the pod name variable": `{"template":{"spec":{"initContainers":[{"name":"prepare",` +
			`"env":[{"name":"TF_SHM_POD_NAME","value":"a"}]}],"containers":[` + worker + `,{"name":"helper"}]}}}`,
		"worker container mounts another volume with the per-pod expression": `{"template":{"spec":{"containers":[` +
			`{"name":"tensorfusion-worker","image":"registry.example/worker:1.2.3","volumeMounts":[{"name":"scratch",` +
			`"mountPath":"/scratch","subPathExpr":"shm/$(TF_SHM_POD_NAMESPACE)/$(TF_SHM_POD_NAME)"}]},{"name":"helper"}]}}}`,
		"worker container mounts it with the former expression": `{"template":{"spec":{"containers":[` +
			`{"name":"tensorfusion-worker","image":"registry.example/worker:1.2.3","volumeMounts":[{"name":"scratch",` +
			`"mountPath":"/scratch","subPathExpr":"shm/$(POD_NAMESPACE)/$(POD_NAME)"}]},{"name":"helper"}]}}}`,
	}
	seen := map[string]string{}
	for name, template := range templates {
		hash := workerHashOfTemplate(template)
		if other, duplicate := seen[hash]; duplicate {
			t.Errorf("templates %q and %q have the same worker hash %s: changing one into the other "+
				"would not replace the worker pods", other, name, hash)
		}
		seen[hash] = name
	}
}

// A definition of the reserved variables on the worker container itself never
// reaches the pod (SetWorkerContainerSpec replaces it), so it does not change
// the hash either: the pods would be replaced by identical ones.
func TestWorkerTemplateHashIgnoresWhatTheWorkerContainerNeverGets(t *testing.T) {
	useControllerHashInputs(t, "32")
	plain := workerHashOfTemplate(`{"template":{"spec":{"containers":[{"name":"tensorfusion-worker"}]}}}`)
	overridden := workerHashOfTemplate(`{"template":{"spec":{"containers":[{"name":"tensorfusion-worker",` +
		`"env":[{"name":"TF_SHM_POD_NAME","value":"other-pod"}]}]}}}`)
	if plain != overridden {
		t.Errorf("worker hash is %s with a discarded variable and %s without", overridden, plain)
	}
}

func hypervisorSpecOf(pool *tfv1.GPUPool) v1.PodSpec {
	spec := v1.PodSpec{}
	AddTFHypervisorConfAfterTemplate(context.Background(), &spec, pool, "", false)
	return spec
}

// The hypervisor pod must start without the Vector configuration: only the
// sidecar reads it. A ConfigMap volume that is not optional keeps every
// container of the pod from being created while the ConfigMap is missing.
func TestHypervisorPodStartsWithoutTheVectorConfigMap(t *testing.T) {
	useControllerHashInputs(t, "32")
	for name, pool := range map[string]*tfv1.GPUPool{
		"without the sidecar": goldenHashPool(),
		"with the sidecar":    goldenVectorHashPool(goldenHypervisorTemplate),
	} {
		t.Run(name, func(t *testing.T) {
			spec := hypervisorSpecOf(pool)
			found := false
			for _, volume := range spec.Volumes {
				if volume.Name != constants.TensorFusionVectorConfigVolumeName {
					continue
				}
				found = true
				if volume.ConfigMap == nil || volume.ConfigMap.Name != constants.TensorFusionVectorConfigName {
					t.Fatalf("unexpected Vector configuration volume: %+v", volume)
				}
				if volume.ConfigMap.Optional == nil || !*volume.ConfigMap.Optional {
					t.Errorf("the Vector configuration volume is not optional: " +
						"the hypervisor would not start without the ConfigMap")
				}
			}
			if !found {
				t.Fatal("no Vector configuration volume")
			}
			if spec.Containers[0].Name != constants.TFContainerNameHypervisor {
				t.Fatalf("first container is %q", spec.Containers[0].Name)
			}
			for _, mount := range spec.Containers[0].VolumeMounts {
				if mount.Name == constants.TensorFusionVectorConfigVolumeName {
					t.Errorf("the hypervisor container mounts the Vector configuration: %+v", mount)
				}
			}
		})
	}
}

// Marking the added volume optional does not change the hypervisor hash
// (TestTemplateHashesDoNotChangeAcrossReleases pins that). A volume of the
// same name in the template is the user's configuration and does.
func TestHypervisorTemplateHashKeepsAVectorConfigVolumeOfTheTemplate(t *testing.T) {
	useControllerHashInputs(t, "32")
	const containers = `"containers":[{"name":"tensorfusion-hypervisor","image":"registry.example/hypervisor:1.2.3"}]`
	templates := map[string]string{
		"no volume": `{"template":{"spec":{` + containers + `}}}`,
		"optional volume": `{"template":{"spec":{` + containers + `,"volumes":[{"name":"vector-config",` +
			`"configMap":{"name":"tensor-fusion-sys-vector-config","optional":true}}]}}}`,
		"volume from another ConfigMap": `{"template":{"spec":{` + containers + `,"volumes":[{"name":"vector-config",` +
			`"configMap":{"name":"my-vector-config"}}]}}}`,
		"optional volume from another ConfigMap": `{"template":{"spec":{` + containers + `,"volumes":[{"name":"vector-config",` +
			`"configMap":{"name":"my-vector-config","optional":true}}]}}}`,
	}
	seenPod, seenPool := map[string]string{}, map[string]string{}
	for name, template := range templates {
		pod, poolWide := hypervisorHashes(goldenVectorHashPool(template))
		for hash, seen := range map[string]map[string]string{pod: seenPod, poolWide: seenPool} {
			if other, duplicate := seen[hash]; duplicate {
				t.Errorf("templates %q and %q have the same hypervisor hash %s", other, name, hash)
			}
			seen[hash] = name
		}
	}
}
