package utils

import (
	context "context"
	"encoding/json"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
)

// HypervisorTemplateHash computes the pool-wide hypervisor rollout hash. It
// includes every ProviderConfig revision so a provider change starts the
// existing GPUPool batched update state machine.
func HypervisorTemplateHash(pool *tfv1.GPUPool) string {
	return hypervisorTemplateHash(pool, true)
}

// HypervisorTemplateHashWithPolicy computes the pool component hash while
// honoring the isolation policy. Dynamic policy deliberately excludes static
// node isolation rules: those rules do not select the Hypervisor process mode
// and changing them must not trigger a rollout.
func HypervisorTemplateHashWithPolicy(pool *tfv1.GPUPool, policy tfv1.IsolationModePolicyType) string {
	return hypervisorTemplateHash(pool, policy != tfv1.IsolationModePolicyDynamic)
}

func hypervisorTemplateHash(pool *tfv1.GPUPool, includeNodeIsolation bool) string {
	spec := hypervisorTemplateSpec(pool)
	revisions := ProviderConfigRevisions(pool)
	var nodeIsolationPolicy string
	if includeNodeIsolation {
		nodeIsolationPolicy = NodeIsolationPolicyHash(pool.Spec.NodeManagerConfig)
	}
	return GetObjectHash(struct {
		Spec                v1.PodSpec
		ProviderRevisions   map[string]string
		NodeIsolationPolicy string
	}{
		Spec:                spec,
		ProviderRevisions:   revisions,
		NodeIsolationPolicy: nodeIsolationPolicy,
	})
}

// HypervisorPodTemplateHash computes the desired hash for one node. Only the
// revision of that node's vendor is included, so a ProviderConfig update in a
// multi-vendor pool does not recreate unrelated hypervisors. The resolved
// isolation mode is node-specific, so policy changes only restart nodes whose
// effective mode changed.
func HypervisorPodTemplateHash(pool *tfv1.GPUPool, vendor string, isolationMode tfv1.IsolationModeType) string {
	spec := hypervisorTemplateSpec(pool)
	revision := ProviderConfigRevision(pool, vendor)
	return GetObjectHash(struct {
		Spec          v1.PodSpec
		Revision      string
		IsolationMode tfv1.IsolationModeType
	}{Spec: spec, Revision: revision, IsolationMode: isolationMode})
}

func hypervisorTemplateSpec(pool *tfv1.GPUPool) v1.PodSpec {
	podTmpl := &v1.PodTemplate{}
	if pool.Spec.ComponentConfig.Hypervisor != nil && pool.Spec.ComponentConfig.Hypervisor.PodTemplate != nil {
		json.Unmarshal(pool.Spec.ComponentConfig.Hypervisor.PodTemplate.Raw, podTmpl) //nolint:errcheck
	}
	spec := podTmpl.Template.Spec
	templateVolumes := spec.Volumes
	AddTFHypervisorConfAfterTemplate(context.Background(), &spec, pool, "", false)
	return hypervisorTemplateHashSpec(spec, templateVolumes)
}

// hypervisorTemplateHashSpec returns the pod spec a hypervisor template hash
// is computed from, for the same reason as templateHashSpec: that this release
// marks the Vector configuration volume it adds as optional is not something
// the user configured, and left in the hash it would restart every hypervisor
// on upgrade. A volume of that name which the template defines itself is the
// user's and is hashed as it is. (One such volume hashes like the added one:
// the same ConfigMap, not optional. Declaring exactly that in the template
// does not roll the hypervisors.)
func hypervisorTemplateHashSpec(spec v1.PodSpec, templateVolumes []v1.Volume) v1.PodSpec {
	if lo.ContainsBy(templateVolumes, func(volume v1.Volume) bool {
		return volume.Name == constants.TensorFusionVectorConfigVolumeName
	}) {
		return spec
	}
	hashed := *spec.DeepCopy()
	for i := range hashed.Volumes {
		if hashed.Volumes[i].Name == constants.TensorFusionVectorConfigVolumeName && hashed.Volumes[i].ConfigMap != nil {
			hashed.Volumes[i].ConfigMap.Optional = nil
		}
	}
	return hashed
}

// WorkerTemplateHash computes hash of the base worker pod template
// including code-level defaults from SetWorkerContainerSpec.
// Only the hypervisor port number is included — other hypervisor config
// changes should not cascade to worker pod recreation.
func WorkerTemplateHash(workerConfig *tfv1.WorkerConfig, hypervisorConfig *tfv1.HypervisorConfig) string {
	podTmpl := &v1.PodTemplate{}
	if workerConfig != nil && workerConfig.PodTemplate != nil {
		json.Unmarshal(workerConfig.PodTemplate.Raw, podTmpl) //nolint:errcheck
	}
	spec := podTmpl.Template.Spec
	if len(spec.Containers) == 0 {
		spec.Containers = []v1.Container{{}}
	}
	// Use a minimal HypervisorConfig containing only the port number,
	// so that unrelated hypervisor changes don't affect the worker hash.
	portOnly := &tfv1.HypervisorConfig{}
	if hypervisorConfig != nil && hypervisorConfig.PortNumber != nil {
		portOnly.PortNumber = hypervisorConfig.PortNumber
	}
	SetWorkerContainerSpec(&spec.Containers[0], &tfv1.WorkloadProfileSpec{}, workerConfig, portOnly, "", false)
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	return GetObjectHash(templateHashSpec(spec))
}

// legacyDataPathWorkerExpr is the value constants.TFDataPathWorkerExpr had
// before it was expressed in variables reserved for TensorFusion.
const legacyDataPathWorkerExpr = "shm/$(POD_NAMESPACE)/$(POD_NAME)"

// templateHashSpec returns the pod spec a template hash is computed from.
//
// A template hash says whether the template a user configured has changed;
// pods whose hash differs are replaced. How this release mounts the per-pod
// data directory (the reserved variables of setWorkerDataPathEnv and the
// expression that uses them) is not part of what the user configured: left in
// the hash, upgrading the operator would replace every running worker pod.
// Those parts are therefore hashed the way they were composed before they
// existed. Pods created earlier keep the former mount, which resolves to the
// same directory, and take the new form whenever they are next created.
// TestTemplateHashesDoNotChangeAcrossReleases pins the result.
//
// Only what SetWorkerContainerSpec itself adds is hashed that way, in the one
// container it composes (the first). The same variable names or expression
// anywhere else in the template are the user's configuration and stay in the
// hash as they are.
func templateHashSpec(spec v1.PodSpec) v1.PodSpec {
	hashed := *spec.DeepCopy()
	if len(hashed.Containers) == 0 {
		return hashed
	}
	worker := &hashed.Containers[0]
	for i := range worker.VolumeMounts {
		if isWorkerDataMount(worker.VolumeMounts[i]) {
			worker.VolumeMounts[i].SubPathExpr = legacyDataPathWorkerExpr
		}
	}
	if lo.ContainsBy(worker.Env, isComposedWorkerDataPathEnv) {
		worker.Env = lo.Reject(worker.Env, func(env v1.EnvVar, _ int) bool { return isComposedWorkerDataPathEnv(env) })
	}
	return hashed
}

// isComposedWorkerDataPathEnv reports whether env is one of the two downward
// API variables setWorkerDataPathEnv adds.
func isComposedWorkerDataPathEnv(env v1.EnvVar) bool {
	if env.Value != "" || env.ValueFrom == nil || env.ValueFrom.FieldRef == nil {
		return false
	}
	switch env.Name {
	case constants.TFShmPodNamespaceEnv:
		return env.ValueFrom.FieldRef.FieldPath == constants.NamespaceFieldRef
	case constants.TFShmPodNameEnv:
		return env.ValueFrom.FieldRef.FieldPath == constants.ResourceNameFieldRef
	}
	return false
}

// ClientTemplateHash computes hash of the base client injection template
// including code-level defaults from AddTFDefaultClientConfBeforePatch,
// plus webhook-stage fields (OperatorEndpoint, PatchToContainer, etc.)
// that also affect the final injected pod spec.
func ClientTemplateHash(pool *tfv1.GPUPool) string {
	pod := &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "baseline"}}}}
	tfInfo := TensorFusionInfo{Profile: &tfv1.WorkloadProfileSpec{}}
	AddTFDefaultClientConfBeforePatch(context.Background(), pod, pool, tfInfo, []int{0})
	clientConfig := pool.Spec.ComponentConfig.Client
	return GetObjectHash(pod.Spec, clientConfig.OperatorEndpoint, clientConfig.PatchToPod, clientConfig.PatchToContainer)
}
