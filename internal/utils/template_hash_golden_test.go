package utils

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

// The template hashes decide whether pods that already run are replaced: a
// workload's worker pods are deleted when the worker hash differs from the one
// they were created with, and a pool's client pods and hypervisors are rolled
// when theirs do. A hash may therefore only change when the template the user
// configured changes, never because a release composes pods differently.
//
// The values pinned in this file are what the previous release computes for
// these inputs in a running controller. A different value means that upgrading
// the operator replaces every running pod of that kind. To change one on
// purpose, compute it with the previous release (this file runs there as it
// is, apart from constants that release does not have) and keep both values,
// as TestHypervisorHashWithVectorChangesOnceInThisRelease does.
func goldenHashPool() *tfv1.GPUPool {
	workerTemplate := `{"template":{"spec":{"containers":[{"name":"tensorfusion-worker",` +
		`"image":"registry.example/worker:1.2.3","env":[{"name":"FOO","value":"bar"}]}]}}}`
	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "golden"},
		Spec: tfv1.GPUPoolSpec{ComponentConfig: &tfv1.ComponentConfig{
			Worker: &tfv1.WorkerConfig{
				Image:       "registry.example/worker:1.2.3",
				PodTemplate: &runtime.RawExtension{Raw: []byte(workerTemplate)},
			},
			Hypervisor: &tfv1.HypervisorConfig{
				Image:       "registry.example/hypervisor:1.2.3",
				PortNumber:  ptr.To[int32](8000),
				PodTemplate: &runtime.RawExtension{Raw: []byte(goldenHypervisorTemplate)},
			},
			Client: &tfv1.ClientConfig{
				Image:            "registry.example/client:1.2.3",
				RemoteModeImage:  "registry.example/client-remote:1.2.3",
				OperatorEndpoint: "http://operator.example:8080",
				PatchToPod:       &runtime.RawExtension{Raw: []byte(`{"spec":{"hostPID":false}}`)},
				PatchToContainer: &runtime.RawExtension{Raw: []byte(`{"env":[{"name":"A","value":"b"}]}`)},
			},
		}},
	}
}

const (
	goldenHypervisorTemplate = `{"template":{"spec":{"containers":[{"name":"tensorfusion-hypervisor",` +
		`"image":"registry.example/hypervisor:1.2.3"}]}}}`
	// The form in which the chart declares the sidecar next to the hypervisor.
	goldenHypervisorTemplateWithSidecar = `{"template":{"spec":{"containers":[{"name":"tensorfusion-hypervisor",` +
		`"image":"registry.example/hypervisor:1.2.3"},` +
		`{"name":"vector","resources":{"requests":{"cpu":"50m","memory":"64Mi"}}}]}}}`
)

// goldenVectorHashPool is goldenHashPool with the Vector sidecar enabled.
func goldenVectorHashPool(hypervisorTemplate string) *tfv1.GPUPool {
	pool := goldenHashPool()
	hypervisor := pool.Spec.ComponentConfig.Hypervisor
	hypervisor.EnableVector = true
	hypervisor.VectorImage = "registry.example/vector:1.2.3"
	hypervisor.PodTemplate = &runtime.RawExtension{Raw: []byte(hypervisorTemplate)}
	return pool
}

func setForTest[T any](t *testing.T, target *T, value T) {
	t.Helper()
	previous := *target
	*target = value
	t.Cleanup(func() { *target = previous })
}

// useControllerHashInputs gives everything the template hashes depend on
// besides the pool the value it has in a running controller, whatever the
// environment of the test process is. The pinned hashes are only meaningful in
// that state: a test binary started with GO_TESTING=true, for one, composes
// pods with a zero termination grace period and computes other hashes than any
// controller does.
//
// kubeMinor is the minor version of the cluster, which the controller always
// knows at start-up (cmd/main.go) and which decides the hypervisor's AppArmor
// profile.
func useControllerHashInputs(t *testing.T, kubeMinor string) {
	t.Helper()
	// Read from the operator's environment when a pod is composed.
	t.Setenv(constants.KubeApiVersionMinorEnv, kubeMinor)
	t.Setenv(constants.HypervisorPodResourcesProxyEnabledEnv, "")
	t.Setenv(constants.VectorGreptimePasswordSecretEnv, "")

	// Derived from the environment when the process starts.
	if !IsTestMode && *constants.GracefulPeriodSeconds != 5 {
		t.Fatalf("the termination grace period of composed pods is %ds, was 5s: "+
			"every worker and hypervisor hash changes with it", *constants.GracefulPeriodSeconds)
	}
	setForTest(t, &constants.GracefulPeriodSeconds, ptr.To(int64(5)))
	setForTest(t, &nvidiaOperatorProgressiveMigrationEnv, false)
	setForTest(t, &isLicensedEnv, false)

	// The hashes are computed without a vendor, so no ProviderConfig applies;
	// a manager left behind by another test must not either.
	previous := provider.GetManager()
	provider.SetGlobalManagerForTesting(nil)
	t.Cleanup(func() { provider.SetGlobalManagerForTesting(previous) })
}

type goldenHash struct {
	name string
	got  string
	want string
}

func checkGoldenHashes(t *testing.T, hashes []goldenHash) {
	t.Helper()
	for _, hash := range hashes {
		t.Logf("%s: %s", hash.name, hash.got)
		if hash.got != hash.want {
			t.Errorf("%s hash is %s, was %s: every running pod of this kind would be replaced on upgrade",
				hash.name, hash.got, hash.want)
		}
	}
}

func hypervisorHashes(pool *tfv1.GPUPool) (pod string, poolWide string) {
	return HypervisorPodTemplateHash(pool, constants.AcceleratorVendorNvidia, tfv1.IsolationModeSoft),
		HypervisorTemplateHash(pool)
}

func TestTemplateHashesDoNotChangeAcrossReleases(t *testing.T) {
	for _, tc := range []struct {
		kubeMinor                 string
		hypervisorPod, hypervisor string
	}{
		{kubeMinor: "29", hypervisorPod: goldenHypervisorPod, hypervisor: goldenHypervisor},
		{kubeMinor: "32", hypervisorPod: goldenHypervisorPodUnconfined, hypervisor: goldenHypervisorUnconfined},
	} {
		t.Run("Kubernetes 1."+tc.kubeMinor, func(t *testing.T) {
			useControllerHashInputs(t, tc.kubeMinor)
			pool := goldenHashPool()
			config := pool.Spec.ComponentConfig
			hypervisorPod, hypervisor := hypervisorHashes(pool)
			checkGoldenHashes(t, []goldenHash{
				{"worker, nothing configured", WorkerTemplateHash(nil, nil), goldenWorkerEmpty},
				{"worker", WorkerTemplateHash(config.Worker, config.Hypervisor), goldenWorker},
				{"client", ClientTemplateHash(pool), goldenClient},
				{"hypervisor pod", hypervisorPod, tc.hypervisorPod},
				{"hypervisor, pool-wide", hypervisor, tc.hypervisor},
			})
		})
	}
}

// With the Vector sidecar enabled, the hypervisor hash of this release differs
// from the previous one: the hypervisor is told to write its metrics where the
// sidecar reads them, the sidecar's default memory limit is raised, and a
// template that declares only the hypervisor container keeps that container
// instead of being discarded. Hypervisors with the sidecar are therefore
// restarted once when the operator is upgraded to this release. That is the
// only restart this release is meant to cause: the values of both releases are
// pinned side by side, and any other value is a regression that restarts them
// once more (or, for the previous release's values, would mean the change is
// not rolled out).
func TestHypervisorHashWithVectorChangesOnceInThisRelease(t *testing.T) {
	for _, tc := range []struct {
		name                                string
		template                            string
		previousPod, previousPoolWide       string
		thisReleasePod, thisReleasePoolWide string
	}{
		{
			name:                "template declares the hypervisor container only",
			template:            goldenHypervisorTemplate,
			previousPod:         previousVectorHypervisorPod,
			previousPoolWide:    previousVectorHypervisor,
			thisReleasePod:      goldenVectorHypervisorPod,
			thisReleasePoolWide: goldenVectorHypervisor,
		},
		{
			name:                "template declares the sidecar too",
			template:            goldenHypervisorTemplateWithSidecar,
			previousPod:         previousVectorSidecarHypervisorPod,
			previousPoolWide:    previousVectorSidecarHypervisor,
			thisReleasePod:      goldenVectorSidecarHypervisorPod,
			thisReleasePoolWide: goldenVectorSidecarHypervisor,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			useControllerHashInputs(t, "32")
			pod, poolWide := hypervisorHashes(goldenVectorHashPool(tc.template))
			checkGoldenHashes(t, []goldenHash{
				{"hypervisor pod with Vector", pod, tc.thisReleasePod},
				{"hypervisor with Vector, pool-wide", poolWide, tc.thisReleasePoolWide},
			})
			if tc.thisReleasePod == tc.previousPod || tc.thisReleasePoolWide == tc.previousPoolWide {
				t.Errorf("the pinned hashes equal the previous release's (%s, %s): "+
					"hypervisors would keep running without the metrics path the sidecar reads",
					tc.previousPod, tc.previousPoolWide)
			}
		})
	}
}

// The inputs useControllerHashInputs fixes do reach the hashes: a hash that
// ignored them would be pinned to one value by accident.
func TestTemplateHashInputsOutsideThePool(t *testing.T) {
	useControllerHashInputs(t, "32")
	pool := goldenVectorHashPool(goldenHypervisorTemplate)
	config := pool.Spec.ComponentConfig
	hashes := func() []string {
		pod, poolWide := hypervisorHashes(pool)
		return []string{WorkerTemplateHash(config.Worker, config.Hypervisor), ClientTemplateHash(pool), pod, poolWide}
	}
	base := hashes()

	for _, tc := range []struct {
		name   string
		change func(t *testing.T)
		// worker, client, hypervisor pod, hypervisor pool-wide
		changes [4]bool
	}{
		{"Secret with the database password", func(t *testing.T) {
			t.Setenv(constants.VectorGreptimePasswordSecretEnv, "release-greptimedb-secret")
		}, [4]bool{false, false, true, true}},
		{"pod-resources proxy", func(t *testing.T) {
			t.Setenv(constants.HypervisorPodResourcesProxyEnabledEnv, constants.TrueStringValue)
		}, [4]bool{false, false, true, true}},
		{"Kubernetes before 1.30", func(t *testing.T) {
			t.Setenv(constants.KubeApiVersionMinorEnv, "29")
		}, [4]bool{false, false, true, true}},
		{"termination grace period of test mode", func(t *testing.T) {
			setForTest(t, &constants.GracefulPeriodSeconds, ptr.To(int64(0)))
		}, [4]bool{true, false, true, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.change(t)
			names := [4]string{"worker", "client", "hypervisor pod", "hypervisor, pool-wide"}
			for i, got := range hashes() {
				if changed := got != base[i]; changed != tc.changes[i] {
					t.Errorf("%s hash: changed=%t, want %t", names[i], changed, tc.changes[i])
				}
			}
		})
	}
}

const (
	// Computed by the previous release.
	goldenWorkerEmpty             = "9949aa4feaea2b3e"
	goldenWorker                  = "bcd8a3a6cb42a467"
	goldenClient                  = "caaffe232ad5289c"
	goldenHypervisorPod           = "4f7e5222fea9c326"
	goldenHypervisor              = "c09697dc417ce143"
	goldenHypervisorPodUnconfined = "25f37570465e5046"
	goldenHypervisorUnconfined    = "c4e7082059ceaae3"

	// With the Vector sidecar: the previous release's values ...
	previousVectorHypervisorPod        = "d7ec02a018e4f3b5"
	previousVectorHypervisor           = "663f8c85b0d7b5da"
	previousVectorSidecarHypervisorPod = "ca9447f47b867732"
	previousVectorSidecarHypervisor    = "75bbedae16beb4f7"
	// ... and this release's.
	goldenVectorHypervisorPod        = "7fcd0b519213a608"
	goldenVectorHypervisor           = "30ff197d4ef4797d"
	goldenVectorSidecarHypervisorPod = "6b88baf69f0a67af"
	goldenVectorSidecarHypervisor    = "63c61e18baa52c3c"
)
