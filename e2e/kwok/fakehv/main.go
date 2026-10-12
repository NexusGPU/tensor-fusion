//go:build e2e

// Command fakehv is the stand-in for the TensorFusion hypervisor on KWOK nodes.
//
// Pods on a KWOK node never run, so the hypervisor pod the operator creates
// there is "Running" (KWOK says so) but registers nothing. fakehv writes what a
// real hypervisor writes to the API server, and nothing else:
//
//  1. Node.status.capacity / allocatable: the 16 tensor-fusion.ai/index_<hex>
//     extended resources, 36 each (the hypervisor's device plugins,
//     pkg/hypervisor/backend/kubernetes/deviceplugin.go);
//  2. one GPU custom resource per fake device: labels, controller owner
//     reference to the GPUNode, status.capacity / available / uuid / gpuModel /
//     vendor / index / numaNode / isolationMode / isolationPolicy /
//     nodeSelector / usedBy / phase=Pending / message
//     (KubeletBackend.mutateGPUResourceState);
//  3. GPUNode.status: totalTFlops, totalVRAM, totalGPUs, managedGPUs,
//     managedGPUDeviceIDs, nodeInfo (APIClient.UpdateGPUNodeStatus).
//
// It acts only on nodes that carry kwok.x-k8s.io/node=fake AND have a GPUNode
// (i.e. TensorFusion claimed them) AND whose hypervisor pod is Running, and only
// after -delay has passed since that pod was first seen Running. The delay
// stands for driver initialisation + hypervisor start-up + device discovery. A
// replaced hypervisor pod (new UID) waits again, like a restarted hypervisor.
//
// Every GPU it creates is labelled e2e.tensor-fusion.ai/fake=true and the node
// is annotated e2e.tensor-fusion.ai/fake-hypervisor-registered-at.
//
// What it does NOT do: no device, no limiter, no worker bookkeeping, no
// metrics, no kubelet. status.available after the first write, runningApps,
// phase Running and everything else stay with the operator, as in production.
//
//	go run -tags e2e -mod=vendor ./e2e/kwok/fakehv -delay 20s
package main

import (
	"context"
	"crypto/sha1"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

const (
	tfDomain   = "tensor-fusion.ai"
	fakeLabel  = "e2e.tensor-fusion.ai/fake"
	fakeRegAt  = "e2e.tensor-fusion.ai/fake-hypervisor-registered-at"
	kwokMarker = "kwok.x-k8s.io/node"
)

var (
	gvkNode    = schema.GroupVersionKind{Version: "v1", Kind: "Node"}
	gvkPod     = schema.GroupVersionKind{Version: "v1", Kind: "Pod"}
	gvkGPU     = schema.GroupVersionKind{Group: tfDomain, Version: "v1", Kind: "GPU"}
	gvkGPUNode = schema.GroupVersionKind{Group: tfDomain, Version: "v1", Kind: "GPUNode"}
)

// device is what a provider library would report for one card.
type device struct {
	Model  string  // GPU.status.gpuModel, as in the chart's public GPU info table
	VRAMMi int64   // MiB
	Tflops float64 // FP16 TFLOPS of the table
}

// Devices by instance family; "*" is the fallback. The numbers are the
// vendor-published ones of charts/tensor-fusion (gpu-public-gpu-info).
var devices = map[string]device{
	"g4dn": {"T4", 15360, 65},
	"g5":   {"A10G", 23028, 125},
	"g6":   {"L4", 23034, 121},
	"*":    {"T4", 15360, 65},
}

type fake struct {
	c       client.Client
	ns      string
	delay   time.Duration
	perNode int
	seen    map[string]time.Time // hypervisor pod UID -> first seen Running
	done    map[string]bool      // hypervisor pod UID -> registered
}

func main() {
	delay := flag.Duration("delay", 20*time.Second, "time between the hypervisor pod turning Running and the GPUs being registered (driver + hypervisor start-up)")
	interval := flag.Duration("interval", time.Second, "poll interval")
	perNode := flag.Int("gpus-per-node", 1, "fake devices per node")
	ns := flag.String("tf-namespace", "tensor-fusion-sys", "namespace of the TensorFusion release (hypervisor pods)")
	flag.Parse()

	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(clientcmd.NewDefaultClientConfigLoadingRules(), &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		log.Fatalf("kubeconfig: %v", err)
	}
	cfg.QPS, cfg.Burst = 50, 100
	cfg.UserAgent = "tf-e2e-fake-hypervisor"
	c, err := client.New(cfg, client.Options{})
	if err != nil {
		log.Fatalf("client: %v", err)
	}
	f := &fake{c: c, ns: *ns, delay: *delay, perNode: *perNode, seen: map[string]time.Time{}, done: map[string]bool{}}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("fake hypervisor: delay=%s gpus-per-node=%d namespace=%s", *delay, *perNode, *ns)
	t := time.NewTicker(*interval)
	defer t.Stop()
	for {
		if err := f.pass(ctx); err != nil && ctx.Err() == nil {
			log.Printf("pass: %v", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func list(ctx context.Context, c client.Client, gvk schema.GroupVersionKind, opts ...client.ListOption) ([]unstructured.Unstructured, error) {
	l := &unstructured.UnstructuredList{}
	l.SetGroupVersionKind(gvk.GroupVersion().WithKind(gvk.Kind + "List"))
	if err := c.List(ctx, l, opts...); err != nil {
		return nil, err
	}
	return l.Items, nil
}

func (f *fake) pass(ctx context.Context) error {
	gpuNodes, err := list(ctx, f.c, gvkGPUNode)
	if err != nil {
		return fmt.Errorf("list GPUNodes: %w", err)
	}
	if len(gpuNodes) == 0 {
		return nil
	}
	pods, err := list(ctx, f.c, gvkPod, client.InNamespace(f.ns), client.MatchingLabels{tfDomain + "/component": "hypervisor"})
	if err != nil {
		return fmt.Errorf("list hypervisor pods: %w", err)
	}
	hvByNode := map[string]*unstructured.Unstructured{}
	for i := range pods {
		p := &pods[i]
		node, _, _ := unstructured.NestedString(p.Object, "spec", "nodeName")
		phase, _, _ := unstructured.NestedString(p.Object, "status", "phase")
		if node != "" && phase == "Running" && p.GetDeletionTimestamp() == nil {
			hvByNode[node] = p
		}
	}
	for i := range gpuNodes {
		gn := &gpuNodes[i]
		if gn.GetDeletionTimestamp() != nil {
			continue
		}
		hv := hvByNode[gn.GetName()]
		if hv == nil {
			continue
		}
		uid := string(hv.GetUID())
		if f.done[uid] {
			continue
		}
		node := &unstructured.Unstructured{}
		node.SetGroupVersionKind(gvkNode)
		if err := f.c.Get(ctx, types.NamespacedName{Name: gn.GetName()}, node); err != nil {
			continue
		}
		// Never touch a node that has a kubelet.
		if node.GetLabels()[kwokMarker] != "fake" && node.GetAnnotations()[kwokMarker] != "fake" {
			continue
		}
		first, ok := f.seen[uid]
		if !ok {
			f.seen[uid] = time.Now()
			log.Printf("node %s: hypervisor pod %s Running, registering GPUs in %s", gn.GetName(), hv.GetName(), f.delay)
			continue
		}
		if time.Since(first) < f.delay {
			continue
		}
		if err := f.register(ctx, gn, node, hv); err != nil {
			log.Printf("node %s: %v (will retry)", gn.GetName(), err)
			continue
		}
		f.done[uid] = true
	}
	return nil
}

func (f *fake) register(ctx context.Context, gn, node, hv *unstructured.Unstructured) error {
	name := gn.GetName()
	pool := ""
	for _, o := range gn.GetOwnerReferences() {
		if o.Controller != nil && *o.Controller {
			pool = o.Name
		}
	}
	if pool == "" && len(gn.GetOwnerReferences()) > 0 {
		pool = gn.GetOwnerReferences()[0].Name
	}
	if pool == "" {
		return fmt.Errorf("GPUNode has no controller owner reference")
	}
	dev, ok := devices[strings.SplitN(node.GetLabels()["node.kubernetes.io/instance-type"], ".", 2)[0]]
	if !ok {
		dev = devices["*"]
	}
	mode, policy := "", ""
	if cs, _, _ := unstructured.NestedSlice(hv.Object, "spec", "containers"); len(cs) > 0 {
		envs, _, _ := unstructured.NestedSlice(cs[0].(map[string]any), "env")
		for _, e := range envs {
			m, _ := e.(map[string]any)
			switch m["name"] {
			case "TF_ISOLATION_MODE":
				mode = fmt.Sprint(m["value"])
			case "TF_ISOLATION_MODE_POLICY":
				policy = fmt.Sprint(m["value"])
			}
		}
	}

	// 1. device plugins: index_0 .. index_f, 36 devices each.
	res := map[string]any{}
	for _, h := range "0123456789abcdef" {
		res[tfDomain+"/index_"+string(h)] = "36"
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	nodePatch, _ := json.Marshal(map[string]any{"status": map[string]any{"capacity": res, "allocatable": res}})
	if err := f.c.Status().Patch(ctx, node, client.RawPatch(types.MergePatchType, nodePatch)); err != nil {
		return fmt.Errorf("register index resources: %w", err)
	}

	// 2. GPU resources.
	var ids []string
	for i := 0; i < f.perNode; i++ {
		sum := sha1.Sum([]byte(fmt.Sprintf("%s/%d", name, i)))
		id := fmt.Sprintf("gpu-%x-%x-%x-%x-%x", sum[0:4], sum[4:6], sum[6:8], sum[8:10], sum[10:16])
		ids = append(ids, id)
		caps, _ := json.Marshal(map[string]any{
			"SupportsPartitioning": false, "SupportsSoftIsolation": true, "SupportsHardIsolation": true,
			"SupportsSnapshot": false, "SupportsMetrics": true, "SupportsRemoting": true,
			"MaxPartitions": 0, "MaxWorkersPerDevice": 0,
		})
		gpu := &unstructured.Unstructured{}
		gpu.SetGroupVersionKind(gvkGPU)
		err := f.c.Get(ctx, types.NamespacedName{Name: id}, gpu)
		if apierrors.IsNotFound(err) {
			gpu.SetName(id)
			gpu.SetLabels(map[string]string{
				// As KubeletBackend.mutateGPUResourceState: a node name may not fit a label value.
				tfDomain + "/managed-by": utils.LabelSafeValue(name), tfDomain + "/gpupool": pool, fakeLabel: "true",
			})
			gpu.SetAnnotations(map[string]string{
				tfDomain + "/last-sync":                   time.Now().Format(time.RFC3339),
				tfDomain + "/virtualization-capabilities": string(caps),
				tfDomain + "/tflops-capacity-source":      "provider-config",
			})
			t := true
			gpu.SetOwnerReferences([]metav1.OwnerReference{{
				APIVersion: tfDomain + "/v1", Kind: "GPUNode", Name: name, UID: gn.GetUID(), Controller: &t, BlockOwnerDeletion: &t,
			}})
			if err = f.c.Create(ctx, gpu); err != nil {
				return fmt.Errorf("create GPU %s: %w", id, err)
			}
		} else if err != nil {
			return err
		}
		capacity := map[string]any{"vram": fmt.Sprintf("%dMi", dev.VRAMMi), "tflops": fmt.Sprintf("%g", dev.Tflops)}
		status := map[string]any{
			"capacity": capacity, "uuid": id, "gpuModel": dev.Model, "vendor": "NVIDIA",
			"index": int64(i), "numaNode": int64(0),
			"nodeSelector": map[string]any{"kubernetes.io/hostname": name},
			"usedBy":       "tensor-fusion", "message": "managed",
		}
		if mode != "" {
			status["isolationMode"] = mode
		}
		if policy != "" {
			status["isolationPolicy"] = policy
		}
		// available and phase belong to the operator once published.
		if _, found, _ := unstructured.NestedMap(gpu.Object, "status", "available"); !found {
			status["available"] = capacity
		}
		if phase, _, _ := unstructured.NestedString(gpu.Object, "status", "phase"); phase == "" || phase == "Unknown" {
			status["phase"] = "Pending"
		}
		patch, _ := json.Marshal(map[string]any{"status": status})
		if err := f.c.Status().Patch(ctx, gpu, client.RawPatch(types.MergePatchType, patch)); err != nil {
			return fmt.Errorf("GPU %s status: %w", id, err)
		}
	}

	// 3. GPUNode status.
	gnStatus := map[string]any{
		"totalTFlops": fmt.Sprintf("%g", dev.Tflops*float64(f.perNode)),
		"totalVRAM":   fmt.Sprintf("%d", dev.VRAMMi*int64(f.perNode)*1024*1024),
		"totalGPUs":   int64(f.perNode), "managedGPUs": int64(f.perNode),
		"managedGPUDeviceIDs": ids,
		"nodeInfo":            map[string]any{"ramSize": "16Gi", "dataDiskSize": "100Gi"},
	}
	if phase, _, _ := unstructured.NestedString(gn.Object, "status", "phase"); phase == "" {
		gnStatus["phase"] = "Pending"
	}
	patch, _ := json.Marshal(map[string]any{"status": gnStatus})
	if err := f.c.Status().Patch(ctx, gn, client.RawPatch(types.MergePatchType, patch)); err != nil {
		return fmt.Errorf("GPUNode status: %w", err)
	}

	mark, _ := json.Marshal(map[string]any{"metadata": map[string]any{"annotations": map[string]any{fakeRegAt: now}}})
	_ = f.c.Patch(ctx, node, client.RawPatch(types.MergePatchType, mark))
	log.Printf("node %s (%s): registered %d x %s (%g TFLOPS, %d Mi) in pool %s, isolation=%q policy=%q",
		name, node.GetLabels()["node.kubernetes.io/instance-type"], f.perNode, dev.Model, dev.Tflops, dev.VRAMMi, pool, mode, policy)
	return nil
}
