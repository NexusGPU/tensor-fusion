/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package kubernetes

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	podresv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
)

const (
	// upstreamKubeletPodResourcesSocket is the kubelet socket we proxy in front of.
	// kubelet hardcodes this path under its --root-dir.
	upstreamKubeletPodResourcesSocket = "/var/lib/kubelet/pod-resources/kubelet.sock"

	// proxySocketPath is where DCGM exporter is expected to dial.
	// Sibling directory of the real kubelet socket; mounted via hostPath into
	// both the hypervisor (as listener) and dcgm-exporter (as client).
	proxySocketDir  = "/var/lib/kubelet/pod-resources-tf"
	proxySocketPath = proxySocketDir + "/kubelet.sock"

	// nvidiaGPUResourceName is what DCGM exporter expects in resource_name.
	nvidiaGPUResourceName = "nvidia.com/gpu"

	upstreamDialTimeout = 5 * time.Second
)

// PodResourcesProxy is a transparent kubelet PodResources gRPC proxy that
// rewrites tensor-fusion device-plugin entries (resource_name=tensor-fusion.ai/index_*,
// device_ids=<dummy>) into the form DCGM exporter understands:
//
//	resource_name = "nvidia.com/gpu"
//	device_ids    = [<real NVML UUID>, ...]
//
// Real UUIDs come from pod annotations written by the TF scheduler, checked in
// this preference order:
//  1. tensor-fusion.ai/container-gpus  (per-container JSON: {"name": ["GPU-xxx"]})
//  2. tensor-fusion.ai/partition-uuids (MIG: "MIG-xxx:GPU-parent,...") — emits MIG-
//  3. tensor-fusion.ai/gpu-ids         (pod-level comma-separated fallback)
//
// All non-TF entries (real nvidia.com/gpu, CPU, memory, DRA) are passed through.
type PodResourcesProxy struct {
	podresv1.UnimplementedPodResourcesListerServer

	upstreamConn *grpc.ClientConn
	upstream     podresv1.PodResourcesListerClient
	cache        *PodCacheManager

	server   *grpc.Server
	lis      net.Listener
	stopOnce sync.Once
}

// StartPodResourcesProxy launches the proxy. Returns an error only when setup
// fails (e.g. proxy socket cannot be listened on). The upstream kubelet socket
// is dialed lazily by grpc.NewClient — if kubelet is down, individual List
// calls will fail but the proxy itself stays up.
//
// The caller (KubeletBackend) owns the lifecycle and must call Stop() to tear
// the proxy down. We deliberately don't watch a context here: Backend.Stop()
// is the single shutdown path, and a ctx-watcher goroutine would race with it
// on exit.
func StartPodResourcesProxy(cache *PodCacheManager) (*PodResourcesProxy, error) {
	if cache == nil {
		return nil, fmt.Errorf("pod cache manager is required")
	}

	if err := os.MkdirAll(proxySocketDir, 0o750); err != nil {
		return nil, fmt.Errorf("mkdir %s: %w", proxySocketDir, err)
	}
	// Stale socket from a previous run blocks net.Listen.
	if err := os.Remove(proxySocketPath); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("remove stale socket %s: %w", proxySocketPath, err)
	}

	// grpc.NewClient is non-blocking; dial happens on first RPC.
	conn, err := grpc.NewClient("unix://"+upstreamKubeletPodResourcesSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial upstream %s: %w", upstreamKubeletPodResourcesSocket, err)
	}

	lis, err := net.Listen("unix", proxySocketPath)
	if err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("listen %s: %w", proxySocketPath, err)
	}
	// DCGM exporter typically runs as root; tighten to 0o660 to keep arbitrary
	// processes from talking to the proxy.
	if err := os.Chmod(proxySocketPath, 0o660); err != nil {
		klog.Warningf("pod-resources proxy: chmod %s: %v", proxySocketPath, err)
	}

	p := &PodResourcesProxy{
		upstreamConn: conn,
		upstream:     podresv1.NewPodResourcesListerClient(conn),
		cache:        cache,
		server:       grpc.NewServer(),
		lis:          lis,
	}
	podresv1.RegisterPodResourcesListerServer(p.server, p)

	go func() {
		klog.Infof("pod-resources proxy listening on %s (upstream %s)",
			proxySocketPath, upstreamKubeletPodResourcesSocket)
		if err := p.server.Serve(lis); err != nil {
			klog.Errorf("pod-resources proxy server exited: %v", err)
		}
	}()
	return p, nil
}

// Stop is idempotent — Backend.Stop() and any future ctx-driven path can both
// call it safely.
func (p *PodResourcesProxy) Stop() {
	p.stopOnce.Do(func() {
		if p.server != nil {
			p.server.GracefulStop()
		}
		if p.upstreamConn != nil {
			_ = p.upstreamConn.Close()
		}
		// Best-effort cleanup; ignore errors.
		_ = os.Remove(proxySocketPath)
	})
}

// List proxies kubelet's response, rewriting TF device-plugin entries in place.
func (p *PodResourcesProxy) List(
	ctx context.Context, req *podresv1.ListPodResourcesRequest,
) (*podresv1.ListPodResourcesResponse, error) {
	dialCtx, cancel := context.WithTimeout(ctx, upstreamDialTimeout)
	defer cancel()
	resp, err := p.upstream.List(dialCtx, req)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return resp, nil
	}

	rewritten := 0
	for _, pr := range resp.PodResources {
		pod := p.lookupPod(pr.GetNamespace(), pr.GetName())
		if pod == nil {
			continue // not TF managed, leave untouched
		}
		for _, cont := range pr.Containers {
			uuids := containerGPUIDs(pod, cont.GetName())
			if len(uuids) == 0 {
				continue
			}
			for i := range uuids {
				uuids[i] = normalizeNVMLUUID(uuids[i])
			}
			for _, cd := range cont.Devices {
				if !isTFDevicePluginResource(cd.GetResourceName()) {
					continue
				}
				cd.ResourceName = nvidiaGPUResourceName
				cd.DeviceIds = append(cd.DeviceIds[:0], uuids...)
				cd.Topology = nil // let DCGM resolve NUMA via NVML
				rewritten++
			}
		}
	}
	if rewritten > 0 {
		klog.V(4).Infof("pod-resources proxy: rewrote %d TF entries", rewritten)
	}
	return resp, nil
}

// Get is a transparent passthrough; we don't need to rewrite single-pod queries
// because DCGM exporter only ever calls List.
func (p *PodResourcesProxy) Get(
	ctx context.Context, req *podresv1.GetPodResourcesRequest,
) (*podresv1.GetPodResourcesResponse, error) {
	return p.upstream.Get(ctx, req)
}

// GetAllocatableResources passes through. The node-level allocatable view
// already reflects the (dummy) TF device-plugin resources; DCGM doesn't use
// this RPC for pod labelling.
func (p *PodResourcesProxy) GetAllocatableResources(
	ctx context.Context, req *podresv1.AllocatableResourcesRequest,
) (*podresv1.AllocatableResourcesResponse, error) {
	return p.upstream.GetAllocatableResources(ctx, req)
}

// lookupPod scans the (per-node) TF pod cache for a pod by namespace/name.
// Cache is keyed by UID; for typical N<256 per node this linear scan is fine.
func (p *PodResourcesProxy) lookupPod(namespace, name string) *corev1.Pod {
	if namespace == "" || name == "" {
		return nil
	}
	for _, pod := range p.cache.GetAllPods() {
		if pod.Namespace == namespace && pod.Name == name {
			return pod
		}
	}
	return nil
}

// isTFDevicePluginResource matches the device-plugin resource names registered
// by pkg/hypervisor/backend/kubernetes/deviceplugin.go:
//
//	tensor-fusion.ai/index_<0..f>
//
// (PodIndexAnnotation + PodIndexDelimiter + %x).
func isTFDevicePluginResource(resourceName string) bool {
	prefix := constants.PodIndexAnnotation + constants.PodIndexDelimiter
	return strings.HasPrefix(resourceName, prefix)
}

// containerGPUIDs returns the real GPU UUIDs assigned to a container, in the
// preference order documented in the design doc:
//  1. tensor-fusion.ai/container-gpus  — JSON {containerName: ["GPU-..."]}
//  2. tensor-fusion.ai/partition-uuids — MIG instances: "MIG-uuid:GPU-parent,..."
//  3. tensor-fusion.ai/gpu-ids         — comma-separated pod-level fallback
//
// For MIG pods we return the MIG-<uuid> form (left side of each pair); DCGM
// exporter resolves it back to a parent GPU and a MIG profile via NVML.
func containerGPUIDs(pod *corev1.Pod, containerName string) []string {
	if pod == nil || pod.Annotations == nil {
		return nil
	}
	if raw := pod.Annotations[constants.ContainerGPUsAnnotation]; raw != "" {
		m := map[string][]string{}
		if err := json.Unmarshal([]byte(raw), &m); err == nil {
			if ids, ok := m[containerName]; ok && len(ids) > 0 {
				return filterEmpty(ids)
			}
		}
	}
	if raw := pod.Annotations[constants.PartitionUUIDsAnnotation]; raw != "" {
		return parsePartitionUUIDs(raw)
	}
	if raw := pod.Annotations[constants.GPUDeviceIDsAnnotation]; raw != "" {
		return filterEmpty(strings.Split(raw, ","))
	}
	return nil
}

// parsePartitionUUIDs extracts the MIG instance UUIDs from a
// tensor-fusion.ai/partition-uuids annotation value. The format is
// comma-separated "partitionUUID:parentGPU" pairs, e.g.
// "MIG-xxx/7/0:GPU-abc,MIG-yyy/3/0:GPU-def".
func parsePartitionUUIDs(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if colon := strings.IndexByte(p, ':'); colon >= 0 {
			p = p[:colon]
		}
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// normalizeNVMLUUID uppercases a leading "gpu-" prefix to match what NVML and
// DCGM exporter emit ("GPU-xxxxxxxx-..."). TF's pod-level fallback annotation
// (tensor-fusion.ai/gpu-ids) stores ids in their kubernetes-name form which is
// lowercased; without this normalization DCGM's UUID-string match silently
// fails and pod/namespace/container labels are dropped from metrics.
//
// MIG- prefixed UUIDs from partition-uuids are returned unchanged — DCGM
// already expects that form for MIG instances.
func normalizeNVMLUUID(id string) string {
	if strings.HasPrefix(id, "gpu-") {
		return "GPU-" + id[len("gpu-"):]
	}
	return id
}

func filterEmpty(in []string) []string {
	out := in[:0]
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// SocketPath returns the path the proxy listens on. Useful for tests and logs.
func (p *PodResourcesProxy) SocketPath() string {
	return proxySocketPath
}
