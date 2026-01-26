# AMD GPU Setup Guide for TensorFusion

This guide covers setting up TensorFusion with AMD GPUs using the TheRock ROCm distribution.

## Overview

TensorFusion supports AMD Instinct datacenter GPUs (MI series) through a provider library that uses AMD SMI for device discovery and metrics collection. This enables:

- **GPU Discovery**: Automatic detection and registration of AMD GPUs
- **Metrics Collection**: Real-time utilization, memory, power, and temperature monitoring
- **Resource Scheduling**: Fractional GPU allocation via VRAM and TFLOPS requests
- **Workload Management**: Integration with Kubernetes for GPU workload scheduling

### Supported GPUs

| Model | VRAM | FP16 TFLOPS | Status |
|-------|------|-------------|--------|
| MI325X | 256GB HBM3e | 1307 | ✓ Tested |
| MI300X | 192GB HBM3 | 1307 | ✓ Supported |
| MI300A | 128GB HBM3 | 979 | ✓ Supported |
| MI250X | 128GB HBM2e | 383 | ✓ Supported |
| MI250 | 128GB HBM2e | 362 | ✓ Supported |
| MI210 | 64GB HBM2e | 362 | ✓ Supported |

## Prerequisites

### Hardware Requirements

- AMD Instinct MI series GPU (MI210, MI250, MI250X, MI300A, MI300X, MI325X)
- PCIe Gen4/Gen5 compatible motherboard
- Adequate power supply (MI325X requires ~750W per GPU)

### Software Requirements

- Linux kernel 5.15+ with AMDGPU driver support
- Kubernetes 1.28+
- Helm 3.x
- Container runtime (Docker/containerd)

## Installation

### Step 1: Install ROCm via TheRock

TensorFusion uses TheRock, AMD's next-generation ROCm tarball distribution. The installation script is included:

```bash
# Install TheRock ROCm (default: 7.11.0rc0 for MI325X)
sudo ./scripts/install_rocm_tarball.sh 7.11.0rc0 gfx94X-dcgpu prereleases

# Verify installation
ls /opt/rocm
rocm-smi
```

**TheRock Version Selection:**

| GPU Family | Architecture | Recommended Version |
|------------|--------------|---------------------|
| MI325X | gfx94X-dcgpu | 7.11.0rc0 |
| MI300X/A | gfx94X-dcgpu | 7.10.0 |
| MI250X | gfx90a | 7.10.0 |
| MI210 | gfx90a | 7.10.0 |

**Release Types:**
- `stable` - Production releases
- `prereleases` - Release candidates
- `nightlies` - Nightly builds (for testing)

### Step 2: Verify GPU Detection

After ROCm installation, verify the GPUs are detected:

```bash
# Check GPU detection
rocm-smi

# Expected output for MI325X:
# ============================ ROCm System Management Interface ============================
# ====================================== GPU INFO ==========================================
# GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
# 0    45°C   75W     1700Mhz 1600Mhz  0%   auto  750W    0%     0%   
# ...

# Verify AMD SMI library
ls /opt/rocm/lib/libamd_smi.so

# Check device files
ls -la /dev/dri/
ls -la /dev/kfd
```

### Step 3: Install AMD Device Plugin (Optional)

For automatic node labeling, install the AMD device plugin:

```bash
kubectl apply -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
```

This adds labels like `amd.com/gpu.product-name=AMD_Instinct_MI325_OAM` to nodes.

### Step 4: Label AMD GPU Nodes

If not using the AMD device plugin, manually label your nodes:

```bash
# Label nodes with AMD GPUs
kubectl label nodes <node-name> amd.com/gpu.product-name=AMD_Instinct_MI325_OAM

# Verify labels
kubectl get nodes -l amd.com/gpu.product-name=AMD_Instinct_MI325_OAM
```

### Step 5: Deploy TensorFusion with AMD Support

```bash
# Install TensorFusion with AMD configuration
helm install tensor-fusion ./charts/tensor-fusion \
  -n tensor-fusion \
  -f ./charts/tensor-fusion/values-amd.yaml \
  --create-namespace

# Verify deployment
kubectl get pods -n tensor-fusion
```

### Step 6: Create TensorFusionCluster

Apply the AMD cluster configuration:

```bash
kubectl apply -f config/samples/v1_tensorfusioncluster_amd.yaml

# Check status
kubectl get tensorfusioncluster -n tensor-fusion
kubectl get gpupool -n tensor-fusion
kubectl get gpus -n tensor-fusion
```

## Configuration

### values-amd.yaml Reference

Key configuration options in `charts/tensor-fusion/values-amd.yaml`:

```yaml
# Node selector for AMD GPU nodes
initialGpuNodeLabelSelector: "amd.com/gpu.product-name=AMD_Instinct_MI325_OAM"

controller:
  image:
    repository: ghcr.io/saienduri/tensor-fusion-operator
    tag: "amd-support"

# GreptimeDB for metrics storage
greptime:
  installStandalone: true
  persistence:
    enabled: true
```

### GPUPool Configuration

Example AMD GPUPool configuration:

```yaml
apiVersion: tensor-fusion.ai/v1
kind: GPUPool
metadata:
  name: amd-gpu-pool
  namespace: tensor-fusion
spec:
  defaultUsingLocalGPU: true
  
  capacityConfig:
    oversubscription:
      vramExpandToHostMem: 50
      tflopsOversellRatio: 200
  
  nodeManagerConfig:
    provisioningMode: AutoSelect
    defaultVendor: AMD
    nodeSelector:
      nodeSelectorTerms:
        - matchExpressions:
            - key: amd.com/gpu.product-name
              operator: In
              values:
                - AMD_Instinct_MI325_OAM
  
  componentConfig:
    hypervisor:
      providerImage:
        AMD: ghcr.io/saienduri/tensor-fusion-amd-provider:latest
      enableVector: true
    worker:
      providerImage:
        AMD: ghcr.io/saienduri/tensor-fusion-amd-provider:latest
```

### Workload Annotations

To request AMD GPU resources for your workloads:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-gpu-workload
  annotations:
    tensor-fusion.ai/enabled: "true"
    tensor-fusion.ai/pool: "amd-gpu-pool"
    tensor-fusion.ai/vram-request: "16Gi"
    tensor-fusion.ai/vram-limit: "32Gi"
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/tflops-limit: "200"
    tensor-fusion.ai/isolation: "shared"
spec:
  containers:
  - name: pytorch
    image: rocm/pytorch:latest
    # Your application config
```

## GPU Verification

### Verify GPU Discovery

After deployment, verify GPUs are discovered:

```bash
# List discovered GPUs
kubectl get gpus -n tensor-fusion

# Check GPU details
kubectl describe gpu <gpu-name> -n tensor-fusion

# Check GPUNode status
kubectl get gpunodes -n tensor-fusion
```

### Verify Hypervisor

Check the hypervisor is running correctly:

```bash
# Check hypervisor pods
kubectl get pods -l tensor-fusion.ai/component=hypervisor -n tensor-fusion

# View hypervisor logs
kubectl logs -l tensor-fusion.ai/component=hypervisor -n tensor-fusion --tail=50
```

### Test Workload Execution

Deploy a test workload to verify GPU access:

```bash
# Deploy test pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: rocm-test
  namespace: tensor-fusion
  labels:
    tensor-fusion.ai/enabled: 'true'
  annotations:
    tensor-fusion.ai/pool: "amd-tensor-fusion-cluster-amd-gpu-pool"
    tensor-fusion.ai/vram-request: "8Gi"
    tensor-fusion.ai/tflops-request: "50"
    tensor-fusion.ai/isolation: "shared"
spec:
  containers:
  - name: test
    image: rocm/pytorch:latest
    command: ["python3", "-c"]
    args:
      - |
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"ROCm available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
  restartPolicy: Never
EOF

# Check pod logs
kubectl logs rocm-test -n tensor-fusion
```

## Metrics and Observability

### GreptimeDB Metrics

TensorFusion stores metrics in GreptimeDB. To query metrics:

```bash
# Port-forward GreptimeDB
kubectl port-forward -n greptimedb svc/greptimedb-standalone 4002:4002

# Query GPU metrics
curl 'http://localhost:4002/v1/sql?db=public&sql=SELECT * FROM tf_gpu_metrics ORDER BY timestamp DESC LIMIT 10'
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `utilization_percent` | GPU compute utilization (0-100%) |
| `memory_used_bytes` | VRAM used in bytes |
| `power_usage_watts` | Current power draw |
| `temperature_celsius` | GPU temperature |
| `pcie_rx_bytes` | PCIe receive throughput |
| `pcie_tx_bytes` | PCIe transmit throughput |

## Troubleshooting

### Common Issues

#### GPU Not Detected

**Symptom:** `kubectl get gpus` shows no GPUs

**Diagnosis:**
```bash
# Check hypervisor logs
kubectl logs -l tensor-fusion.ai/component=hypervisor -n tensor-fusion

# Verify AMD SMI on node
kubectl exec -it <hypervisor-pod> -- rocm-smi
```

**Solutions:**
1. Ensure `/dev/kfd` and `/dev/dri/renderD*` devices exist on the node
2. Verify the AMDGPU kernel driver is loaded: `lsmod | grep amdgpu`
3. Check that the node has the correct labels

#### Provider Image Pull Failed

**Symptom:** Hypervisor pod stuck in `ImagePullBackOff`

**Solution:**
```bash
# Verify image exists
docker pull ghcr.io/saienduri/tensor-fusion-amd-provider:latest

# Check image pull secrets
kubectl get secrets -n tensor-fusion
```

#### VRAM Units Mismatch

**Note:** AMD SMI returns VRAM size in megabytes. The TensorFusion AMD provider automatically converts this to bytes. If you see incorrect VRAM values, ensure you're using the latest provider image.

#### renderD Device Mapping

Each GPU maps to a specific `/dev/dri/renderD*` device. The AMD provider automatically discovers this mapping via `/sys/class/drm`. If workloads can't access the GPU:

```bash
# Check device permissions
ls -la /dev/dri/
ls -la /dev/kfd

# Verify user is in video group
groups
```

#### GPU CR Name Collisions

If you have multiple nodes with GPUs, ensure GPU Custom Resource names don't collide. The TensorFusion hypervisor prepends the node name to GPU identifiers to prevent this.

### Debug Commands

```bash
# Full cluster status
kubectl get tensorfusioncluster,gpupool,gpunode,gpu -n tensor-fusion

# Hypervisor debug logs
kubectl logs -l tensor-fusion.ai/component=hypervisor -n tensor-fusion -f

# Check events
kubectl get events -n tensor-fusion --sort-by='.lastTimestamp'

# Describe failing resources
kubectl describe gpupool <pool-name> -n tensor-fusion
kubectl describe gpunode <node-name> -n tensor-fusion
```

### Getting Help

1. Check the [TensorFusion GitHub Issues](https://github.com/NexusGPU/tensor-fusion/issues)
2. Review hypervisor and controller logs
3. Verify ROCm installation with `rocm-smi` directly on the GPU node

## Limitations

### Current Limitations (Phase 1-5)

1. **No Hard Isolation**: VRAM/TFLOPS limits are enforced at scheduling time but not at runtime. Workloads can exceed their limits.

2. **No Partitioning**: AMD GPUs don't have MIG-like partitioning (unlike NVIDIA). All isolation is soft/shared.

3. **No Snapshot/Resume**: Process state snapshot and resume is not supported.

4. **Local GPU Only**: Remote GPU execution (GPU-over-IP) is planned for Phase 6.

### Planned Features (Phase 6)

- Remote GPU execution via HIP API forwarding
- GPU-over-IP for containers on non-GPU nodes
- HIP client stub for transparent remote execution

## References

- [TheRock ROCm Distribution](https://github.com/ROCm/TheRock)
- [AMD SMI Documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
- [TensorFusion Documentation](https://github.com/NexusGPU/tensor-fusion)
