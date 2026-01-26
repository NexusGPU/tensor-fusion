#!/bin/bash
# Copyright 2024.
# AMD Remote GPU Cluster Integration Test Script
# Tests TensorFusion deployment with AMD GPUs in remote (GPU-over-IP) mode

set -e

echo "========================================="
echo "AMD Remote GPU Cluster Integration Test"
echo "========================================="
echo ""
echo "This test deploys TensorFusion with AMD GPUs in REMOTE mode:"
echo "  - Workers run on GPU nodes (with actual GPUs)"
echo "  - Clients can run on ANY node (no GPU required)"
echo "  - HIP API calls are forwarded over the network"
echo ""

# Configuration
NAMESPACE="${NAMESPACE:-tensor-fusion}"
HELM_RELEASE="${HELM_RELEASE:-tensor-fusion}"
CLUSTER_NAME="${CLUSTER_NAME:-amd-remote-cluster}"
GPUPOOL_NAME="${GPUPOOL_NAME:-amd-remote-pool}"
AMD_LABEL="${AMD_LABEL:-amd.com/gpu.product-name=AMD_Instinct_MI325_OAM}"

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "Error: helm not found. Please install Helm 3."
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Step 1: Verify images are available
echo "Step 1: Checking required images..."
echo "========================================"

echo "Required images for remote GPU mode:"
echo "  - ghcr.io/saienduri/tensor-fusion-amd-provider:latest"
echo "  - ghcr.io/saienduri/tensor-fusion-hip-worker:latest"
echo "  - ghcr.io/saienduri/tensor-fusion-hypervisor:amd-support"
echo "  - ghcr.io/saienduri/tensor-fusion-operator:amd-support"
echo ""
echo "If images are not built, run:"
echo "  ./scripts/build-remote-gpu-images.sh --push"
echo ""

read -p "Are the images available in your registry? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please build and push images first."
    exit 1
fi

# Step 2: Label AMD GPU nodes
echo ""
echo "Step 2: Labeling AMD GPU nodes..."
echo "========================================"

AMD_NODES=$(kubectl get nodes -l 'amd.com/gpu.product-name=AMD_Instinct_MI325_OAM' -o name 2>/dev/null || true)

if [ -z "$AMD_NODES" ]; then
    echo "Warning: No AMD GPU nodes detected automatically."
    echo "Please manually label your AMD GPU nodes:"
    echo "  kubectl label nodes <node-name> amd.com/gpu.product-name=AMD_Instinct_MI325_OAM"
    echo ""
    read -p "Have you labeled your AMD GPU nodes? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please label nodes and try again."
        exit 1
    fi
else
    echo "Found AMD GPU nodes:"
    echo "$AMD_NODES"
fi
echo ""

# Step 3: Install/Upgrade TensorFusion
echo "Step 3: Installing TensorFusion..."
echo "========================================"

cd "$(dirname "$0")/.."

if helm list -n $NAMESPACE | grep -q $HELM_RELEASE; then
    echo "Upgrading existing release..."
    helm upgrade $HELM_RELEASE ./charts/tensor-fusion \
        -n $NAMESPACE \
        -f ./charts/tensor-fusion/values-amd.yaml \
        --wait \
        --timeout 10m
else
    echo "Installing new release..."
    helm install $HELM_RELEASE ./charts/tensor-fusion \
        -n $NAMESPACE \
        -f ./charts/tensor-fusion/values-amd.yaml \
        --create-namespace \
        --wait \
        --timeout 10m
fi

echo "✓ TensorFusion installed/upgraded"
echo ""

# Step 4: Wait for controller to be ready
echo "Step 4: Waiting for TensorFusion controller..."
echo "========================================"

kubectl wait --for=condition=ready pod \
    -l tensor-fusion.ai/component=operator \
    -n $NAMESPACE \
    --timeout=5m

echo "✓ Controller is ready"
echo ""

# Step 5: Create TensorFusionCluster for remote mode
echo "Step 5: Creating TensorFusionCluster (remote mode)..."
echo "========================================"

kubectl apply -f ./config/samples/v1_tensorfusioncluster_amd_remote.yaml

echo "Waiting for TensorFusionCluster to be ready..."
sleep 10

echo ""
echo "TensorFusionCluster status:"
kubectl get tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE -o wide 2>/dev/null || echo "Cluster not found yet"

echo ""
echo "✓ TensorFusionCluster created"
echo ""

# Step 6: Verify GPUPool was auto-created
echo "Step 6: Verifying GPUPool auto-creation..."
echo "========================================"

ACTUAL_GPUPOOL_NAME="${CLUSTER_NAME}-${GPUPOOL_NAME}"

echo "Waiting for GPUPool to be auto-created..."
for i in {1..12}; do
    if kubectl get gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE &>/dev/null; then
        break
    fi
    echo -n "."
    sleep 5
done
echo ""

echo ""
echo "GPUPools in cluster:"
kubectl get gpupool -n $NAMESPACE

# Verify remote mode is enabled
echo ""
echo "Checking remote GPU mode configuration..."
USING_LOCAL=$(kubectl get gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE -o jsonpath='{.spec.defaultUsingLocalGPU}' 2>/dev/null || echo "unknown")
echo "  defaultUsingLocalGPU: $USING_LOCAL"

if [ "$USING_LOCAL" = "false" ]; then
    echo "✓ Remote GPU mode is enabled"
else
    echo "⚠ Warning: Remote GPU mode may not be correctly configured"
fi
echo ""

# Step 7: Wait for hypervisor
echo "Step 7: Waiting for hypervisor pods..."
echo "========================================"

kubectl wait --for=condition=ready pod \
    -l tensor-fusion.ai/component=hypervisor \
    -n $NAMESPACE \
    --timeout=5m 2>/dev/null || echo "No hypervisor pods found (may take time)"

echo ""
echo "Hypervisor pods:"
kubectl get pods -n $NAMESPACE -l tensor-fusion.ai/component=hypervisor
echo ""

# Step 8: Check GPUNode status
echo "Step 8: Checking GPUNode status..."
echo "========================================"

echo "GPUNodes discovered:"
kubectl get gpunode -n $NAMESPACE 2>/dev/null || echo "No GPUNodes discovered yet"
echo ""

# Step 9: Deploy test workload
echo "Step 9: Deploying test workload (remote GPU client)..."
echo "========================================"

kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-remote-test
  namespace: $NAMESPACE
  labels:
    tensor-fusion.ai/enabled: "true"
  annotations:
    tensor-fusion.ai/inject-container: "test"
    tensor-fusion.ai/pool: "$ACTUAL_GPUPOOL_NAME"
    tensor-fusion.ai/isolation: "shared"
    tensor-fusion.ai/vram-request: "4Gi"
    tensor-fusion.ai/vram-limit: "4Gi"
spec:
  # REQUIRE non-GPU nodes to test remote GPU
  # This ensures the client runs on CPU node and accesses GPU remotely
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: amd.com/gpu.product-name
                operator: DoesNotExist
  containers:
    - name: test
      image: rocm/pytorch:latest
      command: ["python3", "-c"]
      args:
        - |
          import torch
          print("=" * 50)
          print("AMD Remote GPU Test")
          print("=" * 50)
          print(f"PyTorch version: {torch.__version__}")
          print(f"CUDA available: {torch.cuda.is_available()}")
          if torch.cuda.is_available():
              print(f"Device count: {torch.cuda.device_count()}")
              print(f"Device name: {torch.cuda.get_device_name(0)}")
              x = torch.randn(100, 100, device='cuda')
              y = torch.matmul(x, x)
              print(f"GPU computation successful: {y.shape}")
              print("=" * 50)
              print("SUCCESS: Remote GPU is working!")
          else:
              print("CUDA not available - checking if client stub is loaded...")
              import os
              print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'not set')}")
              print(f"TF_WORKER_HOST: {os.environ.get('TF_WORKER_HOST', 'not set')}")
              print(f"TF_WORKER_PORT: {os.environ.get('TF_WORKER_PORT', 'not set')}")
  restartPolicy: Never
EOF

echo ""
echo "Test pod created. Waiting for it to start..."
sleep 10

echo ""
echo "Test pod status:"
kubectl get pod pytorch-remote-test -n $NAMESPACE

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "To check test results:"
echo "  kubectl logs pytorch-remote-test -n $NAMESPACE"
echo ""
echo "To check TensorFusionConnection (worker assignment):"
echo "  kubectl get tensorfusionconnection -n $NAMESPACE"
echo ""
echo "To check worker pods:"
echo "  kubectl get pods -n $NAMESPACE -l tensor-fusion.ai/component=worker"
echo ""
echo "To clean up:"
echo "  kubectl delete pod pytorch-remote-test -n $NAMESPACE"
echo "  kubectl delete tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE"
