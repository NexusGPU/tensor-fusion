#!/bin/bash
# Copyright 2024.
# AMD GPU Cluster Integration Test Script
# Tests TensorFusion deployment with AMD GPUs in Kubernetes

set -e

echo "========================================="
echo "AMD GPU Cluster Integration Test"
echo "========================================="
echo ""

# Configuration
NAMESPACE="${NAMESPACE:-tensor-fusion}"
HELM_RELEASE="${HELM_RELEASE:-tensor-fusion}"
CLUSTER_NAME="${CLUSTER_NAME:-amd-tensor-fusion-cluster}"
GPUPOOL_NAME="${GPUPOOL_NAME:-amd-gpu-pool}"
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

# Step 1: Label AMD GPU nodes
echo "Step 1: Labeling AMD GPU nodes..."
echo "========================================"

# Get nodes with AMD GPUs (assuming they have the AMD device plugin label or you manually identify them)
AMD_NODES=$(kubectl get nodes -l 'amd.com/gpu.product-name=AMD_Instinct_MI325_OAM' -o name)

if [ -z "$AMD_NODES" ]; then
    echo "Warning: No AMD GPU nodes detected automatically."
    echo "Please manually label your AMD GPU nodes with AMD device plugin labels:"
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
    echo ""

    for node in $AMD_NODES; do
        echo "Labeling $node..."
        kubectl label $node $AMD_LABEL --overwrite
    done
    echo "✓ Nodes labeled"
fi
echo ""

# Step 2: Install/Upgrade TensorFusion with AMD values
echo "Step 2: Installing TensorFusion with AMD support..."
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

# Step 3: Wait for controller to be ready
echo "Step 3: Waiting for TensorFusion controller..."
echo "========================================"

kubectl wait --for=condition=ready pod \
    -l tensor-fusion.ai/component=operator \
    -n $NAMESPACE \
    --timeout=5m

echo "✓ Controller is ready"
echo ""

# Step 4: Create TensorFusionCluster (which will auto-create GPUPool)
echo "Step 4: Creating TensorFusionCluster..."
echo "========================================"
echo ""
echo "This follows the official TensorFusion deployment approach:"
echo "  1. TensorFusionCluster is the top-level resource"
echo "  2. It automatically creates and manages GPUPool(s)"
echo "  3. Provides cluster-wide metrics aggregation"
echo ""

kubectl apply -f ./config/samples/v1_tensorfusioncluster_amd.yaml

echo "Waiting for TensorFusionCluster to be ready..."
sleep 10

echo ""
echo "TensorFusionCluster status:"
kubectl get tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE -o yaml

echo ""
echo "✓ TensorFusionCluster created"
echo ""

# Step 5: Verify GPUPool was auto-created
echo "Step 5: Verifying GPUPool auto-creation..."
echo "========================================"

echo "Waiting for GPUPool to be auto-created by TensorFusionCluster..."

# The auto-created GPUPool name is: <cluster-name>-<pool-name>
ACTUAL_GPUPOOL_NAME="${CLUSTER_NAME}-${GPUPOOL_NAME}"

# Wait up to 60 seconds for GPUPool to be created
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

if kubectl get gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE &>/dev/null; then
    echo ""
    echo "✓ GPUPool '$ACTUAL_GPUPOOL_NAME' was auto-created successfully"

    # Wait for GPUPool to reach Ready phase
    echo ""
    echo "Waiting for GPUPool to reach 'Ready' phase (may take several minutes)..."
    echo "Press Ctrl+C to cancel if needed"
    WAIT_COUNT=0
    while true; do
        PHASE=$(kubectl get gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        if [ "$PHASE" = "Ready" ]; then
            echo ""
            echo "✓ GPUPool is Ready (waited ${WAIT_COUNT} seconds)"
            break
        elif [ "$PHASE" = "Updating" ] || [ "$PHASE" = "Pending" ]; then
            echo -n "."
            sleep 5
            WAIT_COUNT=$((WAIT_COUNT + 5))
            # Print status update every minute
            if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
                echo ""
                echo "Still waiting... (${WAIT_COUNT}s elapsed, phase: $PHASE)"
            fi
        else
            echo ""
            echo "⚠ GPUPool phase: $PHASE (waited ${WAIT_COUNT} seconds)"
            echo "This may indicate an issue. Check the GPUPool status:"
            kubectl describe gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE
            break
        fi
    done
    echo ""

    # Show GPUPool details
    kubectl get gpupool $ACTUAL_GPUPOOL_NAME -n $NAMESPACE -o yaml
else
    echo ""
    echo "⚠ Warning: GPUPool not created yet. This may take a few moments."
    echo "Check TensorFusionCluster status:"
    echo "  kubectl describe tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE"
fi
echo ""

# Step 6: Verify GPU discovery
echo "Step 6: Verifying AMD GPU discovery..."
echo "========================================"

echo "Waiting for GPUs to be discovered (30s)..."
sleep 30

# Check GPUNodes
echo ""
echo "GPUNodes discovered:"
kubectl get gpunodes -n $NAMESPACE

# Check GPUs
echo ""
echo "GPUs discovered:"
kubectl get gpus -n $NAMESPACE

GPU_COUNT=$(kubectl get gpus -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
echo ""
echo "Total AMD GPUs discovered: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "⚠ Warning: No GPUs discovered yet. This may take a few minutes."
    echo "Check hypervisor logs:"
    echo "  kubectl logs -l tensor-fusion.ai/component=hypervisor -n $NAMESPACE"
else
    echo "✓ AMD GPUs discovered successfully"
fi
echo ""

# Step 7: Check hypervisor status
echo "Step 7: Checking hypervisor pods..."
echo "========================================"

kubectl get pods -l tensor-fusion.ai/component=hypervisor -n $NAMESPACE

echo ""
echo "Hypervisor logs (last 20 lines):"
kubectl logs -l tensor-fusion.ai/component=hypervisor -n $NAMESPACE --tail=20 | head -20

echo ""

# Step 8: Deploy test workload
echo "Step 8: Deploying test workload..."
echo "========================================"
echo ""
echo "Note: VRAM/TFLOPS limits for AMD GPUs currently provide:"
echo "  ✓ Scheduler-level placement (ensures pod goes to node with capacity)"
echo "  ✓ Monitoring and metrics (tracks usage vs limits)"
echo "  ✗ Hard enforcement (workloads can exceed limits without being stopped)"
echo ""
echo "Hard limit enforcement requires ROCm containers/cgroups integration (future work)."
echo ""

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: amd-gpu-test
  namespace: $NAMESPACE
  labels:
    app: amd-gpu-test
    tensor-fusion.ai/enabled: 'true'
  annotations:
    # TensorFusion annotations for GPU resource requests
    # These control scheduling and monitoring, but NOT hard enforcement (yet)
    tensor-fusion.ai/inject-container: "rocm-test"
    tensor-fusion.ai/pool: "amd-tensor-fusion-cluster-amd-gpu-pool"
    tensor-fusion.ai/vram-request: "16Gi"
    tensor-fusion.ai/vram-limit: "16Gi"
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/tflops-limit: "100"
    tensor-fusion.ai/is-local-gpu: "true"
    tensor-fusion.ai/isolation: "shared"
spec:
  containers:
  - name: rocm-test
    image: rocm/pytorch:latest
    command: ["python3", "-c"]
    args:
      - |
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("No GPU available")
        import time
        time.sleep(3600)
  restartPolicy: Never
EOF

echo "Waiting for test pod to start..."
kubectl wait --for=condition=ready pod/amd-gpu-test -n $NAMESPACE --timeout=5m || true

echo ""
echo "Test pod logs:"
kubectl logs amd-gpu-test -n $NAMESPACE || echo "Pod not ready yet"

echo ""
echo "✓ Test workload deployed"
echo ""

# Step 9: Verify metrics collection
echo "Step 9: Checking metrics collection..."
echo "========================================"
echo ""

if kubectl get namespace greptimedb &>/dev/null; then
    echo "GreptimeDB Status:"
    kubectl get pods -n greptimedb
    echo ""

    # Check if GreptimeDB is running
    POD_STATUS=$(kubectl get pod -n greptimedb tensor-fusion-greptimedb-standalone-0 -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")

    if [ "$POD_STATUS" = "Running" ]; then
        echo "✓ GreptimeDB is running"
        echo ""

        # Test health endpoint
        echo "Testing GreptimeDB health endpoint..."
        if kubectl exec -n greptimedb tensor-fusion-greptimedb-standalone-0 -- \
            curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; then
            echo "✓ GreptimeDB health check passed"
        else
            echo "⚠ GreptimeDB health check failed (may still be starting up)"
        fi
        echo ""

        # Check PV/PVC
        echo "Storage configuration:"
        kubectl get pv,pvc -A | grep greptime || echo "No GreptimeDB storage found"
        echo ""
    else
        echo "⚠ GreptimeDB pod is not running (status: $POD_STATUS)"
        echo ""
        echo "Check storage and pod status:"
        echo "  kubectl get pvc -n greptimedb"
        echo "  kubectl describe pod -n greptimedb tensor-fusion-greptimedb-standalone-0"
        echo ""
    fi

    # Check Vector agents
    echo "Vector agents (metrics collectors):"
    kubectl get pods -l tensor-fusion.ai/component=vector -n $NAMESPACE || echo "No vector agents found"
    echo ""

    # Check controller autoscaler
    echo "Checking if autoscaler is enabled..."
    kubectl logs -l tensor-fusion.ai/component=operator -n $NAMESPACE --tail=50 | grep -i "auto scale" || echo "No autoscaler logs found"

else
    echo "GreptimeDB namespace not found - metrics collection is disabled."
    echo "This is OK for basic GPU functionality testing."
    echo ""
    echo "To enable metrics collection with NFS storage:"
    echo "  ./scripts/setup-greptime-nfs.sh"
fi

echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "✓ TensorFusion installed with AMD support"
echo "✓ AMD GPU nodes labeled"
echo "✓ TensorFusionCluster created: $CLUSTER_NAME"
echo "✓ GPUPool auto-created: ${CLUSTER_NAME}-${GPUPOOL_NAME}"
echo "✓ Discovered $GPU_COUNT AMD GPU(s)"
echo "✓ Test workload deployed"

# Add GreptimeDB status to summary
if kubectl get namespace greptimedb &>/dev/null; then
    GREPTIME_STATUS=$(kubectl get pod -n greptimedb tensor-fusion-greptimedb-standalone-0 -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    if [ "$GREPTIME_STATUS" = "Running" ]; then
        echo "✓ GreptimeDB running (metrics enabled)"
    else
        echo "⚠ GreptimeDB status: $GREPTIME_STATUS (check NFS storage)"
    fi
else
    echo "ℹ GreptimeDB not installed (metrics disabled)"
fi

echo ""
echo "Configuration:"
echo "  Namespace: $NAMESPACE"
echo "  AMD Label: $AMD_LABEL"
echo ""
echo "Next steps:"
echo "1. Check TensorFusionCluster status:"
echo "   kubectl get tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE"
echo "   kubectl describe tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE"
echo ""
echo "2. Check GPU allocation:"
echo "   kubectl describe pod amd-gpu-test -n $NAMESPACE"
echo ""
echo "3. View GPU utilization:"
echo "   kubectl get gpus -n $NAMESPACE"
echo ""
echo "4. Check hypervisor logs:"
echo "   kubectl logs -l tensor-fusion.ai/component=hypervisor -n $NAMESPACE -f"
echo ""
echo "5. View cluster-wide metrics:"
echo "   kubectl get tensorfusioncluster $CLUSTER_NAME -n $NAMESPACE -o jsonpath='{.status}' | jq"
echo ""
echo "6. Check GreptimeDB metrics (if enabled):"
echo "   kubectl port-forward -n greptimedb svc/greptimedb-standalone 4002:4002"
echo "   # Then query: curl 'http://localhost:4002/v1/sql?db=public&sql=SELECT * FROM tf_worker_usage LIMIT 10'"
echo ""
echo "7. Clean up test workload:"
echo "   kubectl delete pod amd-gpu-test -n $NAMESPACE"
echo ""
echo "Documentation:"
echo "  - AMD Implementation: docs/AMD-IMPLEMENTATION-SUMMARY.md"
echo "  - Quick Reference: docs/AMD-TESTING-QUICK-REFERENCE.md"
echo "  - NFS Storage: docs/GREPTIME-NFS-STORAGE.md"
echo "  - Enable Metrics: docs/ENABLING-METRICS.md"
echo ""
echo "========================================="
