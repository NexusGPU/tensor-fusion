#!/bin/bash  

# One-click uninstall script for TensorFusion resources  
# Uninstalls in reverse order of the resource hierarchy: starting from lower-level resources to top-level resources  

set -e

echo "Starting to uninstall TensorFusion resources..."  

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl command not found, please make sure it's installed and in your PATH"  
    exit 1  
fi  

# Set namespace variable, default is tensor-fusion-sys
NAMESPACE=${NAMESPACE:-tensor-fusion-sys}
HELM_RELEASE=${HELM_RELEASE:-tensor-fusion-sys}

echo "Using namespace: $NAMESPACE"
echo "Helm release name: $HELM_RELEASE"

# Uninstall TensorFusionConnection resources (all namespaces)
echo "Uninstalling TensorFusionConnection resources..."
kubectl delete tensorfusionconnections --all --all-namespaces || true

# Uninstall TensorFusionWorkload resources (all namespaces)
echo "Uninstalling TensorFusionWorkload resources..."
kubectl delete tensorfusionworkloads --all --all-namespaces || true

# Uninstall WorkloadProfile resources (all namespaces)
echo "Uninstalling WorkloadProfile resources..."
kubectl delete workloadprofiles --all --all-namespaces || true

# Uninstall GPU resources
echo "Uninstalling GPU resources..."
kubectl delete gpus --all || true

# Uninstall GPUNode resources
echo "Uninstalling GPUNode resources..."
kubectl delete gpunodes --all || true

# Uninstall GPUPool resources
echo "Uninstalling GPUPool resources..."
kubectl delete gpupools --all || true

# Uninstall TensorFusionCluster resources
echo "Uninstalling TensorFusionCluster resources..."
kubectl delete tensorfusionclusters --all || true

# Uninstall SchedulingConfigTemplate resources
echo "Uninstalling SchedulingConfigTemplate resources..."
kubectl delete schedulingconfigtemplates --all || true

# Uninstall Webhook secret
echo "Uninstalling Webhook secret ..."
kubectl delete secret tensor-fusion-webhook-secret -n $NAMESPACE || true

# Check if the Helm release exists
echo "Checking for TensorFusion controller installation method..."

if helm status $HELM_RELEASE -n $NAMESPACE &> /dev/null; then
    echo "Helm release found. Using Helm to uninstall the TensorFusion controller..."
    helm uninstall $HELM_RELEASE -n $NAMESPACE || true
else
    echo "No Helm release or command found. Assuming deployment was done via 'helm template | kubectl apply -f' or similar method"
    echo "Deleting TensorFusion controller resources directly..."
    
    # Delete main controller resources
    kubectl delete deployment $HELM_RELEASE-controller -n $NAMESPACE || true
    kubectl delete service $HELM_RELEASE -n $NAMESPACE || true
    kubectl delete serviceaccount $HELM_RELEASE -n $NAMESPACE || true
    
    # Delete RBAC resources
    kubectl delete clusterrole $HELM_RELEASE-role || true
    kubectl delete clusterrolebinding $HELM_RELEASE-rolebinding || true
    kubectl delete clusterrole tensor-fusion-hypervisor-role || true
    kubectl delete clusterrolebinding tensor-fusion-hypervisor-rolebinding || true
    kubectl delete serviceaccount tensor-fusion-hypervisor-sa -n $NAMESPACE || true
    
    # Delete admission webhook resources
    kubectl delete mutatingwebhookconfiguration $HELM_RELEASE-mutating-webhook || true
    kubectl delete service $HELM_RELEASE-webhook -n $NAMESPACE || true
    kubectl delete job $HELM_RELEASE-add-hook-crt -n $NAMESPACE || true
    kubectl delete job $HELM_RELEASE-patch-admission-webhook -n $NAMESPACE || true
    kubectl delete serviceaccount $HELM_RELEASE-webhook-job -n $NAMESPACE || true
    kubectl delete clusterrole $HELM_RELEASE-webhook-job || true
    kubectl delete clusterrolebinding $HELM_RELEASE-webhook-job || true
    kubectl delete role $HELM_RELEASE-webhook-job -n $NAMESPACE || true
    kubectl delete rolebinding $HELM_RELEASE-webhook-job -n $NAMESPACE || true
    
    # Delete ConfigMaps
    kubectl delete configmap $HELM_RELEASE-config -n $NAMESPACE || true
    kubectl delete configmap tensor-fusion-sys-public-gpu-info -n $NAMESPACE || true
    kubectl delete configmap tensor-fusion-sys-vector-config -n $NAMESPACE || true
    
    # Delete Secrets
    kubectl delete secret $HELM_RELEASE-greptimedb-secret -n $NAMESPACE || true
    kubectl delete secret tf-cloud-vendor-credentials -n $NAMESPACE || true
    
    # Delete alert manager resources (if enabled)
    kubectl delete configmap $HELM_RELEASE-alert-manager-config -n $NAMESPACE || true
    kubectl delete statefulset $HELM_RELEASE-alert-manager -n $NAMESPACE || true
    kubectl delete service alert-manager -n $NAMESPACE || true
    kubectl delete service alert-manager-headless -n $NAMESPACE || true
    
    # Delete GreptimeDB resources (if deployed)
    kubectl delete configmap $HELM_RELEASE-greptimedb-standalone -n greptimedb || true
    kubectl delete statefulset $HELM_RELEASE-greptimedb-standalone -n greptimedb || true
    kubectl delete service greptimedb-standalone -n greptimedb || true
fi

# Delete CRDs
echo "Deleting TensorFusion CRDs..."
kubectl delete crd gpunodes.tensor-fusion.ai || true
kubectl delete crd gpus.tensor-fusion.ai || true
kubectl delete crd tensorfusionconnections.tensor-fusion.ai || true
kubectl delete crd tensorfusionclusters.tensor-fusion.ai || true
kubectl delete crd tensorfusionworkloads.tensor-fusion.ai || true
kubectl delete crd workloadprofiles.tensor-fusion.ai || true
kubectl delete crd schedulingconfigtemplates.tensor-fusion.ai || true
kubectl delete crd gpupools.tensor-fusion.ai || true
kubectl delete crd gpunodeclasses.tensor-fusion.ai || true
kubectl delete crd gpuresourcequotas.tensor-fusion.ai || true
kubectl delete crd gpunodeclaims.tensor-fusion.ai || true

# Delete namespace (automatically delete in non-interactive mode)
if [ -t 0 ]; then
    echo -n "Do you want to delete the namespace $NAMESPACE? (y/n): "
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo "Deleting namespace $NAMESPACE..."
        kubectl delete namespace $NAMESPACE || true
    fi
else
    # Non-interactive mode
    echo "Automatically deleting namespace $NAMESPACE in non-interactive mode..."
    kubectl delete namespace $NAMESPACE || true
fi

echo "TensorFusion resource uninstallation completed!"