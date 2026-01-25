# TensorFusion Components with Persistent Storage

## Overview
This document lists all TensorFusion components that require or can use persistent storage.

## Components Requiring Storage

### 1. **GreptimeDB** (Metrics Database)

**Purpose**: Time-series database for GPU metrics, utilization tracking, and autoscaling.

**Storage Configuration**:
```yaml
greptime:
  installStandalone: true
  persistence:
    enabled: true
    useNFS: true
    size: 20Gi
    storageClassName: "nfs-greptime"
    hostPath: /home/z1_ossci/greptimedb
```

**Storage Type**: StatefulSet with volumeClaimTemplates

**Data Stored**:
- GPU utilization metrics
- Worker usage statistics
- Device metrics (temperature, power, memory)
- Historical performance data

**Storage Requirements**:
- **Minimum**: 20Gi
- **Recommended**: 50Gi for production
- **Growth rate**: 1-5GB per week (depends on workload)

**Failure Impact if No Storage**:
- ❌ Pod stuck in `Pending` state
- ❌ Controller crashes: "failed to create metrics provider: connection refused"
- ❌ Autoscaler disabled
- ✅ Core GPU functionality still works (discovery, scheduling, allocation)

**Status in values-amd.yaml**: ✅ **ENABLED** with NFS hostPath

---

### 2. **AlertManager** (Alert Management)

**Purpose**: Handles alerts and notifications for GPU-related events (failures, resource exhaustion, etc.).

**Storage Configuration**:
```yaml
alert:
  enabled: true
  persistence:
    enabled: false  # Currently disabled
    # storageClass: "gp3"
    # size: 10Gi
```

**Storage Type**: StatefulSet with volumeClaimTemplates (when enabled)

**Data Stored**:
- Alert state and history
- Silenced alerts
- Notification group state

**Storage Requirements**:
- **Minimum**: 1Gi
- **Recommended**: 10Gi for production
- **Growth rate**: Minimal (<100MB typically)

**Fallback Behavior (No Storage)**:
Uses `emptyDir` volume (line 54-56 in alert-manager.yaml):
```yaml
- name: storage
  emptyDir: {}
```

**Failure Impact if No Storage**:
- ⚠️ Alert state lost on pod restart
- ⚠️ Silenced alerts reset
- ✅ AlertManager still functions (no alerts stuck)
- ✅ GPU operations unaffected

**Status in values-amd.yaml**: ✅ **DISABLED** (uses emptyDir) - OK for testing

---

## Storage Summary Table

| Component | Required? | Default Size | AMD Config | Impact if Missing |
|-----------|-----------|--------------|------------|-------------------|
| **GreptimeDB** | Optional | 20Gi | ✅ Enabled (NFS) | Controller crashes, no metrics |
| **AlertManager** | Optional | 10Gi | ❌ Disabled (emptyDir) | Alert state lost on restart |

## Current AMD Configuration

### What's Configured ✅
```yaml
# values-amd.yaml
greptime:
  installStandalone: true
  persistence:
    useNFS: true
    hostPath: /home/z1_ossci/greptimedb
    size: 20Gi
```

### What's Using Ephemeral Storage ✅
```yaml
# values-amd.yaml (inherited from values.yaml)
alert:
  enabled: true
  persistence:
    enabled: false  # Uses emptyDir
```

## Do You Need to Configure AlertManager Storage?

### **For Testing**: ❌ **No, keep persistence disabled**
- Alert state isn't critical
- `emptyDir` is sufficient
- Simplifies deployment
- No storage dependency

### **For Production**: ✅ **Yes, consider enabling**

To enable AlertManager persistence, add to `values-amd.yaml`:

```yaml
alert:
  persistence:
    enabled: true
    size: 10Gi
    storageClassName: "nfs-alertmanager"
    # If using NFS hostPath approach like GreptimeDB:
    useNFS: true
    hostPath: /home/z1_ossci/alertmanager
```

**Benefits**:
- Alert state survives pod restarts
- Silenced alerts persist
- Better production reliability

**Requirements**:
- Same NFS setup as GreptimeDB
- Create directory: `sudo mkdir -p /home/z1_ossci/alertmanager`

## Other Components (No Storage Needed)

### ✅ No Persistent Storage Required:

1. **Controller (Deployment)**
   - Stateless
   - No data persistence needed

2. **Hypervisor (DaemonSet)**
   - Runs on each GPU node
   - Discovers local GPUs
   - No persistent state

3. **Worker Pods**
   - Ephemeral workload containers
   - State managed by controller

4. **Vector Agents (DaemonSet)**
   - Metrics collectors
   - Stateless forwarders

5. **Scheduler**
   - Stateless
   - Uses Kubernetes API for state

6. **Admission Webhooks**
   - Stateless
   - Pod mutation only

## Storage Setup Scripts

### For GreptimeDB (Required for Metrics)
```bash
./scripts/setup-greptime-nfs.sh
```

### For AlertManager (Optional)
Create manually if needed:
```bash
# On GPU node or NFS server
sudo mkdir -p /home/z1_ossci/alertmanager
sudo chmod 777 /home/z1_ossci/alertmanager
```

Then update `values-amd.yaml` and reinstall.

## Troubleshooting Storage Issues

### Check What Needs Storage
```bash
# Find all StatefulSets
kubectl get statefulsets -A

# Check PVCs across all namespaces
kubectl get pvc -A

# Find pods with unbound PVCs
kubectl get pods -A | grep Pending
kubectl describe pod <pod-name> -n <namespace>
```

### Check NFS Availability
```bash
# On GPU nodes
df -h | grep z1_ossci
mount | grep z1_ossci

# Test NFS write
sudo touch /home/z1_ossci/test-write
ls -la /home/z1_ossci/test-write
sudo rm /home/z1_ossci/test-write
```

### Storage-Related Errors

**Error**: `pod has unbound immediate PersistentVolumeClaims`
- **Cause**: No PV available or no storage provisioner
- **Fix**: Create PV manually or install storage provisioner

**Error**: `failed to create metrics provider: connection refused`
- **Cause**: GreptimeDB not running (storage issue)
- **Fix**: Check GreptimeDB pod status and PVC binding

**Error**: `chown: changing ownership of '/data': Permission denied`
- **Cause**: NFS directory permissions
- **Fix**: `sudo chmod 777 /home/z1_ossci/<component-name>`

## Recommendations

### For AMD GPU Testing (Current Setup) ✅
```yaml
greptime:
  persistence:
    enabled: true
    useNFS: true  # ✅ Configured

alert:
  persistence:
    enabled: false  # ✅ OK for testing (uses emptyDir)
```

### For Production Deployment 🚀
```yaml
greptime:
  persistence:
    enabled: true
    useNFS: true
    size: 50Gi  # Increase from 20Gi

alert:
  persistence:
    enabled: true  # Enable for production
    useNFS: true
    size: 10Gi
```

## Summary

**Current AMD Configuration**: ✅ **Properly Configured**

- ✅ GreptimeDB has NFS storage (required for metrics)
- ✅ AlertManager uses emptyDir (acceptable for testing)
- ✅ No other components need persistent storage
- ✅ Setup is production-ready for GPU testing

**Action Required**: 🎯 **None for testing!**

Your NFS storage at `/home/z1_ossci/greptimedb` is the only storage needed for AMD GPU testing with metrics enabled.
