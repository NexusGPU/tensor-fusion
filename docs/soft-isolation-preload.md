# Local soft isolation and application preload libraries

Local soft isolation loads the vendor limiter through `/etc/ld.so.preload`.
An entrypoint can therefore set `LD_PRELOAD` to jemalloc or another application
library without removing the limiter from subsequently executed processes.
TensorFusion preserves the application's `LD_PRELOAD` environment variable.

After the middleware init container copies the limiter, one init container per
injected application container runs in that application's image. It reads the
original `/etc/ld.so.preload`, including an explicitly mounted configuration,
and appends the vendor limiter path. The application receives the merged file
as a read-only subPath mount. Each container has its own generated file; the
original configuration volume is not modified.

This path requires a glibc-based application image with `/bin/sh`, `cat`, and
`chmod`. The init container inherits the application's image pull policy,
security context, environment, working directory, and volume mounts. Its output
is an emptyDir, so a read-only image filesystem is supported. The preload file
is a startup snapshot, not a live merge of subsequent ConfigMap changes.
Existing Pods must be recreated to receive the new injection.

To verify a deployed application, inspect `/etc/ld.so.preload` and the actual
application process's `/proc/<pid>/maps`: both the vendor limiter and the
application preload library should be mapped. Changing or clearing
`LD_PRELOAD` alone no longer disables the limiter; this must be accounted for
when diagnosing NVML or `nvidia-smi` problems.

This operator change does not change limiter initialization or the NVIDIA SMI
wrapper in vgpu-provider. A hanging `nvidia-smi` still requires separate runtime
diagnosis; file preloading is not evidence that the hang has been resolved.
