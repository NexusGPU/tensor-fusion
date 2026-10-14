# smprobe

Counts the streaming multiprocessors (SMs) of a GPU that a CUDA workload in the current container can execute on.

TensorFusion hard isolation limits compute in space: the workload's CUDA context gets a fixed share of the GPU's SMs. `nvidia-smi` cannot show this, because its utilization is the fraction of time during which any kernel ran. A pod confined to 18 of 40 SMs still reads close to 100% there while it is busy. `smprobe` shows the SM count directly.

## Run it

Copy the binary into a pod and run it. It needs glibc 2.17 or later and `libcuda.so.1`, nothing else.

```bash
kubectl cp smprobe <pod>:/tmp/smprobe -c <container>
kubectl exec <pod> -c <container> -- /tmp/smprobe
```

Output in a pod with hard isolation at 47% on a Tesla T4:

```text
smprobe 0.1.0
device 0: Tesla T4, compute capability 7.5
SMs of the whole GPU: 40 (built-in table)   SMs the driver reports to this context: 18
probe: 4 launches x 4608 blocks x 64 threads, 200 us per block, kernel image cubin sm_70
SMs used: 18 of 40 (45%)   SM ids 0..17   blocks per SM min 1024 max 1024   35.0 ms
%nsmid (upper bound of SM ids seen by the kernel): 18
map (one character per SM id, # = blocks ran there, . = none):
  ########## ########.. .......... ..........
cuCtxGetExecAffinity(SM_COUNT): not available here (CUDA error 500)
```

The same GPU from a pod with soft isolation, or from any context without an SM limit, gives `SMs used: 40 of 40 (100%)`.

| Option | Meaning |
| --- | --- |
| `-q` | Print only the `SMs used` line |
| `-n N`, `-i SEC` | Take N samples, SEC seconds apart |
| `-v` | Also print how many blocks ran on each SM |
| `-T N` | SM count of the whole GPU, for a model the built-in table does not know |
| `-b`, `-t`, `-s`, `-r` | Blocks per launch, threads per block, microseconds per block, launches per sample |
| `-d N` | Device index |
| `-l PATH` | CUDA driver library to load instead of `libcuda.so.1` |
| `-A` | Skip the `cuCtxGetExecAffinity` query |

## How it works

1. The program loads `libcuda.so.1` with `dlopen()` and uses only the CUDA driver API. In a hard-isolated pod that library is TensorFusion's forwarding client, so the probe runs in the worker's limited context, like the application's own kernels.
2. It launches a small kernel with many thread blocks (256 per SM by default). The GPU spreads the blocks of a launch over every SM the context may use.
3. Thread 0 of each block reads the PTX special register `%smid`, the id of the SM the block runs on, and adds 1 to that SM's counter with an atomic add. Every thread then spins for 200 microseconds, so that blocks overlap and fill all SMs.
4. The host copies the counters back. The number of counters that are not zero is the number of SMs used.

The kernel is written in PTX (`smprobe.ptx`) and embedded twice: as precompiled cubins, one per GPU generation, and as PTX text. The forwarding client accepts a cubin but not PTX text, so the cubin is tried first. The PTX text is the fallback for a GPU generation without a cubin; the native driver compiles it itself.

## Reading the result

- **The total comes from a table.** A context with an SM limit makes the driver report the limited count as the device's SM count (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` and `%nsmid` both say 18 above, and so does `torch.cuda.get_device_properties(0).multi_processor_count`). The size of the whole GPU cannot be read from inside such a context, so `smprobe` takes it from a table of common models. Pass `-T` for a model the table does not know; without it the driver's count is used and the result reads as 100%.
- **SM ids are the context's own numbering.** A limited context sees ids `0..N-1`. Two limited contexts on one GPU both report `0..17`; the ids do not say which physical SMs each one has.
- **It measures the limit, not the load.** The probe saturates the context for a few milliseconds and reports where its own blocks ran. The count is the same whether the application is idle or busy, and a busy neighbour on the same GPU does not change it. It does not show how many SMs the application's kernels occupy at a given moment.
- **Where to run it.** Run it in the application container to see what the application gets. Run in the worker container, it loads the native driver and creates its own context without a limit, which reports the whole GPU: a useful control on the same device.
- **`cuCtxGetExecAffinity`** returns the SM count of a context on the native driver. The forwarding client does not forward it, so the line reads "not available" in a hard-isolated pod.
- With the default options one sample takes 10 to 40 ms of GPU time and about 4 KiB of GPU memory in addition to the context.

## Build

```bash
./build.sh            # needs gcc; uses the checked-in kernels.h
./build.sh kernels    # also regenerates kernels.h from smprobe.ptx; needs ptxas
```

`ptxas` is part of the CUDA toolkit. Without the toolkit, take it from the PyPI wheel `nvidia-cuda-nvcc-cu12` (unzip the wheel and set `PTXAS` to `nvidia/cuda_nvcc/bin/ptxas`). Build on an old distribution such as `ubuntu:20.04` so that the binary runs on newer ones.

| File | Content |
| --- | --- |
| `smprobe.c` | The program |
| `smprobe.ptx` | The probe kernel |
| `kernels.h` | Generated: the PTX text and the cubins for sm_50 to sm_120 |
| `build.sh` | Build script |
| `publish.sh` | Uploads the built binary to an OSS bucket as public-read objects: `SMPROBE_BUCKET=<bucket> ./publish.sh` prints the download URL |
