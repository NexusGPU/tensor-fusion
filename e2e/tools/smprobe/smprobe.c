/*
 * smprobe - count the streaming multiprocessors (SMs) a CUDA workload in this
 * container can execute on.
 *
 * It launches a tiny kernel with many thread blocks. Each block reads the PTX
 * special register %smid (the id of the SM it runs on) and increments a
 * per-SM counter. The number of counters that are not zero is the number of
 * SMs the context was given.
 *
 * The program uses only the CUDA driver API, loaded with dlopen() from
 * libcuda.so.1. The kernel is embedded as precompiled cubins (one per GPU
 * generation) with the PTX text as a fallback, so nothing from the CUDA
 * toolkit is needed at run time.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "kernels.h"

#define SMPROBE_VERSION "0.1.0"

/* Minimal driver API declarations (from the public cuda.h). */
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef unsigned long long CUdeviceptr;

#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE 13
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_EXEC_AFFINITY_TYPE_SM_COUNT 0

typedef struct {
  int type;
  unsigned int sm_count;
  unsigned int pad[6];
} exec_affinity_t;

static CUresult (*cuInit)(unsigned int);
static CUresult (*cuDeviceGetCount)(int *);
static CUresult (*cuDeviceGet)(CUdevice *, int);
static CUresult (*cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
static CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
static CUresult (*cuCtxSetCurrent)(CUcontext);
static CUresult (*cuCtxSynchronize)(void);
static CUresult (*cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*cuModuleUnload)(CUmodule);
static CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*cuMemFree)(CUdeviceptr);
static CUresult (*cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
static CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  CUstream, void **, void **);
static CUresult (*cuCtxGetExecAffinity)(exec_affinity_t *, int); /* optional */

static void *sym(void *lib, const char *name, int required) {
  void *p = dlsym(lib, name);
  if (!p && required) {
    fprintf(stderr, "smprobe: %s not found in the CUDA driver library\n", name);
    exit(2);
  }
  return p;
}

static void *load_driver(const char *path) {
  const char *names[] = {path, "libcuda.so.1", "libcuda.so", NULL};
  void *lib = NULL;
  for (int i = path ? 0 : 1; names[i] && !lib; i++) {
    lib = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
    if (path) break;
  }
  if (!lib) {
    fprintf(stderr, "smprobe: cannot load the CUDA driver library: %s\n", dlerror());
    exit(2);
  }
  cuInit = sym(lib, "cuInit", 1);
  cuDeviceGetCount = sym(lib, "cuDeviceGetCount", 1);
  cuDeviceGet = sym(lib, "cuDeviceGet", 1);
  cuDeviceGetName = sym(lib, "cuDeviceGetName", 1);
  cuDeviceGetAttribute = sym(lib, "cuDeviceGetAttribute", 1);
  cuDevicePrimaryCtxRetain = sym(lib, "cuDevicePrimaryCtxRetain", 1);
  cuDevicePrimaryCtxRelease = sym(lib, "cuDevicePrimaryCtxRelease_v2", 0);
  if (!cuDevicePrimaryCtxRelease) cuDevicePrimaryCtxRelease = sym(lib, "cuDevicePrimaryCtxRelease", 1);
  cuCtxSetCurrent = sym(lib, "cuCtxSetCurrent", 1);
  cuCtxSynchronize = sym(lib, "cuCtxSynchronize", 1);
  cuModuleLoadData = sym(lib, "cuModuleLoadData", 1);
  cuModuleGetFunction = sym(lib, "cuModuleGetFunction", 1);
  cuModuleUnload = sym(lib, "cuModuleUnload", 1);
  cuMemAlloc = sym(lib, "cuMemAlloc_v2", 1);
  cuMemFree = sym(lib, "cuMemFree_v2", 1);
  cuMemsetD32 = sym(lib, "cuMemsetD32_v2", 1);
  cuMemcpyDtoH = sym(lib, "cuMemcpyDtoH_v2", 1);
  cuLaunchKernel = sym(lib, "cuLaunchKernel", 1);
  cuCtxGetExecAffinity = sym(lib, "cuCtxGetExecAffinity", 0);
  return lib;
}

static void check(CUresult r, const char *what) {
  if (r != 0) {
    fprintf(stderr, "smprobe: %s failed, CUDA error %d\n", what, r);
    exit(1);
  }
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void usage(void) {
  puts("smprobe " SMPROBE_VERSION " - count the SMs a CUDA workload in this container can execute on\n"
       "\n"
       "usage: smprobe [options]\n"
       "  -d N     device index (default 0)\n"
       "  -b N     thread blocks per launch (default 256 per SM)\n"
       "  -t N     threads per block (default 64)\n"
       "  -s US    time each block stays on its SM, in microseconds (default 200)\n"
       "  -r N     launches per sample (default 4)\n"
       "  -n N     samples (default 1)\n"
       "  -i SEC   seconds between samples (default 1)\n"
       "  -l PATH  CUDA driver library (default libcuda.so.1)\n"
       "  -T N     SM count of the whole GPU, when the built-in table does not know the model\n"
       "  -v       also print the number of blocks that ran on each SM\n"
       "  -A       skip the cuCtxGetExecAffinity query\n"
       "  -q       print only the summary line of each sample\n"
       "  -h       this help");
}

#define NSLOTS 1024u

/* Physical SM counts of common GPUs. A context with an SM limit makes the
 * driver report the limited count as the device's SM count, so the size of
 * the whole GPU cannot be read from inside such a context. */
static const struct {
  const char *token;
  int sms;
} known_gpus[] = {
    {"T4", 40},   {"V100", 80},    {"A100", 108},   {"A10", 72},     {"A10G", 72},    {"L4", 58},
    {"A40", 84},  {"L40", 142},    {"L40S", 142},   {"3090 Ti", 84}, {"3090", 82},    {"4090", 128},
    {"A6000", 84},
};

/* Whether `token` occurs in `name` as a whole word. */
static int has_word(const char *name, const char *token) {
  size_t n = strlen(token);
  for (const char *p = strstr(name, token); p; p = strstr(p + 1, token)) {
    int left = p == name || !isalnum((unsigned char)p[-1]);
    int right = !isalnum((unsigned char)p[n]);
    if (left && right) return 1;
  }
  return 0;
}

static int physical_sms(const char *name) {
  for (size_t i = 0; i < sizeof(known_gpus) / sizeof(known_gpus[0]); i++)
    if (has_word(name, known_gpus[i].token)) return known_gpus[i].sms;
  return 0;
}

int main(int argc, char **argv) {
  int dev_index = 0, blocks = 0, threads = 64, spin_us = 200, rounds = 4, samples = 1, quiet = 0, affinity = 1, verbose = 0, phys = 0;
  double interval = 1.0;
  const char *lib_path = getenv("SMPROBE_LIBCUDA");
  int c;
  while ((c = getopt(argc, argv, "d:b:t:s:r:n:i:l:T:vAqh")) != -1) {
    switch (c) {
      case 'd': dev_index = atoi(optarg); break;
      case 'b': blocks = atoi(optarg); break;
      case 't': threads = atoi(optarg); break;
      case 's': spin_us = atoi(optarg); break;
      case 'r': rounds = atoi(optarg); break;
      case 'n': samples = atoi(optarg); break;
      case 'i': interval = atof(optarg); break;
      case 'l': lib_path = optarg; break;
      case 'T': phys = atoi(optarg); break;
      case 'v': verbose = 1; break;
      case 'A': affinity = 0; break;
      case 'q': quiet = 1; break;
      case 'h': usage(); return 0;
      default: usage(); return 2;
    }
  }
  if (threads < 1 || threads > 1024 || rounds < 1 || samples < 1 || spin_us < 0 || blocks < 0) {
    usage();
    return 2;
  }

  load_driver(lib_path);
  check(cuInit(0), "cuInit");

  int count = 0;
  check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (dev_index < 0 || dev_index >= count) {
    fprintf(stderr, "smprobe: device %d does not exist (%d visible)\n", dev_index, count);
    return 1;
  }
  CUdevice dev;
  check(cuDeviceGet(&dev, dev_index), "cuDeviceGet");

  char name[128] = "";
  int sm_total = 0, major = 0, minor = 0, clock_khz = 0;
  check(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
  check(cuDeviceGetAttribute(&sm_total, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "cuDeviceGetAttribute(SM count)");
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute(major)");
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute(minor)");
  if (cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev) != 0 || clock_khz <= 0) clock_khz = 1000000;
  if (blocks == 0) blocks = 256 * (sm_total > 0 ? sm_total : 64);
  const char *phys_src = "-T";
  if (phys <= 0) {
    phys = physical_sms(name);
    phys_src = "built-in table";
  }
  if (phys < sm_total) {
    /* Unknown model, or a table entry that is wrong for this variant. */
    phys = sm_total;
    phys_src = "driver";
  }

  CUcontext ctx = NULL;
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  /* Prefer the cubin built for this GPU generation: the newest one of the same
   * major version that is not newer than the device. Fall back to PTX, which
   * the driver compiles itself (a forwarding client may not accept PTX). */
  CUmodule mod = NULL;
  const char *image = "none";
  const struct smprobe_cubin *best = NULL;
  for (unsigned i = 0; i < smprobe_cubin_count; i++) {
    const struct smprobe_cubin *k = &smprobe_cubins[i];
    if (k->major == major && k->minor <= minor && (!best || k->minor > best->minor)) best = k;
  }
  CUresult load = -1;
  if (best) {
    load = cuModuleLoadData(&mod, best->data);
    if (load == 0) image = best->name;
  }
  if (load != 0) {
    CUresult r = cuModuleLoadData(&mod, smprobe_ptx);
    if (r != 0) {
      fprintf(stderr, "smprobe: cannot load the probe kernel for compute capability %d.%d (cubin: %d, PTX: %d)\n", major,
              minor, load, r);
      return 1;
    }
    image = "PTX";
  }
  CUfunction fn;
  check(cuModuleGetFunction(&fn, mod, "smprobe"), "cuModuleGetFunction");

  size_t bytes = (NSLOTS + 2) * sizeof(uint32_t);
  CUdeviceptr dbuf = 0;
  check(cuMemAlloc(&dbuf, bytes), "cuMemAlloc");
  uint32_t *hist = calloc(NSLOTS + 2, sizeof(uint32_t));

  unsigned int nslots = NSLOTS;
  unsigned long long ticks = (unsigned long long)spin_us * (unsigned long long)clock_khz / 1000ull;
  void *args[] = {&dbuf, &nslots, &ticks};

  if (!quiet) {
    printf("smprobe %s\n", SMPROBE_VERSION);
    printf("device %d: %s, compute capability %d.%d\n", dev_index, name, major, minor);
    printf("SMs of the whole GPU: %d (%s)   SMs the driver reports to this context: %d\n", phys, phys_src, sm_total);
    printf("probe: %d launches x %d blocks x %d threads, %d us per block, kernel image %s\n", rounds, blocks, threads,
           spin_us, image);
  }

  for (int s = 0; s < samples; s++) {
    if (s > 0) usleep((useconds_t)(interval * 1e6));
    check(cuMemsetD32(dbuf, 0, NSLOTS + 2), "cuMemsetD32");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    double t0 = now_ms();
    for (int r = 0; r < rounds; r++)
      check(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL), "cuLaunchKernel");
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    double elapsed = now_ms() - t0;
    check(cuMemcpyDtoH(hist, dbuf, bytes), "cuMemcpyDtoH");

    unsigned used = 0, min = 0, max = 0, lo = 0, hi = 0;
    unsigned long long total = 0;
    for (unsigned i = 0; i < NSLOTS; i++) {
      if (!hist[i]) continue;
      if (!used) lo = i, min = hist[i];
      hi = i;
      used++;
      total += hist[i];
      if (hist[i] < min) min = hist[i];
      if (hist[i] > max) max = hist[i];
    }
    unsigned nsmid = hist[NSLOTS], overflow = hist[NSLOTS + 1];
    int denom = phys > 0 ? phys : (int)nsmid;

    printf("SMs used: %u of %d (%.0f%%)   SM ids %u..%u   blocks per SM min %u max %u   %.1f ms\n", used, denom,
           denom ? 100.0 * used / denom : 0.0, lo, hi, min, max, elapsed);
    if (overflow) printf("warning: %u blocks ran on SM ids >= %u and were not counted\n", overflow, NSLOTS);
    if (total + overflow != (unsigned long long)blocks * rounds)
      printf("warning: %llu blocks reported, %llu launched\n", total + overflow, (unsigned long long)blocks * rounds);

    if (!quiet) {
      printf("%%nsmid (upper bound of SM ids seen by the kernel): %u\n", nsmid);
      unsigned top = hi + 1 > (unsigned)denom ? hi + 1 : (unsigned)denom;
      printf("map (one character per SM id, # = blocks ran there, . = none):\n  ");
      for (unsigned i = 0; i < top; i++) {
        putchar(hist[i] ? '#' : '.');
        if (i % 10 == 9 && i + 1 < top) putchar(' ');
      }
      putchar('\n');
      if (verbose) printf("blocks per SM id:\n");
      for (unsigned i = 0; verbose && i < top; i++) {
        if (!hist[i]) continue;
        int bar = max ? (int)(40.0 * hist[i] / max + 0.5) : 0;
        printf("  %4u %8u ", i, hist[i]);
        for (int b = 0; b < bar; b++) putchar('|');
        putchar('\n');
      }
    }
    fflush(stdout);
  }

  if (affinity && !quiet) {
    if (!cuCtxGetExecAffinity) {
      printf("cuCtxGetExecAffinity: not exported by this driver library\n");
    } else {
      exec_affinity_t a;
      memset(&a, 0, sizeof(a));
      CUresult r = cuCtxGetExecAffinity(&a, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
      if (r == 0)
        printf("cuCtxGetExecAffinity(SM_COUNT): %u\n", a.sm_count);
      else
        printf("cuCtxGetExecAffinity(SM_COUNT): not available here (CUDA error %d)\n", r);
    }
  }

  cuMemFree(dbuf);
  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  free(hist);
  return 0;
}
