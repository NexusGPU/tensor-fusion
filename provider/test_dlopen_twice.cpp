/*
 * Test: Can we dlopen the same library twice in the same process?
 * This reproduces the Phase 1 → Phase 2 failure scenario
 */

#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>

int main() {
    const char* libPath = "/usr/local/dcmi/libdcmi.so";

    std::fprintf(stderr, "=== Test 1: First dlopen ===\n");
    void* handle1 = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (handle1) {
        std::fprintf(stderr, "✅ handle1 = %p\n", handle1);
    } else {
        std::fprintf(stderr, "❌ First dlopen failed: %s\n", dlerror());
        return 1;
    }

    std::fprintf(stderr, "\n=== Test 2: Load symbol from handle1 ===\n");
    typedef int (*dcmi_init_fn)(void);
    auto init_fn = reinterpret_cast<dcmi_init_fn>(dlsym(handle1, "dcmi_init"));
    if (init_fn) {
        std::fprintf(stderr, "✅ dcmi_init symbol loaded: %p\n", (void*)init_fn);
        int ret = init_fn();
        std::fprintf(stderr, "   dcmi_init() returned: %d\n", ret);
    } else {
        std::fprintf(stderr, "❌ Failed to load symbol: %s\n", dlerror());
    }

    std::fprintf(stderr, "\n=== Test 3: Second dlopen (same path, before dlclose) ===\n");
    void* handle2 = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (handle2) {
        std::fprintf(stderr, "✅ handle2 = %p\n", handle2);
        std::fprintf(stderr, "   handle1 == handle2? %s\n", handle1 == handle2 ? "yes" : "no");
    } else {
        std::fprintf(stderr, "❌ Second dlopen failed: %s\n", dlerror());
    }

    std::fprintf(stderr, "\n=== Test 4: dlclose handle2 ===\n");
    if (handle2) {
        dlclose(handle2);
        std::fprintf(stderr, "✅ handle2 closed\n");
    }

    std::fprintf(stderr, "\n=== Test 5: Third dlopen (after one dlclose) ===\n");
    void* handle3 = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (handle3) {
        std::fprintf(stderr, "✅ handle3 = %p\n", handle3);
    } else {
        std::fprintf(stderr, "❌ Third dlopen failed: %s\n", dlerror());
    }

    std::fprintf(stderr, "\n=== Test 6: dlclose handle1 (original) ===\n");
    dlclose(handle1);
    std::fprintf(stderr, "✅ handle1 closed\n");

    std::fprintf(stderr, "\n=== Test 7: Fourth dlopen (after all dlclose) ===\n");
    void* handle4 = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
    if (handle4) {
        std::fprintf(stderr, "✅ handle4 = %p\n", handle4);

        // Try to use it
        auto init_fn2 = reinterpret_cast<dcmi_init_fn>(dlsym(handle4, "dcmi_init"));
        if (init_fn2) {
            std::fprintf(stderr, "   dcmi_init symbol loaded: %p\n", (void*)init_fn2);
            int ret = init_fn2();
            std::fprintf(stderr, "   dcmi_init() returned: %d\n", ret);
        }
    } else {
        std::fprintf(stderr, "❌ Fourth dlopen failed: %s\n", dlerror());
    }

    if (handle3) dlclose(handle3);
    if (handle4) dlclose(handle4);

    std::fprintf(stderr, "\n=== Summary ===\n");
    std::fprintf(stderr, "If all tests passed, then dlopen/dlclose works correctly.\n");
    std::fprintf(stderr, "The Phase 2 failure must be due to a different reason.\n");

    return 0;
}
