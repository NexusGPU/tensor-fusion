#!/bin/sh
# Runs inside a gcc container (Job tf-mock-build). Builds the repo's example
# accelerator provider with the three-line rehearsal patch and prints the
# library base64-encoded between markers; mock-gpu.sh turns that into a ConfigMap.
set -eu
P=/w/provider
mkdir -p "$P/example/device_mock" "$P/build"
cd "$P"
cp /src/accelerator.h /src/limiter.h /src/mock_patch.h .
cp /src/accelerator.c example/
cp /src/driver_mock.c /src/driver_mock.h example/device_mock/

A=example/accelerator.c
patch_line() { # patch_line <grep pattern> <sed expression>
  grep -q "$1" "$A" || { echo "PATCH FAILED: '$1' not found in accelerator.c (the example provider changed; update e2e/mock/build.sh)" >&2; exit 3; }
  sed -i "$2" "$A"
}
patch_line '"example-device-%d", deviceIndex' 's/"example-device-%d", deviceIndex/"%s-gpu-%d", tf_mock_host(), deviceIndex/'
patch_line '\*deviceCount = 4;' 's/\*deviceCount = 4;/*deviceCount = tf_mock_count();/'
patch_line 'size_t actualCount = 4;' 's/size_t actualCount = 4;/size_t actualCount = tf_mock_count();/'

CFLAGS="-Wall -Wextra -std=c11 -fPIC -O2 -D_GNU_SOURCE"
gcc $CFLAGS -Iexample/device_mock -pthread -c -o build/driver_mock.o example/device_mock/driver_mock.c
ar rcs build/libdriver_mock.a build/driver_mock.o
gcc $CFLAGS -include mock_patch.h -I. -Iexample/device_mock -c -o build/accelerator_example.o example/accelerator.c
gcc -shared -o build/libaccelerator_example.so build/accelerator_example.o -Lbuild -ldriver_mock -lpthread
ls -l build/libaccelerator_example.so >&2
echo "-----BEGIN MOCK ACCELERATOR-----"
base64 -w0 build/libaccelerator_example.so
echo
echo "-----END MOCK ACCELERATOR-----"
