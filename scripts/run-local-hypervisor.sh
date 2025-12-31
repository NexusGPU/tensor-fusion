export KUBECONFIG="~/.kube/config-local-studio"
export HYPERVISOR_PORT="8033"
export GPU_NODE_NAME="ubuntu"

make build-provider
make build-hypervisor

./bin/hypervisor -accelerator-lib ./provider/build/libaccelerator_example.so -vendor Stub -isolation-mode shared