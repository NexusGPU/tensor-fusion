# Code level benchmark 
# go test -bench=BenchmarkPodWebhookQPS -benchmem -v ./internal/webhook/v1/

# go test -bench=BenchmarkGPUFitPlugin -benchmem ./test/sched/ --benchtime=3s
# go test -bench=BenchmarkScheduler    -benchmem ./test/sched/ --benchtime=5s

# Real world benchmark for Mutating Webhook
cat > /tmp/webhook-body.json << 'EOF'
{
  "kind": "AdmissionReview",
  "apiVersion": "admission.k8s.io/v1",
  "request": {
    "uid": "test-uid-001",
    "kind": {"group": "", "version": "v1", "kind": "Pod"},
    "resource": {"group": "", "version": "v1", "resource": "pods"},
    "name": "test-pod",
    "namespace": "default",
    "operation": "CREATE",
    "object": {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {
        "name": "test-pod",
        "namespace": "default",
        "labels": {
          "app": "test-pod",
          "tensor-fusion.ai/enabled": "true"
        },
        "annotations": {
          "tensor-fusion.ai/tflops-request": "1",
          "tensor-fusion.ai/vram-request": "1Gi",
          "tensor-fusion.ai/tflops-limit": "2",
          "tensor-fusion.ai/vram-limit": "2Gi",
          "tensor-fusion.ai/inject-container": "test-container",
          "tensor-fusion.ai/is-local-gpu": "true"
        }
      },
      "spec": {
        "containers": [{
          "name": "test-container",
          "image": "test-container:1.21"
        }]
      }
    }
  }
}
EOF

export TF_WEBHOOK_IP=10.42.1.80

# validate the request
curl -k -X POST -H "Content-Type: application/json" -d @/tmp/webhook-body.json https://$TF_WEBHOOK_IP:9443/mutate-v1-pod

# wget https://github.com/codesenberg/bombardier/releases/download/v2.0.2/bombardier-linux-amd64
# sudo chmod +x bombardier-linux-amd64 && sudo mv bombardier-linux-amd64 /usr/local/bin/bombardier

# -c concurrency -d duration -l show latency
bombardier  -c 10 -d 15s -l --insecure -H "Content-Type: application/json" -f /tmp/webhook-body.json -m POST https://$TF_WEBHOOK_IP:9443/mutate-v1-pod
bombardier  -c 100 -d 15s -l --insecure -H "Content-Type: application/json" -f /tmp/webhook-body.json -m POST https://$TF_WEBHOOK_IP:9443/mutate-v1-pod


# ======== Test Result on Intel i9 14900K, 4 core 4G mem Tensor Fusion Controller Pod ========

# ubuntu@ubuntu:~$ bombardier  -c 10 -d 15s -l --insecure -H "Content-Type: application/json" -f /tmp/webhook-body.json -m POST https://$TF_WEBHOOK_IP:9443/mutate-v1-pod
# Bombarding https://<TF_WEBHOOK>/mutate-v1-pod for 15s using 10 connection(s)
# [=============================================================================================================================================================================] 15s
# Done!
# Statistics        Avg      Stdev        Max
#   Reqs/sec      8485.70    4979.37   17849.77
#   Latency        1.18ms     3.82ms    46.83ms
#   Latency Distribution
#      50%   606.00us
#      75%   842.00us
#      90%     1.24ms
#      95%     1.65ms
#      99%    33.45ms
#   HTTP codes:
#     1xx - 0, 2xx - 127654, 3xx - 0, 4xx - 0, 5xx - 0
#     others - 0
#   Throughput:    40.01MB/s
# ubuntu@ubuntu:~$ bombardier  -c 100 -d 15s -l --insecure -H "Content-Type: application/json" -f /tmp/webhook-body.json -m POST https://$TF_WEBHOOK_IP:9443/mutate-v1-pod
# Bombarding https://<TF_WEBHOOK>/mutate-v1-pod for 15s using 100 connection(s)
# [=============================================================================================================================================================================] 15s
# Done!
# Statistics        Avg      Stdev        Max
#   Reqs/sec      8457.47    6231.81   22102.60
#   Latency       11.78ms    18.79ms   209.19ms
#   Latency Distribution
#      50%     3.30ms
#      75%     8.30ms
#      90%    51.53ms
#      95%    66.19ms
#      99%    89.08ms
#   HTTP codes:
#     1xx - 0, 2xx - 127308, 3xx - 0, 4xx - 0, 5xx - 0
#     others - 0
#   Throughput:    39.94MB/s