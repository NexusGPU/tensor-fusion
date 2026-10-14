# TensorFusion cluster e2e: infrastructure (terraform/) + test suite (suite/).
#
#   make up            EKS + Karpenter + TensorFusion, dead-man switch armed (TTL=3h)
#   make test-smoke    tier smoke-nogpu (no GPU needed)
#   make rehearsal     mock-GPU rehearsal of the Karpenter + core control path
#   make test-<tier>   core | isolation | karpenter-finops | perf-quick | perf-full
#   make down          destroy + leftover sweep
#
# No cloud account at all (Karpenter on fake KWOK nodes in a kind cluster):
#   make kwok-up kwok-test kwok-down   [KWOK_SSH_HOST=<docker host>]
#
# Alibaba Cloud ACK + Karpenter's Alibaba Cloud provider (real T4 nodes):
#   make ack-up                        VPC + ACK cluster + Karpenter + TensorFusion
#   make ack-test TIER=core            any test-<tier> target against that cluster
#   make ack-down                      delete everything + leftover sweep
#
# The suite runs against whatever KUBECONFIG points at; by default the kubeconfig
# of the terraform stack. For a bare-metal cluster: make test-core KUBECONFIG=~/.kube/bm
#
# See README.md (cases, tiers, cost) and GO-LIVE.md (real-GPU run).

SHELL       := /bin/bash
KUBECONFIG  ?= $(CURDIR)/terraform/.kube/config
RUN_ID      ?= $(shell date -u +%Y%m%d-%H%M%S)
RESULTS     ?= $(CURDIR)/results/$(RUN_ID)
GO          ?= go
# -p 2 and nice: the suite must stay cheap to build on a busy laptop.
GOFLAGS_E2E := -tags e2e -mod=vendor -p 2
TEST_BIN    := $(CURDIR)/results/bin/suite.test
TIMEOUT     ?= 90m
FOCUS       ?=
export KUBECONFIG

INFRA_TARGETS := tools preflight quotas plan up extend deadman cost down install tf-install tf-uninstall \
                 mock-install mock-uninstall gpu-dryrun status sweep sweep-delete prices check

.PHONY: help
help:
	@grep -hE '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'
	@echo; echo "infrastructure targets (delegated to terraform/Makefile):"; $(MAKE) --no-print-directory -C terraform help

.PHONY: $(INFRA_TARGETS)
$(INFRA_TARGETS):
	@$(MAKE) --no-print-directory -C terraform $@

.PHONY: build
build: ## compile the suite (go test -c, build tag e2e)
	@mkdir -p $(dir $(TEST_BIN))
	cd .. && nice -n 19 $(GO) test $(GOFLAGS_E2E) -c -o $(TEST_BIN) ./e2e/suite

# run_tier <ginkgo label filter> <gpu mode: auto|real|mock|none>
define run_tier
	@mkdir -p $(RESULTS); \
	echo ">> tier filter: $(1)   gpu mode: $(2)   results: $(RESULTS)"; \
	cd suite && E2E_RUN_ID=$(RUN_ID) E2E_RESULTS_DIR=$(RESULTS) E2E_GPU_MODE=$${E2E_GPU_MODE:-$(2)} nice -n 10 $(TEST_BIN) \
	  -test.timeout $(TIMEOUT) -test.v \
	  -ginkgo.label-filter='$(1)' $(if $(FOCUS),-ginkgo.focus='$(FOCUS)',) \
	  -ginkgo.junit-report=$(RESULTS)/junit.xml -ginkgo.json-report=$(RESULTS)/ginkgo.json \
	  -ginkgo.no-color -ginkgo.seed=1 -ginkgo.timeout=$(TIMEOUT) 2>&1 | tee $(RESULTS)/run.log | grep -E '^(•|\s*\[|  measure|  note|Ran |PASS|FAIL|ok|---|summary:)|STEP' ; \
	rc=$${PIPESTATUS[0]}; echo; cat $(RESULTS)/summary.md 2>/dev/null | head -60; exit $$rc
endef

.PHONY: test-smoke
test-smoke: build ## tier smoke-nogpu: control plane, admission, Karpenter CPU launch + consolidation (~10 min, < $0.05)
	$(call run_tier,smoke-nogpu,none)

.PHONY: test-rehearsal
test-rehearsal: build ## tier rehearsal: Karpenter paths + core scheduling on mock GPUs (needs `make mock-install`)
	$(call run_tier,rehearsal,mock)

.PHONY: rehearsal
rehearsal: mock-install test-rehearsal ## mock-install + test-rehearsal

.PHONY: test-dryrun
test-dryrun: build ## tier dryrun: GPU-only cases proven as far as possible without a GPU
	$(call run_tier,dryrun,none)

.PHONY: test-core
test-core: build ## tier core: documented core features on real GPUs
	$(call run_tier,core,real)

.PHONY: test-isolation
test-isolation: build ## tier isolation: VRAM / compute limit enforcement, neighbours, runtime limit change
	$(call run_tier,isolation,real)

.PHONY: test-karpenter-finops
test-karpenter-finops: build ## tier karpenter-finops: both Karpenter paths, consolidation, spot fallback, cost accounting
	$(call run_tier,karpenter-finops,real)

.PHONY: test-perf-quick
test-perf-quick: build ## tier perf, quick profile (a few minutes per runtime)
	$(call run_tier,perf && perf-quick,real)

.PHONY: test-perf-full
test-perf-full: build ## tier perf, full design (mainly for the bare-metal host)
	$(call run_tier,perf,real)

# --- tier kwok: Karpenter + TensorFusion on fake nodes, no cloud, no GPU ---------------
# kind + KWOK + Karpenter (kwok provider) + the chart + a fake hypervisor; see
# kwok/up.sh and README "KWOK tier". Runs where docker is: here, or on
# KWOK_SSH_HOST (the checkout is copied there; results are copied back).
.PHONY: kwok-up
kwok-up: ## tier kwok: kind cluster with fake GPU nodes (Karpenter KWOK provider) + TensorFusion + fake hypervisor
	@bash kwok/run.sh up

.PHONY: kwok-test
kwok-test: ## tier kwok: Karpenter path A / path B / FinOps cases on fake nodes (control plane only)
	@bash kwok/run.sh test

.PHONY: kwok-down
kwok-down: ## tier kwok: stop the fake hypervisor, delete the kind cluster
	@bash kwok/run.sh down

# --- ACK: Alibaba Cloud ACK + Karpenter (Alibaba Cloud provider), real GPUs ----------------
# ack/up.sh (VPC, cluster, system node pool), ack/install.sh (Karpenter, NodePools),
# ack/install-tf.sh (the chart). Credentials come from the aliyun CLI profile.
ACK_KUBECONFIG := $(CURDIR)/ack/.work/kubeconfig
ACK_TEST_ENV   := KUBECONFIG=$(ACK_KUBECONFIG) E2E_PRICING_FILE=$(CURDIR)/pricing/alibaba-us-east-1.json \
                  E2E_PRICING_REGION=us-east-1 E2E_FINOPS_GPU_TYPE=ecs.gn6i-c4g1.xlarge E2E_PATHB_INSTANCE_TYPES=ecs.gn6i-c8g1.2xlarge E2E_BURST=3 E2E_ISOLATION_MODES=soft E2E_FIXED_IMAGES=1
TIER ?= smoke

.PHONY: ack-up
ack-up: ## ACK: VPC + ACK managed cluster + Karpenter (Alibaba Cloud provider) + TensorFusion
	@bash ack/run.sh up

.PHONY: ack-registry
ack-registry: ## ACK: install or update the in-cluster image cache (also part of ack-up); ARGS="status" | "warm <image>..." | "purge"
	@bash ack/registry.sh $(ARGS)

.PHONY: ack-warm
ack-warm: ## ACK: keep one GPU node warm with the demo images (placeholder pod); ARGS="down" | "status"
	@bash ack/warm.sh $(or $(ARGS),up)

.PHONY: ack-test
ack-test: ## ACK: run a tier against the ACK cluster: make ack-test TIER=core|isolation|karpenter-finops|perf-quick|smoke [FOCUS=..]
	@$(ACK_TEST_ENV) $(MAKE) --no-print-directory test-$(TIER) KUBECONFIG=$(ACK_KUBECONFIG)

.PHONY: ack-down
ack-down: ## ACK: delete the cluster and everything around it, then list leftovers
	@bash ack/run.sh down

.PHONY: ack-sweep
ack-sweep: ## ACK: list what of the stack still exists (exit 1 when something is left)
	@bash ack/down.sh sweep

.PHONY: test
test: build ## any label filter: make test FILTER='core && !disruptive' [FOCUS=ISO-0]
	$(call run_tier,$(FILTER),auto)

.PHONY: list
list: build ## list every case with its tiers (dry run, no cluster access)
	@cd suite && KUBECONFIG=/dev/null $(TEST_BIN) -ginkgo.dry-run -ginkgo.v -ginkgo.no-color 2>/dev/null \
	  | grep -E ' [A-Z]+(-[A-Z0-9]+)+ .*\[' | sed -E 's/^.* ([A-Z]+(-[A-Z0-9]+)+) /\1  /' | sort

.PHONY: report
report: ## regenerate the comparison report from results/*/summary.json (RUNS="dir1 dir2")
	@python3 tools/report.py $(if $(RUNS),$(RUNS),$(RESULTS))
