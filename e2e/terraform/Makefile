# Throwaway EKS + Karpenter stack for TensorFusion e2e.
#
#   make up [TTL=3h]      infra + Karpenter + TensorFusion, dead-man switch armed for TTL
#   make extend TTL=2h    move the dead-man deadline to now + TTL
#   make cost             spend to date (Cost Explorer) + live estimate
#   make down             destroy everything, then prove nothing is left (sweep)
#
# Everything else is optional convenience. See README.md.

SHELL        := /bin/bash
# Pinned terraform, installed by `make tools` into .tools/ (git-ignored). No brew, no sudo.
TF_VERSION   ?= 1.16.5
TF           ?= $(CURDIR)/.tools/terraform
TF_ARGS      ?=
TTL          ?= 3h
NAME         ?= tf-e2e
REGION       ?= us-west-2
KUBECONFIG   := $(CURDIR)/.kube/config
LEDGER       := $(CURDIR)/../results/infra-ledger.tsv
export KUBECONFIG

# Exports TF_E2E_* from `terraform output install_env` for the helper scripts.
define load_env
	set -euo pipefail; \
	eval "$$($(TF) output -json install_env | jq -r 'to_entries[] | "export \(.key)=\(.value|@sh)"')"
endef

# Environment for the scripts that must also work without terraform state.
define base_env
	TF_E2E_CLUSTER=$(NAME) TF_E2E_REGION=$(REGION) TF_E2E_AWS_PROFILE=$${AWS_PROFILE:-}
endef

# ledger <event> <seconds>
define ledger
	mkdir -p $(dir $(LEDGER)); printf '%s\t%s\t%s\tseconds=%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(NAME)" "$(1)" "$(2)" >> $(LEDGER)
endef

.PHONY: help
help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

$(TF):
	@set -euo pipefail; mkdir -p .tools/dl; cd .tools/dl; \
	os=$$(uname -s | tr A-Z a-z); arch=$$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/'); \
	zip=terraform_$(TF_VERSION)_$${os}_$${arch}.zip; \
	echo ">> downloading terraform $(TF_VERSION) ($${os}/$${arch})"; \
	curl -fsSLO https://releases.hashicorp.com/terraform/$(TF_VERSION)/$$zip; \
	curl -fsSLO https://releases.hashicorp.com/terraform/$(TF_VERSION)/terraform_$(TF_VERSION)_SHA256SUMS; \
	grep " $$zip$$" terraform_$(TF_VERSION)_SHA256SUMS | shasum -a 256 -c -; \
	unzip -o -q $$zip terraform -d ..; cd ..; rm -rf dl

.PHONY: tools
tools: $(TF) ## install the pinned terraform into .tools/ (checksum verified)
	@$(TF) version | head -1

.PHONY: preflight
preflight: $(TF) ## check local tools and AWS identity (no changes)
	@for b in aws kubectl helm jq bc; do command -v $$b >/dev/null || { echo "missing: $$b"; exit 1; }; done
	@aws sts get-caller-identity --query 'Arn' --output text | sed -E 's/[0-9]{12}/<account>/' \
	  || { echo "AWS credentials are not usable (expired SSO session? run: aws sso login)"; exit 1; }

.PHONY: quotas
quotas: ## EC2 vCPU quotas this stack depends on, and pending increase requests
	@for q in L-1216C47A L-34B43A08 L-DB2E81BA L-3819A6DF; do \
	  aws service-quotas get-service-quota --region $(REGION) --service-code ec2 --quota-code $$q \
	    --query '[Quota.QuotaCode,Quota.Value,Quota.QuotaName]' --output text; \
	  aws service-quotas list-requested-service-quota-change-history-by-quota --region $(REGION) --service-code ec2 --quota-code $$q \
	    --query 'RequestedQuotas[?Status!=`CASE_CLOSED`&&Status!=`APPROVED`].[`  requested`,DesiredValue,Status]' --output text; \
	done

.terraform/modules/modules.json: | $(TF)
	nice -n 19 $(TF) init -input=false

.PHONY: init
init: .terraform/modules/modules.json ## terraform init

.PHONY: plan
plan: init ## terraform plan (no changes)
	nice -n 19 $(TF) plan -input=false -var deadman_ttl=$(TTL) $(TF_ARGS)

.PHONY: up
up: preflight init ## create / converge everything; arms the dead-man switch for TTL (default 3h). SKIP_TF=1 skips the TensorFusion stage
	@# Re-arm first: an expired switch would tear the nodes down again while we apply.
	@$(base_env) ./scripts/deadman.sh arm $(TTL)
	@start=$$(date +%s); \
	nice -n 19 $(TF) apply -auto-approve -input=false -var deadman_ttl=$(TTL) $(TF_ARGS) || exit 1; \
	secs=$$(( $$(date +%s) - start )); echo ">> infrastructure + Karpenter up in $${secs}s"; $(call ledger,up-infra,$$secs)
	@if [ -z "$(SKIP_TF)" ]; then $(MAKE) --no-print-directory tf-install; fi
	@$(base_env) ./scripts/deadman.sh status | head -2
	@echo "export KUBECONFIG=$(KUBECONFIG)"

.PHONY: extend
extend: ## move the dead-man deadline to now + TTL (make extend TTL=2h)
	@$(base_env) ./scripts/deadman.sh arm $(TTL)

.PHONY: deadman
deadman: ## dead-man switch status: deadline, last teardown, recent ticks
	@$(base_env) ./scripts/deadman.sh status

.PHONY: cost
cost: ## spend to date for the Project tag (Cost Explorer) + live estimate from running resources
	@$(base_env) ./scripts/cost.sh --ledger $(LEDGER)

.PHONY: down
down: init ## destroy everything, then verify nothing billable is left (sweep output is the proof)
	@$(base_env) ./scripts/cost.sh --ledger $(LEDGER) || true
	@start=$$(date +%s); \
	nice -n 19 $(TF) destroy -auto-approve -input=false $(TF_ARGS) || \
	  { echo ">> destroy failed once; retrying (the destroy hooks have cleaned up by now)"; \
	    nice -n 19 $(TF) destroy -auto-approve -input=false $(TF_ARGS); }; rc=$$?; \
	secs=$$(( $$(date +%s) - start )); echo ">> destroy finished in $${secs}s (rc=$$rc)"; $(call ledger,down,$$secs); \
	rm -f $(KUBECONFIG); \
	$(base_env) ./scripts/sweep.sh

.PHONY: install
install: ## re-run stage 2 only (Karpenter / NodeClasses / NodePools / device plugin)
	@$(load_env); ./scripts/install.sh

.PHONY: tf-install
tf-install: ## stage 3: install / upgrade TensorFusion (TF_CHART, TF_RELEASE_TAG, TF_OPERATOR_TAG, TF_HYPERVISOR_TAG, TF_VGPU_PROVIDER_TAG, TF_AGENT_ID, TF_ENROLL_TOKEN, TF_CLOUD_ENDPOINT, TF_HELM_ARGS)
	@$(load_env); start=$$(date +%s); ./scripts/install-tf.sh; \
	secs=$$(( $$(date +%s) - start )); $(call ledger,tf-install,$$secs)

.PHONY: tf-uninstall
tf-uninstall: ## remove TensorFusion (repo scripts/uninstall.sh), keep the cluster
	@$(load_env); ./scripts/install-tf.sh uninstall

.PHONY: mock-install
mock-install: ## rehearsal mode: mock accelerator + mock GPUPool on the tf-mock NodePool (no GPU needed)
	@$(load_env); ./scripts/mock-gpu.sh install

.PHONY: mock-uninstall
mock-uninstall: ## remove the mock GPUPool and mock accelerator
	@$(load_env); ./scripts/mock-gpu.sh uninstall

.PHONY: gpu-dryrun
gpu-dryrun: ## prove the GPU launch path without GPUs: EC2 RunInstances --dry-run per GPU type + AMI resolution
	@$(load_env); ./scripts/gpu-dryrun.sh

.PHONY: status
status: ## nodes, NodeClaims and TensorFusion resources
	@kubectl get nodes -L node.kubernetes.io/instance-type,karpenter.sh/capacity-type,karpenter.sh/nodepool
	@kubectl get nodepool,nodeclaim,ec2nodeclass 2>/dev/null || true
	@kubectl get tensorfusioncluster,gpupool,gpunodeclaim,gpunode,gpu 2>/dev/null || true

.PHONY: sweep
sweep: ## report leftovers by tag/name; works without terraform state (NAME= REGION=)
	@$(base_env) ./scripts/sweep.sh

.PHONY: sweep-delete
sweep-delete: ## delete leftovers found by sweep (last resort when terraform destroy cannot finish)
	@$(base_env) ./scripts/sweep.sh --delete

.PHONY: prices
prices: ## refresh ../pricing/<region>.json from the AWS Price List API
	@$(base_env) ./scripts/refresh-prices.sh

.PHONY: check
check: $(TF) ## terraform fmt + validate, shell syntax, lambda syntax (no AWS access)
	$(TF) fmt -check -recursive
	nice -n 19 $(TF) init -backend=false -input=false >/dev/null
	nice -n 19 $(TF) validate
	@for f in scripts/*.sh; do bash -n $$f; done
	@command -v shellcheck >/dev/null && shellcheck -S warning scripts/*.sh || true
	@python3 -m py_compile lambda/deadman.py && rm -rf lambda/__pycache__
