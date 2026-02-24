SHELL := /bin/bash

IMAGE ?= boosttrack:cuda12-uv
CONTAINER_NAME ?= boosttrack-dev
GPU ?= all
WORKDIR ?= /workspace

# Remote MLflow URI (already running on a separate VM)
MLFLOW_TRACKING_URI ?=

# hspot defaults
hspot_data_root ?= data/hspot
hspot_gt_root ?= results/gt
TUNE_TRIALS ?= 30
TUNE_GPU_ID ?= 0
TUNE_PRUNING_SEQS ?= 2
TUNE_EXTRA_ARGS ?=
BASELINE_STUDY_NAME ?= hspot_baseline_val
BASELINE_STUDY_DB ?= results/optuna/hspot_baseline_val.db
BASELINE_MLFLOW_EXPERIMENT ?= BoostTrack-Baselines
BASELINE_MLFLOW_RUN_NAME ?= hspot_baseline_val
BASELINE_EXTRA_ARGS ?=

DOCKER_RUN_BASE = docker run --rm --name $(CONTAINER_NAME) --gpus $(GPU) --ipc=host --network=host \
	-v $(PWD):$(WORKDIR) -w $(WORKDIR)

.PHONY: help vm-bootstrap docker-build docker-shell docker-gpu-check \
	hspot-convert hspot-trackeval-setup hspot-trackeval-setup-allow-missing-gt \
	baseline-hspot-val tune-hspot

help:
	@echo "Targets:"
	@echo "  vm-bootstrap          Install Docker + NVIDIA Container Toolkit on Ubuntu 22.04 VM"
	@echo "  docker-build          Build CUDA12 + uv project image"
	@echo "  docker-shell          Open interactive shell inside container"
	@echo "  docker-gpu-check      Verify GPU visibility inside container"
	@echo "  hspot-convert         Convert hspot MOT-format dataset to COCO JSON"
	@echo "  hspot-trackeval-setup Prepare TrackEval GT/seqmaps for hspot"
	@echo "  hspot-trackeval-setup-allow-missing-gt  Same as above, but skips missing test GT files"
	@echo "  baseline-hspot-val    Run default-parameter baseline on hspot val (logs to MLflow if URI set)"
	@echo "  tune-hspot            Run Optuna tuning on hspot (logs to remote MLflow if URI set)"
	@echo ""
	@echo "Key vars:"
	@echo "  IMAGE=$(IMAGE)"
	@echo "  MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)"
	@echo "  hspot_data_root=$(hspot_data_root)"

vm-bootstrap:
	sudo bash scripts/setup_ubuntu2204_cuda12_docker.sh

docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	$(DOCKER_RUN_BASE) \
		-e MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" \
		$(IMAGE) bash

docker-gpu-check:
	$(DOCKER_RUN_BASE) \
		$(IMAGE) nvidia-smi

hspot-convert:
	$(DOCKER_RUN_BASE) \
		$(IMAGE) python3 data/tools/convert_hspot_to_coco.py --data-path $(hspot_data_root) --splits train,val,test

hspot-trackeval-setup:
	$(DOCKER_RUN_BASE) \
		$(IMAGE) bash tools/setup_hspot_trackeval_gt.sh --data-root $(hspot_data_root) --gt-root $(hspot_gt_root)

hspot-trackeval-setup-allow-missing-gt:
	$(DOCKER_RUN_BASE) \
		$(IMAGE) bash tools/setup_hspot_trackeval_gt.sh --data-root $(hspot_data_root) --gt-root $(hspot_gt_root) --allow-missing-gt

baseline-hspot-val:
	$(DOCKER_RUN_BASE) \
		-e MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" \
		$(IMAGE) python3 tools/tune_boosttrack_optuna.py \
		--dataset hspot \
		--benchmark hspot \
		--study-name $(BASELINE_STUDY_NAME) \
		--study-db $(BASELINE_STUDY_DB) \
		--n-trials 1 \
		--pruning-seqs 0 \
		--early-stop-patience 0 \
		--skip-final-test-eval \
		--fixed-defaults \
		--mlflow-experiment $(BASELINE_MLFLOW_EXPERIMENT) \
		--mlflow-run-name $(BASELINE_MLFLOW_RUN_NAME) \
		$${MLFLOW_TRACKING_URI:+--mlflow-tracking-uri $$MLFLOW_TRACKING_URI} \
		$(BASELINE_EXTRA_ARGS)

tune-hspot:
	$(DOCKER_RUN_BASE) \
		-e MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" \
		$(IMAGE) python3 tools/tune_boosttrack_optuna.py \
		--dataset hspot \
		--benchmark hspot \
		--gpu-id $(TUNE_GPU_ID) \
		--n-trials $(TUNE_TRIALS) \
		--pruning-seqs $(TUNE_PRUNING_SEQS) \
		$${MLFLOW_TRACKING_URI:+--mlflow-tracking-uri $$MLFLOW_TRACKING_URI} \
		$(TUNE_EXTRA_ARGS)
