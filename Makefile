SHELL := /bin/bash

IMAGE ?= boosttrack:cuda12-uv
CONTAINER_NAME ?= boosttrack-dev
GPU ?= all
WORKDIR ?= /workspace
SURF_VOL ?= /data/boosttrack_storage
DOCKER_STORAGE_ROOT ?= $(SURF_VOL)/runtime

# Host-side paths (set these to SURF volume locations).
HOST_DATA_ROOT ?= $(SURF_VOL)/data
HOST_RESULTS_ROOT ?= $(SURF_VOL)/results
HOST_WEIGHTS_ROOT ?= $(SURF_VOL)/weights

# Container-visible paths.
data_root ?= $(WORKDIR)/data
results_root ?= $(WORKDIR)/results
weights_root ?= $(WORKDIR)/external/weights

# Remote MLflow URI (already running on a separate VM)
MLFLOW_TRACKING_URI ?=

# hspot defaults
hspot_data_root ?= $(data_root)/hspot
hspot_gt_root ?= $(results_root)/gt
trackers_root ?= $(results_root)/trackers
optuna_root ?= $(results_root)/optuna
TUNE_TRIALS ?= 64
TUNE_GPU_ID ?= 0
TUNE_PRUNING_SEQS ?= 2
TUNE_TIMEOUT_SEC ?= 360000
TUNE_PRUNER_STARTUP_TRIALS ?= 5
TUNE_EARLY_STOP_PATIENCE ?= 10
TUNE_EARLY_STOP_MIN_DELTA ?= 0.01
TUNE_EXTRA_ARGS ?= --mlflow-log-summary-json
BASELINE_STUDY_NAME ?= hspot_baseline_val
BASELINE_STUDY_DB ?= $(optuna_root)/hspot_baseline_val.db
BASELINE_SUMMARY_JSON ?= $(optuna_root)/$(BASELINE_STUDY_NAME)_summary.json
BASELINE_MLFLOW_EXPERIMENT ?= BoostTrack-Baselines
BASELINE_MLFLOW_RUN_NAME ?= hspot_baseline_val
BASELINE_EXTRA_ARGS ?=
TUNE_STUDY_NAME ?= boosttrack_hota_tuning
TUNE_STUDY_DB ?= $(optuna_root)/$(TUNE_STUDY_NAME).db
TUNE_SUMMARY_JSON ?= $(optuna_root)/$(TUNE_STUDY_NAME)_summary.json

DOCKER_GPU_ARGS = $(if $(strip $(GPU)),--gpus $(GPU),)
DOCKER_MLFLOW_ENV = -e MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)"

DOCKER_RUN_BASE = docker run --rm --name $(CONTAINER_NAME) $(DOCKER_GPU_ARGS) --ipc=host --network=host \
	-v "$(PWD):$(WORKDIR)" \
	-v "$(HOST_DATA_ROOT):$(data_root)" \
	-v "$(HOST_RESULTS_ROOT):$(results_root)" \
	-v "$(HOST_WEIGHTS_ROOT):$(weights_root)" \
	-e BOOSTTRACK_DATA_DIR="$(data_root)" \
	-e BOOSTTRACK_GT_FOLDER="$(hspot_gt_root)" \
	-e BOOSTTRACK_TRACKERS_FOLDER="$(trackers_root)" \
	-e BOOSTTRACK_WEIGHTS_DIR="$(weights_root)" \
	-w $(WORKDIR)

.PHONY: help vm-bootstrap docker-build docker-shell docker-gpu-check mlflow-smoke-test \
	hspot-convert hspot-trackeval-setup hspot-trackeval-setup-allow-missing-gt \
	baseline-hspot-val tune-hspot

help:
	@echo "Targets:"
	@echo "  vm-bootstrap          Install Docker + NVIDIA Container Toolkit on Ubuntu 22.04 VM"
	@echo "  docker-build          Build CUDA12 + uv project image"
	@echo "  docker-shell          Open interactive shell inside container"
	@echo "  docker-gpu-check      Verify GPU visibility inside container"
	@echo "  mlflow-smoke-test     Validate remote MLflow logging with a tiny run + artifact"
	@echo "  hspot-convert         Convert hspot MOT-format dataset to COCO JSON"
	@echo "  hspot-trackeval-setup Prepare TrackEval GT/seqmaps for hspot"
	@echo "  hspot-trackeval-setup-allow-missing-gt  Same as above, but skips missing test GT files"
	@echo "  baseline-hspot-val    Run default-parameter baseline on hspot val (logs to MLflow if URI set)"
	@echo "  tune-hspot            Run Optuna tuning on hspot (train pruning, val objective, test final eval)"
	@echo ""
	@echo "Key vars:"
	@echo "  IMAGE=$(IMAGE)"
	@echo "  GPU=$(GPU)  (set to empty on CPU-only VMs)"
	@echo "  MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)"
	@echo "  SURF_VOL=$(SURF_VOL)"
	@echo "  DOCKER_STORAGE_ROOT=$(DOCKER_STORAGE_ROOT)"
	@echo "  HOST_DATA_ROOT=$(HOST_DATA_ROOT)"
	@echo "  HOST_RESULTS_ROOT=$(HOST_RESULTS_ROOT)"
	@echo "  HOST_WEIGHTS_ROOT=$(HOST_WEIGHTS_ROOT)"
	@echo "  data_root=$(data_root)"
	@echo "  results_root=$(results_root)"
	@echo "  weights_root=$(weights_root)"
	@echo "  hspot_data_root=$(hspot_data_root)"
	@echo "  hspot_gt_root=$(hspot_gt_root)"
	@echo "  trackers_root=$(trackers_root)"
	@echo "  optuna_root=$(optuna_root)"
	@echo "  TUNE_TRIALS=$(TUNE_TRIALS)"
	@echo "  TUNE_TIMEOUT_SEC=$(TUNE_TIMEOUT_SEC)"

vm-bootstrap:
	sudo bash scripts/setup_ubuntu2204_cuda12_docker.sh $(if $(DOCKER_STORAGE_ROOT),--storage-root "$(DOCKER_STORAGE_ROOT)",)

docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	$(DOCKER_RUN_BASE) \
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) bash

docker-gpu-check:
	$(DOCKER_RUN_BASE) \
		$(IMAGE) nvidia-smi

mlflow-smoke-test:
	$(DOCKER_RUN_BASE) \
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) python3 tools/mlflow_smoke_test.py

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
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) python3 tools/tune_boosttrack_optuna.py \
		--dataset hspot \
		--benchmark hspot \
		--data-root $(data_root) \
		--gt-folder $(hspot_gt_root) \
		--trackers-folder $(trackers_root) \
		--study-name $(BASELINE_STUDY_NAME) \
		--study-db $(BASELINE_STUDY_DB) \
		--output-json $(BASELINE_SUMMARY_JSON) \
		--n-trials 1 \
		--pruning-seqs 0 \
		--skip-train-pruning \
		--early-stop-patience 0 \
		--skip-final-test-eval \
		--fixed-defaults \
		--mlflow-experiment $(BASELINE_MLFLOW_EXPERIMENT) \
		--mlflow-run-name $(BASELINE_MLFLOW_RUN_NAME) \
		$${MLFLOW_TRACKING_URI:+--mlflow-tracking-uri $$MLFLOW_TRACKING_URI} \
		$(BASELINE_EXTRA_ARGS)

tune-hspot:
	$(DOCKER_RUN_BASE) \
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) python3 tools/tune_boosttrack_optuna.py \
		--dataset hspot \
		--benchmark hspot \
		--data-root $(data_root) \
		--gt-folder $(hspot_gt_root) \
		--trackers-folder $(trackers_root) \
		--study-name $(TUNE_STUDY_NAME) \
		--study-db $(TUNE_STUDY_DB) \
		--output-json $(TUNE_SUMMARY_JSON) \
		--gpu-id $(TUNE_GPU_ID) \
		--n-trials $(TUNE_TRIALS) \
		--pruning-seqs $(TUNE_PRUNING_SEQS) \
		--timeout-sec $(TUNE_TIMEOUT_SEC) \
		--pruner-startup-trials $(TUNE_PRUNER_STARTUP_TRIALS) \
		--early-stop-patience $(TUNE_EARLY_STOP_PATIENCE) \
		--early-stop-min-delta $(TUNE_EARLY_STOP_MIN_DELTA) \
		$${MLFLOW_TRACKING_URI:+--mlflow-tracking-uri $$MLFLOW_TRACKING_URI} \
		$(TUNE_EXTRA_ARGS)
