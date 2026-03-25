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
HOST_CACHE_ROOT ?= $(SURF_VOL)/cache
HOST_TMP_ROOT ?= $(SURF_VOL)/tmp
HOST_CONTAINER_HOME ?= $(SURF_VOL)/container-home

# Container-visible paths.
data_root ?= /data
results_root ?= /results
weights_root ?= /weights
cache_root ?= /cache
tmp_root ?= /tmp-volume
container_home ?= /home/boosttrack

# Remote MLflow URI (already running on a separate VM)
MLFLOW_TRACKING_URI ?= http://ubuntu2204sudo.property-occupa.src.surf-hosted.nl:80

# hspot defaults
hspot_data_root ?= $(data_root)/hspot
hspot_gt_root ?= $(results_root)/gt
trackers_root ?= $(results_root)/trackers
optuna_root ?= $(results_root)/optuna
viz_root ?= $(results_root)/visualizations
TUNE_TRIALS ?= 128
TUNE_GPU_ID ?= 0
TUNE_PRUNING_SEQS ?= 0
TUNE_TIMEOUT_SEC ?= 360000
TUNE_PRUNER_STARTUP_TRIALS ?= 5
TUNE_EARLY_STOP_PATIENCE ?= 20
TUNE_EARLY_STOP_MIN_DELTA ?= 0.002
TUNE_EXTRA_ARGS ?= --mlflow-log-summary-json
BASELINE_STUDY_NAME ?= hspot_baseline_val
BASELINE_STUDY_DB ?= $(optuna_root)/hspot_baseline_val.db
BASELINE_SUMMARY_JSON ?= $(optuna_root)/$(BASELINE_STUDY_NAME)_summary.json
BASELINE_MLFLOW_EXPERIMENT ?= BoostTrack++
BASELINE_MLFLOW_RUN_NAME ?= hspot_baseline_val
BASELINE_EXTRA_ARGS ?=
TUNE_MLFLOW_EXPERIMENT ?= $(BASELINE_MLFLOW_EXPERIMENT)
TUNE_STUDY_NAME ?= hspot_hota_optuna
TUNE_STUDY_DB ?= $(optuna_root)/$(TUNE_STUDY_NAME).db
TUNE_SUMMARY_JSON ?= $(optuna_root)/$(TUNE_STUDY_NAME)_summary.json
VIS_SPLIT ?= val
VIS_TRACKER ?=
VIS_SEQ ?=
VIS_MAX_FRAMES ?= 0

DOCKER_GPU_ARGS = $(if $(strip $(GPU)),--gpus $(GPU),)
DOCKER_MLFLOW_ENV = -e MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)"

DOCKER_RUN_BASE = docker run --rm --name $(CONTAINER_NAME) $(DOCKER_GPU_ARGS) --ipc=host --network=host \
	-v "$(PWD):$(WORKDIR)" \
	-v "$(HOST_DATA_ROOT):$(data_root)" \
	-v "$(HOST_RESULTS_ROOT):$(results_root)" \
	-v "$(HOST_WEIGHTS_ROOT):$(weights_root)" \
	-v "$(HOST_CACHE_ROOT):$(cache_root)" \
	-v "$(HOST_TMP_ROOT):$(tmp_root)" \
	-v "$(HOST_CONTAINER_HOME):$(container_home)" \
	-e BOOSTTRACK_DATA_DIR="$(data_root)" \
	-e BOOSTTRACK_GT_FOLDER="$(hspot_gt_root)" \
	-e BOOSTTRACK_TRACKERS_FOLDER="$(trackers_root)" \
	-e BOOSTTRACK_WEIGHTS_DIR="$(weights_root)" \
	-e HOME="$(container_home)" \
	-e XDG_CACHE_HOME="$(cache_root)" \
	-e TORCH_HOME="$(cache_root)/torch" \
	-e PIP_CACHE_DIR="$(cache_root)/pip" \
	-e UV_CACHE_DIR="$(cache_root)/uv" \
	-e MPLCONFIGDIR="$(cache_root)/matplotlib" \
	-e TMPDIR="$(tmp_root)" \
	-w $(WORKDIR)

.PHONY: help vm-bootstrap docker-build docker-shell docker-gpu-check mlflow-smoke-test \
	surf-layout hspot-convert hspot-trackeval-setup hspot-trackeval-setup-allow-missing-gt \
	baseline-hspot-val tune-hspot visualize-hspot

help:
	@echo "Targets:"
	@echo "  vm-bootstrap          Install Docker + NVIDIA Container Toolkit on Ubuntu 22.04 VM"
	@echo "  surf-layout           Create the default SURF volume directory layout"
	@echo "  docker-build          Build CUDA12 + uv project image"
	@echo "  docker-shell          Open interactive shell inside container"
	@echo "  docker-gpu-check      Verify GPU visibility inside container"
	@echo "  mlflow-smoke-test     Validate remote MLflow logging with a tiny run + artifact"
	@echo "  hspot-convert         Convert hspot MOT-format dataset to COCO JSON"
	@echo "  hspot-trackeval-setup Prepare TrackEval GT/seqmaps for hspot"
	@echo "  hspot-trackeval-setup-allow-missing-gt  Same as above, but skips missing test GT files"
	@echo "  baseline-hspot-val    Run default-parameter baseline on hspot val (logs to MLflow if URI set)"
	@echo "  tune-hspot            Run Optuna tuning on hspot (val objective, test final eval; train pruning off by default)"
	@echo "  visualize-hspot       Render hspot frames with GT and prediction boxes overlaid"
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
	@echo "  HOST_CACHE_ROOT=$(HOST_CACHE_ROOT)"
	@echo "  HOST_TMP_ROOT=$(HOST_TMP_ROOT)"
	@echo "  HOST_CONTAINER_HOME=$(HOST_CONTAINER_HOME)"
	@echo "  data_root=$(data_root)"
	@echo "  results_root=$(results_root)"
	@echo "  weights_root=$(weights_root)"
	@echo "  cache_root=$(cache_root)"
	@echo "  tmp_root=$(tmp_root)"
	@echo "  container_home=$(container_home)"
	@echo "  hspot_data_root=$(hspot_data_root)"
	@echo "  hspot_gt_root=$(hspot_gt_root)"
	@echo "  trackers_root=$(trackers_root)"
	@echo "  optuna_root=$(optuna_root)"
	@echo "  viz_root=$(viz_root)"
	@echo "  TUNE_TRIALS=$(TUNE_TRIALS)"
	@echo "  TUNE_TIMEOUT_SEC=$(TUNE_TIMEOUT_SEC)"
	@echo "  VIS_SPLIT=$(VIS_SPLIT)"
	@echo "  VIS_TRACKER=$(VIS_TRACKER)"
	@echo "  VIS_SEQ=$(VIS_SEQ)"
	@echo "  VIS_MAX_FRAMES=$(VIS_MAX_FRAMES)"

surf-layout:
	mkdir -p "$(DOCKER_STORAGE_ROOT)" "$(HOST_DATA_ROOT)" "$(HOST_RESULTS_ROOT)" "$(HOST_WEIGHTS_ROOT)" "$(HOST_CACHE_ROOT)" "$(HOST_TMP_ROOT)" "$(HOST_CONTAINER_HOME)"

vm-bootstrap: surf-layout
	sudo bash scripts/setup_ubuntu2204_cuda12_docker.sh $(if $(DOCKER_STORAGE_ROOT),--storage-root "$(DOCKER_STORAGE_ROOT)",)

docker-build: surf-layout
	docker build -t $(IMAGE) .

docker-shell: surf-layout
	$(DOCKER_RUN_BASE) \
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) bash

docker-gpu-check: surf-layout
	$(DOCKER_RUN_BASE) \
		$(IMAGE) nvidia-smi

mlflow-smoke-test: surf-layout
	$(DOCKER_RUN_BASE) \
		$(DOCKER_MLFLOW_ENV) \
		$(IMAGE) python3 tools/mlflow_smoke_test.py

hspot-convert: surf-layout
	$(DOCKER_RUN_BASE) \
		$(IMAGE) python3 data/tools/convert_hspot_to_coco.py --data-path $(hspot_data_root) --splits train,val,test

hspot-trackeval-setup: surf-layout
	$(DOCKER_RUN_BASE) \
		$(IMAGE) bash tools/setup_hspot_trackeval_gt.sh --data-root $(hspot_data_root) --gt-root $(hspot_gt_root)

hspot-trackeval-setup-allow-missing-gt: surf-layout
	$(DOCKER_RUN_BASE) \
		$(IMAGE) bash tools/setup_hspot_trackeval_gt.sh --data-root $(hspot_data_root) --gt-root $(hspot_gt_root) --allow-missing-gt

baseline-hspot-val: surf-layout
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

tune-hspot: surf-layout
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
		--mlflow-experiment $(TUNE_MLFLOW_EXPERIMENT) \
		$${MLFLOW_TRACKING_URI:+--mlflow-tracking-uri $$MLFLOW_TRACKING_URI} \
		$(TUNE_EXTRA_ARGS)

visualize-hspot: surf-layout
	@test -n "$(VIS_TRACKER)" || (echo "VIS_TRACKER is required, e.g. VIS_TRACKER=hspot_baseline_val_trial_0005_full_post_gbi" >&2; exit 1)
	$(DOCKER_RUN_BASE) \
		$(IMAGE) python3 tools/visualize_hspot_tracking.py \
		--data-root $(hspot_data_root) \
		--gt-root $(hspot_gt_root) \
		--trackers-root $(trackers_root) \
		--split $(VIS_SPLIT) \
		--tracker-name $(VIS_TRACKER) \
		--output-root $(viz_root) \
		--max-frames $(VIS_MAX_FRAMES) \
		$(if $(strip $(VIS_SEQ)),--seq $(VIS_SEQ),)
