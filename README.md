### Step-by-step: SURF volume-first setup (Ubuntu 22.04)
This repository supports a Docker workflow where datasets, weights, tracking outputs, and Docker/containerd storage can all be placed on a mounted SURF volume.

Note: Ubuntu system packages installed with `apt` (for example Docker binaries) still live on the OS disk. Project/runtime-heavy storage is what this setup redirects to the SURF volume.

1. Mount your SURF volume at `/data/boosttrack_storage` and create a layout:
```shell
export SURF_VOL=/data/boosttrack_storage
mkdir -p "$SURF_VOL"/{repo,data,results,weights,runtime}
git clone <your-repo-url> "$SURF_VOL/repo"
cd "$SURF_VOL/repo"
```

2. Bootstrap Docker + NVIDIA runtime, with Docker/containerd storage on the volume:
```shell
make vm-bootstrap
```
Then re-login (or run `newgrp docker`) and verify:
```shell
make docker-build
make docker-gpu-check
```

3. Optional: override paths if you use a different mount point.
```shell
make help
# or set custom path:
# make tune-hspot SURF_VOL=/my/other/mount
```
Default resolved paths are:
- `HOST_DATA_ROOT=/data/boosttrack_storage/data`
- `HOST_RESULTS_ROOT=/data/boosttrack_storage/results`
- `HOST_WEIGHTS_ROOT=/data/boosttrack_storage/weights`

4. Put hspot data on the volume:
```text
/data/boosttrack_storage/data/hspot/{train,val,test}/<sequence>/{img1,det,gt,seqinfo.ini}
```

5. Put detector weights on the volume:
```text
/data/boosttrack_storage/weights/bytetrack_x_mot17.pth.tar
/data/boosttrack_storage/weights/bytetrack_ablation.pth.tar
/data/boosttrack_storage/weights/bytetrack_x_mot20.tar
```
Weights are loaded through `BOOSTTRACK_WEIGHTS_DIR` (set automatically by the Make targets).

6. Convert hspot annotations to COCO:
```shell
make hspot-convert
```

7. Prepare TrackEval ground-truth layout:
```shell
make hspot-trackeval-setup
```
If your test split has no `gt/gt.txt`:
```shell
make hspot-trackeval-setup-allow-missing-gt
```

8. Configure remote MLflow (optional):
```shell
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000
```
If required:
```shell
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<password>
```

9. Run baseline and tuning:
```shell
make baseline-hspot-val
make tune-hspot
```
This workflow uses `train` for pruning, `val` for the Optuna objective, and `test` for final best-trial evaluation.
If test GT is unavailable:
```shell
make tune-hspot TUNE_EXTRA_ARGS="--skip-final-test-eval"
```

Main outputs (all on `/data/boosttrack_storage/results` by default):
- Baseline Optuna DB: `optuna/hspot_baseline_val.db`
- Baseline summary JSON: `optuna/hspot_baseline_val_summary.json`
- Tuning Optuna DB: `optuna/boosttrack_hota_tuning.db`
- Tuning summary JSON: `optuna/boosttrack_hota_tuning_summary.json`
- Tracking outputs: `trackers/hspot-val/` and `trackers/hspot-test/`
