### Step-by-step: SURF + hspot + MLflow + tuning
This repository includes a Docker workflow for Ubuntu 22.04 CUDA VMs. Dependencies are installed in-container with `uv`.

1. Download the model weights and set up the datasets.

We use the same weights as [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/main). The weights can be downloaded from the [link](https://drive.google.com/drive/folders/15hZcR4bW_Z9hEaXXjeWhQl_jwRKllauG?usp=sharing).

Download the weights and place to `BoostTrack/external/weights` folder.

2. Set up Docker + NVIDIA runtime on the SURF VM:
```shell
git clone <your-repo-url>
cd <repo-folder>
make vm-bootstrap
```
Then re-login (or run `newgrp docker`), and verify:
```shell
make docker-build
make docker-gpu-check
```

3. Place your custom hspot dataset under `data/hspot`:
```text
data/hspot/{train,val,test}/<sequence>/{img1,det,gt,seqinfo.ini}
```

4. Convert hspot annotations to COCO:
```shell
make hspot-convert
```

5. Prepare TrackEval ground-truth layout:
```shell
make hspot-trackeval-setup
```
If your test split has no `gt/gt.txt`:
```shell
make hspot-trackeval-setup-allow-missing-gt
```
This prepares TrackEval data for `train`, `val`, and `test`. TrackEval is already included in this repository (`external/TrackEval`), so no separate TrackEval installation is required.

6. Configure remote MLflow (running on another VM):
```shell
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000
```
If required, also set auth variables such as:
```shell
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<password>
```

7. Run a baseline on the validation split (default BoostTrack hyperparameters, logged to MLflow):
```shell
make baseline-hspot-val
```

8. Run hyperparameter tuning (train pruning, validation HOTA objective, 1 GPU):
```shell
make tune-hspot
```

This workflow uses `train` for early pruning/subset evaluation, `val` for the Optuna objective, and `test` for the final best-parameter evaluation.
If test GT is unavailable, add:
```shell
TUNE_EXTRA_ARGS="--skip-final-test-eval"
```

Main outputs:
- Baseline Optuna DB: `results/optuna/hspot_baseline_val.db`
- Baseline summary JSON: `results/optuna/hspot_baseline_val_summary.json`
- Optuna DB: `results/optuna/boosttrack_hota_tuning.db`
- Tuning summary JSON: `results/optuna/boosttrack_hota_tuning_summary.json`
- Tracking results: `results/trackers/hspot-val/` and `results/trackers/hspot-test/`
