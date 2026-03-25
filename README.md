### Step-by-step: SURF volume-first setup (Ubuntu 22.04)
This repository supports a Docker workflow where the repository, datasets, weights, tracking outputs, runtime caches, temporary files, and Docker/containerd storage can all be placed on a mounted SURF volume.
The Docker image builds the Python environment from `boost-track-env.yml` (with the `prefix:` entry ignored inside the container).

Important: on a standard Ubuntu 22.04 VM you cannot make root-disk usage literally zero, because the base OS, `dpkg` database, and installed system binaries/services live on `/`.
What this setup does redirect to the SURF volume is everything that grows large in practice:
- the git checkout
- datasets
- results, trackers, checkpoints, Optuna databases, and local artifacts
- container `HOME`, `~/.cache`, and `TMPDIR`
- Docker image/layer storage and containerd storage
- bootstrap-time APT package archives, package lists, and temporary download files

The remaining root-disk footprint is the small set of Ubuntu-managed system files for Docker/NVIDIA toolkit installation under paths such as `/usr`, `/etc`, `/lib`, and `/var/lib/dpkg`.

1. Mount your SURF volume at `/data/boosttrack_storage` and create a layout:
```shell
export SURF_VOL=/data/boosttrack_storage
mkdir -p "$SURF_VOL"/{repo,data,results,weights,cache,tmp,container-home,runtime}
git clone <your-repo-url> "$SURF_VOL/repo"
cd "$SURF_VOL/repo"
```

2. Bootstrap Docker + NVIDIA runtime, with Docker/containerd storage plus bootstrap cache/tmp on the volume:
```shell
make vm-bootstrap
```
Validate GPU access in containers:
```shell
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```
Then re-login (or run `newgrp docker`) and verify:
```shell
make docker-build
make docker-gpu-check
```
The bootstrap script uses `SURF_VOL/runtime` for:
- Docker `data-root`
- containerd `root` and `state`
- APT archive cache and package lists used during installation
- temporary host-side download files used during installation

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
- `HOST_CACHE_ROOT=/data/boosttrack_storage/cache`
- `HOST_TMP_ROOT=/data/boosttrack_storage/tmp`
- `HOST_CONTAINER_HOME=/data/boosttrack_storage/container-home`

Inside the container these are exposed as:
- `HOME=/home/boosttrack`
- `BOOSTTRACK_DATA_DIR=/data`
- `BOOSTTRACK_GT_FOLDER=/results/gt`
- `BOOSTTRACK_WEIGHTS_DIR=/weights`
- `XDG_CACHE_HOME=/cache`
- `TORCH_HOME=/cache/torch`
- `TMPDIR=/tmp-volume`

If you keep the repository checkout itself under `SURF_VOL/repo`, the project's local `./cache` directory also remains on the SURF volume.

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
export MLFLOW_TRACKING_URI=http://ubuntu2204sudo.property-occupa.src.surf-hosted.nl:80
```

Smoke-test the connection before running longer jobs:
```shell
make mlflow-smoke-test
```

9. Run baseline and tuning:
```shell
make baseline-hspot-val
make tune-hspot
```
This workflow uses `val` for the Optuna objective and `test` for final best-trial evaluation. Train pruning is disabled by default.
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

In practice, this means the root disk is no longer the place where Docker images, container writable layers, datasets, experiment outputs, or model caches accumulate. Only the Ubuntu-managed system install footprint remains on `/`.
