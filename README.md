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
make baseline-hspot-val MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
```
Optional override:
```shell
make baseline-hspot-val \
  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  BASELINE_MLFLOW_EXPERIMENT=BoostTrack-Baselines \
  BASELINE_MLFLOW_RUN_NAME=hspot_baseline_val_run01
```

8. Run hyperparameter tuning (train pruning, validation HOTA objective, 1 GPU):
```shell
make tune-hspot MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
```
Example with custom settings:
```shell
make tune-hspot \
  MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  TUNE_TRIALS=50 \
  TUNE_EXTRA_ARGS="--mlflow-experiment BoostTrack-hspot --mlflow-run-name surf_vm_run_01 --mlflow-log-summary-json"
```
This workflow uses `train` for early pruning/subset evaluation, `val` for the Optuna objective, and `test` for the final best-parameter evaluation.
If test GT is unavailable, add:
```shell
TUNE_EXTRA_ARGS="--skip-final-test-eval"
```

9. Final evaluation:
- By default, `tools/tune_boosttrack_optuna.py` performs a final test evaluation using the best validation hyperparameters.
- You can also run TrackEval manually:
```shell
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL test \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK hspot \
  --TRACKERS_TO_EVAL <tracker_name>
```
For tuning runs with post-processing enabled (default), `<tracker_name>` is typically `<exp_name>_post_gbi`.

Main outputs:
- Baseline Optuna DB: `results/optuna/hspot_baseline_val.db`
- Baseline summary JSON: `results/optuna/hspot_baseline_val_summary.json`
- Optuna DB: `results/optuna/boosttrack_hota_tuning.db`
- Tuning summary JSON: `results/optuna/boosttrack_hota_tuning_summary.json`
- Tracking results: `results/trackers/hspot-val/` and `results/trackers/hspot-test/`

## Running the experiments and evaluation
### Run BoostTrack
To run the BoostTrack on MOT17 and MOT20 validation sets run the following:
```shell
python main.py --dataset mot17 --exp_name BoostTrack --no_reid --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt
python main.py --dataset mot20 --exp_name BoostTrack --no_reid --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt
```
Note, three resulting folders will be created for each experiment: BoostTrack, BoostTrack_post, BoostTrack_post_gbi. The folders with the suffixes correspond to results with applied linear and gradient boosting interpolation. 

To evaluate the results using TrackEval run:
```shell
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT17  --TRACKERS_TO_EVAL BoostTrack_post_gbi
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrack_post_gbi 
```
### Run BoostTrack+
Similarly, to run the BoostTrack+ run:
```shell
python main.py --dataset mot17 --exp_name BoostTrackPlus --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt
python main.py --dataset mot20 --exp_name BoostTrackPlus --btpp_arg_iou_boost --btpp_arg_no_sb --btpp_arg_no_vt
```
To evaluate the BoostTrack+ results run:
```shell
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT17  --TRACKERS_TO_EVAL BoostTrackPlus_post_gbi
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BoostTrackPlus_post_gbi
```
### Run BoostTrack++
Finally, the default setting is to use BoostTrack++:
```shell
python main.py --dataset mot17 --exp_name BTPP
python main.py --dataset mot20 --exp_name BTPP
```
To evaluate the BoostTrack++ results run:
```shell
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT17  --TRACKERS_TO_EVAL BTPP_post_gbi
python external/TrackEval/scripts/run_mot_challenge.py   --SPLIT_TO_EVAL val   --GT_FOLDER results/gt/   --TRACKERS_FOLDER results/trackers/   --BENCHMARK MOT20  --TRACKERS_TO_EVAL BTPP_post_gbi
```

# Acknowledgements
Our implementation is developed on top of publicly available codes. We thank authors of [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/), [SORT](https://github.com/abewley/sort), [StrongSort](https://github.com/dyhBUPT/StrongSORT), [NCT](https://github.com/Autoyou/Noise-control-multi-object-tracking), [ByteTrack](https://github.com/ifzhang/ByteTrack/) for making their code available. 

# Citation

If you find our work useful, please cite our papers: 
```
@article{stanojevic2024boostTrack,
  title={BoostTrack: boosting the similarity measure and detection confidence for improved multiple object tracking},
  author={Stanojevic, Vukasin D and Todorovic, Branimir T},
  journal={Machine Vision and Applications},
  issn = {0932-8092},
  year={2024},
  volume={35},
  number = {3},
  doi={10.1007/s00138-024-01531-5}
}

@article{stanojevic2024btpp,
      title={BoostTrack++: using tracklet information to detect more objects in multiple object tracking},
      author={Vuka\v{s}in Stanojevi\'c and Branimir Todorovi\'c},
      year={2024},
      eprint={2408.13003},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      doi={https://doi.org/10.48550/arXiv.2408.13003}
}
```

## A bug notice:
There is a bug in calculating shape similarity. 
It is supposed to be calculated as 
```math
S^{shape}_{d_i, t_j} = c_{d_i, t_j} \cdot \exp \biggl(-\big(\frac{|D_i^w - T_j^w|}{\text{max}(D_i^w, T_j^w)}  + \frac{|D_i^h - T_j^h|}{\text{max}(D_i^h, T_j^h)}\big)\biggr).
```
However, in the code, the equation is implemented (it is multiplied by detection-tracklet confidence later) as
```
np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))
```
instead of 
```
np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dh, th)))
```
Dividing both additions by the shorter dimension, i.e. width, penalizes shape mismatch more. Hyperparameters $\lambda_{IoU}, \lambda_{MhD}$ and $\lambda_{shape}$ are tuned to work with the original implementation, and using the correct implementation produces slightly worse results.
For this reason, we keep the implementation with the bug as function shape_similarity_v1, used by the default, and we provide the correct implementation in function shape_similarity_v2 (see file assoc.py).
Correct implementation can be used by passing the --s_sim_corr flag.

Changing the shape similarity implementation affects the results. We provide new results corresponding to the tables 1, 2 and 3 in the [following response](https://github.com/vukasin-stanojevic/BoostTrack/issues/8).

We thank Luong Duc Trong for detecting the bug.
