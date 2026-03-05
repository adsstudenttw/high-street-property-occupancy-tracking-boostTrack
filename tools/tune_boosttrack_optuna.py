#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.trial import TrialState

from default_settings import BoostTrackSettings, GeneralSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
FLOAT_PARAM_KEYS = {
    "det_thresh",
    "iou_threshold",
    "lambda_iou",
    "lambda_mhd",
    "lambda_shape",
    "dlo_boost_coef",
}
INT_PARAM_KEYS = {"min_hits", "max_age"}
BOOL_INT_PARAM_KEYS = {"use_dlo_boost", "use_duo_boost"}
ALL_PARAM_KEYS = FLOAT_PARAM_KEYS | INT_PARAM_KEYS | BOOL_INT_PARAM_KEYS


def parse_args():
    """Parse CLI arguments for tuning, evaluation, and MLflow logging."""
    parser = argparse.ArgumentParser(
        "Optuna-based BoostTrack hyperparameter tuning (validation HOTA) + final test evaluation."
    )
    parser.add_argument("--dataset", type=str, default="mot17", help="Dataset name passed to main.py.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="TrackEval benchmark name (default: inferred from --dataset).",
    )
    parser.add_argument("--study-name", type=str, default="boosttrack_hota_tuning")
    parser.add_argument("--study-db", type=str, default="results/optuna/boosttrack_hota_tuning.db")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="Single GPU id to use.")
    parser.add_argument("--gt-folder", type=str, default="results/gt/")
    parser.add_argument("--trackers-folder", type=str, default="results/trackers/")
    parser.add_argument(
        "--pruning-seqs",
        type=int,
        default=2,
        help="Evaluate this many train sequences first and prune weak trials early. 0 disables subset stage.",
    )
    parser.add_argument(
        "--skip-train-pruning",
        action="store_true",
        help="Skip the train subset stage and evaluate trials directly on val.",
    )
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop study if no best-trial improvement after this many completed trials. 0 disables.",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    # Runtime switches forwarded to main.py.
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--no-cmc", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--s-sim-corr", action="store_true")
    parser.add_argument("--btpp-arg-iou-boost", action="store_true")
    parser.add_argument("--btpp-arg-no-sb", action="store_true")
    parser.add_argument("--btpp-arg-no-vt", action="store_true")

    # Search space.
    parser.add_argument("--det-thresh-min", type=float, default=0.2)
    parser.add_argument("--det-thresh-max", type=float, default=0.8)
    parser.add_argument("--iou-thresh-min", type=float, default=0.1)
    parser.add_argument("--iou-thresh-max", type=float, default=0.7)
    parser.add_argument("--min-hits-min", type=int, default=1)
    parser.add_argument("--min-hits-max", type=int, default=7)
    parser.add_argument("--max-age-min", type=int, default=15)
    parser.add_argument("--max-age-max", type=int, default=120)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--dlo-boost-min", type=float, default=0.3)
    parser.add_argument("--dlo-boost-max", type=float, default=0.9)

    parser.add_argument("--skip-final-test-eval", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--fixed-defaults",
        action="store_true",
        help="Fix tuned hyperparameters to default BoostTrack settings for the selected dataset.",
    )
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=[],
        help=(
            "Fix/override one tuned parameter as key=value. "
            "Valid keys: det_thresh,iou_threshold,min_hits,max_age,lambda_iou,lambda_mhd,lambda_shape,"
            "dlo_boost_coef,use_dlo_boost,use_duo_boost. Can be repeated."
        ),
    )

    # MLflow (supports external hosted tracking servers).
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (e.g. https://mlflow.myorg.com). If omitted, MLflow is disabled.",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="BoostTrack-Optuna")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument(
        "--mlflow-tag",
        action="append",
        default=[],
        help="Extra MLflow tag in key=value format. Can be passed multiple times.",
    )
    parser.add_argument("--mlflow-log-optuna-db", action="store_true")
    parser.add_argument("--mlflow-log-summary-json", action="store_true")
    return parser.parse_args()


def infer_benchmark(dataset):
    """Infer TrackEval benchmark name from a dataset identifier."""
    mapping = {"mot17": "MOT17", "mot20": "MOT20", "hspot": "hspot"}
    return mapping.get(dataset.lower(), dataset.upper())


def default_tuning_params(dataset):
    """Return default BoostTrack hyperparameters for the selected dataset."""
    dataset_key = dataset.lower()
    det_thresh = GeneralSettings.dataset_specific_settings.get(dataset_key, {}).get(
        "det_thresh", GeneralSettings.values["det_thresh"]
    )
    dlo_boost_coef = BoostTrackSettings.dataset_specific_settings.get(dataset_key, {}).get(
        "dlo_boost_coef", BoostTrackSettings.values["dlo_boost_coef"]
    )
    return {
        "det_thresh": float(det_thresh),
        "iou_threshold": float(GeneralSettings.values["iou_threshold"]),
        "min_hits": int(GeneralSettings.values["min_hits"]),
        "max_age": int(GeneralSettings.values["max_age"]),
        "lambda_iou": float(BoostTrackSettings.values["lambda_iou"]),
        "lambda_mhd": float(BoostTrackSettings.values["lambda_mhd"]),
        "lambda_shape": float(BoostTrackSettings.values["lambda_shape"]),
        "dlo_boost_coef": float(dlo_boost_coef),
        "use_dlo_boost": int(bool(BoostTrackSettings.values["use_dlo_boost"])),
        "use_duo_boost": int(bool(BoostTrackSettings.values["use_duo_boost"])),
    }


def parse_fixed_params(raw_items):
    """Parse repeated fixed parameter overrides from key=value inputs."""
    fixed_params = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --fixed-param '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in ALL_PARAM_KEYS:
            raise ValueError(
                f"Invalid --fixed-param key '{key}'. "
                "Expected one of det_thresh,iou_threshold,min_hits,max_age,lambda_iou,"
                "lambda_mhd,lambda_shape,dlo_boost_coef,use_dlo_boost,use_duo_boost."
            )
        if raw_value == "":
            raise ValueError(f"Invalid --fixed-param '{item}'. Value cannot be empty.")
        if key in FLOAT_PARAM_KEYS:
            value = float(raw_value)
        elif key in INT_PARAM_KEYS:
            value = int(raw_value)
        else:
            lowered = raw_value.lower()
            if lowered in {"0", "false"}:
                value = 0
            elif lowered in {"1", "true"}:
                value = 1
            else:
                raise ValueError(
                    f"Invalid --fixed-param '{item}'. Boolean parameters expect 0/1/true/false."
                )
        fixed_params[key] = value
    return fixed_params


def resolve_fixed_params(args):
    """Resolve fixed parameter values from defaults and explicit key=value overrides."""
    fixed_params = {}
    if args.fixed_defaults:
        fixed_params.update(default_tuning_params(args.dataset))
    fixed_params.update(parse_fixed_params(args.fixed_param))
    return fixed_params


def parse_mlflow_tags(raw_tags):
    """Convert repeated key=value tag arguments into an MLflow tag dictionary."""
    tags = {}
    for item in raw_tags:
        if "=" not in item:
            raise ValueError(f"Invalid --mlflow-tag '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "":
            raise ValueError(f"Invalid --mlflow-tag '{item}'. Key cannot be empty.")
        tags[key] = value
    return tags


def init_mlflow(args):
    """Initialize MLflow client if tracking is enabled, otherwise return disabled state."""
    tracking_uri = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None, None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking was requested but mlflow is not installed. "
            "Install dependencies from boost-track-env.yml or pip install mlflow."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    return mlflow, tracking_uri


def run_cmd(cmd, env):
    """Execute a subprocess command in the repository root with the provided environment."""
    print("$", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def cleanup_experiment(trackers_folder, benchmark, split_name, exp_name):
    """Remove previous tracker output folders for a given experiment name and split."""
    split_root = Path(trackers_folder) / f"{benchmark}-{split_name}"
    for suffix in ("", "_post", "_post_gbi"):
        target = split_root / f"{exp_name}{suffix}"
        if target.exists():
            shutil.rmtree(target)


def tracker_name_from_exp(exp_name, no_post):
    """Resolve the tracker folder name used by TrackEval for a given experiment."""
    return exp_name if no_post else f"{exp_name}_post_gbi"


def parse_hota(summary_file):
    """Read TrackEval summary output and return the scalar HOTA value."""
    if not summary_file.exists():
        raise RuntimeError(f"TrackEval summary not found: {summary_file}")
    with summary_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=" ")
        rows = [[cell for cell in row if cell != ""] for row in reader if row]
    if len(rows) < 2:
        raise RuntimeError(f"Unexpected summary format in {summary_file}")
    headers = rows[0]
    values = rows[1]
    result = dict(zip(headers, values))
    if "HOTA" not in result:
        raise RuntimeError(f"HOTA field missing in {summary_file}")
    return float(result["HOTA"])


def load_seqmap_sequences(gt_folder, benchmark, split_name):
    """Load sequence names from a TrackEval seqmap file."""
    seqmap = Path(gt_folder) / "seqmaps" / f"{benchmark}-{split_name}.txt"
    if not seqmap.exists():
        raise RuntimeError(
            f"Seqmap not found at {seqmap}. Set --pruning-seqs 0 or provide GT/seqmaps for {benchmark}-{split_name}."
        )
    sequences = []
    with seqmap.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0 or not row:
                continue
            seq = row[0].strip()
            if seq:
                sequences.append(seq)
    if not sequences:
        raise RuntimeError(f"No sequences found in {seqmap}")
    return sequences


def build_main_cmd(args, exp_name, split_name, params, seq_subset):
    """Build the command line to run `main.py` for one trial configuration."""
    cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--exp_name",
        exp_name,
        "--split",
        split_name,
    ]
    if split_name == "test":
        cmd.append("--test_dataset")
    if args.no_reid:
        cmd.append("--no_reid")
    if args.no_cmc:
        cmd.append("--no_cmc")
    if args.no_post:
        cmd.append("--no_post")
    if args.s_sim_corr:
        cmd.append("--s_sim_corr")
    if args.btpp_arg_iou_boost:
        cmd.append("--btpp_arg_iou_boost")
    if args.btpp_arg_no_sb:
        cmd.append("--btpp_arg_no_sb")
    if args.btpp_arg_no_vt:
        cmd.append("--btpp_arg_no_vt")
    if seq_subset:
        cmd += ["--seqs", ",".join(seq_subset)]

    cmd += ["--hp_det_thresh", f"{params['det_thresh']:.6f}"]
    cmd += ["--hp_iou_threshold", f"{params['iou_threshold']:.6f}"]
    cmd += ["--hp_min_hits", str(params["min_hits"])]
    cmd += ["--hp_max_age", str(params["max_age"])]
    cmd += ["--hp_lambda_iou", f"{params['lambda_iou']:.6f}"]
    cmd += ["--hp_lambda_mhd", f"{params['lambda_mhd']:.6f}"]
    cmd += ["--hp_lambda_shape", f"{params['lambda_shape']:.6f}"]
    cmd += ["--hp_dlo_boost_coef", f"{params['dlo_boost_coef']:.6f}"]
    cmd += ["--hp_use_dlo_boost", str(params["use_dlo_boost"])]
    cmd += ["--hp_use_duo_boost", str(params["use_duo_boost"])]
    return cmd


def build_eval_cmd(args, tracker_name, split_name, seq_subset):
    """Build the TrackEval command line for HOTA evaluation."""
    cmd = [
        sys.executable,
        "external/TrackEval/scripts/run_mot_challenge.py",
        "--SPLIT_TO_EVAL",
        split_name,
        "--GT_FOLDER",
        args.gt_folder,
        "--TRACKERS_FOLDER",
        args.trackers_folder,
        "--BENCHMARK",
        args.benchmark,
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--METRICS",
        "HOTA",
        "--PRINT_CONFIG",
        "False",
        "--PLOT_CURVES",
        "False",
        "--OUTPUT_DETAILED",
        "False",
        "--TIME_PROGRESS",
        "False",
    ]
    if seq_subset:
        cmd += ["--SEQ_INFO"] + seq_subset
    return cmd


def evaluate_run(args, env, exp_name, split_name, seq_subset, params):
    """Run tracking + TrackEval and return HOTA for one split."""
    cleanup_experiment(args.trackers_folder, args.benchmark, split_name, exp_name)
    run_cmd(build_main_cmd(args, exp_name, split_name, params, seq_subset), env=env)

    tracker_name = tracker_name_from_exp(exp_name, args.no_post)
    run_cmd(build_eval_cmd(args, tracker_name, split_name, seq_subset), env=env)

    summary_file = (
        Path(args.trackers_folder)
        / f"{args.benchmark}-{split_name}"
        / tracker_name
        / "pedestrian_summary.txt"
    )
    return parse_hota(summary_file)


class EarlyStoppingCallback:
    def __init__(self, patience, min_delta):
        """Create a callback that stops study optimization after stagnation."""
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_count = 0

    def __call__(self, study, trial):
        """Update stagnation counters and stop the study when patience is exceeded."""
        if self.patience <= 0 or trial.state != TrialState.COMPLETE:
            return
        value = trial.value
        if self.best is None or value > self.best + self.min_delta:
            self.best = value
            self.bad_count = 0
            return
        self.bad_count += 1
        if self.bad_count >= self.patience:
            print(
                f"Early stopping study: no improvement greater than {self.min_delta} "
                f"for {self.bad_count} completed trials."
            )
            study.stop()


def resolve_output_paths(args):
    """Resolve and create output paths for summary JSON and Optuna SQLite storage."""
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else REPO_ROOT / "results" / "optuna" / f"{args.study_name}_summary.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.study_db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path}"
    return output_json, db_path, storage


def build_runtime_env(args):
    """Build the subprocess environment and pin execution to the selected GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return env


def start_parent_mlflow_run(mlflow, args, tracking_uri, mlflow_tags, fixed_params):
    """Start and initialize the parent MLflow run for the whole tuning session."""
    if mlflow is None:
        return False
    parent_run_name = args.mlflow_run_name or f"{args.study_name}_{args.dataset}"
    mlflow.start_run(run_name=parent_run_name)
    mlflow.set_tags(
        {
            "dataset": args.dataset,
            "benchmark": args.benchmark,
            "optimizer": "optuna_tpe",
            "pruner": "median",
            "objective": "HOTA",
        }
    )
    if mlflow_tags:
        mlflow.set_tags(mlflow_tags)
    mlflow.log_params(
        {
            "study_name": args.study_name,
            "n_trials": args.n_trials,
            "timeout_sec": args.timeout_sec if args.timeout_sec is not None else -1,
            "seed": args.seed,
            "gpu_id": args.gpu_id,
            "train_split": TRAIN_SPLIT,
            "eval_split": VAL_SPLIT,
            "final_split": TEST_SPLIT,
            "pruning_seqs": args.pruning_seqs,
            "skip_train_pruning": int(args.skip_train_pruning),
            "pruner_startup_trials": args.pruner_startup_trials,
            "pruner_warmup_steps": args.pruner_warmup_steps,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "no_reid": int(args.no_reid),
            "no_cmc": int(args.no_cmc),
            "no_post": int(args.no_post),
            "s_sim_corr": int(args.s_sim_corr),
            "btpp_arg_iou_boost": int(args.btpp_arg_iou_boost),
            "btpp_arg_no_sb": int(args.btpp_arg_no_sb),
            "btpp_arg_no_vt": int(args.btpp_arg_no_vt),
            "mlflow_tracking_uri": tracking_uri,
            "fixed_defaults": int(args.fixed_defaults),
        }
    )
    if fixed_params:
        mlflow.set_tag("fixed_params", json.dumps(fixed_params, sort_keys=True))
    return True


def determine_train_subset(args, mlflow):
    """Select train subset sequences used for pruning-stage evaluation."""
    if args.skip_train_pruning or args.pruning_seqs <= 0:
        return None
    all_train_sequences = load_seqmap_sequences(args.gt_folder, args.benchmark, TRAIN_SPLIT)
    train_subset = all_train_sequences[: min(args.pruning_seqs, len(all_train_sequences))]
    print(f"Pruning subset ({TRAIN_SPLIT}): {train_subset}")
    if mlflow is not None:
        mlflow.set_tag("train_pruning_subset", ",".join(train_subset))
    return train_subset


def create_study(args, storage):
    """Create or load an Optuna study configured with TPE sampler and median pruner."""
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.pruner_startup_trials)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials, n_warmup_steps=args.pruner_warmup_steps
    )
    return optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


def suggest_trial_params(trial, args, fixed_params):
    """Sample one trial's hyperparameters from the configured search space."""
    params = {
        "det_thresh": trial.suggest_float(
            "det_thresh",
            float(fixed_params["det_thresh"]) if "det_thresh" in fixed_params else args.det_thresh_min,
            float(fixed_params["det_thresh"]) if "det_thresh" in fixed_params else args.det_thresh_max,
        ),
        "iou_threshold": trial.suggest_float(
            "iou_threshold",
            float(fixed_params["iou_threshold"]) if "iou_threshold" in fixed_params else args.iou_thresh_min,
            float(fixed_params["iou_threshold"]) if "iou_threshold" in fixed_params else args.iou_thresh_max,
        ),
        "min_hits": trial.suggest_int(
            "min_hits",
            int(fixed_params["min_hits"]) if "min_hits" in fixed_params else args.min_hits_min,
            int(fixed_params["min_hits"]) if "min_hits" in fixed_params else args.min_hits_max,
        ),
        "max_age": trial.suggest_int(
            "max_age",
            int(fixed_params["max_age"]) if "max_age" in fixed_params else args.max_age_min,
            int(fixed_params["max_age"]) if "max_age" in fixed_params else args.max_age_max,
        ),
        "lambda_iou": trial.suggest_float(
            "lambda_iou",
            float(fixed_params["lambda_iou"]) if "lambda_iou" in fixed_params else args.lambda_min,
            float(fixed_params["lambda_iou"]) if "lambda_iou" in fixed_params else args.lambda_max,
        ),
        "lambda_mhd": trial.suggest_float(
            "lambda_mhd",
            float(fixed_params["lambda_mhd"]) if "lambda_mhd" in fixed_params else args.lambda_min,
            float(fixed_params["lambda_mhd"]) if "lambda_mhd" in fixed_params else args.lambda_max,
        ),
        "lambda_shape": trial.suggest_float(
            "lambda_shape",
            float(fixed_params["lambda_shape"]) if "lambda_shape" in fixed_params else args.lambda_min,
            float(fixed_params["lambda_shape"]) if "lambda_shape" in fixed_params else args.lambda_max,
        ),
        "dlo_boost_coef": trial.suggest_float(
            "dlo_boost_coef",
            float(fixed_params["dlo_boost_coef"]) if "dlo_boost_coef" in fixed_params else args.dlo_boost_min,
            float(fixed_params["dlo_boost_coef"]) if "dlo_boost_coef" in fixed_params else args.dlo_boost_max,
        ),
        "use_dlo_boost": trial.suggest_categorical(
            "use_dlo_boost", [int(fixed_params["use_dlo_boost"])] if "use_dlo_boost" in fixed_params else [0, 1]
        ),
        "use_duo_boost": trial.suggest_categorical(
            "use_duo_boost", [int(fixed_params["use_duo_boost"])] if "use_duo_boost" in fixed_params else [0, 1]
        ),
    }
    return params


def run_trial(args, env, mlflow, train_subset, trial, params):
    """Execute one Optuna trial, including train pruning stage and full validation stage."""
    trial_run_started = False
    trial_run_status = "FAILED"
    trial_prefix = f"{args.study_name}_trial_{trial.number:04d}"

    if mlflow is not None:
        mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=True)
        trial_run_started = True
        mlflow.set_tags({"optuna_trial_number": trial.number})
        mlflow.log_params(params)

    try:
        if train_subset:
            prune_exp = f"{trial_prefix}_prune"
            prune_hota = evaluate_run(args, env, prune_exp, TRAIN_SPLIT, train_subset, params)
            trial.report(prune_hota, step=0)
            trial.set_user_attr("subset_train_hota", prune_hota)
            if mlflow is not None:
                mlflow.log_metric("subset_train_hota", prune_hota, step=0)
            if trial.should_prune():
                if mlflow is not None:
                    mlflow.set_tag("trial_state", "PRUNED")
                trial_run_status = "KILLED"
                raise optuna.TrialPruned(f"Pruned after subset HOTA={prune_hota:.4f}")

        full_exp = f"{trial_prefix}_full"
        val_hota = evaluate_run(args, env, full_exp, VAL_SPLIT, None, params)
        trial.report(val_hota, step=1)
        trial.set_user_attr("val_exp_name", full_exp)
        if mlflow is not None:
            mlflow.log_metric("val_hota", val_hota, step=1)
            mlflow.set_tags({"trial_state": "COMPLETE", "val_exp_name": full_exp})
        trial_run_status = "FINISHED"
        return val_hota
    except optuna.TrialPruned:
        if mlflow is not None:
            mlflow.set_tag("trial_state", "PRUNED")
        if trial_run_status != "KILLED":
            trial_run_status = "KILLED"
        raise
    except Exception:
        if mlflow is not None:
            mlflow.set_tag("trial_state", "FAILED")
        trial_run_status = "FAILED"
        raise
    finally:
        if mlflow is not None and trial_run_started:
            mlflow.end_run(status=trial_run_status)


def build_objective(args, env, mlflow, train_subset, fixed_params):
    """Build the Optuna objective closure bound to runtime context."""
    def objective(trial):
        """Evaluate a sampled trial and return validation HOTA."""
        params = suggest_trial_params(trial, args, fixed_params)
        return run_trial(args, env, mlflow, train_subset, trial, params)

    return objective


def run_optimization(args, study, objective):
    """Run Optuna optimization with early-stopping callback support."""
    callbacks = [EarlyStoppingCallback(args.early_stop_patience, args.early_stop_min_delta)]
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout_sec, callbacks=callbacks)


def get_best_trial(study):
    """Return the best completed trial or raise when none exist."""
    if len(study.trials) == 0 or study.best_trial is None:
        raise RuntimeError("No successful Optuna trials were completed.")
    return study.best_trial


def evaluate_best_on_test(args, env, best_params, mlflow):
    """Evaluate the best validation parameters on the test split if enabled."""
    if args.skip_final_test_eval:
        return None, None
    final_test_exp = f"{args.study_name}_best_test"
    test_hota = evaluate_run(args, env, final_test_exp, TEST_SPLIT, None, best_params)
    print(f"Final test HOTA: {test_hota:.4f}")
    if mlflow is not None:
        mlflow.log_metric("test_hota", test_hota)
    return final_test_exp, test_hota


def build_summary(
    args, study, best_trial, best_params, final_test_exp, test_hota, db_path, tracking_uri, fixed_params
):
    """Assemble the final summary payload for JSON persistence and logging."""
    return {
        "study_name": args.study_name,
        "dataset": args.dataset,
        "benchmark": args.benchmark,
        "gpu_id": args.gpu_id,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "best_trial_number": best_trial.number,
        "best_val_hota": best_trial.value,
        "best_params": best_params,
        "best_val_exp_name": best_trial.user_attrs.get("val_exp_name"),
        "final_test_exp_name": final_test_exp,
        "final_test_hota": test_hota,
        "optuna_db": str(db_path),
        "mlflow_tracking_uri": tracking_uri,
        "fixed_params": fixed_params,
    }


def save_summary(summary, output_json):
    """Write the final summary JSON to disk."""
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved tuning summary to {output_json}")


def log_parent_mlflow_results(mlflow, args, summary, output_json, db_path):
    """Log final best metrics, tags, and optional artifacts to the parent MLflow run."""
    if mlflow is None:
        return
    mlflow.log_metric("best_val_hota", summary["best_val_hota"])
    mlflow.set_tags(
        {
            "best_trial_number": summary["best_trial_number"],
            "best_val_exp_name": summary.get("best_val_exp_name", ""),
        }
    )
    if args.mlflow_log_summary_json:
        mlflow.log_artifact(str(output_json))
    if args.mlflow_log_optuna_db:
        mlflow.log_artifact(str(db_path))


def main():
    """Orchestrate the full tuning workflow from setup to final reporting."""
    args = parse_args()
    args.benchmark = args.benchmark or infer_benchmark(args.dataset)
    fixed_params = resolve_fixed_params(args)
    mlflow_tags = parse_mlflow_tags(args.mlflow_tag)
    output_json, db_path, storage = resolve_output_paths(args)
    env = build_runtime_env(args)
    mlflow, tracking_uri = init_mlflow(args)
    parent_run_started = False
    parent_run_status = "FAILED"

    try:
        parent_run_started = start_parent_mlflow_run(
            mlflow, args, tracking_uri, mlflow_tags, fixed_params
        )
        train_subset = determine_train_subset(args, mlflow)
        study = create_study(args, storage)
        objective = build_objective(args, env, mlflow, train_subset, fixed_params)
        run_optimization(args, study, objective)
        best_trial = get_best_trial(study)
        best_params = dict(best_trial.params) 
        final_test_exp, test_hota = evaluate_best_on_test(args, env, best_params, mlflow)
        summary = build_summary(
            args, study, best_trial, best_params, final_test_exp, test_hota, db_path, tracking_uri, fixed_params
        )
        save_summary(summary, output_json)
        log_parent_mlflow_results(mlflow, args, summary, output_json, db_path)
        parent_run_status = "FINISHED"
    finally:
        if mlflow is not None and parent_run_started:
            mlflow.end_run(status=parent_run_status)


if __name__ == "__main__":
    main()
