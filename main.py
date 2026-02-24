import os
import shutil
import time

import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack

"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""


def get_main_args():
    parser = make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used")
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used")

    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")

    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost.")
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used.")
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threhold should NOT be used for the detection confidence boost.")

    parser.add_argument(
        "--no_post",
        action="store_true",
        help="do not run post-processing.",
    )
    parser.add_argument(
        "--seqs",
        type=str,
        default=None,
        help="Optional comma-separated sequence names to process (used for quick subset runs).",
    )
    parser.add_argument("--hp_det_thresh", type=float, default=None)
    parser.add_argument("--hp_iou_threshold", type=float, default=None)
    parser.add_argument("--hp_min_hits", type=int, default=None)
    parser.add_argument("--hp_max_age", type=int, default=None)
    parser.add_argument("--hp_lambda_iou", type=float, default=None)
    parser.add_argument("--hp_lambda_mhd", type=float, default=None)
    parser.add_argument("--hp_lambda_shape", type=float, default=None)
    parser.add_argument("--hp_dlo_boost_coef", type=float, default=None)
    parser.add_argument("--hp_use_dlo_boost", type=int, choices=[0, 1], default=None)
    parser.add_argument("--hp_use_duo_boost", type=int, choices=[0, 1], default=None)

    args = parser.parse_args()
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "hspot":
        args.result_folder = os.path.join(args.result_folder, "HSPOT-val")

    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
    return args


def main():
    # Set dataset and detector
    args = get_main_args()
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr

    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    # Optional runtime hyperparameter overrides (used by tuning scripts).
    if args.hp_det_thresh is not None:
        GeneralSettings.values['det_thresh'] = args.hp_det_thresh
        if args.dataset in GeneralSettings.dataset_specific_settings:
            GeneralSettings.dataset_specific_settings[args.dataset]['det_thresh'] = args.hp_det_thresh
    if args.hp_iou_threshold is not None:
        GeneralSettings.values['iou_threshold'] = args.hp_iou_threshold
    if args.hp_min_hits is not None:
        GeneralSettings.values['min_hits'] = args.hp_min_hits
    if args.hp_max_age is not None:
        GeneralSettings.values['max_age'] = args.hp_max_age
    if args.hp_lambda_iou is not None:
        BoostTrackSettings.values['lambda_iou'] = args.hp_lambda_iou
    if args.hp_lambda_mhd is not None:
        BoostTrackSettings.values['lambda_mhd'] = args.hp_lambda_mhd
    if args.hp_lambda_shape is not None:
        BoostTrackSettings.values['lambda_shape'] = args.hp_lambda_shape
    if args.hp_dlo_boost_coef is not None:
        BoostTrackSettings.values['dlo_boost_coef'] = args.hp_dlo_boost_coef
        if args.dataset in BoostTrackSettings.dataset_specific_settings:
            BoostTrackSettings.dataset_specific_settings[args.dataset]['dlo_boost_coef'] = args.hp_dlo_boost_coef
    if args.hp_use_dlo_boost is not None:
        BoostTrackSettings.values['use_dlo_boost'] = bool(args.hp_use_dlo_boost)
    if args.hp_use_duo_boost is not None:
        BoostTrackSettings.values['use_duo_boost'] = bool(args.hp_use_duo_boost)

    detector_path, size = get_detector_path_and_im_size(args)
    det = detector.Detector("yolox", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size)
    seq_filter = None
    if args.seqs:
        seq_filter = {seq.strip() for seq in args.seqs.split(",") if seq.strip()}

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0
    # See __getitem__ of dataset.MOTDataset
    for (img, np_img), label, info, idx in loader:
        # Frame info
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        if seq_filter is not None and video_name not in seq_filter:
            continue

        # Hacky way to skip SDP and DPM when testing
        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        img = img.cuda()

        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()

            tracker = BoostTrack(video_name=video_name)

        pred = det(img, tag)
        start_time = time.time()

        if pred is None:
            continue
        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)
    # Save detector results
    det.dump_cache()
    if tracker is not None:
        tracker.dump_cache()
    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if not args.no_post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        interval = 1000  # i.e. no max interval
        utils.dti(post_folder_data, post_folder_data, n_dti=interval, n_min=25)

        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

        res_folder = os.path.join(args.result_folder, args.exp_name, "data")
        post_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "data")

        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(res_folder):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)

            GBInterpolation(
                path_in=in_path,
                path_out=out_path2,
                interval=interval
            )
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")


if __name__ == "__main__":
    main()
