import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_id_list(raw):
    if raw is None or raw.strip() == "":
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip() != ""}


def parse_args():
    parser = argparse.ArgumentParser("Convert hspot MOT-format data to COCO JSON.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hspot",
        help="Root path of hspot dataset.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split folders to process under <data-path>.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Output annotations directory. Defaults to <data-path>/annotations.",
    )
    parser.add_argument(
        "--person-class-ids",
        type=str,
        default="1",
        help="Comma-separated class ids treated as person. Leave empty to keep all classes.",
    )
    parser.add_argument(
        "--ignored-class-ids",
        type=str,
        default="2,7,8,12",
        help="Comma-separated class ids to ignore.",
    )
    parser.add_argument(
        "--require-marked",
        action="store_true",
        help="If set, keep only rows with mark/conf >= 1 (column 7).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=None,
        help="If set, keep only rows with visibility >= this threshold (column 9).",
    )
    parser.add_argument(
        "--keep-original-track-ids",
        action="store_true",
        help="Use original track ids directly instead of global reindexing per sequence.",
    )
    return parser.parse_args()


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def sorted_image_files(img_dir):
    files = [f for f in os.listdir(img_dir) if is_image_file(f)]

    def key_fn(name):
        stem = Path(name).stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(files, key=key_fn)


def load_mot_annotations(path):
    if not os.path.exists(path):
        return np.zeros((0, 10), dtype=np.float32)
    anns = np.loadtxt(path, dtype=np.float32, delimiter=",", ndmin=2)
    if anns.size == 0:
        return np.zeros((0, 10), dtype=np.float32)
    return anns


def should_keep_row(
    row, person_class_ids, ignored_class_ids, require_marked, min_visibility
):
    mark = float(row[6]) if row.shape[0] > 6 else 1.0
    cls_id = int(row[7]) if row.shape[0] > 7 else 1
    vis = float(row[8]) if row.shape[0] > 8 else 1.0

    if require_marked and mark < 1:
        return False
    if min_visibility is not None and vis < min_visibility:
        return False
    if cls_id in ignored_class_ids:
        return False
    if person_class_ids and cls_id not in person_class_ids:
        return False
    return True


def convert_split(
    data_path,
    split,
    out_path,
    person_class_ids,
    ignored_class_ids,
    require_marked,
    min_visibility,
    keep_original_track_ids,
):
    split_root = os.path.join(data_path, split)

    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }

    if not os.path.isdir(split_root):
        raise RuntimeError(f"Split directory does not exist: {split_root}")

    seqs = sorted([seq for seq in os.listdir(split_root) if not seq.startswith(".")])
    image_count = 0
    ann_count = 0
    video_count = 0
    next_track_id = 1
    track_id_map = {}

    for seq in seqs:
        seq_path = os.path.join(split_root, seq)
        img_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")

        if not os.path.isdir(img_dir):
            continue

        files = sorted_image_files(img_dir)
        if len(files) == 0:
            continue

        video_count += 1
        out["videos"].append({"id": video_count, "file_name": seq})

        frame_to_image_id = {}
        split_start_idx = len(out["images"])
        split_image_ids = []

        for offset, file_name in enumerate(files, start=1):
            image_path = os.path.join(img_dir, file_name)
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]

            image_id = image_count + offset
            split_image_ids.append(image_id)

            frame_num_by_order = offset
            stem = Path(file_name).stem
            frame_num_by_name = int(stem) if stem.isdigit() else frame_num_by_order

            frame_to_image_id[frame_num_by_order] = image_id
            frame_to_image_id[frame_num_by_name] = image_id

            image_info = {
                "file_name": f"{seq}/img1/{file_name}",
                "id": image_id,
                "frame_id": frame_num_by_order,
                "prev_image_id": -1,
                "next_image_id": -1,
                "video_id": video_count,
                "height": height,
                "width": width,
            }
            out["images"].append(image_info)

        for idx, image_id in enumerate(split_image_ids):
            out_idx = split_start_idx + idx
            if idx > 0:
                out["images"][out_idx]["prev_image_id"] = split_image_ids[idx - 1]
            if idx < len(split_image_ids) - 1:
                out["images"][out_idx]["next_image_id"] = split_image_ids[idx + 1]

        if split != "test":
            anns = load_mot_annotations(gt_path)
            for row in anns:
                frame_id = int(row[0])
                if frame_id not in frame_to_image_id:
                    continue
                if not should_keep_row(
                    row,
                    person_class_ids,
                    ignored_class_ids,
                    require_marked,
                    min_visibility,
                ):
                    continue

                x, y, w, h = [float(v) for v in row[2:6]]
                if w <= 0 or h <= 0:
                    continue

                local_track_id = int(row[1])
                if keep_original_track_ids:
                    track_id = local_track_id
                else:
                    key = (seq, local_track_id)
                    if key not in track_id_map:
                        track_id_map[key] = next_track_id
                        next_track_id += 1
                    track_id = track_id_map[key]

                ann_count += 1
                out["annotations"].append(
                    {
                        "id": ann_count,
                        "category_id": 1,
                        "image_id": frame_to_image_id[frame_id],
                        "track_id": track_id,
                        "bbox": [x, y, w, h],
                        "conf": float(row[6]) if row.shape[0] > 6 else 1.0,
                        "iscrowd": 0,
                        "area": float(w * h),
                    }
                )

        image_count += len(files)
        print(f"{split} | {seq}: {len(files)} images")

    print(
        f"loaded {split} for {len(out['images'])} images and {len(out['annotations'])} samples"
    )
    json.dump(out, open(out_path, "w"))


def main():
    args = parse_args()
    data_path = args.data_path
    out_dir = (
        args.out_path
        if args.out_path is not None
        else os.path.join(data_path, "annotations")
    )
    os.makedirs(out_dir, exist_ok=True)

    splits = [x.strip() for x in args.splits.split(",") if x.strip() != ""]
    person_class_ids = parse_id_list(args.person_class_ids)
    ignored_class_ids = parse_id_list(args.ignored_class_ids)

    for split in splits:
        output_path = os.path.join(out_dir, f"{split}.json")
        convert_split(
            data_path=data_path,
            split=split,
            out_path=output_path,
            person_class_ids=person_class_ids,
            ignored_class_ids=ignored_class_ids,
            require_marked=args.require_marked,
            min_visibility=args.min_visibility,
            keep_original_track_ids=args.keep_original_track_ids,
        )


if __name__ == "__main__":
    main()
