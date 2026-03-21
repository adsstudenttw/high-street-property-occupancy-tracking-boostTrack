#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


Color = Tuple[int, int, int]
BoxRow = Tuple[int, int, float, float, float, float]

GT_COLOR: Color = (0, 200, 0)
PRED_COLOR: Color = (0, 140, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay hspot GT and tracking predictions on sequence frames."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/hspot",
        help="hspot dataset root containing train/val/test sequence folders.",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default="/results/gt",
        help="TrackEval GT root containing hspot-<split>/<seq>/gt/gt.txt.",
    )
    parser.add_argument(
        "--trackers-root",
        type=str,
        default="/results/trackers",
        help="Tracking outputs root containing hspot-<split>/<tracker>/data/<seq>.txt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--tracker-name",
        type=str,
        required=True,
        help="Tracker folder name inside hspot-<split>, e.g. hspot_baseline_val_trial_0005_full_post_gbi.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default=None,
        help="Optional single sequence to visualize. Defaults to all available sequence files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/results/visualizations",
        help="Directory where annotated frames will be written.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on frames per sequence. 0 means all frames.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.5,
        help="OpenCV font scale used for labels.",
    )
    return parser.parse_args()


def load_mot_rows(path: Path) -> Dict[int, List[BoxRow]]:
    rows: Dict[int, List[BoxRow]] = {}
    if not path.exists():
        return rows

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 6:
                continue
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            rows.setdefault(frame_id, []).append((frame_id, track_id, x, y, w, h))
    return rows


def draw_boxes(
    image,
    rows: List[BoxRow],
    color: Color,
    label_prefix: str,
    font_scale: float,
) -> None:
    for _, track_id, x, y, w, h in rows:
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}:{track_id}"
        label_origin = (x1, max(16, y1 - 6))
        cv2.putText(
            image,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )


def sequence_names(args: argparse.Namespace, tracker_data_dir: Path) -> List[str]:
    if args.seq:
        return [args.seq]
    return sorted(path.stem for path in tracker_data_dir.glob("*.txt"))


def render_sequence(
    seq: str,
    args: argparse.Namespace,
    pred_rows: Dict[int, List[BoxRow]],
    gt_rows: Dict[int, List[BoxRow]],
    output_dir: Path,
) -> None:
    img_dir = Path(args.data_root) / args.split / seq / "img1"
    if not img_dir.is_dir():
        raise RuntimeError(f"Image directory does not exist: {img_dir}")

    image_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}],
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    )
    if args.max_frames > 0:
        image_files = image_files[: args.max_frames]

    seq_output_dir = output_dir / seq
    seq_output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, image_path in enumerate(image_files, start=1):
        frame_id = int(image_path.stem) if image_path.stem.isdigit() else frame_index
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        draw_boxes(image, gt_rows.get(frame_id, []), GT_COLOR, "GT", args.font_scale)
        draw_boxes(image, pred_rows.get(frame_id, []), PRED_COLOR, "P", args.font_scale)

        out_path = seq_output_dir / image_path.name
        cv2.imwrite(str(out_path), image)


def main() -> None:
    args = parse_args()
    tracker_data_dir = (
        Path(args.trackers_root) / f"hspot-{args.split}" / args.tracker_name / "data"
    )
    if not tracker_data_dir.is_dir():
        raise RuntimeError(f"Tracker data directory does not exist: {tracker_data_dir}")

    output_dir = Path(args.output_root) / f"hspot-{args.split}" / args.tracker_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequence_names(args, tracker_data_dir):
        pred_file = tracker_data_dir / f"{seq}.txt"
        gt_file = Path(args.gt_root) / f"hspot-{args.split}" / seq / "gt" / "gt.txt"
        pred_rows = load_mot_rows(pred_file)
        gt_rows = load_mot_rows(gt_file)
        render_sequence(seq, args, pred_rows, gt_rows, output_dir)
        print(f"Rendered {seq} to {output_dir / seq}")


if __name__ == "__main__":
    main()
