#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="data/hspot"
GT_ROOT="results/gt"
ALLOW_MISSING_GT=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/setup_hspot_trackeval_gt.sh [--data-root PATH] [--gt-root PATH] [--allow-missing-gt]

Description:
  Prepares TrackEval ground-truth layout for hspot:
    - <gt-root>/hspot-val/<SEQ>/gt/gt.txt
    - <gt-root>/hspot-val/<SEQ>/seqinfo.ini
    - <gt-root>/hspot-test/<SEQ>/gt/gt.txt
    - <gt-root>/hspot-test/<SEQ>/seqinfo.ini
    - <gt-root>/seqmaps/hspot-val.txt
    - <gt-root>/seqmaps/hspot-test.txt

Options:
  --data-root PATH       Source root (default: data/hspot)
  --gt-root PATH         Destination TrackEval GT root (default: results/gt)
  --allow-missing-gt     Skip sequences missing gt/gt.txt instead of failing
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --gt-root)
      GT_ROOT="$2"
      shift 2
      ;;
    --allow-missing-gt)
      ALLOW_MISSING_GT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

prepare_split() {
  local split="$1"
  local src_split="$DATA_ROOT/$split"
  local dst_split="$GT_ROOT/hspot-$split"
  local seqmap_file="$GT_ROOT/seqmaps/hspot-$split.txt"
  local tmp_seqmap
  local copied_count=0
  local skipped_count=0

  if [[ ! -d "$src_split" ]]; then
    echo "Missing split directory: $src_split" >&2
    exit 1
  fi

  mkdir -p "$dst_split" "$GT_ROOT/seqmaps"
  tmp_seqmap="$(mktemp)"
  echo "name" > "$tmp_seqmap"

  while IFS= read -r seq_path; do
    local seq
    local src_seq
    local src_gt
    local src_ini
    local dst_seq

    seq="$(basename "$seq_path")"
    src_seq="$src_split/$seq"
    src_gt="$src_seq/gt/gt.txt"
    src_ini="$src_seq/seqinfo.ini"
    dst_seq="$dst_split/$seq"

    if [[ ! -f "$src_ini" ]]; then
      echo "Missing seqinfo.ini: $src_ini" >&2
      exit 1
    fi

    if [[ ! -f "$src_gt" ]]; then
      if [[ "$ALLOW_MISSING_GT" -eq 1 ]]; then
        echo "Skipping $split/$seq (missing gt): $src_gt"
        skipped_count=$((skipped_count + 1))
        continue
      fi
      echo "Missing gt file: $src_gt" >&2
      echo "If this is expected, rerun with --allow-missing-gt" >&2
      exit 1
    fi

    mkdir -p "$dst_seq/gt"
    cp "$src_gt" "$dst_seq/gt/gt.txt"
    cp "$src_ini" "$dst_seq/seqinfo.ini"
    echo "$seq" >> "$tmp_seqmap"
    copied_count=$((copied_count + 1))
  done < <(find "$src_split" -mindepth 1 -maxdepth 1 -type d | sort)

  mv "$tmp_seqmap" "$seqmap_file"
  echo "Prepared split '$split': copied=$copied_count skipped=$skipped_count"
  echo "  seqmap: $seqmap_file"
  echo "  gt dir: $dst_split"
}

prepare_split "val"
prepare_split "test"

echo "Done. You can now run Optuna tuning with:"
echo "  python3 tools/tune_boosttrack_optuna.py --dataset hspot --benchmark hspot --gpu-id 0"
