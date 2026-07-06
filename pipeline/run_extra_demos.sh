#!/usr/bin/env bash
# Render two ADDITIONAL demo clips from segments of the already-downloaded
# set03 videos, without overwriting the existing demo_out/ files.
#
# The demo script names its output by --video-id, and that same id drives the
# ego-speed lookup from pie_annotations.pkl. So we render each segment with the
# CORRECT id into a scratch dir, then move + rename the result into demo_out/
# under a fresh name. Originals (demo_video_0012/0016.mp4) stay untouched.
#
# Usage:  bash run_extra_demos.sh
set -euo pipefail
# Run from the repository root (this script lives in pipeline/).
cd "$(dirname "$0")/.."
source .venv/bin/activate

WEIGHTS="paper_and_artifacts/runs/bilstm_baseline"
OUT="pipeline/demo_out"
SCRATCH="pipeline/demo_out/_scratch"
mkdir -p "$OUT" "$SCRATCH"

# segment: video_id  raw_clip  start_frame  max_frames  output_name
run_segment () {
  local vid="$1" clip="$2" start="$3" maxf="$4" name="$5"
  echo ""
  echo "=============================================================="
  echo " Rendering '$name'  ($vid frames $start-$((start+maxf)))"
  echo "=============================================================="
  python pipeline/10_yolo_bytetrack_demo.py --stage demo \
    --video "$clip" --video-id "$vid" \
    --start-frame "$start" --max-frames "$maxf" \
    --weights-dir "$WEIGHTS" --dump-csv --out-dir "$SCRATCH"

  # Move the produced artifacts into demo_out/ with the new name.
  mv "$SCRATCH/demo_${vid}.mp4" "$OUT/${name}.mp4"
  if [ -f "$SCRATCH/demo_${vid}_predictions.csv" ]; then
    mv "$SCRATCH/demo_${vid}_predictions.csv" "$OUT/${name}_predictions.csv"
  fi
  # keep one sample frame for reference, drop the rest of the scratch PNGs
  rm -f "$SCRATCH/"demo_${vid}_f*.png
  echo "  -> $OUT/${name}.mp4"
}

# Clip 1: busiest crowd in set03 (21 crossing pedestrians)
run_segment "video_0016" "PIE_clips/set03/video_0016.mp4" 2816 900 "demo_busy_crowd_v0016"

# Clip 2: moving vehicle + crossers, different segment than the existing one
run_segment "video_0012" "PIE_clips/set03/video_0012.mp4" 6026 900 "demo_crossers_v0012"

rmdir "$SCRATCH" 2>/dev/null || true
echo ""
echo "Done. New clips:"
ls -lh "$OUT"/demo_busy_crowd_v0016.mp4 "$OUT"/demo_crossers_v0012.mp4
