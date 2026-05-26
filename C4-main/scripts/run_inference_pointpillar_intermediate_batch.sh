#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-opencood/logs/pointpillar_intermediate_bootstrap_v2xset}"
VALIDATE_DIR="${VALIDATE_DIR:-v2xset/validate}"
FRAME_INDEX="${FRAME_INDEX:-0}"
NUM_FRAMES="${NUM_FRAMES:-3}"
MAX_FRAMES="${MAX_FRAMES:-16}"
COLOR_MODE="${COLOR_MODE:-intensity}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/inference_pointpillar_intermediate_batch}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
POINT_SIZE="${POINT_SIZE:-1.0}"
BACKGROUND="${BACKGROUND:-dark}"
HEADLESS="${HEADLESS:-1}"

if [[ -f "$MODEL_DIR/config.yaml" ]] && [[ -f "$MODEL_DIR/latest.pth" ]]; then
  echo "Using existing model_dir: $MODEL_DIR"
else
  conda run -n torch118 python scripts/prepare_pointpillar_intermediate_bootstrap.py \
    --target_dir "$MODEL_DIR" \
    --validate_dir "$VALIDATE_DIR"
fi

EFFECTIVE_MAX_FRAMES="$MAX_FRAMES"
if [[ "$FRAME_INDEX" =~ ^- ]]; then
  EFFECTIVE_MAX_FRAMES=0
else
  min_needed=$(( FRAME_INDEX + NUM_FRAMES ))
  if [[ "$EFFECTIVE_MAX_FRAMES" -le 0 || "$EFFECTIVE_MAX_FRAMES" -lt "$min_needed" ]]; then
    EFFECTIVE_MAX_FRAMES="$min_needed"
  fi
fi

cmd=(
  conda run -n torch118 python opencood/tools/inference.py
  --model_dir "$MODEL_DIR"
  --fusion_method intermediate
  --frame_index "$FRAME_INDEX"
  --num_frames "$NUM_FRAMES"
  --color_mode "$COLOR_MODE"
  --save_vis_dir "$OUTPUT_DIR"
  --width "$WIDTH"
  --height "$HEIGHT"
  --point_size "$POINT_SIZE"
  --background "$BACKGROUND"
  --max_frames "$EFFECTIVE_MAX_FRAMES"
)

if [[ "$HEADLESS" == "1" ]]; then
  cmd+=(--headless)
fi

printf 'Running: '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
