# Frame2Video Command Examples

## Dry-run a Continuous OpenCOOD Sequence

```bash
python frame2video/frame2video.py \
  --input OpenCOOD-main/logs/trust/test6 \
  --pattern "late_constant_frame_*.png" \
  --output Results/videos/trust_test6_5fps.mp4 \
  --fps 5 \
  --dry-run
```

## Convert Adaptive Key Frames With OpenCV Fallback

Use this when FFmpeg is not installed in the current environment.

```bash
python frame2video/frame2video.py \
  --input Results/adaptive/baseline_no_trust/vis \
  --pattern "late_constant_frame_*.png" \
  --output Results/adaptive/videos/baseline_keyframes_opencv.mp4 \
  --fps 1 \
  --backend opencv \
  --overwrite
```

## Convert With FFmpeg

```bash
python frame2video/frame2video.py \
  --input Results/adaptive/trust_physical_sensitive/vis \
  --pattern "late_constant_frame_*.png" \
  --output Results/adaptive/videos/sensitive_keyframes.mp4 \
  --fps 1 \
  --backend ffmpeg \
  --overwrite
```

## Select a Frame Window

```bash
python frame2video/frame2video.py \
  --input OpenCOOD-main/logs/trust/test6 \
  --pattern "late_constant_frame_*.png" \
  --output Results/videos/test6_190_195.mp4 \
  --fps 5 \
  --start-frame 190 \
  --end-frame 195 \
  --dry-run
```
