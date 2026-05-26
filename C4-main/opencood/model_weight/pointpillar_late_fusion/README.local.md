This directory contains the downloaded official OPV2V PointPillar late-fusion
checkpoint and has been adapted for local runtime testing.

Runtime layout:
- `config.yaml`: configured for `LateFusionDataset` + `point_pillar`.
- `net_epoch30.pth`: downloaded checkpoint.
- `latest.pth`: symlink to `net_epoch30.pth` for loader/script compatibility.

Local adaptation:
- `root_dir`: `v2xset/validate`
- `validate_dir`: `v2xset/validate`

Important:
- The checkpoint was trained for OPV2V late fusion.
- Running it on local V2XSet data is useful for testing the late-fusion
  pipeline and visualization flow, but it is not a formal V2XSet late-fusion
  benchmark result.

Recommended smoke-test command:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/model_weight/pointpillar_late_fusion \
  --fusion_method late \
  --frame_index 1 \
  --num_frames 1 \
  --max_frames 2 \
  --save_vis_dir logs/inference/official_late \
  --color_mode constant \
  --headless
```
