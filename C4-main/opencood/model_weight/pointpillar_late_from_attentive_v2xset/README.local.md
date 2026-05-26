This directory is derived from `pointpillar_attentive_fusion_runtime_v2xset`.

Purpose:
- Run late-fusion inference with `LateFusionDataset` and `point_pillar`.
- Reuse the same `latest.pth` only as a practical compatibility test.
- This is not an official late-fusion trained checkpoint.

Changed from the original config:
- `fusion.core_method`: `IntermediateFusionDataset` -> `LateFusionDataset`
- `model.core_method`: `point_pillar_intermediate` -> `point_pillar`
- `name`: `pointpillar_late_from_attentive_v2xset`

Keep these geometry fields aligned unless you intentionally rebuild the full
late-fusion experiment config:
- `preprocess.cav_lidar_range`
- `postprocess.anchor_args.cav_lidar_range`
- `model.args.lidar_range`
