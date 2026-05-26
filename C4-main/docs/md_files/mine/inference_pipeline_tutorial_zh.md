# 推理框架跑通教程

本文档覆盖从模型目录准备到效果图导出的完整链路，默认环境为 `torch118`。

## 1. 预训练权重现状

- 官方 Box 条目：`pointpillar_attentive_fusion.zip`
- 文件大小：`45,188,266` bytes，约 `43.1 MiB / 45.2 MB`
- 当前仓库里记录的 Google Drive 镜像已失效；CLI 自动下载未跑通。

因此，当前教程分两条线：

1. **先用 bootstrap checkpoint 跑通推理链路**  
   这会生成一个可加载的随机初始化 `model_dir`，用于验证“模型加载 -> 推理 -> 出图”整条框架。
2. **后续若你手动从浏览器拿到官方 zip**  
   只要把 `config.yaml` 和权重文件替换进同一个 `model_dir`，下面的推理脚本可以直接复用。

## 2. 解析你下载的官方权重 zip

如果你已经把 `pointpillar_attentive_fusion.zip` 放到 `opencood/model_weight/`，先执行：

```bash
conda run -n torch118 python scripts/prepare_downloaded_pointpillar_model.py \
  --zip_path opencood/model_weight/pointpillar_attentive_fusion.zip \
  --variant pointpillar_attentive_fusion \
  --target_dir opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset \
  --validate_dir v2xset/validate \
  --force
```

它会做三件事：

1. 解压 zip
2. 提取真正可用的 `config.yaml` 和 `latest.pth`
3. 把 `validate_dir` 改到你本地的 `v2xset/validate`

准备完成后，`MODEL_DIR` 用这个目录：

```bash
opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset
```

## 3. 准备 bootstrap model_dir

```bash
conda run -n torch118 python scripts/prepare_pointpillar_intermediate_bootstrap.py \
  --target_dir opencood/logs/pointpillar_intermediate_bootstrap_v2xset \
  --validate_dir v2xset/validate
```

输出目录里会生成：

- `config.yaml`
- `latest.pth`

这个 checkpoint 只用于跑通框架，不代表有效检测精度。

如果你暂时没有官方权重，才走这一条。

## 4. 单帧推理导出

```bash
MODEL_DIR=opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset \
bash scripts/run_inference_pointpillar_intermediate_single.sh
```

默认行为：

- `frame_index=0`
- `num_frames=1`
- `max_frames=16`
- `headless=1`
- 输出到 `logs/inference_pointpillar_intermediate_single`

如果你要改参数：

```bash
FRAME_INDEX=2 \
NUM_FRAMES=1 \
COLOR_MODE=z-value \
OUTPUT_DIR=logs/infer_single_z \
bash scripts/run_inference_pointpillar_intermediate_single.sh
```

## 5. 多帧效果图导出

```bash
MODEL_DIR=opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset \
bash scripts/run_inference_pointpillar_intermediate_batch.sh
```

默认导出前 3 帧到 `logs/inference_pointpillar_intermediate_batch`。

如果你想取最后一帧或最后几帧：

```bash
FRAME_INDEX=-1 \
NUM_FRAMES=1 \
MAX_FRAMES=0 \
OUTPUT_DIR=logs/infer_last \
bash scripts/run_inference_pointpillar_intermediate_batch.sh
```

注意：`FRAME_INDEX=-1` 表示最后一帧；这时必须把 `MAX_FRAMES=0`，否则脚本会提前停止。

对于正数帧索引，脚本现在会自动把 `MAX_FRAMES` 提高到足够覆盖目标帧，不需要手动算。

## 6. 只看预测框或只看 GT

只看预测框：

```bash
PRED_ONLY=1 \
MODEL_DIR=opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset \
OUTPUT_DIR=logs/infer_pred_only \
bash scripts/run_inference_pointpillar_intermediate_single.sh
```

只看 GT：

```bash
GT_ONLY=1 \
MODEL_DIR=opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset \
OUTPUT_DIR=logs/infer_gt_only \
bash scripts/run_inference_pointpillar_intermediate_single.sh
```

## 7. 结果如何检查

重点看两类目录：

- 模型目录：`opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset`
- 图片目录：你传给 `OUTPUT_DIR` 的路径

验证标准：

1. 控制台出现 `Loading Model from checkpoint`
2. 能看到 `samples found`
3. 输出目录里出现 PNG
4. PNG 中能看到点云，且 `GT_ONLY=1` 时能稳定看到 GT 框

## 8. 常见问题

### 1) 无桌面环境，Open3D 截图失败

现在推理脚本支持 `--headless` 回退；脚本默认已经开启，不需要额外处理。

### 2) 想用官方预训练而不是 bootstrap

手动下载官方 zip 后，把其中的 `config.yaml` 和 `*.pth` 放到一个新目录里，然后直接把：

```bash
MODEL_DIR=你的官方模型目录
```

传给同一套运行脚本即可。
