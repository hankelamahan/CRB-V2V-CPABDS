# OpenCOOD Frame2Video 设计方案

本文档用于设计一个“帧图片转视频”的小工具，服务于当前项目中 `OpenCOOD-main/logs`、`Results/adaptive/*/vis` 等目录生成的可视化 PNG 图片。目标是把 OpenCOOD 推理输出的逐帧图片按时间顺序合成为 MP4/GIF 等视频，方便汇报、演示和实验对比。

## 1. 调研结论

常见帧转视频方案主要有三类：

| 方案 | 代表接口 | 优点 | 不足 | 适合本项目程度 |
|---|---|---|---|---|
| FFmpeg 命令行 | `ffmpeg -framerate 10 -i 'img-%03d.png' out.mp4` | 编码稳定、速度快、MP4 兼容性好、参数成熟 | 需要系统安装 `ffmpeg`；连续编号输入最方便，稀疏帧需要额外处理 | 最推荐 |
| OpenCV VideoWriter | `cv2.VideoWriter(output, fourcc, fps, frameSize)` | Python 内部可控，便于逐帧读写、resize、加文字 | 编码兼容性依赖 OpenCV 构建；MP4 fourcc 有平台差异 | 适合作为 fallback 或高级处理后端 |
| MoviePy ImageSequenceClip | `ImageSequenceClip(files, fps=24).write_videofile(...)` | 接口简单，适合剪辑、拼接、加字幕 | 依赖更重，速度通常不如直接 FFmpeg | 可选，不作为默认实现 |

官方文档要点：

- FFmpeg `image2` 支持用 `-framerate` 读取图片序列，并可通过 `-start_number` 指定起始编号；官方示例为 `ffmpeg -framerate 10 -i 'img-%03d.jpeg' out.mkv`。
- FFmpeg 文档也提醒：对于 `image2` 这类输入，优先使用输入侧 `-framerate`，不要混淆为输出侧 `-r`。
- OpenCV `VideoWriter` 需要指定输出文件名、fourcc、fps、frameSize，并通过 `write()` 逐帧写入。
- MoviePy 支持 `ImageSequenceClip` 读取图片序列，`write_videofile(..., fps=...)` 导出视频。

综合本项目需求，推荐设计为：

> **Python CLI 包装器 + FFmpeg 默认编码后端 + OpenCV/Pillow 做图片扫描和尺寸校验。**

这样既能保持 FFmpeg 的视频编码质量，又能让用户通过 Python 命令方便地指定输入目录、FPS、输出文件、排序方式、是否递归、是否 resize 等参数。

## 2. 本地 OpenCOOD 图片特点

已检查到的典型路径包括：

```text
OpenCOOD-main/logs/trust/test6/late_constant_frame_00190.png
OpenCOOD-main/logs/trust/test6/late_constant_frame_00191.png
OpenCOOD-main/logs/trust/test6/late_constant_frame_00192.png

OpenCOOD-main/logs/batch/early_constant_frame_00020.png
OpenCOOD-main/logs/batch/early_constant_frame_00021.png

Results/adaptive/baseline_no_trust/vis/late_constant_frame_00000.png
Results/adaptive/baseline_no_trust/vis/late_constant_frame_00099.png
Results/adaptive/baseline_no_trust/vis/late_constant_frame_00100.png
```

观察到的特点：

- 图片通常是 `.png`。
- 分辨率通常是 `1920x1080`。
- 命名常见模式：`<fusion>_<color>_frame_<00000>.png`。
- 有些目录是连续帧，如 `00190-00199`。
- 有些目录是抽样关键帧，如 `00000,00099,00100,00105,00119,00120,00199`，并不连续。
- 也存在非 frame 命名图片，如 `scenario_1_167.png`、`early_endFrame.png`，不能简单全部混入同一个视频。

因此工具必须支持：

- 按 glob 匹配目标帧，例如 `--pattern "*_frame_*.png"`。
- 自然排序，确保 `frame_00099` 在 `frame_00100` 前面。
- 非连续帧合成视频，不能强依赖源文件编号连续。
- 输出到用户指定目录，例如 `Results/adaptive/videos/*.mp4`。

## 3. 总体设计

目录建议：

```text
frame2video/
  README.md                 # 本设计文档
  frame2video.py             # 后续实现主入口
  requirements.txt           # 可选依赖说明
  examples/
    commands.md              # 常用命令示例，可后续添加
```

后续实现时，主入口建议为：

```bash
python frame2video/frame2video.py \
  --input OpenCOOD-main/logs/trust/test6 \
  --pattern "late_constant_frame_*.png" \
  --output Results/videos/test6.mp4 \
  --fps 5
```

核心数据流：

```text
输入目录
  -> 按 pattern 收集图片
  -> 自然排序
  -> 过滤/截取 frame index
  -> 校验尺寸和文件可读性
  -> 构建临时连续序列或 concat 列表
  -> 调用 FFmpeg 编码
  -> 输出 mp4/gif
  -> 清理临时文件
```

## 4. 推荐后端策略

### 4.1 默认后端：FFmpeg

适合大多数 OpenCOOD 可视化图片。

推荐编码命令形态：

```bash
ffmpeg -y \
  -framerate 5 \
  -i /tmp/frame2video_xxx/frame_%06d.png \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p \
  output.mp4
```

参数含义：

| 参数 | 含义 | 推荐默认值 |
|---|---|---|
| `-framerate` | 输入图片序列按多少 FPS 解释 | 用户传入 `--fps`，默认 5 或 10 |
| `-c:v libx264` | H.264 编码，兼容性好 | `libx264` |
| `-crf` | 画质控制，越小越清晰、文件越大 | 18 |
| `-preset` | 编码速度/压缩率权衡 | `medium` |
| `-pix_fmt yuv420p` | 提高播放器兼容性 | `yuv420p` |
| `-y` | 覆盖已有输出 | 由 `--overwrite` 控制 |

为什么采用“临时连续序列”：

- OpenCOOD 输出经常不是从 0 开始，例如 `00190-00199`。
- 有些实验只导出关键帧，编号不连续。
- FFmpeg 的 `%06d` 输入对连续序列最自然。
- 工具可以把排序后的源图片软链接或复制为临时目录中的 `frame_000000.png, frame_000001.png...`，从而统一处理连续和非连续场景。

### 4.2 备选后端：OpenCV

当系统没有 FFmpeg，或者后续需要逐帧叠加文字、统一 resize、画面拼接时，可以增加 OpenCV 后端：

```python
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
writer.write(frame)
writer.release()
```

注意点：

- 所有帧必须是相同尺寸。
- OpenCV 读取是 BGR 格式。
- MP4 编码器可用性受平台影响，常见 fourcc 为 `mp4v`、`avc1`、`MJPG`。

### 4.3 不建议默认使用 MoviePy

MoviePy 的 `ImageSequenceClip` 很方便，但对当前需求而言偏重：

- 当前只是图片序列转视频，不需要复杂剪辑。
- FFmpeg 命令行更直接、更快。
- MoviePy 内部也依赖 FFmpeg 进行视频导出。

因此 MoviePy 只作为后续“加片头、拼接多个视频、加字幕”的可选扩展。

## 5. CLI 接口设计

建议主命令：

```bash
python frame2video/frame2video.py [options]
```

### 5.1 必需参数

| 参数 | 示例 | 说明 |
|---|---|---|
| `--input` | `OpenCOOD-main/logs/trust/test6` | 输入图片目录 |
| `--output` | `Results/videos/test6.mp4` | 输出视频路径 |

### 5.2 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--pattern` | `*_frame_*.png` | 图片匹配模式 |
| `--fps` | `5` | 每秒播放帧数 |
| `--recursive` | false | 是否递归扫描子目录 |
| `--sort` | `natural` | 排序方式：`natural/name/mtime` |
| `--start-frame` | 空 | 只保留编号大于等于该值的帧 |
| `--end-frame` | 空 | 只保留编号小于等于该值的帧 |
| `--stride` | `1` | 每隔多少帧取一张 |
| `--limit` | 空 | 最多使用多少张图片 |
| `--backend` | `ffmpeg` | `ffmpeg/opencv` |
| `--codec` | `libx264` | FFmpeg 视频编码器 |
| `--crf` | `18` | FFmpeg 画质参数 |
| `--preset` | `medium` | FFmpeg 编码预设 |
| `--pix-fmt` | `yuv420p` | 输出像素格式 |
| `--overwrite` | false | 是否覆盖已有输出 |
| `--dry-run` | false | 只打印将使用的帧和命令，不生成视频 |

### 5.3 尺寸处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--resize` | 空 | 指定输出尺寸，如 `1920x1080` |
| `--resize-mode` | `fit` | `fit/stretch/crop` |
| `--pad-color` | `black` | fit 模式下的填充颜色 |
| `--allow-size-mismatch` | false | 是否允许输入帧尺寸不一致 |

建议第一版先实现：

- `--input`
- `--output`
- `--pattern`
- `--fps`
- `--overwrite`
- `--dry-run`
- `--backend ffmpeg`

其余参数可以在第二版扩展。

## 6. 图片排序与帧号解析

OpenCOOD 常见文件名：

```text
late_constant_frame_00190.png
early_constant_frame_00020.png
intermediate_intensity_frame_00001.png
```

推荐使用正则提取末尾 frame id：

```text
.*_frame_(\d+)\.(png|jpg|jpeg)$
```

排序规则：

1. 如果能解析出 frame id，优先按 frame id 升序。
2. 如果解析失败，退回自然排序。
3. 如果用户指定 `--sort mtime`，按文件修改时间排序。

这样可以同时支持：

- `late_constant_frame_00190.png`
- `scenario_1_167.png`
- 其他普通图片序列

建议 `--pattern` 默认不要匹配所有 PNG，而是：

```text
*_frame_*.png
```

这样能避免把 `early_endFrame.png`、`scenario_1_167.png` 等非连续展示图误加入视频。

## 7. 常用命令示例

### 7.1 把 trust/test6 的后 10 帧转为 5 FPS 视频

```bash
python frame2video/frame2video.py \
  --input OpenCOOD-main/logs/trust/test6 \
  --pattern "late_constant_frame_*.png" \
  --output Results/videos/trust_test6_5fps.mp4 \
  --fps 5 \
  --overwrite
```

### 7.2 把 adaptive baseline 的关键帧转成慢速视频

```bash
python frame2video/frame2video.py \
  --input Results/adaptive/baseline_no_trust/vis \
  --pattern "late_constant_frame_*.png" \
  --output Results/adaptive/baseline_no_trust/baseline_keyframes.mp4 \
  --fps 1 \
  --overwrite
```

### 7.3 只预览将会使用哪些帧

```bash
python frame2video/frame2video.py \
  --input Results/adaptive/trust_physical_sensitive/vis \
  --pattern "late_constant_frame_*.png" \
  --output Results/adaptive/trust_physical_sensitive/sensitive.mp4 \
  --fps 1 \
  --dry-run
```

### 7.4 使用 FFmpeg 原生命令处理连续编号图片

如果图片是完全连续编号，也可以绕过 Python 工具直接用 FFmpeg：

```bash
ffmpeg -y \
  -framerate 5 \
  -start_number 190 \
  -i "OpenCOOD-main/logs/trust/test6/late_constant_frame_%05d.png" \
  -c:v libx264 \
  -crf 18 \
  -pix_fmt yuv420p \
  Results/videos/test6_direct.mp4
```

但该命令只适合连续序列，不适合 `00000,00099,00100...` 这类关键帧抽样目录。

## 8. 错误处理设计

第一版应明确处理以下错误：

| 场景 | 处理方式 |
|---|---|
| 输入目录不存在 | 直接报错，打印路径 |
| 没有匹配图片 | 报错并提示检查 `--pattern` |
| 输出文件已存在且未 `--overwrite` | 报错，避免误覆盖 |
| 未安装 FFmpeg | 报错并提示安装或切换 `--backend opencv` |
| 图片尺寸不一致 | 默认报错；后续可通过 `--resize` 修复 |
| 图片损坏或不可读 | 报错并列出文件名 |
| FPS <= 0 | 报错 |

输出日志建议包含：

```text
Input dir: ...
Pattern: ...
Matched frames: 10
First frame: late_constant_frame_00190.png
Last frame: late_constant_frame_00199.png
Resolution: 1920x1080
FPS: 5
Duration: 2.00s
Output: ...
Backend: ffmpeg
```

## 9. 实现建议

### 9.1 核心函数划分

后续 `frame2video.py` 建议拆成以下函数：

```python
parse_args()
collect_frames(input_dir, pattern, recursive)
extract_frame_index(path)
sort_frames(paths, mode)
filter_frames(paths, start_frame, end_frame, stride, limit)
validate_frames(paths, allow_size_mismatch=False)
prepare_temp_sequence(paths, temp_dir)
build_ffmpeg_command(temp_pattern, output, fps, codec, crf, preset, pix_fmt)
run_command(command)
write_video_opencv(paths, output, fps, resize=None)
main()
```

### 9.2 临时目录策略

推荐使用 `tempfile.TemporaryDirectory()`：

```text
/tmp/frame2video_xxxxx/
  frame_000000.png -> 原图软链接
  frame_000001.png -> 原图软链接
  frame_000002.png -> 原图软链接
```

优先使用软链接，失败时退回复制。这样可以：

- 支持非连续源帧。
- 支持源帧从任意编号开始。
- 避免修改 OpenCOOD 原始日志目录。

### 9.3 输出格式

第一版优先支持：

| 格式 | 推荐方式 |
|---|---|
| `.mp4` | FFmpeg + `libx264` |
| `.avi` | FFmpeg 或 OpenCV + `MJPG` |
| `.gif` | FFmpeg palettegen/paletteuse，第二版再加 |

MP4 是默认推荐格式，兼容演示软件、浏览器和微信/邮件传播。

## 10. 为什么该设计适合当前项目

当前 OpenCOOD 使用方式经常是：

```bash
python opencood/tools/inference.py \
  --save_vis_dir logs/trust/test6 \
  --frame_index 190 \
  --num_frames 10 \
  --headless
```

生成结果一般已经是统一尺寸 PNG，帧数不大，文件命名有固定 frame id。因此最适合：

- 用 Python 做收集、排序、校验；
- 用 FFmpeg 做最终编码；
- 保持命令行接口简单；
- 避免侵入 OpenCOOD 代码框架。

这个工具应作为项目外部辅助脚本存在，不需要修改 `OpenCOOD-main/opencood` 的算法代码。

## 11. 后续实现优先级

建议按以下顺序实现：

1. **MVP**：单目录、glob 匹配、自然排序、FPS、MP4 输出、FFmpeg 后端。
2. **鲁棒性**：尺寸校验、dry-run、overwrite、start/end/stride/limit。
3. **兼容性**：OpenCV 后端、自动 resize、递归扫描。
4. **展示增强**：添加帧号水印、实验名标题、多个实验横向拼接。
5. **批处理**：自动扫描 `logs/trust/*`，每个目录生成一个视频。

## 12. 参考资料

- FFmpeg 官方文档：`https://ffmpeg.org/ffmpeg.html`
- FFmpeg all/image2 文档：`https://www.ffmpeg.org/ffmpeg-all.html`
- OpenCV VideoWriter 官方文档：`https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html`
- MoviePy ImageSequenceClip 官方文档：`https://zulko.github.io/moviepy/user_guide/loading.html`
- MoviePy 渲染/导出官方文档：`https://zulko.github.io/moviepy/user_guide/rendering.html`
