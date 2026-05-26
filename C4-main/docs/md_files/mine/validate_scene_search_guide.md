# Validate 特定场景检索脚本使用说明

本文档说明如何在 `validate` 数据集中筛选特定场景，并把筛出的候选帧交给现有可视化脚本做人工确认。

当前实现采用两段式流程：

1. 用元数据和启发式规则做粗筛。
2. 用可视化脚本逐帧确认目标场景。

适用场景包括：

- 找 `3` 辆协同感知车辆的场景
- 找带 `RSU` 的车路协同场景
- 找低速、目标较多、疑似有遮挡或阻塞物的帧

运行前提：

- 请在安装了 OpenCOOD 依赖的环境里执行
- 仓库推荐环境名是 `opencood`
- 当前机器上可直接使用 `torch118`
- 如果你在系统默认 `python` 下运行，可能会遇到 `PyYAML` 缺失

## 1. 脚本位置

检索脚本路径：

- [search_validate_scene.py](/home/wcp/c4/OpenCOOD-main/opencood/tools/search_validate_scene.py)

## 2. 数据组织方式

当前 `validate` 数据按下面的结构组织：

```text
validate/
  scenario_id/
    cav_id/
      000001.yaml
      000001.pcd
      ...
```

含义如下：

- `scenario_id`：场景目录，例如 `2021_09_09_22_21_11`
- `cav_id`：协同体目录
- 正 ID：车端 CAV
- 负 ID：路侧单元 RSU，例如 `-1`
- `timestamp`：时间帧，例如 `000291`

脚本会按场景顺序扫描，并额外生成一个 `dataset_index`。这个字段和现有可视化脚本的 `--frame_index` 对应，可以直接用于定位目标帧。

## 3. 脚本默认行为

如果不显式传 `--validate_dir`，脚本会按下面顺序尝试：

1. `v2xset/validate`
2. `opv2v_data_dumping/validate`

如果都不存在，再报错。

默认输出文件：

```bash
logs/validate_scene_index.csv
```

## 4. 如何运行

### 4.1 全量扫描 validate

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --output_path logs/Search/validate_scene_index.csv
```

### 4.2 只查找 3 车协同场景

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --target_cav_count 3 \
  --output_path logs/validate_3cav.csv
```

### 4.3 查找 3 车协同 + 低速帧

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --target_cav_count 3 \
  --max_ego_speed 5 \
  --output_path logs/validate_3cav_low_speed.csv
```

### 4.4 查找 3 车协同 + 疑似阻塞候选帧

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --target_cav_count 3 \
  --max_ego_speed 5 \
  --candidate_only \
  --candidate_type blocking \
  --output_path logs/validate_3cav_blocking.csv
```

### 4.5 只看含 RSU 的车路协同场景

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --require_rsu \
  --output_path logs/validate_with_rsu.csv
```

### 4.6 只扫描一个场景或一帧

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --scenario_id 2021_09_09_22_21_11 \
  --output_path logs/one_scenario.csv
```

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --scenario_id 2021_09_09_22_21_11 \
  --timestamp 000291 \
  --output_path logs/one_frame.csv
```

## 5. 参数说明

常用参数如下：

- `--validate_dir`
  - `validate` 数据目录
- `--hypes_yaml`
  - 可选。若未传 `--validate_dir`，则从 YAML 中读取 `validate_dir`
- `--output_path`
  - 输出 CSV 路径
- `--target_cav_count`
  - 只保留指定数量 CAV 的场景，例如 `3`
- `--require_rsu`
  - 只保留含 RSU 的场景
- `--exclude_rsu`
  - 只保留不含 RSU 的场景
- `--min_object_count`
  - 目标数下限
- `--max_ego_speed`
  - ego 速度上限
- `--scenario_id`
  - 只看一个场景
- `--timestamp`
  - 只看一个时间帧
- `--candidate_only`
  - 只保留启发式候选帧
- `--candidate_type`
  - 候选类型，可选 `either / blocking / dense`
- `--max_results`
  - 控制终端里预览打印多少行

## 6. 输出字段说明

CSV 中的主要字段如下：

- `dataset_index`
  - 数据集线性帧索引，可直接传给 `vis_data_sequence.py --frame_index`
- `frame_index_in_scenario`
  - 当前帧在场景内的序号
- `scenario_id`
  - 场景目录名
- `timestamp`
  - 时间帧编号
- `ego_cav_id`
  - 当前场景中用作 ego 的 CAV ID
- `cav_count`
  - 正 ID CAV 数量
- `rsu_count`
  - RSU 数量
- `has_rsu`
  - 是否含 RSU
- `frame_count`
  - 场景总帧数
- `cav_ids`
  - 当前场景全部 CAV ID
- `rsu_ids`
  - 当前场景全部 RSU ID
- `ego_speed`
  - ego 车速度
- `object_count`
  - 当前帧 `vehicles` 中目标总数
- `nearby_object_count`
  - ego 周围近距离目标数
- `nearby_static_count`
  - ego 周围近距离静态目标数
- `front_object_count`
  - ego 前方近距离目标数
- `front_static_count`
  - ego 前方近距离静态目标数
- `nearest_object_distance`
  - 最近目标距离
- `nearest_front_static_distance`
  - 最近前方静态目标距离
- `candidate_static_object_dense`
  - 是否命中“静态目标密集”候选
- `candidate_blocking_object`
  - 是否命中“前方静态目标阻塞”候选
- `candidate_reasons`
  - 命中候选的原因标签
- `yaml_path`
  - 对应 ego YAML 文件路径

## 7. 推荐检索流程

如果你想找“有路障、3 辆协同感知车辆”的场景，建议按下面步骤做：

1. 先按 `3` 车协同筛：

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --target_cav_count 3 \
  --output_path logs/step1_3cav.csv
```

2. 再叠加低速和候选约束：

```bash
conda run -n torch118 python opencood/tools/search_validate_scene.py \
  --validate_dir v2xset/validate \
  --target_cav_count 3 \
  --max_ego_speed 5 \
  --candidate_only \
  --candidate_type either \
  --output_path logs/step2_3cav_candidates.csv
```

3. 打开输出 CSV，重点看下面几列：

- `dataset_index`
- `scenario_id`
- `timestamp`
- `ego_speed`
- `front_static_count`
- `candidate_reasons`

4. 选择若干候选帧，进入可视化确认。

## 8. 如何把候选帧交给可视化脚本

假设在 CSV 中筛到：

- `dataset_index = 123`

则可以直接调用现有可视化脚本：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --hypes_yaml opencood/hypes_yaml/visualization.yaml \
  --fusion_method early \
  --frame_index 123 \
  --save_path logs/frame_123.png
```

如果想连续看几帧：

```bash
conda run -n torch118 python opencood/visualization/vis_data_sequence.py \
  --hypes_yaml opencood/hypes_yaml/visualization.yaml \
  --fusion_method early \
  --frame_index 123 \
  --num_frames 5 \
  --save_dir logs/frame_123_seq
```

说明：

- 这里的 `--frame_index` 对应搜索脚本输出的 `dataset_index`
- 当前 `visualization.yaml` 默认指向 `v2xset/validate`
- 如果你换了数据集目录，需要同步调整 YAML 中的 `validate_dir`

## 9. “路障”字段的真实含义

这里要特别注意：

- 当前 OpenCOOD 这套数据接口主要稳定暴露的是 `vehicles`
- 脚本没有直接读取到一个可靠的 `barrier / cone / obstacle` 显式类别字段
- 因此 `candidate_blocking_object` 不是“明确存在路障”的语义标签
- 它只是“前方存在静态目标，且值得人工确认”的候选标记

补充说明：

- 在当前仓库自带的 `v2xset/validate` 上，`3 车协同 + max_ego_speed 5 + candidate_only` 能筛到非空结果
- 如果把 `max_ego_speed` 收紧到 `0.5`，结果可能为空，这属于数据本身分布导致的正常现象

也就是说，这个脚本做的是：

- 候选召回

不是：

- 精确语义级路障识别

## 10. 推荐观察哪些参数

针对“找特定场景”这件事，优先关注下面几类参数：

- 协同规模：
  - `cav_count`
  - `rsu_count`
- 场景拥挤度：
  - `object_count`
  - `nearby_object_count`
- 静态阻塞倾向：
  - `nearby_static_count`
  - `front_static_count`
  - `candidate_reasons`
- 运动状态：
  - `ego_speed`
- 精确定位：
  - `scenario_id`
  - `timestamp`
  - `dataset_index`

## 11. 已知限制

当前版本有这些限制：

- 候选规则是启发式的，不是标注级真值检索
- “路障”需要人工确认
- 指标基于 ego 视角 YAML 中的 `vehicles`
- 如果换成别的数据集，字段语义可能不同，需要重新检查

## 12. 建议的实际用法

对于算法组，比较稳妥的方式不是直接找“唯一正确答案”，而是先把候选帧缩到一个较小集合，再人工确认并沉淀为白名单。

建议最终额外整理一份人工确认表，至少包含：

- `scenario_id`
- `timestamp`
- `dataset_index`
- `reason`
- `confirmed_by`
- `notes`

这样后续复现实验或对比融合算法时，可以直接复用同一批目标场景。
