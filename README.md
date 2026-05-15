# OVDAS

OVDAS: Open-Vocabulary Drone Auto-annotation for Small-object Detection

本项目是《计算机视觉》课程大作业，研究方向为基于 Grounding DINO 与 SAM 的无人机航拍/遥感小目标半自动标注及检测模型训练。项目目标是在 15 天内完成一个可复现、可展示、可写入报告的实验闭环。

## 项目目标

- 使用开放词汇检测模型 Grounding DINO，根据文本 prompt 在无人机航拍图像中检测目标。
- 使用 SAM 对 Grounding DINO 的检测框进行分割修正，提升自动标注质量。
- 将检测框或分割结果转换为 YOLO 格式标签。
- 使用自动标签训练 YOLO 检测模型，并与人工标签 baseline 对比。
- 评估自动标注质量、检测性能和小目标场景下的失败案例。

## 主流程

```text
VisDrone 数据准备
  -> Grounding DINO 开放词汇检测
  -> SAM mask 修正
  -> 自动生成 YOLO 标签
  -> 自动标签质量评估
  -> YOLO 检测模型训练
  -> 定量评估与可视化分析
  -> 课程报告
```

## 本地 WSL 与远程 GPU 服务器分工

本项目中 Codex 主要用于本地 WSL 环境下的代码开发与轻量调试。由于本地主机算力有限，大规模推理、SAM 批量分割和 YOLO 正式训练需要在远程 GPU 服务器上运行。

本地 WSL 负责：

- 代码编写、重构和配置检查；
- 小样本数据格式转换；
- 单张或少量图片的流程调试；
- 单元测试、可视化检查和 README/报告材料整理。

远程 GPU 服务器负责：

- 全量 Grounding DINO 推理；
- 全量 SAM 分割；
- YOLO 正式训练；
- 多阈值、多 prompt 消融实验；
- 最终评估和批量可视化生成。

## 推荐数据集

主数据集推荐使用 VisDrone 的 Object Detection in Images 任务。默认使用 8 个类别：

```text
pedestrian
people
bicycle
car
van
truck
bus
motor
```

原始数据集不应提交到仓库。请将 VisDrone 放到 `data/raw/VisDrone/`，后续 Day 2 会编写转换脚本，将处理后的 YOLO 数据放到 `data/processed/visdrone/`。

## 环境安装

本阶段只准备基础依赖，不安装 Grounding DINO、SAM，也不下载模型权重。

```bash
pip install -r requirements.txt
```

或者使用 conda：

```bash
conda env create -f environment.yml
conda activate ovdas
```

## 15 天开发计划

| Day | 目标 | 主要产出 |
| --- | --- | --- |
| 1 | 项目初始化、目录结构、基础配置 | README、requirements、configs、data 说明 |
| 2 | VisDrone 数据解析与 YOLO 格式转换 | `tools/convert_visdrone_to_yolo.py` |
| 3 | Grounding DINO 单图推理封装 | 单图 JSON 与可视化输出 |
| 4 | Grounding DINO 批量推理脚本 | 服务器批量推理入口 |
| 5 | SAM 单图与批量分割修正 | mask 与 refined bbox |
| 6 | 自动 YOLO 标签生成 | `outputs/auto_labels/` 与标签统计 |
| 7 | 自动标注质量评估 | IoU、precision、recall、按类别统计 |
| 8 | YOLO baseline 训练准备 | YOLO 数据配置与服务器脚本 |
| 9 | YOLO 训练与初步评估 | manual/auto 训练结果 |
| 10 | 消融实验 | DINO-only、DINO+SAM、阈值/prompt 对比 |
| 11 | 小目标专项分析 | size-based 指标与失败案例 |
| 12 | 可视化整理 | pipeline、示例、指标图 |
| 13 | 报告初稿 | `report/draft.md` |
| 14 | 报告完善与代码清理 | README、结果表、最终图 |
| 15 | 最终打包 | 报告 PDF 和代码压缩包 |

## 当前 Day 1 进度

- 已建立基础项目目录结构。
- 已创建基础依赖文件：`requirements.txt`、`environment.yml`。
- 已创建基础配置：`configs/default.yaml`、`configs/classes_visdrone.yaml`。
- 已创建数据放置说明：`data/README.md`。
- 已创建 `.gitignore`，排除数据集、权重、训练输出和压缩包等大文件。

## Day 1 自检

在本地 WSL 中运行以下命令，检查关键目录、基础文件和 YAML 配置是否完整：

```bash
python3 tools/check_project_setup.py
```

### 小样本清单

运行以下命令扫描 `data/samples/images/` 与 `data/samples/labels/`，生成 `results/tables/sample_manifest.csv`：

```bash
python3 tools/list_samples.py
```

也可以直接运行本地调试入口，它会同时检查项目结构并生成小样本清单：

```bash
bash scripts/local_debug.sh
```

## Day 2 数据转换

将 VisDrone DET 原始标注转换为 YOLO 格式标签。输入目录需要包含 `images/` 和 `annotations/`。

本地 WSL 可以先用 val 集做小规模转换测试：

```bash
python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-val \
  --out-root data/processed/visdrone \
  --split val \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images \
  --limit 20
```

服务器或数据准备阶段可转换 train 全量数据：

```bash
python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-train \
  --out-root data/processed/visdrone \
  --split train \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images
```

输出目录：

```text
data/processed/visdrone/images/<split>/
data/processed/visdrone/labels/<split>/
```

转换完成后，可以可视化检查 YOLO 标签是否和图像对齐：

```bash
python3 tools/visualize_yolo_labels.py \
  --data-root data/processed/visdrone \
  --split val \
  --classes-config configs/classes_visdrone.yaml \
  --out-dir results/visualizations/manual_labels_val \
  --limit 20
```

## Day 3 单图推理

Grounding DINO 单图推理用于本地 WSL 的少量图片调试。大规模 Grounding DINO 推理必须放到远程 GPU 服务器运行，不要在本地 WSL 上跑全量数据。

运行前需要将 `--config-file` 和 `--checkpoint` 替换为实际的 Grounding DINO 配置文件和权重路径。

```bash
python3 tools/run_grounding_dino_single.py \
  --image data/samples/images/example.jpg \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --output-dir outputs/debug_grounding \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cpu \
  --config-file configs/grounding_dino_placeholder.py \
  --checkpoint checkpoints/grounding_dino_placeholder.pth
```

输出内容：

```text
outputs/debug_grounding/<image_stem>_grounding_dino.jpg
outputs/debug_grounding/<image_stem>_grounding_dino.json
```

## Day 4 Grounding DINO 批量推理

Grounding DINO 批量推理用于生成后续 SAM 修正和自动标签转换所需的中间结果。本地 WSL 只建议用 `--limit 5` 到 `--limit 20` 做小样本检查；val/train 全量推理应在远程 GPU 服务器执行。

服务器 val debug 推荐先跑前 20 张，确认 prompt、阈值、权重路径和输出格式正常：

```bash
bash scripts/server_run_grounding_dino_val_debug.sh
```

等价完整命令如下：

```bash
python3 tools/run_grounding_dino_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --output-dir outputs/grounding_dino_json/val_debug \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda \
  --config-file external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --checkpoint checkpoints/groundingdino_swint_ogc.pth \
  --limit 20 \
  --skip-existing
```

每张成功处理的图片会在输出目录生成一份 JSON 和一张可视化图：

```text
outputs/grounding_dino_json/val_debug/<image_stem>_grounding_dino.json
outputs/grounding_dino_json/val_debug/<image_stem>_grounding_dino.jpg
```

`--skip-existing` 会在对应 JSON 已存在时跳过该图片，方便服务器中断后续跑。`outputs/` 是大规模中间结果目录，已被 `.gitignore` 排除，不要提交或打包全量推理结果；最终报告只保留少量关键 JSON、CSV 或可视化示例即可。

## Day 5 SAM bbox 修正

SAM refine 读取 Day 4 的 Grounding DINO JSON，从每个 detection 的 `bbox_xyxy` 调用 SAM 生成 mask，再由 mask 反推 `refined_bbox_xyxy`。输出 JSON 会保留原始 `bbox_xyxy`、`score`、`phrase` 等字段，并新增：

```text
refined_bbox_xyxy
mask_area
sam_score
mask_path
refine_status
```

默认不保存每个 mask PNG，避免 `outputs/` 过大；需要检查 mask 时再加 `--save-mask --mask-output-dir ...`。

服务器运行前需要确保已安装 SAM Python 包，并将 SAM 权重放到 `--sam-checkpoint` 指定路径，例如：

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

单图调试命令如下。真实 SAM 推理建议在远程 GPU 服务器运行；本地 WSL 主要用于 `--help`、语法检查和极少量样本验证。

```bash
python3 tools/run_sam_refine_single.py \
  --image data/processed/visdrone/images/val/0000001_02999_d_0000005.jpg \
  --dino-json outputs/grounding_dino_json/val_debug/0000001_02999_d_0000005_grounding_dino.json \
  --output-json outputs/sam_refine_json/val_debug/0000001_02999_d_0000005_sam_refine.json \
  --vis-output results/visualizations/sam_refine_val_debug/0000001_02999_d_0000005_sam_refine.jpg \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device cuda
```

服务器 val debug 推荐先对 Day 4 debug 输出跑前 20 张：

```bash
bash scripts/server_run_sam_refine_val_debug.sh
```

等价完整命令如下：

```bash
python3 tools/run_sam_refine_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --dino-json-dir outputs/grounding_dino_json/val_debug \
  --output-json-dir outputs/sam_refine_json/val_debug \
  --vis-output-dir results/visualizations/sam_refine_val_debug \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device cuda \
  --limit 20 \
  --skip-existing
```

如需保存 mask PNG：

```bash
python3 tools/run_sam_refine_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --dino-json-dir outputs/grounding_dino_json/val_debug \
  --output-json-dir outputs/sam_refine_json/val_debug \
  --vis-output-dir results/visualizations/sam_refine_val_debug \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device cuda \
  --limit 20 \
  --skip-existing \
  --save-mask \
  --mask-output-dir outputs/sam_masks/val_debug
```

batch 脚本会输出 summary：

```text
total_images
processed_images
skipped_images
failed_images
total_detections
refined_detections
output_dir
```

空 detections 会正常生成空结果和可视化图；单张图片失败会记录到 `outputs/sam_refine_json/val_debug/sam_refine_batch_failures.txt`，不会中断整个 batch。

## Day 6 自动标签生成

自动标签生成脚本将 Grounding DINO 或 SAM refine JSON 转换为 YOLO txt 标签：

```text
class_id x_center y_center width height
```

坐标会裁剪到图像范围内并归一化到 0 到 1。类别映射优先读取 `configs/classes_visdrone.yaml`，默认 8 类为：

```text
pedestrian -> 0
people -> 1
bicycle -> 2
car -> 3
van -> 4
truck -> 5
bus -> 6
motor -> 7
```

脚本会清洗 phrase 的大小写、空格和末尾标点；无法映射的类别会跳过并计入统计。空 detections 或过滤后无标签的图片仍会生成空 txt，方便后续 YOLO 训练流程保持样本对齐。

DINO+SAM 自动标签 debug 推荐命令：

```bash
bash scripts/server_generate_auto_labels_val_debug.sh
```

等价完整命令如下：

```bash
python3 tools/generate_yolo_labels_from_auto.py \
  --json-dir outputs/sam_refine_json/val_debug \
  --image-dir data/processed/visdrone/images/val \
  --out-label-dir outputs/auto_labels/sam_refine_val_debug/labels \
  --classes-config configs/classes_visdrone.yaml \
  --bbox-key refined_bbox_xyxy \
  --fallback-bbox-key bbox_xyxy \
  --score-threshold 0.35 \
  --min-box-area 4 \
  --stats-csv results/tables/auto_label_statistics_sam_refine_val_debug.csv \
  --limit 20 \
  --skip-existing
```

DINO-only 自动标签生成使用原始 `bbox_xyxy`：

```bash
python3 tools/generate_yolo_labels_from_auto.py \
  --json-dir outputs/grounding_dino_json/val_debug \
  --image-dir data/processed/visdrone/images/val \
  --out-label-dir outputs/auto_labels/dino_val_debug/labels \
  --classes-config configs/classes_visdrone.yaml \
  --bbox-key bbox_xyxy \
  --fallback-bbox-key bbox_xyxy \
  --score-threshold 0.35 \
  --min-box-area 4 \
  --stats-csv results/tables/auto_label_statistics_dino_val_debug.csv \
  --limit 20 \
  --skip-existing
```

如需构造可直接给 YOLO 使用的 image/label 子目录，可以加 `--copy-images --out-image-dir outputs/auto_labels/sam_refine_val_debug/images`。默认不复制图片，避免重复占用空间。

summary 字段包括：

```text
total_json_files
processed_files
failed_files
total_detections
kept_labels
skipped_low_score
skipped_unknown_class
skipped_invalid_bbox
skipped_small_box
empty_label_files
output_label_dir
```

统计 CSV 会写入 summary 指标和每类保留标签数量，例如 `results/tables/auto_label_statistics_sam_refine_val_debug.csv`。`outputs/auto_labels/` 属于中间结果目录，不应提交或打包全量文件。

## 后续运行命令占位

以下命令是后续开发占位，不应在 Day 6 运行自动标签质量评估或 YOLO 训练。

```bash
# Day 8-9: YOLO 训练，服务器运行，待实现
yolo detect train data=configs/yolo_visdrone_auto.yaml model=yolov8s.pt

# Day 7/11: 评估与分析，待实现
python3 tools/evaluate_auto_labels.py --config configs/default.yaml
python3 tools/analyze_small_objects.py --config configs/default.yaml
```

## 结果目录

- `outputs/`: 保存中间输出，如 Grounding DINO JSON、SAM masks、自动标签和调试结果。
- `results/tables/`: 保存实验表格。
- `results/metrics/`: 保存指标文件。
- `results/visualizations/`: 保存可视化结果。
- `results/failure_cases/`: 保存失败案例。
- `figures/`: 保存报告用图和流程图素材。
- `report/`: 保存课程报告草稿和最终文档。
