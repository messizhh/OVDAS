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

## 后续运行命令占位

以下命令是后续开发占位，不应在 Day 4 运行 SAM、自动标签生成或 YOLO 训练。

```bash
# Day 5: SAM 修正，待实现
python3 tools/run_sam_refine_batch.py --config configs/default.yaml

# Day 6: 自动标签生成，待实现
python3 tools/generate_yolo_labels_from_auto.py --config configs/default.yaml

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
