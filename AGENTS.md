# AGENTS.md

## 0. 项目定位

本项目是《计算机视觉》课程大作业，目标是在 15 天内完成一个高质量、可复现、可展示、可写入报告的计算机视觉项目。

项目方向：

> 基于 Grounding DINO 与 SAM 的无人机航拍/遥感小目标半自动标注及检测模型训练研究

核心思想：

1. 使用开放词汇检测模型 Grounding DINO 根据文本提示自动检测无人机/遥感图像中的目标。
2. 使用 SAM 对 Grounding DINO 的检测框进行分割修正。
3. 将检测框或分割结果转换为 YOLO 格式标签。
4. 使用自动生成的标签训练下游检测器，例如 YOLOv8 / YOLOv10 / RT-DETR。
5. 在人工标注的验证集/测试集上评估检测效果。
6. 重点分析小目标、密集目标、复杂背景、域差异带来的影响。
7. 最终形成完整代码、实验结果、可视化图和课程报告。

本项目不是简单跑通某个开源 baseline，而是要形成一个完整的实验闭环：

```text
数据集准备
  -> Grounding DINO 开放词汇检测
  -> SAM 分割修正
  -> 自动生成 YOLO 标签
  -> 标签质量评估
  -> 训练下游检测模型
  -> 定量评估
  -> 可视化分析
  -> 报告撰写
```

---

## 1. 最重要的开发约束

### 1.1 本地环境限制

Codex 只能在用户主机的 WSL 环境中运行，不能直接部署或运行在远程 GPU 服务器上。

因此：

- 本地 WSL 主要用于：
  - 代码编写；
  - 代码重构；
  - 小样本调试；
  - 数据格式转换脚本验证；
  - 单张或少量图片推理测试；
  - CPU 或轻量 GPU 实验；
  - 单元测试；
  - 配置文件检查；
  - README、报告素材、实验表格脚本生成。

- 本地 WSL 不应用于：
  - 大规模 Grounding DINO 推理；
  - 大规模 SAM 分割；
  - YOLO 长时间训练；
  - 全量数据集训练；
  - 多模型大规模消融实验；
  - 高显存模型训练。

### 1.2 服务器运行限制

吃配置的实验需要在远程 GPU 服务器上运行，例如：

- Grounding DINO 批量推理；
- SAM 批量分割；
- 自动标注大规模生成；
- YOLOv8s / YOLOv8m / YOLOv10 / RT-DETR 训练；
- 多阈值消融实验；
- 多模型对比实验；
- 大规模评估；
- 可视化批量生成。

Codex 在写代码时必须考虑：

1. 所有重实验脚本都应该可以通过命令行在服务器运行。
2. 不要把路径写死成本地路径。
3. 所有路径应通过配置文件或命令行参数传入。
4. 所有实验应支持断点续跑或跳过已生成结果。
5. 所有输出应保存到明确的 `outputs/`、`runs/` 或 `results/` 目录中。
6. 服务器运行脚本应尽量提供 `.sh` 启动文件，方便用户复制到服务器执行。
7. 所有训练、推理和评估命令都要在 README 或 `scripts/` 中保留。

---

## 2. 15 天完成要求

本项目必须在 15 天内完成，不允许无限扩展。

所有开发都必须优先服务于课程大作业得分，而不是追求学术论文级复杂度。

### 2.1 时间优先级

项目优先级如下：

1. 能完整跑通主流程。
2. 能生成可写入报告的定量结果。
3. 能生成清晰漂亮的可视化结果。
4. 能完成至少 2 到 3 个有意义的消融实验。
5. 能形成结构清晰、可复现的代码仓库。
6. 能完成高质量报告。

如果时间不足，优先保留主线，不要盲目增加模型。

### 2.2 推荐 15 天计划

#### Day 1：项目初始化

目标：

- 建立项目目录结构；
- 明确数据集；
- 确定任务类别；
- 写好 README 初稿；
- 配置本地 WSL 开发环境；
- 准备小样本数据。

产出：

- `README.md`
- `requirements.txt`
- `configs/default.yaml`
- `data/samples/`
- 基础目录结构

#### Day 2：数据集解析

目标：

- 支持 VisDrone 或 DIOR 数据集读取；
- 完成原始标注格式到 YOLO 格式的转换；
- 支持 train/val/test 划分；
- 能可视化人工标注。

产出：

- `tools/convert_visdrone_to_yolo.py`
- `tools/visualize_labels.py`
- `data/visdrone_yolo/`
- 标注可视化图片

#### Day 3：Grounding DINO 单图推理

目标：

- 封装 Grounding DINO 推理接口；
- 支持文本 prompt；
- 支持单张图片推理；
- 输出 bbox、score、phrase；
- 保存可视化图片和 JSON 结果。

产出：

- `src/open_vocab/grounding_dino_infer.py`
- `tools/run_grounding_dino_single.py`
- `outputs/debug_grounding/`

#### Day 4：Grounding DINO 批量推理脚本

目标：

- 支持批量推理；
- 支持 box threshold 和 text threshold；
- 支持跳过已处理图片；
- 支持保存中间结果；
- 提供服务器运行脚本。

产出：

- `tools/run_grounding_dino_batch.py`
- `scripts/server_run_grounding_dino.sh`
- `outputs/grounding_dino_json/`

注意：

- 本地只用 5 到 20 张图片验证。
- 全量批量推理必须放到服务器执行。

#### Day 5：SAM 单图与批量分割

目标：

- 读取 Grounding DINO 的 bbox；
- 调用 SAM 生成 mask；
- 从 mask 反推更精确的 bbox；
- 保存 mask、bbox、可视化结果。

产出：

- `src/segmentation/sam_refine.py`
- `tools/run_sam_refine_single.py`
- `tools/run_sam_refine_batch.py`
- `scripts/server_run_sam_refine.sh`

注意：

- 本地只跑少量图片。
- 批量 SAM 必须放到服务器运行。

#### Day 6：自动标签生成

目标：

- 将 Grounding DINO 或 Grounding DINO + SAM 的结果转换成 YOLO 标签；
- 支持类别映射；
- 支持置信度过滤；
- 支持空标签处理；
- 支持统计每类标注数量。

产出：

- `tools/generate_yolo_labels_from_auto.py`
- `configs/classes_visdrone.yaml`
- `outputs/auto_labels/`
- `results/label_statistics.csv`

#### Day 7：自动标注质量评估

目标：

- 将自动标签与人工标签进行匹配；
- 计算 IoU；
- 统计 precision、recall、matched count、false positives、false negatives；
- 按类别统计；
- 按目标尺寸统计 small / medium / large。

产出：

- `tools/evaluate_auto_labels.py`
- `results/auto_label_quality.csv`
- `results/auto_label_quality_by_class.csv`
- `results/auto_label_quality_by_size.csv`

这是报告的重要高分部分，必须完成。

#### Day 8：YOLO 基线训练准备

目标：

- 准备 YOLO 格式数据集；
- 写好训练配置；
- 支持人工标签训练；
- 支持自动标签训练；
- 生成服务器训练脚本。

产出：

- `configs/yolo_visdrone_manual.yaml`
- `configs/yolo_visdrone_auto.yaml`
- `scripts/server_train_yolo_manual.sh`
- `scripts/server_train_yolo_auto.sh`

注意：

- 本地只测试命令是否能启动。
- 正式训练必须在服务器运行。

#### Day 9：YOLO 训练与初步评估

目标：

- 在服务器上训练人工标签 YOLO baseline；
- 在服务器上训练自动标签 YOLO；
- 保存训练日志、权重、指标。

产出：

- `runs/manual_yolo/`
- `runs/auto_yolo/`
- `results/yolo_eval_manual.csv`
- `results/yolo_eval_auto.csv`

#### Day 10：消融实验

目标：

至少完成 2 到 3 个消融实验：

1. Grounding DINO only vs Grounding DINO + SAM；
2. 不同置信度阈值；
3. 不同 prompt；
4. 自动标签训练 vs 少量人工标签训练；
5. YOLOv8n vs YOLOv8s。

推荐最小消融：

- `DINO-only`
- `DINO+SAM`
- `DINO+SAM+confidence filter`

产出：

- `results/ablation_threshold.csv`
- `results/ablation_prompt.csv`
- `results/ablation_method.csv`

#### Day 11：小目标专项分析

目标：

- 按目标大小统计检测性能；
- 分析小目标漏检；
- 生成小目标可视化图；
- 生成失败案例图。

产出：

- `tools/analyze_small_objects.py`
- `results/small_object_analysis.csv`
- `results/failure_cases/`
- `results/visualizations/small_objects/`

#### Day 12：可视化整理

目标：

- 生成报告中需要的图：
  - 方法流程图；
  - 自动标注示例图；
  - SAM 修正前后对比图；
  - YOLO 检测结果图；
  - 失败案例图；
  - 指标柱状图或折线图；
  - 小目标分析图。

产出：

- `figures/pipeline.png`
- `figures/auto_annotation_examples/`
- `figures/sam_refinement_examples/`
- `figures/detection_results/`
- `figures/failure_cases/`
- `figures/metrics_charts/`

#### Day 13：报告初稿

目标：

完成报告初稿。

报告结构：

```text
摘要
1 引言
2 数据集介绍
3 方法设计
4 实验设置
5 实验结果与分析
6 可视化与失败案例分析
7 总结
8 组员分工
```

产出：

- `report/draft.md`
- `report/figures/`
- `report/tables/`

#### Day 14：报告完善与代码清理

目标：

- 补充实验表格；
- 补充分析文字；
- 检查图表编号；
- 检查代码可运行性；
- 清理无用文件；
- 补充 README；
- 写清楚本地和服务器运行方式。

产出：

- `README.md`
- `report/final.docx` 或 `report/final.pdf`
- `results/`
- `scripts/`

#### Day 15：最终打包

目标：

- 确认报告命名；
- 确认代码压缩包命名；
- 删除无关大文件；
- 保留必要权重下载说明；
- 检查可复现性。

产出：

```text
姓名-课题编号.pdf
姓名-课题编号-代码.zip
```

如果是自选课题，课题编号应使用 `5`。

---

## 3. 项目目录规范

推荐目录结构：

```text
project-root/
├── AGENTS.md
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── default.yaml
│   ├── classes_visdrone.yaml
│   ├── yolo_visdrone_manual.yaml
│   ├── yolo_visdrone_auto.yaml
│   └── experiment.yaml
├── data/
│   ├── samples/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── datasets/
│   │   ├── visdrone.py
│   │   └── yolo_format.py
│   ├── open_vocab/
│   │   ├── grounding_dino_infer.py
│   │   └── prompts.py
│   ├── segmentation/
│   │   └── sam_refine.py
│   ├── detection/
│   │   └── yolo_train_utils.py
│   ├── evaluation/
│   │   ├── bbox_metrics.py
│   │   ├── auto_label_eval.py
│   │   └── small_object_eval.py
│   └── utils/
│       ├── io.py
│       ├── visualization.py
│       ├── bbox.py
│       └── logging.py
├── tools/
│   ├── convert_visdrone_to_yolo.py
│   ├── visualize_labels.py
│   ├── run_grounding_dino_single.py
│   ├── run_grounding_dino_batch.py
│   ├── run_sam_refine_single.py
│   ├── run_sam_refine_batch.py
│   ├── generate_yolo_labels_from_auto.py
│   ├── evaluate_auto_labels.py
│   ├── analyze_small_objects.py
│   └── make_report_figures.py
├── scripts/
│   ├── local_debug.sh
│   ├── server_run_grounding_dino.sh
│   ├── server_run_sam_refine.sh
│   ├── server_train_yolo_manual.sh
│   ├── server_train_yolo_auto.sh
│   ├── server_eval_yolo.sh
│   └── server_run_all.sh
├── outputs/
│   ├── grounding_dino_json/
│   ├── sam_masks/
│   ├── auto_labels/
│   └── debug/
├── results/
│   ├── tables/
│   ├── metrics/
│   ├── visualizations/
│   └── failure_cases/
├── figures/
│   ├── pipeline/
│   ├── examples/
│   ├── charts/
│   └── report/
├── report/
│   ├── draft.md
│   ├── final.docx
│   └── final.pdf
└── tests/
    ├── test_bbox.py
    ├── test_yolo_label.py
    └── test_dataset.py
```

---

## 4. 编码规范

### 4.1 通用要求

所有 Python 代码必须满足：

- 使用 Python 3.10+；
- 尽量使用 type hints；
- 函数要有清晰 docstring；
- 不写超长函数；
- 不把路径硬编码到函数内部；
- 不在库文件中直接执行实验；
- 实验入口统一放在 `tools/` 或 `scripts/`；
- 能用配置文件控制的参数，不要写死；
- 对关键步骤输出日志；
- 对不存在的文件、空标注、无检测结果等情况做容错处理。

### 4.2 命令行规范

所有工具脚本都应支持命令行参数，例如：

```bash
python tools/run_grounding_dino_batch.py \
  --image-dir data/processed/visdrone/images/train \
  --output-dir outputs/grounding_dino_json/train \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda
```

不要写成只能在 IDE 中运行的代码。

### 4.3 配置文件规范

配置文件优先使用 YAML。

示例：

```yaml
project:
  name: open_vocab_drone_small_object_detection
  seed: 42

data:
  dataset: visdrone
  root: data/processed/visdrone
  image_dir: images/train
  label_dir: labels/train
  classes:
    - pedestrian
    - people
    - bicycle
    - car
    - van
    - truck
    - bus
    - motor

grounding_dino:
  prompt: "pedestrian. people. bicycle. car. van. truck. bus. motor."
  box_threshold: 0.35
  text_threshold: 0.25
  device: cuda

sam:
  model_type: vit_h
  checkpoint: checkpoints/sam_vit_h_4b8939.pth
  device: cuda

auto_label:
  min_box_area: 4
  confidence_threshold: 0.35
  use_sam_refine: true

yolo:
  model: yolo8s.pt
  epochs: 100
  imgsz: 1024
  batch: 16
  device: 0
```

---

## 5. 本地 WSL 与服务器分工

### 5.1 本地 WSL 允许执行的任务

Codex 可以在本地 WSL 中完成：

```text
代码生成
代码重构
配置文件编写
小样本数据转换
单图推理接口测试
少量图片可视化
标签格式检查
CSV 统计脚本测试
README 编写
报告草稿辅助
```

本地调试建议使用：

```text
data/samples/
```

样本数量建议：

- 5 张图片用于快速单元测试；
- 20 张图片用于小规模流程验证；
- 不超过 100 张图片用于本地轻量实验。

### 5.2 必须服务器执行的任务

以下任务默认必须在服务器上执行：

```text
全量 Grounding DINO 推理
全量 SAM 分割
YOLO 正式训练
RT-DETR 正式训练
多阈值大规模消融
多 prompt 大规模消融
最终模型评估
批量可视化生成
```

### 5.3 脚本必须同时支持本地和服务器

每个重实验脚本都要做到：

- 本地可用小样本测试；
- 服务器可用全量数据运行；
- 参数一致；
- 输出目录一致；
- 日志清晰；
- 可以跳过已经完成的结果。

示例：

```bash
# 本地 WSL 小样本调试
bash scripts/local_debug.sh

# 服务器全量运行
bash scripts/server_run_all.sh
```

---

## 6. 数据集策略

### 6.1 首选数据集

主数据集优先使用：

```text
VisDrone
```

推荐任务：

```text
Object Detection in Images
```

推荐类别：

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

### 6.2 可选补充数据集

如果主流程完成较早，可以增加：

```text
DIOR 小样本子集
```

DIOR 只作为补充展示或泛化测试，不作为必须完成项。

### 6.3 数据集处理原则

- 不要把原始大数据集提交到代码压缩包。
- `data/README.md` 中说明数据集下载方式和目录放置方式。
- 所有数据转换脚本必须可复现。
- 处理后的标签格式必须清晰。
- 类别映射必须写入配置文件。

---

## 7. 实验设计优先级

### 7.1 必须完成的实验

必须完成以下实验：

#### 实验一：人工标签 YOLO baseline

目的：

- 获得上限参考；
- 验证数据集和训练流程正常。

输出：

```text
results/tables/yolo_manual_baseline.csv
```

#### 实验二：自动标签质量评估

目的：

- 评估 Grounding DINO / Grounding DINO + SAM 自动标签质量；
- 计算与人工标签的 IoU 匹配情况。

输出：

```text
results/tables/auto_label_quality.csv
```

#### 实验三：自动标签训练 YOLO

目的：

- 验证自动标注是否能训练出有效检测模型。

输出：

```text
results/tables/yolo_auto_label_results.csv
```

#### 实验四：小目标专项分析

目的：

- 分析小目标检测难点；
- 体现项目主题。

输出：

```text
results/tables/small_object_analysis.csv
```

### 7.2 推荐消融实验

优先做以下消融：

1. Grounding DINO only vs Grounding DINO + SAM；
2. 不同 box threshold；
3. 不同 prompt；
4. 自动标签 vs 少量人工标签；
5. YOLOv8n vs YOLOv8s。

如果时间不够，至少完成前 2 个。

---

## 8. 评价指标

### 8.1 检测模型指标

必须支持：

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
```

### 8.2 自动标注质量指标

必须支持：

```text
Mean IoU
Matched GT Count
False Positive Count
False Negative Count
Auto-label Precision
Auto-label Recall
Per-class Statistics
Small / Medium / Large Object Statistics
```

### 8.3 小目标划分

默认按 COCO 风格划分：

```text
small: area < 32 * 32
medium: 32 * 32 <= area < 96 * 96
large: area >= 96 * 96
```

对于无人机图像，也可以额外使用相对面积：

```text
small: bbox_area / image_area < 0.001
medium: 0.001 <= bbox_area / image_area < 0.01
large: bbox_area / image_area >= 0.01
```

代码中应支持配置切换。

---

## 9. 可视化要求

项目必须生成以下可视化结果：

1. 原始人工标注可视化；
2. Grounding DINO 检测结果；
3. SAM mask 修正结果；
4. 自动生成 YOLO 标签可视化；
5. YOLO 最终检测结果；
6. 失败案例；
7. 小目标漏检案例；
8. 指标对比图。

可视化图片应保存到：

```text
figures/
results/visualizations/
```

用于报告的图片应复制或导出到：

```text
report/figures/
```

---

## 10. 报告导向规则

所有代码和实验都要服务于最终报告。

报告中必须能回答以下问题：

1. 本项目解决什么计算机视觉问题？
2. 为什么遥感/无人机小目标检测困难？
3. 为什么使用开放词汇检测模型？
4. Grounding DINO 在项目中起什么作用？
5. SAM 在项目中起什么作用？
6. 自动标签如何转换为 YOLO 格式？
7. 自动标签质量如何评估？
8. 自动标签能否训练出有效检测模型？
9. 小目标检测效果如何？
10. 项目有哪些失败案例和局限？
11. 如果继续改进，可以从哪些方向优化？

---

## 11. README 要求

README 必须包含：

```text
项目简介
方法流程图
环境安装
数据集准备
本地 WSL 调试方式
服务器运行方式
Grounding DINO 推理命令
SAM 分割命令
自动标签生成命令
YOLO 训练命令
评估命令
结果目录说明
主要实验结果
常见问题
```

必须明确说明：

> 本项目中 Codex 主要用于本地 WSL 环境下的代码开发与轻量调试。由于本地主机算力有限，大规模推理、SAM 批量分割和 YOLO 正式训练需要在远程 GPU 服务器上运行。

---

## 12. 权重与大文件管理

不要把以下内容提交进代码压缩包：

```text
原始数据集
大型模型权重
训练生成的全部中间文件
完整 runs 目录
大量 mask 图片
大规模推理 JSON
```

可以保留：

```text
少量 sample 图片
少量 sample 标签
少量 demo 输出
关键结果 CSV
关键可视化图
训练日志摘要
README 下载说明
```

建议 `.gitignore` 包含：

```gitignore
data/raw/
data/processed/
checkpoints/
weights/
runs/
outputs/
*.pt
*.pth
*.onnx
*.engine
*.zip
*.tar
*.tar.gz
__pycache__/
.vscode/
.DS_Store
```

---

## 13. 服务器脚本规范

服务器脚本必须写在 `scripts/` 目录下。

示例：

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python tools/run_grounding_dino_batch.py \
  --image-dir data/processed/visdrone/images/train \
  --output-dir outputs/grounding_dino_json/train \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda \
  --skip-existing
```

训练脚本示例：

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

yolo detect train \
  model=yolov8s.pt \
  data=configs/yolo_visdrone_auto.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=16 \
  project=runs \
  name=yolov8s_auto_labels
```

---

## 14. 错误处理要求

代码必须处理以下情况：

- 图片读取失败；
- 标注文件为空；
- 某张图片没有检测框；
- SAM 没有生成有效 mask；
- bbox 越界；
- bbox 宽高为 0；
- 类别无法映射；
- 输出目录不存在；
- 服务器中断后重复运行；
- CUDA 不可用时给出清晰错误提示。

不要因为单张图片失败导致整个批处理崩溃。应记录失败文件并继续运行。

---

## 15. 日志与结果记录

每个批量脚本都应输出日志。

建议保存：

```text
logs/
```

日志应包含：

```text
运行时间
配置参数
处理图片数量
成功数量
失败数量
跳过数量
输出目录
错误文件列表
```

实验结果表格统一保存为 CSV：

```text
results/tables/
```

---

## 16. 禁止事项

不要做以下事情：

1. 不要只写一个 notebook 完成全部项目。
2. 不要把路径写死为某台机器的绝对路径。
3. 不要把所有代码塞进一个巨大 Python 文件。
4. 不要只展示可视化图片而没有定量指标。
5. 不要只调用 Grounding DINO/SAM，不做下游评估。
6. 不要只训练 YOLO，不体现开放词汇自动标注。
7. 不要在本地 WSL 进行全量重训练。
8. 不要让正式实验依赖交互式手动操作。
9. 不要提交大型数据集和大型权重。
10. 不要无限扩展功能，必须服务于 15 天内完成。

---

## 17. 高分优先级

如果时间紧张，按以下顺序保证质量：

### 第一优先级：完整主流程

```text
VisDrone 小样本
-> Grounding DINO
-> SAM
-> YOLO 标签
-> YOLO 训练
-> 测试评估
```

### 第二优先级：实验表格

至少要有：

```text
人工标签 YOLO baseline 表
自动标签质量评估表
自动标签训练 YOLO 对比表
小目标分析表
```

### 第三优先级：可视化

至少要有：

```text
自动标注图
SAM 修正图
YOLO 检测图
失败案例图
```

### 第四优先级：消融实验

至少完成：

```text
DINO only vs DINO + SAM
不同置信度阈值
```

### 第五优先级：补充模型或补充数据集

只有主流程和报告都稳定后，才考虑：

```text
YOLOv8m
YOLOv10
RT-DETR
DIOR 子集
```

---

## 18. 最终交付物

最终应交付：

```text
报告 PDF 或 Word
代码压缩包
README
实验结果表格
关键可视化图
运行脚本
配置文件
```

如果使用自选课题，编号为：

```text
5
```

命名示例：

```text
姓名-5.pdf
姓名-5-代码.zip
```

多人组队时：

```text
姓名1-姓名2-姓名3-5.pdf
姓名1-姓名2-姓名3-5-代码.zip
```

---

## 19. 给 Codex 的执行原则

当用户要求修改或生成代码时，Codex 应优先：

1. 判断该任务属于本地轻量调试还是服务器重实验。
2. 如果是服务器重实验，应生成可在服务器运行的脚本，而不是尝试在本地完成。
3. 每次新增功能时，同时考虑：
   - 输入路径；
   - 输出路径；
   - 配置参数；
   - 日志；
   - 异常处理；
   - README 运行命令。
4. 尽量保持项目可复现。
5. 避免过度工程化。
6. 所有工作围绕 15 天内完成高分大作业展开。
7. 如果某个功能会明显拖慢项目进度，应提醒用户降低优先级。
8. 优先产出能写入报告的实验结果和图表。
9. 对于高显存、大数据、大训练任务，只生成脚本和说明，不在本地 WSL 强行运行。
10. 每次改动都应尽量保持已有接口兼容。

---

## 20. 项目一句话目标

在 15 天内完成一个基于 Grounding DINO + SAM + YOLO 的无人机/遥感小目标半自动标注与检测系统，形成完整实验闭环、定量结果、可视化分析和高质量课程报告。
