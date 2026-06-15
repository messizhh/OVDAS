# AGENTS.md

## 0. 当前阶段与最高优先级

本项目是《计算机视觉》课程大作业，当前已经完成基础版 OVDAS 流程，现进入为期 **10 天**的“改进方法提出、真实实验验证、最新方法对照与实验报告重写”阶段。

项目方向保持不变：

> 基于 Grounding DINO、SAM 与 YOLOv8s 的无人机航拍小目标半自动标注及检测研究

当前阶段的主方法暂定名为：

> **OVDAS-Tile：基于全图-切片多尺度推理、候选融合、尺寸感知过滤与选择性 SAM 的无人机小目标自动标注方法**

本文件是当前阶段的最高优先级规则。当本文件与旧 README、旧实验计划、旧 15 天安排或历史提示冲突时，以本文件为准。

当前阶段必须完成三项课程要求：

1. 给出可解释的改进方法，并通过真实实验验证效果；对比必须包含项目已有方法，可在条件允许时增加 2025 年及之后的相关领域方法。
2. 实验报告必须包含简要方法介绍、数据库介绍、实验设置、数据处理、参数设置，并重点分析实验结果。
3. 报告必须尽量体现实验由项目成员实际完成，例如保留服务器运行命令、终端输出、训练过程、结果截图、日志和 CSV。

---

## 1. 当前项目状态

### 1.1 已完成的基础流程

项目已经具备：

```text
VisDrone 标注转换
-> 人工标签可视化
-> Grounding DINO 单图与批量推理
-> SAM 框提示细化
-> 自动 YOLO 标签生成
-> 自动标签质量评估
-> YOLOv8s 下游训练
-> 人工标注验证集评估
-> 小/中/大目标专项分析
-> PR 曲线与混淆矩阵
-> 失败案例导出
-> 权重 hash 校验
```

当前阶段不得重新从零搭建已有流程，应在现有实现上进行最小、可复现的扩展。

### 1.2 已有可信基线结果

以下结果必须保留，不得覆盖、修改或伪造：

| 方法 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Small Recall |
|---|---:|---:|---:|---:|---:|
| Manual YOLOv8s | 0.649 | 0.526 | 0.564 | 0.352 | 0.5245 |
| DINO-only YOLOv8s | 0.454 | 0.209 | 0.283 | 0.199 | 0.0858 |
| DINO+SAM YOLOv8s | 0.436 | 0.236 | 0.276 | 0.187 | 0.0896 |

现有结果说明：

- 人工标签模型仍是当前性能上限参考；
- 自动标签流程能够训练出有效检测器，但与人工标签存在明显差距；
- 自动标签模型的主要短板是 Recall，尤其是 Small Recall；
- SAM 主要修正已有框，不能恢复 Grounding DINO 没有提出的目标；
- 新改进应优先提升小目标候选覆盖率，同时控制低阈值带来的误检。

### 1.3 已完成的 OVDAS-Tile 工程实现

项目已新增：

- 全图与重叠切片 Grounding DINO 推理；
- 640×640 切片和 20% overlap；
- 切片框映射回原图；
- 全图与切片候选融合；
- 同类别 NMS；
- 尺寸感知置信度过滤；
- 选择性 SAM；
- 固定 200 张实验子集；
- A-E 子集消融脚本；
- 统一自动标签评估脚本；
- 相应单元测试。

服务器 5 张真实 GPU 冒烟测试已经成功完成：

```text
processed_images: 5
failed_images: 0
raw_boxes: 992
merged_boxes: 433
invalid boxes: 0
out-of-bounds boxes: 0
```

### 1.4 当前阻塞问题

当前检测结果中出现以下组合短语：

```text
truck bus
car van truck bus
car truck
van truck bus
car truck bus
van bus
```

它们会导致 `class_id=None` 和 `class_name=None`。在通用组合类别解析规则修复、测试并重新冒烟验证前，禁止开始 200 张正式消融。

---

## 2. 六篇近期论文对本项目的启示

本阶段的设计与报告需要参考以下 2025-2026 年工作，但不得把未复现的方法写成自己的实验结果。

### 2.1 Light-Weight Cross-Modal Enhancement Method with Benchmark Construction for UAV-based Open-Vocabulary Object Detection（2025）

核心信息：

- 提出 UAV-Label Engine、UAVDE-2M、UAVCAP-15K 和 CAGE；
- CAGE 使用交叉注意力、门控机制和 FiLM 调制增强视觉-文本融合；
- 在 YOLO-World-v2 中验证，并报告 VisDrone 零样本性能提升；
- 论文训练依赖大规模 UAV 数据和 4×A800、100 epochs，超出本课程 10 天复现预算。

对本项目的启示：

- 无人机 OVD 的核心问题包括自然图像到航拍图像的域差异、微小目标和语义错配；
- 报告应分析精度、效率和部署成本，而不能只分析 mAP；
- CAGE 可作为近期相关工作和未来的跨模态融合方向，不作为本阶段强制复现方法。

### 2.2 Cross-View Open-Vocabulary Object Detection in Aerial Imagery（2025）

核心信息：

- 使用 aerial-ground 对比对齐和多实例文本词袋关联缓解跨视角域差异；
- 在 xView、DOTA-v2、VisDrone、DIOR、HRRSD 上评估；
- 说明简单微调可能造成灾难性遗忘，结构化跨视角对齐更有效；
- 需要额外的 aerial-ground correspondence 和较大规模训练。

对本项目的启示：

- 报告中应把“航拍视角域差异”列为基础问题，而不只讨论小目标；
- 类别名称及其词汇变化可能影响开放词汇检测；
- 该方法适合相关工作和未来研究，不适合在 10 天内完整复现。

### 2.3 OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images（2025）

核心信息：

- 支持文本和图像 prompt；
- 使用 alignment head 与 fusion head 平衡速度和精度；
- 使用预训练、微调、自训练三阶段流程；
- 在自训练阶段使用阈值、NMS 和视觉-文本相似度进行伪标签过滤；
- ORSD+ 包含约 47 万张图像、200 类，完整复现成本高。

对本项目的启示：

- 尺寸感知过滤、重复框合并和伪标签质量控制具有合理的研究依据；
- 本项目可以借鉴“多阶段过滤”思想，但不复制其大规模训练体系；
- 可选扩展是对中、大目标增加视觉-文本相似度过滤，但只有主实验完成后才能考虑。

### 2.4 Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation（2026）

核心信息：

- 比较 Grounding DINO、OWLv2、YOLO-World、YOLOE、LLMDet；
- OWLv2 获得较高召回，但产生大量 False Positive；
- 大词表会造成严重语义混淆；
- 将词表从 80 类缩小到每图约 3.2 类后性能显著提升；
- aerial prefix 和 synonym expansion 未带来稳定收益。

对本项目的启示：

- 不能默认“更长 prompt”或“更多同义词”一定更好；
- 可在固定 200 张子集上增加一次低成本 prompt 分组实验；
- 应重点记录 Precision-Recall 权衡、False Positive 和 False Negative；
- 外部模型若只提高 Recall 但误检失控，不能宣称整体更优。

### 2.5 AD-Det: Boosting Object Detection in UAV Images with Focused Small Objects and Balanced Tail Classes（2025）

核心信息：

- 使用 coarse-to-fine 框架；
- ASOE 从高分辨率特征图中定位小目标密集区域并进行局部放大；
- DCC 使用动态类别均衡 copy-paste 改善长尾类别；
- 全图结果和局部结果通过 NMS 融合；
- 在 VisDrone 和 UAVDT 上真实验证，并包含完整消融。

对本项目的启示：

- 全图与局部高分辨率推理融合是合理的小目标改进方向；
- OVDAS-Tile 可以视为无需重新训练特征区域选择器的轻量 coarse-to-fine 方案；
- 当前阶段采用固定重叠切片，不实现 ASOE 和 DCC；
- 报告必须诚实说明固定切片会处理更多背景，效率低于自适应区域方法。

### 2.6 Open-Vocabulary Object Detection in UAV Imagery: A Review and Future Perspectives（2025）

核心信息：

- 将 UAV OVOD 方法概括为伪标签方法和视觉-语言融合方法；
- 系统总结了小目标、密集背景、视角差异、尺度变化和恶劣成像条件；
- 指出伪标签方法的主要风险是错误传播；
- 指出高性能多阶段方法通常具有较高延迟和资源开销。

对本项目的启示：

- OVDAS 属于面向 UAV 的伪标签自动标注路线；
- 标签质量评估是核心实验，不能只报告下游 YOLO 指标；
- 报告需要同时分析漏检、误检、语义错配、定位误差与计算成本。

### 2.7 近期方法的复现分级

#### 必须真实运行

```text
Manual YOLOv8s
DINO-only
DINO+SAM
DINO+Tile
OVDAS-Tile
```

#### 可选真实外部基线

满足以下条件时，可选择一个近期通用 OVD 模型：

```text
优先候选：YOLOE-11 或 OWLv2
备选：YOLO-World
```

准入条件：

- 官方代码和权重可以在半天内完成环境准备；
- 使用相同 8 类、相同图像清单、相同人工标签和相同匹配规则；
- 只比较自动标签质量和推理时间；
- 不要求再训练 100 epochs 下游 YOLO；
- 不得占用主实验时间。

#### 只做文献对比

```text
CAGE
Cross-View OVD
OpenRSD
AD-Det
```

原因是它们依赖大规模额外数据、多阶段训练或不同的检测框架，无法在 10 天内进行公平、完整复现。

---

## 3. 改进方法候选与实施边界

### 3.1 主改进：OVDAS-Tile

主流程：

```text
输入 VisDrone 图像
  ├─ 全图 Grounding DINO
  └─ 重叠切片 Grounding DINO
          ↓
    切片框映射回原图
          ↓
    phrase 清洗、alias 与组合类别解析
          ↓
    全图/切片候选融合
          ↓
    class-aware NMS
          ↓
    尺寸感知置信度过滤
          ↓
    选择性 SAM refine
          ↓
    YOLO 自动标签
          ↓
    自动标签质量评估
          ↓
    最佳配置训练 YOLOv8s
```

核心动机：

- 全图推理保留全局语境和中、大目标；
- 切片放大小目标，改善候选覆盖；
- NMS 合并相邻切片的重复目标；
- 小目标使用更低阈值以优先保留；
- 中、大目标使用更高阈值控制误检；
- 小目标跳过 SAM，避免低分辨率 mask 导致框消失或异常；
- 中、大目标可使用 SAM 改善几何边界。

### 3.2 默认参数

```yaml
tile:
  size: 640
  overlap_ratio: 0.20
  include_full_image: true

grounding_dino:
  prompt: "pedestrian. people. bicycle. car. van. truck. bus. motor."
  box_threshold: 0.25
  text_threshold: 0.25

merge:
  method: class_aware_nms
  iou_threshold: 0.50

size_aware_filter:
  small_area_ratio: 0.001
  medium_area_ratio: 0.01
  small_score_threshold: 0.20
  medium_score_threshold: 0.25
  large_score_threshold: 0.35

selective_sam:
  min_refine_area_px: 1024

evaluation:
  match_iou_threshold: 0.50
```

参数不是最终结论。只有固定 200 张子集验证后，才能锁定全量配置。

### 3.3 次要改进候选：Prompt 分组推理

该实验来自 2026 年比较研究对语义混淆的启示，仅作为低成本补充消融。

允许比较：

```text
P0：完整 8 类 prompt
P1：人类/非机动车/机动车三组 prompt
P2：单类别逐次 prompt（只在极小子集或 200 张上）
```

限制：

- 不做无穷同义词搜索；
- 不把 aerial prefix 作为主要创新；
- 只允许在主 A-E 消融完成后增加；
- 若运行成本超过半天，立即取消；
- 所有 prompt 结果必须经过统一类别映射和 NMS。

### 3.4 可选改进：视觉-文本语义过滤

受 OpenRSD 伪标签过滤启发，可在主实验完成后探索：

- 仅对 medium/large 候选框裁剪区域；
- 使用可获得的视觉-文本编码器计算框图像与类别文本相似度；
- 删除明显语义不一致候选；
- 不对极小框强制使用该过滤，因为极小 crop 的语义特征不稳定。

该项不是必须完成项，禁止在主实验不完整时实施。

### 3.5 本阶段不实施的复杂改进

```text
CAGE 模块重新训练
Cross-View 对比对齐训练
OpenRSD 多阶段预训练/自训练
AD-Det ASOE+DCC 完整复现
新建百万级 UAV 开放词汇数据集
多个外部模型的完整下游训练
```

这些方向写入相关工作、局限性或未来工作，不得冒充当前实现。

---

## 4. 研究问题与可验证假设

### 4.1 必须回答的研究问题

1. 全图+切片是否比全图 DINO-only 提高整体 Recall 和 Small Recall？
2. 在相同阈值下，切片本身带来的收益是多少？
3. box threshold 从 0.35 降到 0.25 后，Recall、Precision 和预测框数量如何变化？
4. 尺寸感知过滤能否在保留小目标的同时减少误检？
5. 选择性 SAM 是否比全量 SAM 更适合小目标密集的 UAV 图像？
6. 改进后的自动标签 F1 和 Mean IoU 是否提高？
7. 自动标签改进能否传递到下游 YOLOv8s？
8. OVDAS-Tile 带来的额外时间和显存开销是否可接受？
9. 主要失败是否来自漏检、语义混淆、重复框、框偏移还是背景误检？

### 4.2 预注册假设

在正式实验前记录以下预期，但不得把预期写成结果：

- H1：DINO+Tile 的 Small Recall 高于 DINO-only；
- H2：DINO+Tile-0.25 的 Recall 高于 DINO+Tile-0.35，但 Precision 可能下降；
- H3：OVDAS-Tile 的 F1 高于无过滤的低阈值切片方法；
- H4：选择性 SAM 不会显著降低 Small Recall；
- H5：OVDAS-Tile 的推理时间明显高于 DINO-only；
- H6：上游标签质量提升不保证下游 mAP 必然提升。

实验结果无论是否支持假设都必须保留。

---

## 5. 必须比较的方法与公平性

### 5.1 项目内主表

最终主表至少包含：

```text
Manual YOLOv8s
DINO-only YOLOv8s
DINO+SAM YOLOv8s
OVDAS-Tile YOLOv8s
```

DINO+Tile 可以作为上游自动标签质量和消融结果展示；若时间和算力允许，也可训练下游模型，但不是必须。

### 5.2 固定 200 张 A-E 消融

固定条件：

```text
数据集：VisDrone train
图像数：200
seed：42
固定 image list：outputs/experiment_subsets/visdrone_train_seed42_200.txt
人工标签：同一 labels/train
匹配阈值：IoU=0.5
类别：相同 8 类
模型权重：相同 Grounding DINO 和 SAM 权重
```

必须比较：

| 编号 | 方法 | 配置 | 验证目的 |
|---|---|---|---|
| A | DINO-only | 全图，box=0.35，无 SAM | 原项目上游基线 |
| B | DINO+SAM | 全图，box=0.35，全量 SAM | SAM 原始效果 |
| C | DINO+Tile-0.35 | 全图+切片，box=0.35，无 SAM | 切片独立贡献 |
| D | DINO+Tile-0.25 | 全图+切片，box=0.25，无 SAM | 低阈值召回与误检 |
| E | OVDAS-Tile | 低阈值、尺寸过滤、选择性 SAM | 完整方法 |

禁止为 A-E 各自训练 100 epochs。200 张阶段只评估自动标签质量。

### 5.3 可追加消融

只有 A-E 完成后，最多增加两项：

```text
F：OVDAS-Tile 无选择性 SAM
G：OVDAS-Tile + prompt 分组
```

不允许无限搜索 tile size、overlap、阈值或 prompt。

### 5.4 公平性要求

所有方法必须尽可能保持：

- 相同图像；
- 相同类别定义；
- 相同人工标签；
- 相同匹配代码；
- 相同 IoU；
- 相同统计脚本；
- 相同下游 YOLOv8s 预训练权重；
- 相同 epochs、imgsz、batch、seed 和验证集。

论文中的数值只可放在“相关工作/文献对比”表中，不可与本项目实测数值直接混成同一公平实验表。不同数据集、类别、训练数据和评估指标必须明确标注“不可直接比较”。

---

## 6. 数据集与数据处理规则

### 6.1 主数据集

```text
VisDrone2019-DET
train: 6471 images
val: 548 images
```

项目实际使用 8 类：

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

### 6.2 200 张固定子集

要求：

- 从 train 中使用 seed=42 固定抽取；
- 已生成 200 张清单；
- 当前统计约 11093 个人工目标；
- 所有方法必须使用完全相同清单；
- 不允许重新随机抽样以获得更好结果。

### 6.3 数据处理必须记录

报告和日志必须记录：

- 原始 VisDrone 标注到 YOLO 的转换规则；
- ignored region 和未使用类别的处理；
- bbox 裁剪、归一化和越界处理；
- 空标签处理；
- 切片尺寸与重叠；
- tile 到原图坐标映射；
- 类别 alias 与组合短语规则；
- NMS 类型和阈值；
- size-aware filter 阈值；
- SAM 跳过条件；
- 自动标签到下游 YOLO 数据集的构建方式。

### 6.4 小目标划分

必须同时支持：

```text
COCO absolute area:
small < 32^2
medium: 32^2 to 96^2
large >= 96^2
```

和相对面积：

```text
small: bbox_area / image_area < 0.001
medium: 0.001 <= ratio < 0.01
large: ratio >= 0.01
```

报告主表使用项目当前已统一的划分方式，另一种作为补充分析时必须明确标注。

---

## 7. 评价指标

### 7.1 自动标签质量

必须输出：

```text
GT count
Prediction count
Matched count
True Positive
False Positive
False Negative
Precision
Recall
F1
Mean IoU
Average predictions per image
Total inference time
Average inference time per image
Per-class statistics
Small / Medium / Large statistics
Unknown phrase count
Invalid box count
Out-of-bounds box count
Failed image count
```

### 7.2 下游检测模型

必须输出：

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
Small Recall
Medium Recall
Large Recall
PR curve
Confusion matrix
Per-class AP/Recall（可获得时）
```

### 7.3 计算开销

至少记录：

```text
总推理时间
平均每张图时间
每张图平均 tile 数
raw boxes
merged boxes
最终标签框数量
SAM 实际调用数量
SAM skipped_small 数量
GPU 型号
PyTorch/CUDA 版本
峰值显存（可选）
```

### 7.4 改进有效性判断

不能只看 Small Recall。建议优先级：

1. 自动标签总体 F1；
2. Small Recall；
3. Overall Recall；
4. Precision；
5. Mean IoU；
6. Prediction count 与 GT count 差距；
7. 下游 mAP@0.5:0.95；
8. 推理时间。

若 Small Recall 提高但 Precision 严重下降，应写成“召回提升但误检代价较大”。

若自动标签质量提升但下游模型没有提升，应分析：

- 错误伪标签传播；
- 类别混淆；
- 定位误差；
- 标签数量失衡；
- YOLO 对噪声标签的敏感性；
- 改进是否只覆盖少数困难样本。

---

## 8. 10 天执行计划

### Day 1：规则更新、组合类别修复与 5 张复验

任务：

- 用本文件替换旧 `AGENTS.md`；
- 修复组合类别短语解析；
- 补充单元测试；
- 本地 compile、pytest、diff check；
- 本地 commit/push；
- 服务器 pull；
- 使用新输出目录重跑 5 张冒烟测试。

完成标准：

```text
测试全部通过
5/5 成功
unknown classes=0
invalid boxes=0
out-of-bounds boxes=0
JSON=5
可视化=5
```

证据：

- 本地 pytest 截图；
- Git commit 截图；
- 服务器命令和完成统计；
- 修复前后 unknown count 对比。

### Day 2：200 张 A-E 推理与 SAM

任务：

- 运行 A-E 五组；
- 每组独立输出目录；
- 记录开始/结束时间；
- 检查每组成功数、失败数和 JSON 数量；
- 不训练 YOLO。

证据：

- `nvidia-smi`；
- 五组启动参数；
- 进度日志；
- 200/200 完成统计。

### Day 3：200 张统一评估、消融分析与最终配置

任务：

- 生成 summary、by-class、by-size CSV；
- 分析 A→B、A→C、C→D、D→E；
- 最多追加一到两组小范围实验；
- 锁定最终全量配置；
- 导出同图五方法对比和失败案例。

完成标准：

- 明确每个组件的独立作用；
- 选出一个全量配置；
- 写出选择理由；
- 停止无限调参。

### Day 4：6471 张 train 全量自动标签

任务：

- 使用最终配置全量运行；
- 支持 `--skip-existing`；
- 保存 JSON、必要 SAM 输出、YOLO 标签和统计；
- 统计每类、每尺寸、空标签和 unknown。

完成标准：

```text
6471 张完成或有明确失败列表
图像与标签一一对应
无越界框
无静默类别丢失
完整日志保存
```

### Day 5：548 张 val 自动标签质量验证

任务：

- 在 val 使用同一最终配置；
- 计算总体、类别和尺寸指标；
- 与 DINO-only 和 DINO+SAM 统一比较；
- 输出效率表和可视化。

完成标准：

- 得到最终自动标签质量主表；
- 得到 small/medium/large 表；
- 得到典型成功、漏检、误检和语义混淆案例。

### Day 6：构建下游数据集并启动 YOLOv8s

任务：

- 构建独立 OVDAS-Tile YOLO 数据集；
- 避免目录级软链接导致 Ultralytics 标签路径误读；
- 检查图片数、标签数、类别范围、空标签；
- 启动 100 epochs 训练。

固定条件：

```yaml
model: yolov8s.pt
epochs: 100
imgsz: 1024
batch: 16
seed: 与已有基线一致
val: 人工标注 VisDrone val
```

### Day 7：训练完成、正式评估与 hash

任务：

- 完成或继续断点训练；
- 导出最终指标；
- 生成 PR 曲线、混淆矩阵和预测图；
- 运行 small/medium/large 分析；
- 检查权重 hash；
- 确认未混用其他模型权重。

禁止：

- 用低于 100 epochs 的结果与已有 100 epochs 基线直接等价比较；
- 用同一权重冒充不同方法；
- 只截图而不保存 CSV。

### Day 8：近期方法对照、效率与失败分析

优先顺序：

1. 完成项目内主实验；
2. 完成效率、类别、尺寸和失败案例分析；
3. 时间允许时运行一个外部 OVD 模型；
4. 整理六篇近期论文的相关工作对照表。

外部模型实验停止条件：

- 环境配置超过半天；
- 权重不可稳定下载；
- 评估接口无法在一天内统一；
- 会延误报告和主实验。

### Day 9：实验报告重写

任务：

- 完成方法、数据集、实验设置和结果章节；
- 插入真实截图；
- 把所有数字与 CSV 对齐；
- 加入近期方法相关工作；
- 重点写结果分析，而不是堆砌模型原理。

### Day 10：审查、复现检查与最终交付

任务：

- 检查报告数字与源文件；
- 检查图表标题、单位和坐标；
- 检查 README 命令；
- 清理无关中间文件；
- 保存少量 demo；
- 生成最终 PDF 和代码包；
- 完成一次最终可复现性检查。

---

## 9. 实验报告要求

### 9.1 推荐结构

```text
摘要
1 引言
2 相关工作
  2.1 开放词汇目标检测
  2.2 UAV/遥感开放词汇检测
  2.3 UAV 小目标与 coarse-to-fine 检测
3 数据库与数据处理
  3.1 VisDrone 数据集
  3.2 类别选择与标注转换
  3.3 小目标分布与类别不平衡
4 基础方法
  4.1 Grounding DINO
  4.2 SAM
  4.3 自动标签生成
  4.4 YOLOv8s 下游训练
5 改进方法 OVDAS-Tile
  5.1 设计动机
  5.2 全图-切片多尺度推理
  5.3 类别归一化与候选融合
  5.4 尺寸感知过滤
  5.5 选择性 SAM
6 实验设置
  6.1 硬件与软件环境
  6.2 数据划分
  6.3 参数设置
  6.4 对比方法
  6.5 评价指标
7 实验结果与分析
  7.1 已有方法对比
  7.2 200 张组件消融
  7.3 完整 val 自动标签质量
  7.4 下游 YOLOv8s 结果
  7.5 小目标专项分析
  7.6 类别分析
  7.7 效率分析
  7.8 失败案例
  7.9 与 2025-2026 近期方法的联系与差异
8 局限性与未来工作
9 总结
10 组员分工
```

### 9.2 方法介绍要求

方法介绍应简洁，重点解释：

- 为什么 DINO-only 漏检小目标；
- 为什么 SAM 不能恢复漏检目标；
- 为什么切片能提高小目标像素占比；
- 为什么要保留全图分支；
- 为什么需要 NMS；
- 为什么不同尺寸使用不同阈值；
- 为什么 small 框跳过 SAM。

不得把六篇论文的方法大段复制为本项目方法。

### 9.3 数据库介绍要求

至少包括：

- VisDrone 来源与场景；
- train/val 数量；
- 原始类别和项目使用的 8 类；
- 航拍小目标、密集目标、遮挡、复杂背景和尺度变化；
- 小目标比例或项目实际统计；
- 标注格式转换过程。

### 9.4 实验设置要求

必须清楚列出：

- GPU、CPU、Python、PyTorch、CUDA；
- Grounding DINO 配置与权重；
- SAM 类型与权重；
- prompt；
- box/text threshold；
- tile size 和 overlap；
- NMS 阈值；
- size-aware filter；
- SAM 跳过面积；
- YOLOv8s 训练参数；
- 随机种子；
- IoU 匹配规则；
- 小/中/大目标定义。

### 9.5 结果分析要求

结果章节是报告重点，至少回答：

- 切片是否真正提高 Recall；
- 增加的预测框中有多少是 TP、多少是 FP；
- 低阈值的收益和代价；
- 过滤是否恢复 Precision；
- SAM 对不同尺寸目标的效果；
- 哪些类别改善最大；
- 哪些类别仍语义混淆；
- 上游改进是否传递到下游；
- 计算时间增加多少；
- 相比 AD-Det、OpenRSD、CAGE 等近期方法，本项目的优势和局限是什么。

### 9.6 近期方法文献对比规则

可以建立“近期工作对比”表，列出：

```text
论文
年份
基础模型
主要问题
核心技术
数据集
是否在 VisDrone 评估
是否真实复现
与本项目关系
```

必须明确：

- 文献结果不是本项目实测结果；
- 不同训练数据和指标不可直接比较；
- CAGE、Cross-View、OpenRSD、AD-Det 主要用于研究定位；
- 若实际运行 YOLOE/OWLv2，单独列出“同协议实测外部基线”。

---

## 10. “自己完成实验”的证据要求

### 10.1 必须保留的截图

1. 服务器 `nvidia-smi`；
2. Python、PyTorch、CUDA 版本；
3. Git commit 和服务器代码版本；
4. 固定 200 张子集生成命令与统计；
5. 组合短语修复前后对比；
6. A-E 五组运行参数；
7. 推理进度与最终成功/失败数；
8. 200 张 CSV 终端打印；
9. 全量 6471 张完成统计；
10. 548 张 val 评估输出；
11. YOLOv8s 训练启动参数；
12. 训练 epoch 中间过程；
13. 第 100 epoch 或最终结果；
14. PR 曲线、混淆矩阵；
15. 权重 hash；
16. 同一图像的 Manual GT、DINO-only、DINO+SAM、DINO+Tile、OVDAS-Tile 对比；
17. 至少三类失败案例；
18. 出错与修复记录，例如 BERT 缓存、组合类别映射。

### 10.2 截图命名建议

```text
report/figures/evidence/
  env_nvidia_smi.png
  env_torch_cuda.png
  subset_seed42_200.png
  phrase_fix_before_after.png
  ablation_run_A_E.png
  ablation_summary.png
  full_train_autolabel_done.png
  val_quality_eval.png
  yolo_train_start.png
  yolo_epoch100.png
  yolo_final_metrics.png
  weight_hash_check.png
```

### 10.3 截图使用原则

- 截图是运行证据，不替代 CSV；
- 截图中的路径、时间和参数应可读；
- 不截取无法判断来源的孤立数字；
- 不修改终端输出数字；
- 报告正文引用截图时解释其作用；
- 日志原文件必须保留。

---

## 11. 本地 WSL 与服务器分工

### 11.1 本地 WSL

允许：

```text
代码修改
单元测试
静态检查
配置文件与脚本编写
5-20 张轻量验证
CSV/绘图脚本开发
README 与报告修改
Git commit / push
```

禁止：

```text
6471 张 Grounding DINO 全量推理
全量 SAM
YOLOv8s 100 epochs 正式训练
多模型大规模消融
```

### 11.2 远程 GPU 服务器

负责：

```text
5 张真实 GPU 冒烟测试
200 张 A-E 消融
6471 张 train 自动标签
548 张 val 自动标签评估
SAM 批处理
YOLOv8s 正式训练
最终评估与批量可视化
```

### 11.3 Git 工作流

```text
本地修改
-> 本地测试
-> git commit
-> git push
-> 服务器临时 stash 必要 tracked 修改
-> git pull --ff-only
-> git stash pop
-> 服务器运行实验
```

服务器上的 `outputs/`、`runs/`、大型日志、权重和大批量图片不上传 GitHub。

---

## 12. 编码、测试与脚本规范

### 12.1 Codex skill 使用

根据任务类型，在提示词开头明确调用相关 skill：

```text
新功能或修复：Use the karpathy-coding skill.
测试驱动修改：Use the tdd skill.
完成后审查：Use the review skill.
环境或报错排查：Use the diagnose skill.
```

只调用当前任务真正需要的 skill。

### 12.2 Python 规范

- Python 3.10+；
- 尽量使用 type hints；
- 必要函数有 docstring；
- 避免超长函数；
- 库代码不直接启动实验；
- 路径从参数或配置传入；
- 不为少数样例硬编码；
- 旧接口保持兼容；
- 支持空结果；
- 支持单图失败后继续；
- 明确记录 failed files。

### 12.3 必须测试的内容

```text
切片覆盖完整图像
20% overlap
边缘 tile
小图与长图
tile bbox 到原图映射
bbox 裁剪
class-aware NMS
不同类别不互相抑制
组合类别短语解析
未知类别保留为 unknown
selective SAM skipped_small
SAM 失败 fallback
YOLO 标签范围
JSON 字段兼容
```

### 12.4 本地检查命令

```bash
python -m compileall src tools
python -m pytest -q
git diff --check
bash -n scripts/*.sh
```

### 12.5 服务器脚本规范

所有正式脚本应：

```bash
set -euo pipefail
```

并支持：

- 参数或环境变量指定路径；
- `CUDA_VISIBLE_DEVICES`；
- `OMP_NUM_THREADS` 使用合法正整数；
- `--skip-existing`；
- 独立输出目录；
- `tee` 保存日志；
- 记录 Git commit；
- 记录 Python/PyTorch/CUDA；
- 记录开始和结束时间；
- 记录成功、失败、跳过数量。

---

## 13. 输出目录与命名规范

推荐保留：

```text
outputs/experiment_subsets/
outputs/tile_ablation_subset/
outputs/ovdas_tile_train/
outputs/ovdas_tile_val/
outputs/auto_labels/ovdas_tile/
runs/yolov8s_ovdas_tile/
results/tables/
results/visualizations/ovdas_tile/
results/failure_cases/ovdas_tile/
figures/charts/
report/figures/
report/tables/
logs/
```

200 张方法目录必须互相独立：

```text
outputs/tile_ablation_subset/dino_only_035/
outputs/tile_ablation_subset/dino_sam_035/
outputs/tile_ablation_subset/dino_tile_035/
outputs/tile_ablation_subset/dino_tile_025/
outputs/tile_ablation_subset/ovdas_tile/
```

结果表建议：

```text
results/tables/tile_ablation_subset_summary.csv
results/tables/tile_ablation_subset_by_class.csv
results/tables/tile_ablation_subset_by_size.csv
results/tables/ovdas_tile_val_summary.csv
results/tables/ovdas_tile_val_by_class.csv
results/tables/ovdas_tile_val_by_size.csv
results/tables/yolo_four_model_comparison.csv
results/tables/efficiency_comparison.csv
results/tables/recent_method_literature_comparison.csv
```

---

## 14. 错误处理要求

必须处理：

- 图片读取失败；
- 标注文件不存在或为空；
- 无检测框；
- bbox 非法或越界；
- 类别无法映射；
- 组合 phrase；
- tile 坐标错误；
- NMS 输入为空；
- SAM 空 mask；
- SAM 异常；
- CUDA 不可用；
- Hugging Face 模型未缓存；
- 服务器网络中断；
- 输出目录已有部分结果；
- Ultralytics 标签路径误读；
- 训练中断和 resume；
- CSV 缺列或重复 append。

单张失败不得导致整个批处理崩溃。失败必须进入日志并在最终统计中可见。

---

## 15. 日志、元数据与可复现性

每个正式实验必须保存：

```text
method_name
Git commit
command line
config path
image list
random seed
model checkpoint path
model checkpoint hash
prompt
thresholds
tile settings
NMS settings
SAM settings
start time
end time
total images
processed/skipped/failed
output directory
```

所有关键数字应能追溯到：

```text
CSV
JSON metadata
训练 results.csv
日志
模型验证输出
```

禁止仅凭人工抄写数字形成报告主表。

---

## 16. 权重与大文件管理

不得提交到 GitHub 或最终代码包：

```text
原始 VisDrone 数据集
Grounding DINO/SAM/YOLO 大型权重
完整 outputs
完整 runs
大规模 JSON
大量 mask
Hugging Face 缓存
external 中的大型仓库缓存
```

可以提交或打包：

```text
代码
配置
运行脚本
关键 CSV
少量 demo JSON
少量对比图
日志摘要
README
AGENTS.md
论文引用信息
```

注意清理：

```text
*:Zone.Identifier
*.aux
*.log
*.out
*.toc
*.synctex*
```

其中正式实验日志不得误删，应复制必要摘要到交付目录。

---

## 17. 高分优先级与停止规则

### 第一优先级

```text
修复组合类别
完成 A-E 200 张消融
锁定最佳 OVDAS-Tile
完成全量标签和 val 评估
完成一组 OVDAS-Tile YOLOv8s 训练
```

### 第二优先级

```text
小目标分析
类别分析
效率分析
失败案例
完整报告
真实运行证据
```

### 第三优先级

```text
一个近期外部 OVD 实测基线
Prompt 分组消融
视觉-文本语义过滤
```

### 立即停止扩展的条件

- Day 3 仍未完成 200 张消融；
- Day 5 仍未开始全量 train；
- Day 7 仍未获得下游正式结果；
- 外部模型环境配置超过半天；
- 新功能无法直接回答课程要求；
- 新实验不能形成公平定量对比。

---

## 18. 禁止事项

1. 不伪造实验数据、CSV、截图、日志或训练结果。
2. 不覆盖已有 Manual、DINO-only、DINO+SAM 结果。
3. 不混用模型权重、标签目录、预测目录或缓存。
4. 不在组合类别映射未修复时运行正式 200 张消融。
5. 不在 200 张结果未分析时运行全量 6471 张改进实验。
6. 不为每个消融配置训练 100 epochs。
7. 不无限搜索参数。
8. 不把论文中的结果冒充本项目实测结果。
9. 不直接声称本项目优于 CAGE、OpenRSD、Cross-View 或 AD-Det，除非在同协议下真实复现。
10. 不只展示截图而缺少 CSV 和定量指标。
11. 不只报告 mAP 而忽略 Recall、Small Recall 和误检。
12. 不因为结果不理想而删除实验。
13. 不在本地 WSL 强行执行正式重实验。
14. 不提交大型数据、权重和完整运行目录。
15. 不在主报告未完成时盲目增加多个最新模型。

---

## 19. 最终交付物

最终至少交付：

```text
实验报告 PDF
代码压缩包
README.md
AGENTS.md
环境与运行说明
关键配置文件
服务器运行脚本
关键 CSV 表格
关键可视化图
少量运行证据截图
必要日志摘要
```

推荐代码包保留：

```text
src/
tools/
scripts/
configs/
tests/
results/tables/
report/figures/关键图片
README.md
AGENTS.md
requirements.txt 或 environment.yml
```

---

## 20. 给 Codex 的执行原则

Codex 每次接到任务时必须：

1. 先判断任务属于本地开发还是服务器重实验；
2. 根据任务选择合适的 skill；
3. 修改前阅读当前实现和测试；
4. 优先进行最小范围改动；
5. 新增功能必须补测试；
6. 不破坏已有接口和结果；
7. 所有重实验只生成脚本和准确命令，由用户在服务器运行；
8. 不声称未实际运行的命令已经成功；
9. 不伪造指标；
10. 完成后汇报修改文件、测试命令、真实结果和潜在风险；
11. 所有工作必须服务于 10 天内完成真实改进验证和高质量实验报告。

---

## 21. 项目一句话目标

在 10 天内完成并真实验证 OVDAS-Tile，证明或否定“全图-切片多尺度候选融合、尺寸感知过滤与选择性 SAM”对 VisDrone 自动标注和下游小目标检测的作用，并以项目已有方法为主要基线、以 2025-2026 年 UAV 开放词汇检测研究为学术参照，形成可复现、可审查且具有真实运行证据的课程实验报告。
