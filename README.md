# OVDAS

**OVDAS: Open-Vocabulary Drone Auto-annotation for Small-object Detection**

OVDAS is a computer vision course project for **semi-automatic annotation and detector training in drone-view small-object detection**.  
The project builds a full experimental pipeline based on **Grounding DINO + SAM + YOLOv8s** on the VisDrone detection task.

> 中文简介：本项目面向无人机航拍小目标检测，探索使用 Grounding DINO 进行开放词汇自动检测、使用 SAM 进行 mask-based 框体细化，并将自动生成的伪标签用于训练 YOLOv8s 检测器。项目重点不仅是跑通模型，还包括自动标签质量评估、小目标专项分析、失败案例可视化和权重可信性校验。

---

## Highlights

- **Open-vocabulary auto-annotation** with Grounding DINO.
- **SAM-based geometric refinement** from detection boxes to refined bounding boxes.
- **YOLO-format pseudo-label generation** for DINO-only and DINO+SAM settings.
- **YOLOv8s detector training** on manual labels and two auto-label variants.
- **Small-object analysis** by object size groups: small / medium / large.
- **Tensor-level weight hash checking** to verify that model comparisons are reliable.
- **Report-ready figures and CSV tables** for course report and GitHub presentation.

---

## Pipeline

The overall workflow is:

```text
VisDrone data preparation
    -> Grounding DINO open-vocabulary detection
    -> SAM mask-based bbox refinement
    -> YOLO pseudo-label generation
    -> YOLOv8s training
    -> YOLO validation and prediction export
    -> Auto-label quality evaluation
    -> Small-object analysis and visualization
    -> Report tables and figures
```

A report-ready pipeline figure is provided in the final report assets.

---

## Main Results

All models are evaluated on the **VisDrone validation set with manual labels**.

| Model | Training labels | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| YOLOv8s Manual | Manual annotations | 0.649 | 0.526 | 0.564 | 0.352 |
| YOLOv8s DINO-only | Grounding DINO pseudo labels | 0.454 | 0.209 | 0.283 | 0.199 |
| YOLOv8s DINO+SAM | Grounding DINO + SAM pseudo labels | 0.436 | 0.236 | 0.276 | 0.187 |

### Small-object Recall

| Model | Small recall | Medium recall | Large recall |
|---|---:|---:|---:|
| YOLOv8s Manual | 0.524513 | 0.852641 | 0.902950 |
| YOLOv8s DINO-only | 0.085776 | 0.492312 | 0.792578 |
| YOLOv8s DINO+SAM | 0.089556 | 0.513132 | 0.804948 |

**Observation.**  
The auto-label pipeline can produce a complete training workflow, but the pseudo-label models are still much weaker than the manual-label baseline. The main bottleneck is **low recall on dense small objects**. SAM refinement slightly improves recall but does not improve the overall mAP under the current setting.

---

## Dataset

This project uses **VisDrone2019-DET** for object detection in drone-view images.

The selected 8 categories are:

| YOLO ID | Class |
|---:|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | bus |
| 7 | motor |

### Data Statistics

| Label source | Train images | Train label files | Boxes |
|---|---:|---:|---:|
| Manual | 6471 | 6471 | 335146 |
| DINO-only | 6471 | 6471 | 93408 |
| DINO+SAM | 6471 | 6471 | 93408 |

The auto-label boxes are about **27.9%** of the manual boxes, which explains the recall gap in downstream YOLO training.

> The original VisDrone dataset is not included in this repository. Please download it from the official source and place it under `data/raw/VisDrone/`.

---

## Repository Structure

```text
OVDAS/
├── configs/
│   ├── classes_visdrone.yaml
│   ├── yolo_visdrone_manual.yaml
│   ├── yolo_visdrone_auto_dino_only.yaml
│   └── yolo_visdrone_auto_dino_sam.yaml
├── data/
│   ├── raw/                         # ignored, original datasets
│   └── processed/                   # ignored, processed YOLO-format data
├── external/                        # ignored, external model repos
├── checkpoints/                     # ignored, model weights
├── outputs/                         # ignored, intermediate DINO/SAM/auto-label outputs
├── runs/                            # ignored, YOLO training runs
├── results/
│   ├── tables/
│   ├── yolo_eval/
│   ├── yolo_predictions/
│   └── visualizations/
├── figures/
│   └── charts/
├── report/
│   ├── tables/
│   └── figures/
├── scripts/
│   ├── local_debug.sh
│   ├── server_run_grounding_dino_train.sh
│   ├── server_run_sam_refine_train.sh
│   ├── server_generate_auto_labels_train.sh
│   ├── server_prepare_auto_yolo_dataset.sh
│   ├── server_train_yolo_auto.sh
│   ├── server_check_yolo_weights.sh
│   └── server_after_retrain_analysis.sh
├── src/
│   ├── evaluation/
│   ├── open_vocab/
│   └── segmentation/
└── tools/
    ├── convert_visdrone_to_yolo.py
    ├── visualize_yolo_labels.py
    ├── run_grounding_dino_single.py
    ├── run_grounding_dino_batch.py
    ├── run_sam_refine_single.py
    ├── run_sam_refine_batch.py
    ├── generate_yolo_labels_from_auto.py
    ├── prepare_auto_yolo_dataset.py
    ├── evaluate_auto_labels.py
    ├── analyze_small_objects.py
    ├── check_yolo_weight_hashes.py
    └── make_report_figures.py
```

Large files such as datasets, checkpoints, training outputs, and intermediate auto-label JSONs are intentionally ignored by Git.

---

## Environment

The core experiments were run on a remote GPU server.

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, about 32GB VRAM |
| Python | 3.12.3 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| YOLO framework | Ultralytics YOLOv8.2.103 |
| Detector | YOLOv8s |
| Image size | 1024 |
| Batch size | 16 |
| Epochs | 100 |

Install basic dependencies:

```bash
pip install -r requirements.txt
```

or use conda:

```bash
conda env create -f environment.yml
conda activate ovdas
```

Grounding DINO and SAM should be installed according to their official repositories. Their large weights should be placed in `checkpoints/` and should not be committed.

---

## Local and Server Workflow

This project separates local development and remote computation.

### Local WSL

Use local WSL for:

- code editing;
- configuration checking;
- small-sample debugging;
- table and figure generation;
- report organization.

### Remote GPU Server

Use the server for:

- full Grounding DINO inference;
- full SAM refinement;
- YOLO training;
- YOLO validation and prediction export;
- small-object analysis;
- large-scale visualization.

---

## Usage

### 1. Convert VisDrone Annotations to YOLO Format

```bash
python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-train \
  --out-root data/processed/visdrone \
  --split train \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images
```

Validation split:

```bash
python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-val \
  --out-root data/processed/visdrone \
  --split val \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images
```

Visualize converted labels:

```bash
python3 tools/visualize_yolo_labels.py \
  --data-root data/processed/visdrone \
  --split val \
  --classes-config configs/classes_visdrone.yaml \
  --out-dir results/visualizations/manual_labels_val \
  --limit 20
```

---

### 2. Run Grounding DINO

Single-image debugging:

```bash
python3 tools/run_grounding_dino_single.py \
  --image data/samples/images/example.jpg \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --output-dir outputs/debug_grounding \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda \
  --config-file external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --checkpoint checkpoints/groundingdino_swint_ogc.pth
```

Server batch inference:

```bash
bash scripts/server_run_grounding_dino_train.sh
```

Expected output:

```text
outputs/grounding_dino_json/train/*.json
outputs/grounding_dino_json/train/*.jpg
```

---

### 3. Run SAM Refinement

```bash
bash scripts/server_run_sam_refine_train.sh
```

Expected output:

```text
outputs/sam_refine_json/train/*.json
```

The SAM-refined JSON keeps the original DINO fields and adds:

```text
refined_bbox_xyxy
mask_area
sam_score
mask_path
refine_status
```

If SAM refinement fails for a detection, the label generator falls back to the original DINO box.

---

### 4. Generate YOLO Pseudo Labels

```bash
bash scripts/server_generate_auto_labels_train.sh
```

Expected output:

```text
outputs/auto_labels/dino_only/train/*.txt
outputs/auto_labels/dino_sam/train/*.txt
results/tables/auto_label_stats_train_dino_only.csv
results/tables/auto_label_stats_train_dino_sam.csv
```

---

### 5. Prepare Auto-label YOLO Datasets

```bash
bash scripts/server_prepare_auto_yolo_dataset.sh
```

This creates YOLO-format datasets for:

```text
data/processed/visdrone_auto_yolo_dino_only/
data/processed/visdrone_auto_yolo_dino_sam/
```

> Important engineering note: avoid using directory-level symlinks for `images/train` that resolve back to the manual dataset image directory. Ultralytics may infer the label path from the resolved image path and accidentally read the manual label directory. Use real directories or hard-linked image files instead.

---

### 6. Train YOLOv8s Models

Manual baseline example:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=configs/yolo_visdrone_manual.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=16 \
  device=0 \
  project=runs \
  name=yolov8s_manual_visdrone
```

Auto-label training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH=16 RUN_DINO_ONLY=1 RUN_DINO_SAM=0 bash scripts/server_train_yolo_auto.sh

BATCH=16 RUN_DINO_ONLY=0 RUN_DINO_SAM=1 bash scripts/server_train_yolo_auto.sh
```

---

### 7. Check YOLO Weight Hashes

To avoid accidental weight reuse or wrong dataset paths:

```bash
bash scripts/server_check_yolo_weights.sh
```

Final tensor-level hash check:

| Pair | diff tensors | overall max abs diff |
|---|---:|---:|
| Manual vs DINO-only | 297 | 6.142578125 |
| Manual vs DINO+SAM | 297 | 5.75390625 |
| DINO-only vs DINO+SAM | 297 | 6.18359375 |

---

### 8. Run Final Analysis

```bash
bash scripts/server_after_retrain_analysis.sh
```

This script runs:

1. weight hash checking;
2. YOLO validation;
3. prediction TXT export;
4. small-object analysis;
5. report figure generation.

---

## Report Figures and Tables

Main report tables:

```text
report/tables/yolo_manual_baseline.csv
report/tables/yolo_auto_label_results.csv
report/tables/yolo_three_model_comparison.csv
report/tables/small_object_analysis.csv
report/tables/ablation_method.csv
report/tables/auto_label_quality.csv
```

Main report figures:

```text
report/figures/yolo_main_metrics.png
report/figures/three_model_map_comparison.png
report/figures/small_object_recall.png
report/figures/ablation_method.png
```

Additional YOLO evaluation visualizations:

```text
results/yolo_eval/manual_val/
results/yolo_eval/auto_dino_only_val/
results/yolo_eval/auto_dino_sam_val/
```

---

## Key Findings

1. **Auto-label training is feasible but not a replacement for manual labels.**  
   DINO-only and DINO+SAM can train YOLOv8s detectors, but their mAP is much lower than the manual-label baseline.

2. **Small-object recall is the main bottleneck.**  
   Manual small recall is 0.524513, while DINO-only and DINO+SAM reach only 0.085776 and 0.089556.

3. **SAM refinement slightly improves recall but not mAP.**  
   DINO+SAM improves recall from 0.209 to 0.236, but mAP@0.5 decreases from 0.283 to 0.276.

4. **Engineering validation is necessary.**  
   Tensor-level weight hashing and prediction checks are useful for detecting incorrect dataset paths or accidental weight reuse.

---

## Limitations

- Grounding DINO misses many dense small objects in VisDrone-style scenes.
- SAM refinement only adjusts geometry; it cannot recover objects that DINO did not detect.
- Prompt and threshold ablations were not fully explored in the final report.
- The current pipeline is more suitable for semi-automatic annotation assistance than direct replacement of human annotation.

---

## Future Work

- Use image slicing and multi-scale inference to improve small-object recall.
- Explore threshold tuning and prompt engineering systematically.
- Add a human-in-the-loop review stage for pseudo-label correction.
- Compare YOLOv8s with stronger detectors such as YOLOv8m, YOLOv10, RT-DETR, or small-object-specialized detectors.
- Evaluate whether SAM masks can be used beyond bounding-box refinement, such as mask-aware filtering.

---

## References

- Liu S. et al. **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**. arXiv:2303.05499, 2023.
- Kirillov A. et al. **Segment Anything**. ICCV, 2023.
- Zhu P. et al. **Vision Meets Drones: A Challenge**. arXiv:1804.07437, 2018.
- Ultralytics. **YOLOv8 Documentation**. <https://docs.ultralytics.com>

---

## Acknowledgement

This repository was developed as a computer vision course project.  
The original datasets and pretrained model weights are not redistributed in this repository.
