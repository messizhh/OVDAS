# Data Directory

本目录只保存数据放置说明和少量本地调试样本，不提交完整原始数据集或大规模处理结果。

## 原始数据

原始 VisDrone 数据集应该放到：

```text
data/raw/VisDrone/
```

`data/raw/` 已被 `.gitignore` 排除，请不要将完整数据集提交到仓库。

## 本地小样本

用于本地 WSL 轻量调试的小样本图片放到：

```text
data/samples/images/
```

对应标签放到：

```text
data/samples/labels/
```

建议只放 5 到 20 张图片用于快速测试数据转换、标注可视化和脚本 smoke test。

## 处理后数据

后续 Day 2 的 VisDrone 转 YOLO 脚本会将处理后的 YOLO 数据放到：

```text
data/processed/visdrone/
```

`data/processed/` 已被 `.gitignore` 排除，避免提交转换后的大规模数据。
