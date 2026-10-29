# 上游来源与许可

本仓库不打包作者原始源码、模型权重、数据集、视频或本地虚拟环境。`scripts/bootstrap.py` 按固定提交与 SHA-256 从上游获取所需材料；`external/`、`weights/`、`runs/` 均被 Git 忽略。

- GLAD 源码和权重来自 https://github.com/WindyLab/Global-Local-MAV-Detection 。本项目的候选网络结构和预处理遵循其已发布权重接口。上游 README 带 MIT 徽章，但核查的根目录没有独立 LICENSE 文件；本项目不替上游授予或改写模型许可。
- YOLOv5 v6.0 来自 https://github.com/ultralytics/yolov5 ，其源码自带 GPL-3.0 LICENSE，下载时保留上游文件。分发组合程序时需遵守适用的上游许可。
- 本项目为独立工程复现，不是作者官方发行版。论文方法、公开实现和本项目自行补齐的部分分别在 docs/reproduction_scope.md 中标明。
