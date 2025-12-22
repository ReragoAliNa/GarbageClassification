# 基于深度学习（MobileNetV2）的智能垃圾分类系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📝 项目简介
本项目旨在开发一个轻量级、高准确率的智能垃圾分类系统。通过使用 **MobileNetV2** 迁移学习技术，系统能够对 20 类常见生活垃圾进行精准识别，并支持在消费级显卡（如 RTX 3060）或嵌入式设备上进行实时推理。

### 核心亮点：
- **闪电训练方案**：优化数据采样逻辑，支持在 30 分钟内完成 20 类的模型训练。
- **轻量级架构**：模型参数量小，推理延迟低于 20ms。
- **高鲁棒性**：通过多种数据增强手段（旋转、色彩抖动等），提升模型在复杂背景下的表现。

---

## 🛠️ 环境配置

### 硬件要求
- **GPU**: NVIDIA GeForce RTX 3060 或更高（显存 $\ge$ 6GB）
- **内存**: 16GB RAM

### 软件依赖
1. 创建虚拟环境：
   ```bash
   conda create -n garbage_env python=3.9 -y
   conda activate garbage_env
   ```
2. 安装依赖（注意 NumPy 版本兼容性）：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install "numpy<2" matplotlib pyyaml tqdm tensorboardX torchmetrics
   ```

---

## 📂 数据集准备
项目采用“4大类20小类”的标准进行训练。原始数据存放于 `data/raw` 文件夹下。

### 目录结构
```text
GarbageClassification/
├── data/
│   ├── raw/                # 原始图片（按类别分文件夹）
│   ├── train/              # 自动生成的训练集
│   ├── val/                # 自动生成的验证集
│   └── test/               # 自动生成的测试集
├── configs/
│   └── config.yaml         # 训练超参数配置
├── src/
│   ├── model.py            # MobileNetV2 模型结构
│   └── split_data_final.py # 数据极速拆分脚本
├── train.py                # 训练主程序
└── predict.py              # 单张图片预测脚本
```

---

## 🚀 快速开始

### 1. 数据极速划分
为了在 30 分钟内完成训练，使用采样限额脚本（每类限 300 张）：
```bash
python src/split_data_final.py
```

### 2. 开始模型训练
```bash
python train.py
```

### 3. 实时监控进度
```bash
tensorboard --logdir=logs
```

### 4. 单张图片推理
```bash
python predict.py --image_path test_image.jpg
```

---

## 📊 实验结果分析

### 训练曲线
模型在 RTX 3060 上训练 10-15 个 Epoch 即可实现收敛。
- **验证集准确率 (Val Acc)**: 约 **92%+**
- **单张推理延迟**: < **20ms**

### 性能评估
系统对规则物体（如易拉罐、塑料瓶）表现极佳。针对相似度较高的厨余垃圾，通过引入余弦退火学习率策略（Cosine Annealing）显著提升了精细特征的提取能力。

---

## 👨‍💻 贡献
项目由 ReragoAliNa 开发完成。欢迎提交 Issue 或 Pull Request 进行交流。

---

### 💡 提示：
- **大文件注意**：本项目已通过 `.gitignore` 排除 `data/` 文件夹，请开发者自行准备数据集。

---
