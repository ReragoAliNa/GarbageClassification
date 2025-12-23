# 基于 MobileNetV2 & 迁移学习的智能垃圾分类系统

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![GPU Training](https://img.shields.io/badge/GPU-RTX%203060-green.svg)](https://developer.nvidia.com/cuda-zone)

## 📝 项目简介
本项目旨在开发一个针对 **20类常见垃圾（重点涵盖厨余垃圾）** 的实时识别系统。通过使用轻量级卷积神经网络 **MobileNetV2** 和 **迁移学习** 技术，系统能够在极短的训练时间内实现高精度的分类。

针对大规模数据集（11GB），项目提出并实现了“**闪电训练方案**”，在 RTX 3060 平台上仅需不到 30 分钟即可完成迭代。

## 🚀 核心亮点
- **极致性能**：在测试集中，部分典型类别（如蛋挞、西红柿、蔬菜）实现 **100% 置信度** 的精准识别。
- **闪电拆分**：自研 `split_data_final.py` 脚本，支持对海量原始数据进行采样限额，防止训练过载。
- **Top-K 推理**：预测脚本支持展示前 5 个最相关的类别及其概率分布。
- **环境适配**：完美解决 Windows 环境下的 GBK 编码报错及 NumPy 2.0 兼容性问题。

## 📂 项目结构
```text
GarbageClassification/
├── checkpoints/          # 最优模型权重文件 (.pth)
├── configs/              # 训练参数配置文件 (config.yaml)
├── data/                 # 数据集存放（由脚本自动划分）
│   ├── train/            # 训练集 (70%)
│   ├── val/              # 验证集 (20%)
│   └── test/             # 测试集 (10%)
├── docs/                 # 实验报告、混淆矩阵、预测截图
├── src/                  # 核心插件代码
│   ├── model.py          # MobileNetV2 结构重构
│   ├── dataset.py        # 数据加载与增强
│   └── split_data_final.py # 数据极速拆分脚本
├── train.py              # 模型训练主程序
├── evaluate.py           # 性能评估与混淆矩阵生成
└── predict.py            # Top-K 图像推理演示
```

## 🛠️ 环境安装
```bash
# 建议使用虚拟环境
conda activate garbage_env

# 安装核心依赖 (针对 CUDA 11.8 优化)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" matplotlib pyyaml tqdm seaborn scikit-learn
```

## 📈 实验流程

### 1. 数据极速划分
将原始图片放入 `data/raw` 后运行（每类采样 300 张）：
```bash
python src/split_data_final.py
```

### 2. 启动模型训练
```bash
python train.py
```

### 3. 生成性能报告
```bash
python evaluate.py
```

### 4. 单张图片推理 (Top-5)
```bash
python predict.py
```

## 如何查看监控看板？
**启动服务**：在项目根目录下打开终端，输入：
```Bash
tensorboard --logdir=logs --port=6006
```
**访问界面**：在浏览器中打开 `http://localhost:6006`。
**关键操作**：
Smoothing：建议调节至 `0.6` 以平滑曲线。
Data Table：勾选右侧 `Enable data table` 以查看各阶段精确数值。

## 📊 实验结果分析
- **测试集总准确率**: **82.87%**
- **典型分类表现**: 
  - **蛋挞**: 100% 准确率
  - **蔬菜**: 96.43% 准确率
- **收敛性**: 在迁移学习加持下，模型在第 10 个 Epoch 后 Loss 稳定在 0.13 左右，表现出极佳的泛化能力。

---
