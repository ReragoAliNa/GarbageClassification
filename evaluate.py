import torch
import torch.nn as nn
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import GarbageClassifier

# ==========================================
# 1. 解决中文显示问题 (防止图片出现方块)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def evaluate():
    # --- 1. 环境与配置加载 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    docs_dir = os.path.join(base_dir, "docs")
    
    # 自动创建 docs 文件夹用于存放实验结果
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    # 加载配置 (显式指定 utf-8 解决编码报错)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动评估 | 使用设备: {device}")

    # --- 2. 测试集准备 (对应 224 分辨率) ---
    test_transform = transforms.Compose([
        transforms.Resize((config.get('image_size', 336), config.get('image_size', 336))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(base_dir, "data", "test")
    if not os.path.exists(test_dir):
        print(f"❌ 错误: 找不到测试集目录 {test_dir}，请先运行数据拆分脚本。")
        return

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.get('batch_size', 32), 
        shuffle=False, 
        num_workers=4
    )
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📊 测试集加载成功: 共 {num_classes} 个类别")

    # --- 3. 加载训练好的最优模型 ---
    model = GarbageClassifier(num_classes=num_classes).to(device)
    model_path = os.path.join(base_dir, "checkpoints", "garbage_classifier_best.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 未在 {model_path} 发现训练好的权重文件。")
        return

    # 加载权重到当前设备
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. 批量推理 ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="正在对测试集进行考核"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 5. 生成指标报告 (Text Report) ---
    print("\n" + "="*40)
    print("📋 实验性能指标明细")
    print("="*40)
    # digits=4 提高精度显示
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # 将文本报告保存至 docs 文件夹
    report_save_path = os.path.join(docs_dir, "evaluation_report.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report)

    # --- 6. 绘制并保存混淆矩阵 (Confusion Matrix) ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 11)) # 针对20类设置较大画布
    
    sns.heatmap(
        cm, 
        annot=True, # 显示具体数值
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.xlabel('预测类别 (Predicted)', fontsize=12)
    plt.ylabel('真实类别 (True)', fontsize=12)
    plt.title('20类智能垃圾分类系统 - 混淆矩阵', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片到 docs，设置 300 DPI 保证插入 Word 时高清
    cm_save_path = os.path.join(docs_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=300)
    
    print(f"\n✅ 评估完成！结果已存入项目 docs/ 文件夹：")
    print(f"1. 详细文本报告: docs/evaluation_report.txt")
    print(f"2. 混淆矩阵高清图: docs/confusion_matrix.png")
    
    plt.show()

if __name__ == "__main__":
    evaluate()