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

# 解决中文显示问题（如果你的类别是中文名，请取消下面两行的注释）
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False

def evaluate():
    # --- 1. 环境与配置加载 ---
    # 获取项目根目录，确保路径正确
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    docs_dir = os.path.join(base_dir, "docs")
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    # 加载配置，显式指定 utf-8 编码
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 评估启动 | 使用设备: {device}")

    # --- 2. 测试集数据准备 ---
    # 保持与训练时一致的预处理方案
    test_transform = transforms.Compose([
        transforms.Resize((config.get('image_size', 224), config.get('image_size', 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 路径指向 data/test
    test_dir = os.path.join(base_dir, "data", "test")
    if not os.path.exists(test_dir):
        print(f"❌ 错误: 找不到测试集目录 {test_dir}")
        return

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.get('batch_size', 64), 
        shuffle=False, 
        num_workers=8
    )
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"📊 待评估类别数: {num_classes}")

    # --- 3. 加载训练好的模型 ---
    model = GarbageClassifier(num_classes=num_classes).to(device)
    model_path = os.path.join(base_dir, "checkpoints", "garbage_classifier_best.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 权重文件不存在 -> {model_path}")
        return

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. 执行推理获取预测结果 ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="推理中"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- 5. 计算并打印分类报告 ---
    print("\n" + "="*30)
    print("📈 性能指标报告")
    print("="*30)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # 保存文本报告到 docs
    report_save_path = os.path.join(docs_dir, "evaluation_report.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report)

    # --- 6. 绘制混淆矩阵热图 ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 11)) # 20类建议画布稍大一点
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Garbage Classification System', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片到 docs
    cm_save_path = os.path.join(docs_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=300) # 提高清晰度
    print(f"\n✅ 评估结果已成功保存至 docs 文件夹：")
    print(f"1. 详细报告: {report_save_path}")
    print(f"2. 可视化图: {cm_save_path}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    evaluate()