import torch
import yaml
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import GarbageClassifier

# 1. 中文支持与映射
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

ID_TO_CHINESE = {
    "0": "八宝粥", "1": "冰激凌", "2": "地瓜", "3": "榴莲", "4": "橙子", 
    "5": "炒饭类", "6": "粉条", "7": "糖葫芦", "8": "腊肠", "9": "茶叶", 
    "10": "草莓", "11": "菠萝", "12": "蔬菜", "13": "蛋挞", "14": "蛋糕", 
    "15": "西红柿", "16": "豆腐", "17": "豌豆", "18": "饼干", "19": "鸡蛋"
}

def predict_top_k(image_path, model_path, config_path, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # --- 关键：确保类别排序与 ImageFolder 完美一致 ---
    train_dir = os.path.join(os.path.dirname(config_path), "..", "data", "train")
    # PyTorch 默认是按字符串顺序排序的: '0', '1', '10', '11'...'2'
    folder_names = sorted(os.listdir(train_dir))
    
    # --- 加载模型 ---
    model = GarbageClassifier(num_classes=len(folder_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 图像预处理 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_raw = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_raw).unsqueeze(0).to(device)

    # --- 推理与 Top-K 计算 ---
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        # 获取前 k 个最高概率及其索引
        top_probs, top_indices = torch.topk(prob, k)

    # 转换数据到 CPU
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # 映射名称
    top_names = [ID_TO_CHINESE.get(folder_names[idx], f"未知({folder_names[idx]})") for idx in top_indices]

    # --- 可视化：左图右表 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左侧显示图片
    ax1.imshow(img_raw)
    ax1.set_title(f"检测图片: {os.path.basename(image_path)}", fontsize=14)
    ax1.axis('off')

    # 右侧显示柱状图
    y_pos = np.arange(k)
    bars = ax2.barh(y_pos, top_probs, align='center', color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names, fontsize=12)
    ax2.invert_yaxis()  # 最高概率在最上方
    ax2.set_xlabel('置信度 (Probability)', fontsize=12)
    ax2.set_title(f'Top {k} 相关类别预测', fontsize=14)

    # 在柱状图上标注具体百分比
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width*100:.2f}%', 
                 va='center', ha='left', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join("docs", "top_k_prediction.png")
    os.makedirs("docs", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ 结果已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    predict_top_k(
        image_path=r"E:\GarbageClassification\test.jpg", # 传入那张西红柿
        model_path="checkpoints/garbage_classifier_best.pth",
        config_path="configs/config.yaml",
        k=5 # 列出前 5 个最相关的
    )