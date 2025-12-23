import torch
import yaml
import os
from PIL import Image
from src.model import GarbageClassifier
from torchvision import transforms

def predict(image_path, model_path, config_path):
    # 1. 加载配置并指定编码
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 【关键】获取类别名称列表 (必须与训练时的 ImageFolder 顺序一致)
    # 假设你的训练数据在 data/train 下
    train_dir = os.path.join(os.path.dirname(config_path), "..", "data", "train")
    if os.path.exists(train_dir):
        # 字母排序是 PyTorch 的默认逻辑
        class_names = sorted(os.listdir(train_dir))
    else:
        class_names = [f"类别_{i}" for i in range(config['num_classes'])]

    # 3. 加载模型
    model = GarbageClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 图像预处理 (必须与训练时完全一致)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 确保是闪电模式的 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. 推理
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, 1)
    
    # 获取类别名称
    pred_label = class_names[pred_idx.item()]
    
    print("-" * 30)
    print(f"📷 检测图片: {os.path.basename(image_path)}")
    print(f"🎯 预测结果: {pred_label} (索引: {pred_idx.item()})")
    print(f"📈 置信度: {conf.item()*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    predict(r"E:\GarbageClassification\data\test\西红柿\b0b3836b05d5_1156.jpg", 
            "checkpoints/garbage_classifier_best.pth", 
            "configs/config.yaml")