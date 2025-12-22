import torch
import yaml
from PIL import Image
from src.model import GarbageClassifier
from src.dataset import transforms

def predict(image_path, model_path, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = GarbageClassifier(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, 1)
        
    print(f"Prediction: {pred_idx.item()}, Confidence: {conf.item()*100:.2f}%")

if __name__ == "__main__":
    predict("test.jpg", "checkpoints/garbage_classifier_best.pth", "configs/config.yaml")