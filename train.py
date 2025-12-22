import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from src.dataset import get_dataloaders
from src.model import GarbageClassifier
from src.trainer import Trainer

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "configs", "config.yaml")

def main():
    # 1. 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 数据与指标
    train_loader, val_loader, class_names = get_dataloaders(config)
    config['num_classes'] = len(class_names)
    print(f"检测到数据集包含 {config['num_classes']} 个类别。")
    accuracy_metric = Accuracy(task="multiclass", num_classes=config['num_classes']).to(device)
    
    # 3. 模型初始化
    model = GarbageClassifier(num_classes=config['num_classes'], freeze_layers=config['freeze_layers']).to(device)
    
    # 4. 优化器分层设置
    optimizer = optim.AdamW([
    {"params": [p for i, p in enumerate(model.base_model.features.parameters()) if i < config['freeze_layers']], "lr": float(config['learning_rate_freeze'])},
    {"params": [p for i, p in enumerate(model.base_model.features.parameters()) if i >= config['freeze_layers']], "lr": float(config['learning_rate_base'])},
    {"params": model.base_model.classifier.parameters(), "lr": float(config['learning_rate_base'])}
], weight_decay=float(config['weight_decay']))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(config['log_dir'])
    
    # 5. 训练循环
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, accuracy_metric)
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = trainer.train_one_epoch(epoch)
        val_loss, val_acc = trainer.validate()
        scheduler.step()
        
        # 记录日志
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/garbage_classifier_best.pth")
            print(f"New Best Model Saved with Acc: {val_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()