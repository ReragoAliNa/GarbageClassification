import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=config['image_size'], scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试集仅预处理
    val_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=config['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config['val_dir'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, train_dataset.classes