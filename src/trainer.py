import torch
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, writer, accuracy_metric):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.accuracy_metric = accuracy_metric

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_acc = 0, 0
        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch} Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_acc += self.accuracy_metric(outputs, labels).item() * images.size(0)
            
        return total_loss / len(self.train_loader.dataset), total_acc / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss, total_acc = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_acc += self.accuracy_metric(outputs, labels).item() * images.size(0)
        return total_loss / len(self.val_loader.dataset), total_acc / len(self.val_loader.dataset)