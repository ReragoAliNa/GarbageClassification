import torch.nn as nn
from torchvision import models

class GarbageClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_layers=100):
        super(GarbageClassifier, self).__init__()
        # 1. 加载预训练模型
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        
        # 2. 冻结层逻辑
        for i, (name, param) in enumerate(self.base_model.features.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # 3. 替换分类器 (注意：这里去掉了 AdaptiveAvgPool2d 和 Flatten)
        # 因为 MobileNetV2 的内部 forward 已经帮我们做好了池化和展平
        self.base_model.classifier = nn.Sequential(
            nn.Linear(1280, 512),       # 直接对接 1280 维的展平向量
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 直接调用 base_model 即可，它内部会按顺序跑 features -> pooling -> classifier
        return self.base_model(x)