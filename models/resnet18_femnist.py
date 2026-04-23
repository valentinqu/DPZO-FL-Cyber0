import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ZO_ResNet18_FEMNIST(nn.Module):
    def __init__(self):
        super(ZO_ResNet18_FEMNIST, self).__init__()
        
        # 1. 加载ImageNet上预训练的 ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. 冻结骨干网络的所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3.将原来的 1000 分类改成 FEMNIST 的 62 分类
        num_features = self.backbone.fc.in_features  # 获取输入维度 (对于 ResNet18 是 512)
        self.backbone.fc = nn.Linear(num_features, 62)

    def forward(self, x):
        # x 的原始尺寸: [batch_size, 1, 28, 28]
        
        # 转换 1: 复制通道 (1 -> 3)
        x = x.repeat(1, 3, 1, 1) 
        
        # 转换 2: 放大图片 (28x28 -> 224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.backbone(x)