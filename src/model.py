import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class DentalImplantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
        
    def load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        new_state_dict = {}
        model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        for k, v in model_state.items():
            if k in self.state_dict():
                if self.state_dict()[k].shape == v.shape:
                    new_state_dict[k] = v
                else:
                    print(f"Skipping parameter {k} due to shape mismatch")
        
        self.load_state_dict(new_state_dict, strict=False)