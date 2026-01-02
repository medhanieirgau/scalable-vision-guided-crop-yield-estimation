import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def build_resnet50():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 1)
    return model