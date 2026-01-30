import torch
import torch.nn as nn
from torchvision import models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ChestXrayModel, self).__init__()
        
        # Load a pre-trained DenseNet121
        # Note: weights='DEFAULT' is the modern way, but pretrained=True is legacy but still often works or aliases.
        # Use weights argument if using newer torchvision.
        try:
            self.model = models.densenet121(weights='DEFAULT' if pretrained else None)
        except:
            # Fallback for older torchvision versions
            self.model = models.densenet121(pretrained=pretrained)
        
        # Freezing layers (Optional: often we fine-tune everything or freeze early layers)
        # For now, we won't freeze, but this can be added.
        
        # Replace the classifier
        # DenseNet121 classifier is a Linear layer with 1024 input features
        num_features = self.model.classifier.in_features
        
        # We output logits (no sigmoid here) because we'll use BCEWithLogitsLoss
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
