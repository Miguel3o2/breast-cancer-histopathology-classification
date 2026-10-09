
import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features  # 2048
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.num_features = num_features
        
        print(f"Created ResNetClassifier:")
        print(f"  Backbone: ResNet50 ({'pretrained' if pretrained else 'random init'})")
        print(f"  Features: {num_features}")
        print(f"  Classes: {num_classes}")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x, return_features=False):
        if return_features:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            features = torch.flatten(x, 1)  # (B, 2048)
            
            logits = self.backbone.fc(features)
            return logits, features
        else:
            return self.backbone(x)
    
    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze the FC layer
                param.requires_grad = False
        
        print("✓ Backbone frozen (feature extraction mode)")
        self._print_trainable_params()
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        print("✓ Backbone unfrozen (fine-tuning mode)")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == '__main__':
    print("Testing ResNetClassifier")
    print("="*60)
    
    model = ResNetClassifier(num_classes=2, pretrained=False, dropout=0.5)
    
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Output: {output.shape}")  # Should be (4, 2)
    
    logits, features = model(dummy_input, return_features=True)
    print(f"  Features: {features.shape}")  # Should be (4, 2048)
    
    print("\nTesting freeze/unfreeze:")
    model.freeze_backbone()
    
    total, trainable = count_parameters(model)
    print(f"  After freeze: {trainable:,} trainable (should be ~8K)")
    
    model.unfreeze_backbone()
    total, trainable = count_parameters(model)
    print(f"  After unfreeze: {trainable:,} trainable (should be ~25M)")
    
    print("\n✓ ResNetClassifier working correctly!")
    print("\nNext step: Create utils/metrics.py")
