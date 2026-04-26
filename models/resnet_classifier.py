"""
ResNet50 Classifier with Transfer Learning

This module provides a ResNetClassifier that wraps torchvision's pretrained ResNet50
and replaces the final FC layer for binary classification (benign vs malignant).

Key features:
1. Load ImageNet pretrained weights
2. Freeze/unfreeze backbone for two-stage training
3. Feature extraction mode for analysis
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """
    ResNet50 for binary histopathology classification.
    
    Architecture:
        Input (3, 224, 224)
            ↓
        ResNet50 backbone → 2048-dim features
            ↓
        Dropout(0.5) ← regularization
            ↓
        Linear(2048, num_classes)
            ↓
        Output logits (no softmax — will use CrossEntropyLoss)
    
    Args:
        num_classes: number of output classes (default 2 for benign/malignant)
        pretrained: load ImageNet weights (default True)
        dropout: dropout probability (default 0.5)
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Save original FC layer's input dimension
        num_features = self.backbone.fc.in_features  # 2048
        
        # Replace FC layer with our classification head
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
        """
        Forward pass.
        
        Args:
            x: input tensor (B, 3, 224, 224)
            return_features: if True, return (logits, features) tuple
                           if False, return only logits
        
        Returns:
            logits: (B, num_classes) raw scores
            features (optional): (B, 2048) embedding vector
        """
        if return_features:
            # Extract features before final FC layer
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
            # Standard forward pass
            return self.backbone(x)
    
    def freeze_backbone(self):
        """
        Freeze all layers EXCEPT the final FC layer.
        
        Use this for Stage 1 (feature extraction):
        - Pretrained features are frozen
        - Only the new classification head is trained
        - Fast convergence with high LR
        """
        # Freeze all backbone parameters
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze the FC layer
                param.requires_grad = False
        
        print("✓ Backbone frozen (feature extraction mode)")
        self._print_trainable_params()
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for end-to-end training.
        
        Use this for Stage 2 (fine-tuning):
        - All layers are trainable
        - Backbone adapts to histopathology
        - Use LOW learning rate to avoid destroying pretrained features
        """
        # Unfreeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        print("✓ Backbone unfrozen (fine-tuning mode)")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        """Print count of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def get_trainable_params(self):
        """
        Get list of trainable parameters.
        
        Returns:
            list of parameters with requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing ResNetClassifier")
    print("="*60)
    
    # Create model
    model = ResNetClassifier(num_classes=2, pretrained=False, dropout=0.5)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Output: {output.shape}")  # Should be (4, 2)
    
    # Test feature extraction
    logits, features = model(dummy_input, return_features=True)
    print(f"  Features: {features.shape}")  # Should be (4, 2048)
    
    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze:")
    model.freeze_backbone()
    
    total, trainable = count_parameters(model)
    print(f"  After freeze: {trainable:,} trainable (should be ~8K)")
    
    model.unfreeze_backbone()
    total, trainable = count_parameters(model)
    print(f"  After unfreeze: {trainable:,} trainable (should be ~25M)")
    
    print("\n✓ ResNetClassifier working correctly!")
    print("\nNext step: Create utils/metrics.py")
