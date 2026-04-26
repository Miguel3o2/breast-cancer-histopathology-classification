"""
U-Net Segmenter with Dual Outputs

Architecture:
- Encoder-decoder with skip connections
- Two output heads:
  1. Classification: global pooling → FC → 2 classes (benign/malignant)
  2. Segmentation: pixel-wise → 2 channels (background/tumor)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetSegmenter(nn.Module):
    """
    U-Net with dual outputs for classification and segmentation.
    
    Args:
        in_channels: input channels (3 for RGB)
        num_classes: output classes for classification (2 for benign/malignant)
        seg_classes: output classes for segmentation (2 for background/tumor)
    """
    
    def __init__(self, in_channels=3, num_classes=2, seg_classes=2):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output heads
        # 1. Segmentation output (pixel-wise)
        self.seg_out = nn.Conv2d(64, seg_classes, 1)
        
        # 2. Classification output (image-level)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self.num_classes = num_classes
        self.seg_classes = seg_classes
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: input (B, 3, H, W)
            return_features: if True, return intermediate features
        
        Returns:
            cls_logits: (B, num_classes) classification logits
            seg_logits: (B, seg_classes, H, W) segmentation logits
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 64, H, W)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)  # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)  # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)  # (B, 512, H/8, W/8)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)  # (B, 1024, H/16, W/16)
        
        # Classification from bottleneck features
        cls_features = self.global_pool(b).view(b.size(0), -1)  # (B, 1024)
        cls_logits = self.cls_fc(cls_features)  # (B, num_classes)
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Segmentation output
        seg_logits = self.seg_out(d1)  # (B, seg_classes, H, W)
        
        if return_features:
            return cls_logits, seg_logits, cls_features
        
        return cls_logits, seg_logits


if __name__ == '__main__':
    print("Testing UNetSegmenter")
    print("=" * 70)
    
    model = UNetSegmenter(in_channels=3, num_classes=2, seg_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    cls_out, seg_out = model(dummy_input)
    
    print(f"\nForward pass test:")
    print(f"  Input: {dummy_input.shape}")
    print(f"  Classification output: {cls_out.shape}")  # (2, 2)
    print(f"  Segmentation output: {seg_out.shape}")  # (2, 2, 224, 224)
    
    print("\n✓ UNetSegmenter working correctly!")
    print("\nNext step: Create models/losses.py")
