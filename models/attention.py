"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. Feature Attention - learns to weight image vs clinical features
2. Channel Attention - weights different feature channels
3. Spatial Attention - weights different spatial regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    """
    Attention mechanism for fusing image and clinical features.
    
    Learns dynamic weights:
    - When to trust image features (clear histology)
    - When to trust clinical features (ambiguous image)
    
    Input:
        img_features: (B, D_img) - e.g., (B, 2048)
        clin_features: (B, D_clin) - e.g., (B, 128)
    
    Output:
        fused_features: (B, D_img) - weighted combination
        attention_weights: (B, 2) - [weight_img, weight_clin]
    """
    
    def __init__(self, img_dim=2048, clin_dim=128, hidden_dim=256):
        super().__init__()
        
        # Project clinical features to match image dimension
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, img_dim)
        )
        
        # Attention network
        # Takes concatenated features, outputs attention scores
        self.attention = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # 2 scores: [img, clin]
        )
    
    def forward(self, img_features, clin_features):
        """
        Args:
            img_features: (B, D_img)
            clin_features: (B, D_clin)
        
        Returns:
            fused: (B, D_img)
            weights: (B, 2)
        """
        # Project clinical to same dimension as image
        clin_proj = self.clin_proj(clin_features)  # (B, D_img)
        
        # Concatenate for attention computation
        combined = torch.cat([img_features, clin_proj], dim=1)  # (B, D_img*2)
        
        # Compute attention scores
        scores = self.attention(combined)  # (B, 2)
        
        # Softmax to get weights that sum to 1
        weights = F.softmax(scores, dim=1)  # (B, 2)
        
        # Weighted fusion
        weight_img = weights[:, 0].unsqueeze(1)  # (B, 1)
        weight_clin = weights[:, 1].unsqueeze(1)  # (B, 1)
        
        fused = weight_img * img_features + weight_clin * clin_proj  # (B, D_img)
        
        return fused, weights


class ChannelAttention(nn.Module):
    """
    Channel-wise attention (Squeeze-and-Excitation).
    
    Learns which feature channels are most important.
    Useful for image features where different channels capture different patterns.
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            x_weighted: (B, C, H, W) with channel attention applied
        """
        B, C, _, _ = x.shape
        
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(B, C)  # (B, C)
        
        # Excitation: learn channel weights
        y = self.fc(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        
        # Scale features by learned weights
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism.
    
    Learns which spatial regions to focus on.
    Useful for highlighting tumor regions.
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            x_weighted: (B, C, H, W) with spatial attention applied
        """
        # Pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Learn spatial weights
        y = self.conv(y)  # (B, 1, H, W)
        weights = self.sigmoid(y)  # (B, 1, H, W)
        
        # Apply weights
        return x * weights


class MultimodalFusionModule(nn.Module):
    """
    Complete fusion module combining all attention mechanisms.
    
    Pipeline:
    1. Channel attention on image features
    2. Spatial attention on image features
    3. Feature-level fusion of image + clinical
    """
    
    def __init__(self, img_channels=1024, img_dim=2048, clin_dim=128):
        super().__init__()
        
        self.channel_attn = ChannelAttention(img_channels)
        self.spatial_attn = SpatialAttention()
        self.feature_attn = FeatureAttention(img_dim, clin_dim)
    
    def forward(self, img_feature_map, img_feature_vec, clin_features):
        """
        Args:
            img_feature_map: (B, C, H, W) - spatial features from U-Net
            img_feature_vec: (B, D_img) - global features for classification
            clin_features: (B, D_clin) - clinical feature vector
        
        Returns:
            fused_features: (B, D_img) - for classification
            attention_weights: (B, 2) - [img_weight, clin_weight]
            attended_map: (B, C, H, W) - for visualization
        """
        # Apply channel and spatial attention to feature map
        attended_map = self.channel_attn(img_feature_map)
        attended_map = self.spatial_attn(attended_map)
        
        # Fuse image and clinical features
        fused_features, attn_weights = self.feature_attn(
            img_feature_vec,
            clin_features
        )
        
        return fused_features, attn_weights, attended_map


if __name__ == '__main__':
    print("Testing Attention Mechanisms")
    print("=" * 70)
    
    B = 4  # batch size
    
    # Test Feature Attention
    print("\n1. Feature Attention:")
    img_feat = torch.randn(B, 2048)
    clin_feat = torch.randn(B, 128)
    
    attn = FeatureAttention(img_dim=2048, clin_dim=128)
    fused, weights = attn(img_feat, clin_feat)
    
    print(f"   Input: img {img_feat.shape}, clin {clin_feat.shape}")
    print(f"   Output: fused {fused.shape}, weights {weights.shape}")
    print(f"   Sample weights: {weights[0].detach().numpy()}")
    print(f"   Weights sum to 1: {weights[0].sum():.3f}")
    
    # Test Channel Attention
    print("\n2. Channel Attention:")
    x = torch.randn(B, 512, 14, 14)
    ch_attn = ChannelAttention(512)
    x_attended = ch_attn(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_attended.shape}")
    
    # Test Spatial Attention
    print("\n3. Spatial Attention:")
    x = torch.randn(B, 512, 14, 14)
    sp_attn = SpatialAttention()
    x_attended = sp_attn(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_attended.shape}")
    
    # Test Complete Fusion Module
    print("\n4. Complete Fusion Module:")
    img_map = torch.randn(B, 1024, 14, 14)
    img_vec = torch.randn(B, 2048)
    clin_vec = torch.randn(B, 128)
    
    fusion = MultimodalFusionModule(img_channels=1024, img_dim=2048, clin_dim=128)
    fused, weights, attended_map = fusion(img_map, img_vec, clin_vec)
    
    print(f"   Fused features: {fused.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"   Attended map: {attended_map.shape}")
    
    print("\n✓ All attention mechanisms working correctly!")
    print("\nNext step: Create models/multimodal_net.py")
