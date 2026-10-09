
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    
    def __init__(self, img_dim=2048, clin_dim=128, hidden_dim=256):
        super().__init__()
        
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, img_dim)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # 2 scores: [img, clin]
        )
    
    def forward(self, img_features, clin_features):
        clin_proj = self.clin_proj(clin_features)  # (B, D_img)
        
        combined = torch.cat([img_features, clin_proj], dim=1)  # (B, D_img*2)
        
        scores = self.attention(combined)  # (B, 2)
        
        weights = F.softmax(scores, dim=1)  # (B, 2)
        
        weight_img = weights[:, 0].unsqueeze(1)  # (B, 1)
        weight_clin = weights[:, 1].unsqueeze(1)  # (B, 1)
        
        fused = weight_img * img_features + weight_clin * clin_proj  # (B, D_img)
        
        return fused, weights


class ChannelAttention(nn.Module):
    
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
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)  # (B, C)
        y = self.fc(y).view(B, C, 1, 1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        y = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        y = self.conv(y)  # (B, 1, H, W)
        weights = self.sigmoid(y)  # (B, 1, H, W)
        
        return x * weights


class MultimodalFusionModule(nn.Module):
    
    def __init__(self, img_channels=1024, img_dim=2048, clin_dim=128):
        super().__init__()
        
        self.channel_attn = ChannelAttention(img_channels)
        self.spatial_attn = SpatialAttention()
        self.feature_attn = FeatureAttention(img_dim, clin_dim)
    
    def forward(self, img_feature_map, img_feature_vec, clin_features):
        attended_map = self.channel_attn(img_feature_map)
        attended_map = self.spatial_attn(attended_map)
        
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
    
    print("\n2. Channel Attention:")
    x = torch.randn(B, 512, 14, 14)
    ch_attn = ChannelAttention(512)
    x_attended = ch_attn(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_attended.shape}")
    
    print("\n3. Spatial Attention:")
    x = torch.randn(B, 512, 14, 14)
    sp_attn = SpatialAttention()
    x_attended = sp_attn(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {x_attended.shape}")
    
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
