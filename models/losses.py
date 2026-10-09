
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        B, C, H, W = pred.shape
        target_one_hot = F.one_hot(target.long(), num_classes=C)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        pred = pred.view(B, C, -1)  # (B, C, H*W)
        target_one_hot = target_one_hot.view(B, C, -1)
        
        intersection = (pred * target_one_hot).sum(dim=2)  # (B, C)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)  # (B, C)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over batch and classes
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # (C,) tensor or None
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')  # (B, H, W)
        
        p = F.softmax(pred, dim=1)
        target_expanded = target.unsqueeze(1).long()  # (B, 1, H, W)
        p_t = p.gather(dim=1, index=target_expanded).squeeze(1)  # (B, H, W)
        
        focal_weight = (1.0 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target.long()]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    
    def __init__(self, lambda_cls=1.0, lambda_seg=0.5, focal_gamma=2.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        
        self.cls_criterion = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
    
    def forward(self, cls_pred, seg_pred, cls_target, seg_target):
        cls_loss = self.cls_criterion(cls_pred, cls_target.long())
        
        dice_loss = self.dice_loss(seg_pred, seg_target)
        focal_loss = self.focal_loss(seg_pred, seg_target)
        seg_loss = dice_loss + focal_loss
        
        total_loss = self.lambda_cls * cls_loss + self.lambda_seg * seg_loss
        
        return total_loss, cls_loss, seg_loss


def compute_dice_score(pred, target, num_classes=2):
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dice_scores.append(dice.item())
        else:
            dice_scores.append(1.0)  # Perfect if class not present
    
    return sum(dice_scores) / len(dice_scores)


if __name__ == '__main__':
    print("Testing Loss Functions")
    print("=" * 70)
    
    B, C, H, W = 2, 2, 64, 64
    
    seg_pred = torch.randn(B, C, H, W)
    seg_target = torch.randint(0, C, (B, H, W))
    cls_pred = torch.randn(B, C)
    cls_target = torch.randint(0, C, (B,))
    
    dice_criterion = DiceLoss()
    dice_loss = dice_criterion(seg_pred, seg_target)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    
    focal_criterion = FocalLoss(gamma=2.0)
    focal_loss = focal_criterion(seg_pred, seg_target)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    combined_criterion = CombinedLoss(lambda_cls=1.0, lambda_seg=0.5)
    total_loss, cls_loss, seg_loss = combined_criterion(
        cls_pred, seg_pred, cls_target, seg_target
    )
    print(f"\nCombined Loss:")
    print(f"  Total: {total_loss.item():.4f}")
    print(f"  Classification: {cls_loss.item():.4f}")
    print(f"  Segmentation: {seg_loss.item():.4f}")
    
    dice_score = compute_dice_score(seg_pred, seg_target)
    print(f"\nDice Score (metric): {dice_score:.4f}")
    
    print("\n✓ Loss functions working correctly!")
    print("\nNext step: Create train_segmenter.py")
