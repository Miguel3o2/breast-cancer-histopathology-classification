"""
Train ResNet50 Classifier — Two-Stage Training

Stage 1 (epochs 1-5):  Feature extraction (frozen backbone, high LR)
Stage 2 (epochs 6-25): Fine-tuning (unfrozen, low LR)

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 30 --batch_size 64
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data import get_dataloaders
from models.resnet_classifier import ResNetClassifier
from utils.metrics import compute_all_metrics, print_metrics, torch_to_numpy


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='  Train', leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    
    return running_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    
    all_labels, all_preds, all_probs = [], [], []
    
    pbar = tqdm(dataloader, desc='  Val  ', leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        y_true, y_pred, y_probs = torch_to_numpy(outputs, labels)
        
        all_labels.append(y_true)
        all_preds.append(y_pred)
        all_probs.append(y_probs)
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    return compute_all_metrics(all_labels, all_preds, all_probs)


def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data
    train_dl, val_dl, test_dl, class_weights = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224
    )
    
    # Model
    model = ResNetClassifier(num_classes=2, pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = GradScaler()
    
    history = {'train_loss': [], 'val_metrics': []}
    best_val_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"\n{'='*60}\nTraining: {args.epochs} epochs\n{'='*60}\n")
    
    # STAGE 1: Feature Extraction
    freeze_until = args.freeze_epochs
    if freeze_until > 0:
        print(f"STAGE 1: Feature Extraction (epochs 1-{freeze_until})")
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.lr_stage1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_until, eta_min=args.lr_stage1/100)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        
        # Switch to STAGE 2
        if epoch == freeze_until + 1:
            print(f"\nSTAGE 2: Fine-Tuning (epochs {freeze_until+1}-{args.epochs})")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_stage2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - freeze_until), eta_min=1e-6)
        
        # Train & validate
        train_loss = train_epoch(model, train_dl, criterion, optimizer, scaler, device)
        val_metrics = validate_epoch(model, val_dl, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss {train_loss:.4f}  "
              f"acc {val_metrics['accuracy']:.4f}  "
              f"sens {val_metrics['sensitivity']:.4f}  "
              f"spec {val_metrics['specificity']:.4f}  "
              f"auc {val_metrics['auc']:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}", end='')
        
        # Save best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_metrics': val_metrics
            }, 'checkpoints/resnet50_best.pt')
            print("  ← best", end='')
        print()
    
    # Final test evaluation
    print(f"\n{'='*60}\nFinal Test Set Evaluation\n{'='*60}")
    checkpoint = torch.load('checkpoints/resnet50_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate_epoch(model, test_dl, device)
    print_metrics(test_metrics, title="Test Set Results")
    
    # Save history
    np.save('training_history_resnet50.npy', history)
    print(f"\n✓ Training complete! Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--freeze_epochs', default=5, type=int, help='Epochs for Stage 1 (feature extraction)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr_stage1', default=1e-3, type=float, help='LR for Stage 1 (frozen backbone)')
    parser.add_argument('--lr_stage2', default=1e-4, type=float, help='LR for Stage 2 (fine-tuning)')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_workers', default=2, type=int)
    args = parser.parse_args()
    
    train(args)
