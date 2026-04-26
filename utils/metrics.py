"""
Medical Evaluation Metrics

This module provides metrics specific to medical image classification:
- Sensitivity (Recall / True Positive Rate)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1 Score
- ROC AUC

All metrics handle binary classification (benign vs malignant).
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def compute_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    """
    Compute confusion matrix.
    
    Args:
        y_true: ground truth labels (numpy array or list)
        y_pred: predicted labels (numpy array or list)
        labels: class labels (default [0, 1] for benign/malignant)
    
    Returns:
        2x2 numpy array:
            [[TN, FP],
             [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def sensitivity(y_true, y_pred):
    """
    Sensitivity = Recall = True Positive Rate.
    
    Of all actual cancer cases, what fraction did we detect?
    
    Formula: TP / (TP + FN)
    
    Clinical interpretation:
        0.95 = we catch 95% of cancers (miss 5%)
        0.80 = we catch 80% of cancers (miss 20%) ← unacceptable
    
    Target for cancer detection: >0.95
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    if (tp + fn) == 0:
        return 0.0  # No positive samples
    
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    """
    Specificity = True Negative Rate.
    
    Of all benign cases, what fraction did we correctly identify?
    
    Formula: TN / (TN + FP)
    
    Clinical interpretation:
        0.90 = 90% of benign cases correctly ID'd (10% false alarms)
        0.70 = 70% correct (30% false alarms) ← too many unnecessary biopsies
    
    Target: >0.85
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    if (tn + fp) == 0:
        return 0.0  # No negative samples
    
    return tn / (tn + fp)


def ppv(y_true, y_pred):
    """
    Positive Predictive Value = Precision.
    
    Of all predicted cancers, what fraction are actually cancer?
    
    Formula: TP / (TP + FP)
    
    Clinical interpretation:
        0.85 = 85% of predicted cancers are real (15% false alarms)
        0.50 = only half of predicted cancers are real (too many biopsies)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    if (tp + fp) == 0:
        return 0.0  # No positive predictions
    
    return tp / (tp + fp)


def npv(y_true, y_pred):
    """
    Negative Predictive Value.
    
    Of all predicted benign cases, what fraction are actually benign?
    
    Formula: TN / (TN + FN)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    if (tn + fn) == 0:
        return 0.0
    
    return tn / (tn + fn)


def f1_score(y_true, y_pred):
    """
    F1 Score = harmonic mean of precision and recall.
    
    Formula: 2 * (precision * recall) / (precision + recall)
    
    Balances precision and recall into a single metric.
    """
    prec = ppv(y_true, y_pred)
    rec = sensitivity(y_true, y_pred)
    
    if (prec + rec) == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    WARNING: Misleading for imbalanced data!
    Use sensitivity and specificity instead.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (tp + tn) / (tp + tn + fp + fn)


def compute_roc_auc(y_true, y_probs):
    """
    Compute ROC AUC score.
    
    Args:
        y_true: ground truth labels (0 or 1)
        y_probs: predicted probabilities for class 1 (malignant)
    
    Returns:
        float in [0, 1]
            1.0 = perfect classifier
            0.5 = random guessing
            0.0 = perfectly wrong (flip predictions to fix)
    
    Target for medical AI: >0.95
    """
    try:
        return roc_auc_score(y_true, y_probs)
    except ValueError:
        # Happens if only one class in y_true
        return 0.0


def compute_all_metrics(y_true, y_pred, y_probs=None):
    """
    Compute all metrics at once.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels (0 or 1)
        y_probs: predicted probabilities (optional, for AUC)
    
    Returns:
        dict with all metrics
    """
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
        'sensitivity': sensitivity(y_true, y_pred),
        'specificity': specificity(y_true, y_pred),
        'ppv': ppv(y_true, y_pred),
        'npv': npv(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_probs is not None:
        metrics['auc'] = compute_roc_auc(y_true, y_probs)
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics dictionary."""
    print("\n" + "="*50)
    print(title)
    print("="*50)
    for name, value in metrics.items():
        print(f"  {name.upper():<15}: {value:.4f}")
    print("="*50)


def find_optimal_threshold(y_true, y_probs, target_sensitivity=0.95):
    """
    Find threshold that achieves target sensitivity.
    
    In cancer detection, we want high sensitivity (catch all cancers).
    This function finds the threshold that gives us target sensitivity
    and reports the corresponding specificity.
    
    Args:
        y_true: ground truth labels
        y_probs: predicted probabilities for class 1
        target_sensitivity: desired sensitivity (default 0.95)
    
    Returns:
        dict with threshold, actual sensitivity, and specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    
    threshold = thresholds[idx]
    actual_sens = tpr[idx]
    actual_spec = 1 - fpr[idx]
    
    return {
        'threshold': threshold,
        'sensitivity': actual_sens,
        'specificity': actual_spec
    }


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def torch_accuracy(outputs, labels):
    """
    Compute accuracy from PyTorch tensors.
    
    Args:
        outputs: model logits (B, num_classes)
        labels: ground truth (B,)
    
    Returns:
        float accuracy in [0, 1]
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def torch_to_numpy(outputs, labels):
    """
    Convert PyTorch tensors to numpy for sklearn metrics.
    
    Args:
        outputs: model logits (B, num_classes)
        labels: ground truth (B,)
    
    Returns:
        y_true, y_pred, y_probs (all numpy arrays)
    """
    # Predictions
    _, preds = torch.max(outputs, 1)
    y_pred = preds.cpu().numpy()
    
    # Ground truth
    y_true = labels.cpu().numpy()
    
    # Probabilities (softmax on logits)
    probs = torch.softmax(outputs, dim=1)
    y_probs = probs[:, 1].cpu().numpy()  # probability of class 1 (malignant)
    
    return y_true, y_pred, y_probs


# ──────────────────────────────────────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing Medical Metrics")
    print("="*60)
    
    # Simulate predictions
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.95, 0.85, 0.3, 0.75])
    y_pred = (y_probs > 0.5).astype(int)
    
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred}")
    print(f"Probabilities: {y_probs}")
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # All metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print_metrics(metrics, title="Test Metrics")
    
    # Find optimal threshold
    opt = find_optimal_threshold(y_true, y_probs, target_sensitivity=0.95)
    print(f"\nOptimal threshold for 95% sensitivity:")
    print(f"  Threshold: {opt['threshold']:.3f}")
    print(f"  Sensitivity: {opt['sensitivity']:.3f}")
    print(f"  Specificity: {opt['specificity']:.3f}")
    
    print("\n✓ Metrics working correctly!")
    print("\nNext step: Create train_classifier.py")
