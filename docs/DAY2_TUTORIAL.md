# DAY 2 TUTORIAL — Transfer Learning & Medical Metrics

## What You're Learning Today

By the end of Day 2, you'll understand:
1. Why transfer learning works and when to use it
2. Feature extraction vs fine-tuning strategies
3. Mixed precision training (FP16) for 2x speedup
4. Why accuracy is misleading in medical AI
5. How to interpret sensitivity, specificity, and ROC curves

---

## Part 1: Transfer Learning — Standing on Giants' Shoulders

### What is Transfer Learning?

Imagine you learned to play the piano. Now you want to learn the organ. You don't start from zero — your knowledge of music theory, rhythm, and finger coordination **transfers**.

Transfer learning in computer vision:
1. Train a CNN on **ImageNet** (1.4 million natural images: cats, dogs, cars)
2. The early layers learn **universal features**: edges, textures, shapes
3. The later layers learn **task-specific features**: cat ears, car wheels
4. When we switch to **histopathology**, edges and textures are still useful!
5. We only need to retrain the final layers for cancer vs benign

### ResNet50 Architecture

**ResNet50** has 50 layers organized into 5 stages:

```
Input (3x224x224)
    ↓
Conv1 + MaxPool  →  64 channels
    ↓
Stage 1  →  256 channels     ← learns edges, textures
Stage 2  →  512 channels     ← learns patterns
Stage 3  → 1024 channels     ← learns parts (for ImageNet: fur, wheels)
Stage 4  → 2048 channels     ← learns objects (for ImageNet: cats, cars)
    ↓
AvgPool → 2048-dim feature vector
    ↓
FC Layer → 1000 classes (ImageNet)
```

**For our task:**
- Keep Conv1 through Stage 4 (frozen or fine-tuned)
- Replace FC layer with: `Linear(2048, 2)` for benign/malignant
- Train ONLY the new FC layer initially (feature extraction)
- Then unfreeze and fine-tune all layers (fine-tuning)

### Two-Stage Training Strategy

**Stage 1: Feature Extraction (5 epochs)**
- Freeze ResNet50 backbone (Conv1 → Stage4)
- Train ONLY the new classification head
- High learning rate: 1e-3
- Fast convergence: ~85-88% accuracy

**Stage 2: Fine-Tuning (20 epochs)**
- Unfreeze entire network
- Train ALL layers end-to-end
- Low learning rate: 1e-4 (10x smaller)
- Slow refinement: ~88-92% accuracy

**Why two stages?**
If you unfreeze everything immediately with a high LR, the random weights in the new FC layer create huge gradients that **destroy** the pretrained features. Feature extraction first lets the new head converge, THEN you fine-tune carefully.

---

## Part 2: Mixed Precision Training (FP16)

### The Problem with FP32

Standard PyTorch uses **FP32** (32-bit floating point):
- Every tensor: 4 bytes per number
- A 224x224 RGB image: 224 × 224 × 3 × 4 = 602,112 bytes
- Batch of 32 images: ~19 MB just for input
- Add gradients, activations, optimizer states → **8-16 GB VRAM**

### The Solution: Mixed Precision

**Mixed Precision** uses **FP16** (16-bit) for most operations:
- 2 bytes per number instead of 4
- **2x less memory** → can double batch size
- **2x faster** on modern GPUs (Tensor Cores)
- BUT: Risk of numerical instability (values too small = underflow)

**How PyTorch solves instability:**

1. **Loss scaling:** Multiply loss by a large number (e.g., 512) before backward pass
   - Prevents tiny gradients from rounding to zero
   - Unscales before optimizer step

2. **Dynamic loss scaling:** Automatically adjusts scale factor
   - If gradients overflow (NaN/Inf): reduce scale
   - If stable for N steps: increase scale

**In practice:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Backward with scaled loss
    scaler.scale(loss).backward()
    
    # Optimizer step with unscaling
    scaler.step(optimizer)
    scaler.update()
```

**Expected speedup:**
- Training time: 2x faster
- Memory usage: 50% less
- Accuracy: identical (or better due to implicit regularization)

---

## Part 3: Why Accuracy is Misleading in Medical AI

### The Imbalance Problem

BreakHis dataset:
- Benign: 2,480 images (31%)
- Malignant: 5,429 images (69%)

**Naive baseline:**
```python
def predict_all_malignant(image):
    return "malignant"
```

Accuracy: **69%** without looking at the image!

### What Actually Matters in Cancer Detection

**Clinical perspective:**

1. **Sensitivity (Recall):** Of all actual cancer cases, how many did we catch?
   - Formula: TP / (TP + FN)
   - **Target: >95%** — missing cancer is catastrophic

2. **Specificity:** Of all benign cases, how many did we correctly identify?
   - Formula: TN / (TN + FP)
   - **Target: >90%** — false alarms waste resources but aren't deadly

3. **Precision (PPV):** Of all predicted cancers, how many are real?
   - Formula: TP / (TP + FP)
   - High precision = fewer unnecessary biopsies

**The tradeoff:**
- High sensitivity → more false alarms (lower precision)
- High specificity → might miss some cancers (lower sensitivity)
- We optimize for **high sensitivity** because missing cancer kills

### Confusion Matrix

```
                  Predicted
                Benign  Malignant
Actual  Benign    TN       FP      ← Specificity = TN/(TN+FP)
        Malignant FN       TP      ← Sensitivity = TP/(TP+FN)
```

**Best case scenario:**
- TP: high (catch most cancers)
- TN: high (correctly ID benign)
- FP: low (few false alarms)
- FN: **very low** (miss almost no cancers)

### ROC Curve

**ROC = Receiver Operating Characteristic**

- X-axis: False Positive Rate = FP / (FP + TN)
- Y-axis: True Positive Rate = TP / (TP + FN) = Sensitivity

The model outputs a probability: `P(malignant) = 0.85`

We threshold at 0.5: if P > 0.5 → predict malignant

**ROC curve shows:** What if we change the threshold?
- Threshold = 0.9 → fewer FP, but more FN (high specificity, low sensitivity)
- Threshold = 0.3 → more FP, but fewer FN (low specificity, high sensitivity)

**AUC (Area Under Curve):**
- Perfect classifier: AUC = 1.0
- Random guessing: AUC = 0.5
- **Target for medical AI: AUC > 0.95**

In practice, we pick the threshold that gives us **95% sensitivity** and report the corresponding specificity.

---

## Part 4: Learning Rate Scheduling

### Why Schedule the Learning Rate?

Fixed LR problems:
- Too high: Training diverges (loss explodes)
- Too low: Convergence is slow
- Just right initially becomes too high later

### Cosine Annealing with Warmup

**Warmup (first 5% of training):**
- Start at LR = 0
- Linear increase to target LR
- Prevents large updates with random weights

**Cosine Annealing (remaining 95%):**
- Smooth decay following cosine curve
- LR goes from max → near-zero
- Formula: `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * t / T))`

**Visualization:**
```
LR
│     Warmup      Cosine Annealing
│     ┌───        ╱───╲
│    ╱           ╱     ╲___
│   ╱           ╱          ╲___
│  ╱           ╱               ╲___
│ ╱           ╱                    ╲___
└─────────────────────────────────────── Epochs
  0    2      5                      25
```

**Why this works:**
- Early: Large updates explore loss landscape
- Middle: Medium updates refine solution
- Late: Tiny updates fine-tune details

---

## Part 5: Model Checkpointing Strategy

**What to save:**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # for mixed precision
    'best_val_acc': best_val_acc,
    'metrics': {
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_sensitivity': sensitivity,
        'val_specificity': specificity,
        'val_auc': auc_score
    }
}
```

**Two checkpoint files:**
1. `checkpoints/resnet50_best.pt` — highest validation accuracy
2. `checkpoints/resnet50_last.pt` — last epoch (for resuming training)

**Why save optimizer state?**
If training crashes at epoch 15, you can resume from epoch 15 with the same momentum and LR schedule.

---

## Part 6: Expected Training Dynamics

### Feature Extraction (Epochs 1-5)

```
Epoch 01/25  train_loss 0.4521  val_acc 0.8234  val_auc 0.8912  lr 1.00e-03
Epoch 02/25  train_loss 0.3156  val_acc 0.8523  val_auc 0.9145  lr 1.00e-03
Epoch 03/25  train_loss 0.2891  val_acc 0.8698  val_auc 0.9267  lr 1.00e-03
Epoch 04/25  train_loss 0.2734  val_acc 0.8756  val_auc 0.9312  lr 1.00e-03
Epoch 05/25  train_loss 0.2645  val_acc 0.8789  val_auc 0.9334  lr 1.00e-03
```

**What's happening:**
- New FC layer learns to map pretrained features to benign/malignant
- Fast improvement (frozen backbone is already good)
- Plateaus at ~87-88% (limit of frozen features)

### Fine-Tuning (Epochs 6-25)

```
Epoch 06/25  train_loss 0.2489  val_acc 0.8834  val_auc 0.9401  lr 1.00e-04 ← unfroze
Epoch 10/25  train_loss 0.1923  val_acc 0.8998  val_auc 0.9523  lr 8.09e-05
Epoch 15/25  train_loss 0.1645  val_acc 0.9123  val_auc 0.9612  lr 5.00e-05
Epoch 20/25  train_loss 0.1512  val_acc 0.9187  val_auc 0.9645  lr 1.91e-05
Epoch 25/25  train_loss 0.1467  val_acc 0.9201  val_auc 0.9658  lr 1.00e-06
```

**What's happening:**
- Backbone adapts to histopathology (learns tissue-specific features)
- Slow but steady improvement
- Final accuracy: 91-92%

### If Training Goes Wrong

**Loss exploding (NaN):**
- LR too high → reduce by 10x
- Gradient clipping helps: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

**Accuracy stuck at 69%:**
- Model predicting all malignant
- Check class weights are applied
- Increase weight for benign class

**Training too slow:**
- Enable mixed precision
- Increase batch size
- Reduce num_workers if CPU-bound

---

## What We're Building Today

**File 1: `models/resnet_classifier.py`**
- ResNetClassifier wrapper class
- Freeze/unfreeze methods
- Forward pass with optional feature extraction

**File 2: `utils/metrics.py`**
- Medical metrics: sensitivity, specificity, PPV, NPV
- Confusion matrix computation
- ROC curve and AUC

**File 3: `utils/lr_finder.py`**
- Learning rate range test
- Find optimal LR automatically

**File 4: `train_classifier.py`**
- Two-stage training loop
- Mixed precision
- Checkpointing
- Metrics logging

**File 5: `notebooks/02_training_curves.ipynb`**
- Visualize loss/accuracy over time
- ROC curves
- Confusion matrix heatmap

---

## Expected Outcome

After today:
```python
from models import ResNetClassifier
import torch

model = ResNetClassifier(num_classes=2, pretrained=True)
checkpoint = torch.load('checkpoints/resnet50_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference on new image
image = transform(PIL.Image.open('tissue.png'))
output = model(image.unsqueeze(0))
prob = torch.softmax(output, dim=1)[0, 1].item()
print(f"Probability of malignancy: {prob:.2%}")
```

And you'll understand:
- Why we got 92% instead of 88% (fine-tuning worked)
- Why sensitivity is 95% but specificity is 89% (we optimized for catching cancer)
- Why batch size 64 is 2x faster than 32 (mixed precision enabled)

---

## Time Estimate

- Write model code: 30 min
- Write training loop: 45 min
- Write metrics: 20 min
- First training run (25 epochs): 60-90 min (depending on GPU)
- Analysis notebook: 30 min
- **Total: ~4-5 hours**

---

Ready for the code? I'll generate all 5 files with detailed comments explaining every design decision.
