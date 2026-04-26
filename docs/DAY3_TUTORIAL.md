# DAY 3 TUTORIAL — U-Net Segmentation for Tumor Boundaries

## What You're Learning Today

By the end of Day 3, you'll understand:
1. Why segmentation helps classification
2. How U-Net architecture works (encoder-decoder with skip connections)
3. Dice loss vs Cross-Entropy for segmentation
4. Focal loss for handling class imbalance
5. How to generate synthetic masks for histopathology

---

## Part 1: Why Add Segmentation?

**So far (Days 1-2):**
- ResNet50 classifier: 88-92% accuracy
- Works well, but only gives a binary label: benign/malignant
- Doesn't tell us WHERE the tumor is

**Adding segmentation gives:**
1. **Spatial localization** — highlight tumor regions
2. **Better features** — segmentation forces the model to learn precise boundaries
3. **Interpretability** — clinicians can see what the model detected
4. **Multi-task regularization** — joint training improves both tasks

### Real-World Use Case

A pathologist examines a slide and needs to:
1. **Classify:** Is this cancer? (classification)
2. **Delineate:** Mark the tumor boundaries (segmentation)
3. **Measure:** Calculate tumor size, shape features

Your model will do all three.

---

## Part 2: U-Net Architecture

**Invented for:** Biomedical image segmentation (2015)
**Key innovation:** Skip connections that preserve spatial detail

### Architecture Overview

```
Input (3, 224, 224)
    ↓
Encoder (Downsampling)
    Conv Block 1  →  64 channels  ──┐
    MaxPool ↓                        │ Skip connection
    Conv Block 2  → 128 channels  ──┤
    MaxPool ↓                        │
    Conv Block 3  → 256 channels  ──┤
    MaxPool ↓                        │
    Conv Block 4  → 512 channels  ──┤
    MaxPool ↓                        │
                                     │
Bottleneck                           │
    Conv Block 5  → 1024 channels    │
                                     │
Decoder (Upsampling)                 │
    UpConv + Skip ← 512 channels  ←──┘
    UpConv + Skip ← 256 channels  ←──┘
    UpConv + Skip ← 128 channels  ←──┘
    UpConv + Skip ←  64 channels  ←──┘
    ↓
Output (2, 224, 224) — benign vs malignant per pixel
```

### Why Skip Connections Matter

**Without skips:**
- Encoder: downsample 224→112→56→28→14
- Decoder: upsample 14→28→56→112→224
- Problem: Lost high-res details during downsampling

**With skips:**
- Copy high-res features from encoder directly to decoder
- Decoder gets BOTH:
  - Semantic info from bottleneck (what it is)
  - Spatial info from encoder (where it is precisely)

**Result:** Sharp, accurate boundaries

---

## Part 3: Segmentation Masks for Histopathology

**Challenge:** BreakHis doesn't have pixel-level tumor masks
**Solution:** Generate synthetic masks based on the image-level label

### Synthetic Mask Generation Strategy

For **malignant images:**
1. Detect high-intensity regions (purple nuclei in H&E staining)
2. Apply morphological operations (dilation, erosion)
3. Create tumor mask = regions with abnormal nuclear density

For **benign images:**
- Mask = all zeros (no tumor)

**Why this works:**
- Malignant tissue has densely packed, irregular nuclei
- H&E staining makes nuclei purple/blue
- Thresholding + morphology captures tumor regions

**Limitation:**
- Not perfect (no ground truth)
- But good enough for multi-task learning
- Model learns "suspicious regions" even if mask is imperfect

---

## Part 4: Dice Loss

**Why not just Cross-Entropy?**

Cross-Entropy treats each pixel independently:
```python
CE = -Σ(y_true * log(y_pred))  # sum over all pixels
```

**Problem:** In histopathology, tumor regions are often SMALL
- Background: 90% of pixels
- Tumor: 10% of pixels
- Model can get 90% accuracy by predicting all background!

**Dice Loss** optimizes for overlap directly:

```
Dice = 2 × |Pred ∩ True| / (|Pred| + |True|)

Dice Loss = 1 - Dice
```

**Why it's better:**
- Doesn't care about pixel count
- Only cares about overlap
- Forces model to match tumor boundaries, not just pixel counts

### Visual Intuition

```
Ground Truth:    Prediction:      Dice Score:
    ###              ###              
   #####            #####            High (good)
    ###              ###              

Ground Truth:    Prediction:      Dice Score:
    ###              
   #####                 ###        Low (bad)
    ###                 #####       
                         ###        
```

Even if pixel accuracy is 70% (backgrounds match), Dice score is low because tumor regions don't overlap.

---

## Part 5: Focal Loss

**Problem:** Even with Dice loss, some pixels are easy (obvious background), some are hard (tumor boundaries)

**Focal Loss idea:** Focus learning on hard examples

```
FL = -α × (1 - p)^γ × log(p)
```

Where:
- `p` = predicted probability for correct class
- `γ` = focusing parameter (default 2)
- `α` = class balance weight

**How it works:**

| Prediction | Is Easy? | Weight |
|------------|----------|--------|
| p = 0.9 (confident correct) | Yes | (1-0.9)² = 0.01 (low) |
| p = 0.5 (uncertain) | No | (1-0.5)² = 0.25 (medium) |
| p = 0.1 (confident wrong) | No | (1-0.1)² = 0.81 (high) |

Model focuses 81x more on hard examples than easy ones.

### Combined Dice + Focal Loss

```python
total_loss = dice_loss + λ × focal_loss
```

- Dice: ensures good overlap
- Focal: fixes hard boundary pixels
- λ: balance (usually 0.5-1.0)

---

## Part 6: Multi-Task Learning

**Single-task (Day 2):**
```
Image → ResNet50 → Classification
```

**Multi-task (Day 3):**
```
              ┌→ Classification head → benign/malignant
Image → U-Net ┤
              └→ Segmentation head → pixel-wise mask
```

**Benefits:**
1. **Regularization:** Segmentation task prevents overfitting
2. **Better features:** Model learns spatial structure
3. **Efficiency:** One network, two outputs
4. **Clinical utility:** Both diagnosis AND localization

**Loss balancing:**
```python
total_loss = λ_cls × classification_loss + λ_seg × segmentation_loss
```

Typical: λ_cls = 1.0, λ_seg = 0.5

---

## Part 7: Evaluation Metrics for Segmentation

### Dice Score (primary)
```
Dice = 2TP / (2TP + FP + FN)
```
- Range: [0, 1], higher is better
- 0.0 = no overlap
- 1.0 = perfect overlap
- Target: >0.78 for histopathology

### IoU (Intersection over Union)
```
IoU = TP / (TP + FP + FN)
```
- Alternative to Dice
- Slightly harsher (Dice = 2×IoU / (1+IoU))
- IoU 0.7 ≈ Dice 0.82

### Hausdorff Distance
- Maximum distance between boundaries
- Measures worst-case error
- Useful for clinical applications (tumor margins)

---

## Part 8: Visualizations

**Three key visualizations:**

1. **Original + Prediction Overlay**
```
[Original Image] [Prediction Overlay] [Ground Truth]
```
Shows where model detects tumors

2. **Dice Score per Image**
```
Histogram: how many images achieve Dice > 0.8?
```
Detects failure modes

3. **Failure Cases**
```
Show images with Dice < 0.5
Analyze: too much false positive? missing tumor?
```

---

## What We're Building Today

**File 1: `data/generate_masks.py`**
- Generate synthetic segmentation masks
- Thresholding + morphology
- Save as .npy files

**File 2: `models/unet_segmenter.py`**
- U-Net architecture
- Encoder-decoder with skip connections
- Two output heads (classification + segmentation)

**File 3: `models/losses.py`**
- Dice loss implementation
- Focal loss implementation
- Combined loss

**File 4: `train_segmenter.py`**
- Multi-task training loop
- Joint optimization
- Segmentation metrics

**File 5: `notebooks/03_segmentation_vis.ipynb`**
- Visualize predictions
- Overlay masks on images
- Failure case analysis

---

## Expected Outcome

After today:
```python
from models import UNetSegmenter

model = UNetSegmenter(num_classes=2)
cls_output, seg_output = model(image)

# cls_output: (B, 2) — benign/malignant logits
# seg_output: (B, 2, H, W) — pixel-wise predictions
```

And you'll understand:
- Why Dice > Cross-Entropy for segmentation
- How skip connections preserve boundaries
- Why multi-task learning improves classification
- How to visualize and debug segmentation

---

## Time Estimate

- Generate masks: 20 min
- Write U-Net: 40 min
- Write losses: 30 min
- Training (20 epochs): 60-90 min
- Visualizations: 30 min
- **Total: ~4-5 hours**

---

Ready for the code? I'll generate all 5 files with detailed comments.
