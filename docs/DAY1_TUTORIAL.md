# DAY 1 TUTORIAL — Understanding Histopathology & Data Pipeline

## What You're Learning Today

By the end of Day 1, you'll understand:
1. What histopathology images are and why they're different from regular photos
2. How magnification levels affect what the model can see
3. Why medical image augmentation needs different rules than natural images
4. How to build a production-grade PyTorch Dataset

---

## Part 1: Understanding Histopathology Images

### What is Histopathology?

When a doctor suspects cancer, they take a **biopsy** — a tiny tissue sample. That tissue is:
1. Preserved in chemicals (fixation)
2. Embedded in wax
3. Sliced into 4-micrometer-thin sections (thinner than a human hair)
4. Stained with dyes (usually H&E — hematoxylin and eosin)
5. Examined under a microscope

The pathologist looks at the **cellular structure** — cell shapes, nucleus sizes, how cells cluster — to determine if it's cancerous.

### The BreakHis Dataset

- **7,909 images** of breast tissue
- **2 main classes:** Benign (non-cancerous) vs Malignant (cancerous)
- **8 sub-classes:**
  - Benign: adenosis, fibroadenoma, tubular adenoma, phyllodes tumor
  - Malignant: ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma
- **4 magnification levels:** 40x, 100x, 200x, 400x

### Why Magnification Matters

**40x (low magnification):**
- Shows tissue architecture — how cells organize into structures
- Good for detecting abnormal growth patterns
- Like looking at a city from an airplane

**100x (medium-low):**
- Shows clusters of cells
- Balance between structure and detail
- Like looking at a neighborhood

**200x (medium-high):**
- Individual cells become visible
- Nuclear shapes start to matter
- Like looking at individual buildings

**400x (high magnification):**
- Cell nuclei are clearly visible
- Can see abnormal nuclear shapes (pleomorphism)
- Mitotic figures (cells dividing) are countable
- Like looking at individual rooms

**Key insight:** Different magnifications reveal different features. A tumor might be obvious at 40x (abnormal architecture) but ambiguous at 400x (individual cells look normal). That's why pathologists scan at multiple levels.

---

## Part 2: Medical Image Augmentation — What's Different?

In natural image classification (cats vs dogs), you can:
- Flip horizontally ✓
- Flip vertically ✓
- Rotate 90° ✓
- Change brightness ✓
- Add noise ✓

In **histopathology**, the rules change:

### Valid Augmentations:
1. **Horizontal flip** ✓ — tissue orientation is arbitrary
2. **Vertical flip** ✓ — same reason
3. **90° rotations** ✓ — slides can be rotated under the microscope
4. **Slight color jitter** ✓ — staining intensity varies between labs
5. **Gaussian noise (subtle)** ✓ — simulates scanner noise

### Invalid Augmentations:
1. **Arbitrary rotations (45°, 60°)** ✗ — creates interpolation artifacts that look like tissue damage
2. **Extreme color shifts** ✗ — purple nuclei turning green would confuse the model about cell types
3. **Perspective transforms** ✗ — tissue slides are flat, not 3D scenes
4. **Cutout/random erasing** ✗ — removing tissue regions destroys diagnostic information
5. **Mixup/CutMix** ✗ — blending two tissue samples creates impossible histology

**Why this matters:** If you train with invalid augmentations, the model learns artifacts instead of biology. It might achieve 95% validation accuracy but fail completely on real clinical data from a different scanner.

---

## Part 3: Class Imbalance in Medical Data

BreakHis has **unequal class distribution:**
- Malignant: ~5,429 images (69%)
- Benign: ~2,480 images (31%)

This is **realistic** — cancer biopsies are more common because doctors only biopsy when they're already suspicious.

### Handling Imbalance:

**Stratified sampling:**
- Ensures train/val/test splits have the same class ratio
- Prevents val set from being 90% malignant by chance

**Class weights:**
- Give higher loss penalty for misclassifying the minority class
- Formula: weight = total_samples / (num_classes × class_count)

**Metrics that matter:**
- Accuracy is **misleading** (always predicting malignant = 69% accuracy)
- Use: Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity

---

## Part 4: File Organization Strategy

```
data/
├── raw/                    ← downloaded BreakHis dataset (don't modify)
├── processed/              ← train/val/test splits (generated)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── dataset.py              ← PyTorch Dataset class
├── transforms.py           ← augmentation pipelines
└── preprocess.py           ← create train/val/test splits
```

**Why this structure:**
- Raw data is immutable (can always regenerate processed data)
- CSV files make debugging easy (can inspect splits in Excel)
- Separation of concerns (transforms vs dataset vs preprocessing)

---

## Part 5: PyTorch Dataset Design Pattern

A PyTorch Dataset needs 3 methods:

```python
class BreakHisDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # Load the CSV with image paths and labels
        # Store transform pipeline
        
    def __len__(self):
        # Return number of samples
        
    def __getitem__(self, idx):
        # Load image at index idx
        # Apply transforms
        # Return (image_tensor, label)
```

**Key design choices:**

1. **Lazy loading:** Load images in `__getitem__`, not `__init__`
   - Why: Loading 7,909 images into RAM at once = 10+ GB memory
   - With lazy loading: only load what's needed per batch

2. **Transform composition:** Use `torchvision.transforms.Compose`
   - Why: Clean separation between train augmentation vs val/test preprocessing

3. **Error handling:** Wrap image loading in try/except
   - Why: Corrupted images shouldn't crash training mid-epoch

---

## What We're Building Today

**File 1: `data/preprocess.py`**
- Scans the BreakHis folder structure
- Creates train/val/test CSV files with stratification
- 70% train, 15% val, 15% test

**File 2: `data/transforms.py`**
- Train augmentation pipeline (flips, rotations, color jitter)
- Val/test preprocessing pipeline (just resize + normalize)

**File 3: `data/dataset.py`**
- BreakHisDataset class
- Handles loading images from CSV
- Applies transforms

**File 4: `notebooks/01_eda.ipynb`**
- Visualize sample images
- Check class distribution
- Verify augmentations are working

---

## Expected Outcome

After today, you'll be able to:
```python
from data.dataset import get_dataloaders

train_dl, val_dl, test_dl = get_dataloaders(
    batch_size=32,
    num_workers=4
)

# Get a batch
images, labels = next(iter(train_dl))
print(images.shape)   # torch.Size([32, 3, 224, 224])
print(labels.shape)   # torch.Size([32])
```

And you'll understand:
- Why 224x224 is the standard size (pretrained ResNet expects it)
- Why we normalize with ImageNet mean/std (transfer learning requirement)
- Why stratification matters (prevents biased splits)

---

## Time Estimate

- Download dataset: 10 min
- Write preprocessing script: 20 min
- Write transforms: 15 min
- Write dataset class: 30 min
- EDA notebook: 30 min
- **Total: ~2 hours**

---

Ready for the code? I'll generate all 4 files in order, with detailed comments explaining every line.
