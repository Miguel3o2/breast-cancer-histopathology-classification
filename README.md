# Breast Cancer Histopathology Classification

A PyTorch project for classifying breast histopathology patches from the BreakHis dataset as `benign` or `malignant`, with a reproducible data pipeline, baseline ResNet50 training workflow, and a FastAPI inference service.

## What This Repo Currently Includes

- BreakHis dataset download and organization scripts
- Patient-level train/validation/test splitting
- Image augmentation and PyTorch `Dataset`/`DataLoader` utilities
- ResNet50 transfer-learning baseline for binary classification
- Medical evaluation metrics such as sensitivity, specificity, PPV, NPV, F1, and ROC-AUC
- Synthetic extensions for segmentation masks and clinical features
- FastAPI prediction endpoint wired to the trained classifier checkpoint
- Docker configuration for serving the classifier API

## Current Status

This repository is best described as a strong medical-imaging prototype with a working baseline classification pipeline.

Implemented and runnable today:
- Data preparation for BreakHis
- Baseline classifier training with `train_classifier.py`
- Saved classifier checkpoint at `checkpoints/resnet50_best.pt`
- FastAPI inference path for the trained classifier

Partially scaffolded / exploratory:
- Synthetic segmentation mask generation
- Synthetic clinical feature generation
- Attention module for multimodal fusion
- README/tutorial content for later-day extensions that are not fully wired into one end-to-end training script yet

## Dataset

Source: BreakHis (Breast Cancer Histopathological Database)

Current processed split sizes in this project:
- Train: 5,630 images
- Validation: 1,130 images
- Test: 1,149 images

Labels:
- `benign`
- `malignant`

Additional metadata retained in CSV files:
- magnification
- subtype
- filename

## Saved Baseline Checkpoint

This repo currently contains a trained baseline checkpoint:
- `checkpoints/resnet50_best.pt`

Saved validation metrics from that checkpoint:
- Accuracy: `0.8478`
- Sensitivity: `0.9620`
- Specificity: `0.5524`
- PPV: `0.8476`
- NPV: `0.8488`
- F1: `0.9011`
- ROC-AUC: `0.8695`

These numbers should be treated as baseline prototype results from the current training run, not final production-grade clinical performance.

## Recommended Python Version

Use Python `3.10` to `3.12`.

The local `venv` currently present in this folder uses Python `3.14`, which is not a good default target for this project because PyTorch availability is inconsistent there. For a clean public setup, create a fresh environment on Python 3.10-3.12.

## Setup

```powershell
cd multimodal-cancer-detection
python -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install opencv-python
```

If you are running on CPU only, install the CPU build of PyTorch instead.

## Data Pipeline

1. Download and organize BreakHis:

```powershell
python data\download_dataset.py
```

2. Create patient-level splits:

```powershell
python data\preprocess.py
```

3. Smoke-test transforms and dataset loading:

```powershell
python data\transforms.py
python data\dataset.py
```

## Train the Baseline Classifier

```powershell
python train_classifier.py --batch_size 4 --num_workers 0
```

If GPU memory is limited, reduce batch size further:

```powershell
python train_classifier.py --batch_size 2 --num_workers 0
```

Expected output artifact:
- `checkpoints/resnet50_best.pt`

## Run the API

Start the server:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, check health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Send a sample prediction request using the included PowerShell helper:

```powershell
.\TEST_API.ps1
```

## Project Structure

```text
api/
  main.py
  schemas.py

data/
  download_dataset.py
  preprocess.py
  transforms.py
  dataset.py
  clinical_features.py
  generate_masks.py

models/
  resnet_classifier.py
  unet_segmenter.py
  attention.py
  losses.py

utils/
  metrics.py

train_classifier.py
TEST_API.ps1
Dockerfile
docker-compose.yml
```

## Limitations

- The full multimodal training pipeline is not yet completed in a single training script.
- Segmentation masks are synthetic, not expert-annotated.
- Clinical features are simulated for experimentation because BreakHis does not include real patient metadata.
- The current API serves classifier predictions only; segmentation and multimodal fusion are not yet exposed as production-ready endpoints.
- This project is suitable for portfolio/demo use and learning, but not for real clinical deployment.


## References

- Spanhol et al., `A Dataset for Breast Cancer Histopathological Image Classification`, IEEE TBME 2016
- Ronneberger et al., `U-Net: Convolutional Networks for Biomedical Image Segmentation`, MICCAI 2015
- Selvaraju et al., `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization`, ICCV 2017

## License

MIT
