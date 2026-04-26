# Portfolio And CV Copy

## GitHub Repository Description

Breast cancer histopathology classification project using PyTorch, ResNet50 transfer learning, FastAPI, and the BreakHis dataset. Includes patient-level data splitting, augmentation, baseline training, evaluation metrics, and a prototype inference API.

## Portfolio Project Summary

### Short Version
Built a medical-imaging classification project for breast histopathology using PyTorch and FastAPI. I created a reproducible data pipeline for the BreakHis dataset, trained a ResNet50 transfer-learning baseline, tracked medical metrics such as sensitivity and ROC-AUC, and exposed the model through a simple inference API.

### Longer Version
This project explores breast cancer detection from histopathology image patches using the BreakHis dataset. I implemented dataset download and organization, patient-level train/validation/test splitting to reduce leakage, image augmentation, a ResNet50-based binary classifier, and a FastAPI serving layer for inference. I also added experimental extensions for synthetic segmentation masks, synthetic clinical features, and multimodal attention modules to explore how the project could evolve beyond a baseline classifier.

## CV Bullet Options

- Built a breast histopathology classification pipeline in PyTorch using the BreakHis dataset, including patient-level splitting, augmentation, transfer learning with ResNet50, and medical evaluation metrics.
- Developed a FastAPI inference service for benign vs malignant histopathology patch classification and connected it to a trained baseline checkpoint for local serving.
- Trained and evaluated a medical-imaging baseline model with validation metrics including 0.848 accuracy, 0.962 sensitivity, 0.901 F1, and 0.870 ROC-AUC on the current saved checkpoint.
- Explored extensions for multimodal medical AI by adding synthetic clinical-feature generation, synthetic segmentation masks, and attention-based fusion components.

## One-Line Resume Version

Built a PyTorch + FastAPI breast histopathology classification prototype on BreakHis with patient-level data splitting, ResNet50 transfer learning, and API-based inference.

## Interview Talking Points

- The strongest completed path in the repo is the baseline classifier pipeline, from raw data to training to API inference.
- I focused on patient-level splitting because image-level splitting would leak information across train and test in medical datasets.
- I tracked sensitivity and specificity in addition to accuracy because missing malignant cases matters more than aggregate accuracy in this setting.
- I treated segmentation and multimodal fusion as exploratory extensions, and I would present them honestly as partially implemented rather than as a finished production system.

## Honest Positioning

Use phrasing like:
- medical-imaging prototype
- baseline histopathology classifier
- exploratory multimodal extension work
- FastAPI inference prototype

Avoid claiming:
- production-grade clinical system
- fully deployed multimodal platform
- expert-validated segmentation pipeline
- real-world diagnostic deployment
