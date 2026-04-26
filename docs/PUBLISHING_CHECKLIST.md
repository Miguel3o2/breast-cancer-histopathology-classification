# Publishing Checklist

## Before Pushing To GitHub

- Use a fresh Python 3.10-3.12 environment.
- Install dependencies from `requirements.txt` plus the appropriate PyTorch build.
- Re-run the smoke tests:
  - `python data\transforms.py`
  - `python data\dataset.py`
- Confirm the training path works:
  - `python train_classifier.py --batch_size 2 --num_workers 0`
- Confirm the API path works:
  - `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
  - `Invoke-RestMethod http://localhost:8000/health`
  - `.\TEST_API.ps1`
- Keep raw dataset files, generated CSVs, masks, checkpoints, and local environments out of Git.

## Suggested Git Commands

```powershell
git add .
git commit -m "Prepare breast cancer classification project for portfolio release"
```

## Suggested GitHub Repo Name

- `breast-cancer-histopathology-classification`
- `breakhis-classification-fastapi`
- `medical-imaging-breakhis-classifier`

## Suggested Portfolio Framing

Use this as a medical-imaging prototype / baseline classifier project, not as a completed clinical multimodal platform.

## Suggested CV Framing

Lead with the baseline classifier, data pipeline, and API. Mention segmentation and multimodal work as exploratory extensions.
