"""
FastAPI Application for Model Serving.

Endpoints:
- POST /predict: Get prediction for an image
- GET /health: Health check
- GET /: Basic API message
"""

import base64
import os
import time
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from api.schemas import PredictionRequest, PredictionResponse
from data.transforms import get_val_transforms
from models import ResNetClassifier

app = FastAPI(
    title="Breast Cancer Detection API",
    description="Histopathology classification service for benign vs malignant prediction.",
    version="1.0.0",
)

DEFAULT_MODEL_PATH = "checkpoints/resnet50_best.pt"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_val_transforms(img_size=224)
CLASS_NAMES = ["benign", "malignant"]

model = None
model_load_error = None


def load_model():
    """Load the trained classifier checkpoint if present."""
    global model, model_load_error

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        model = None
        model_load_error = f"Checkpoint not found: {model_path}"
        print(f"WARNING: {model_load_error}")
        return

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        classifier = ResNetClassifier(num_classes=2, pretrained=False, dropout=0.5)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.to(DEVICE)
        classifier.eval()

        model = classifier
        model_load_error = None
        print(f"Model loaded from {model_path}")
    except Exception as exc:
        model = None
        model_load_error = str(exc)
        print(f"WARNING: Could not load model: {exc}")


load_model()


@app.get("/")
def root():
    return {"message": "Breast Cancer Detection API. Visit /docs for documentation."}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
        "model_load_error": model_load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run image classification on a base64-encoded histopathology image."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")

    start_time = time.time()

    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probabilities).item())
        probability = float(probabilities[pred_idx].item())
        uncertainty = float(1.0 - probability)

        if probability >= 0.9:
            confidence = "high"
        elif probability >= 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        processing_time_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=CLASS_NAMES[pred_idx],
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
