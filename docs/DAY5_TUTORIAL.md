# DAY 5 TUTORIAL — Explainability & Production Deployment

## What You're Learning Today

1. Grad-CAM for visual explanations (regulatory requirement)
2. FastAPI for model serving (REST API)
3. Docker containerization (deployment)
4. Weights & Biases experiment tracking (reproducibility)
5. Model versioning and checkpointing

---

## Part 1: Grad-CAM (Gradient-weighted Class Activation Mapping)

### Why Explainability is Non-Negotiable

**FDA/regulatory requirements:**
- Medical AI must explain its decisions
- Clinicians need to verify the model is looking at the right regions
- Can't deploy a "black box" in healthcare

**Grad-CAM shows:** Which parts of the image the model focused on to make its prediction.

### How Grad-CAM Works

**Intuition:** 
Backpropagate the class score to see which pixels contributed most.

**Steps:**
1. Forward pass: image → prediction
2. Backward pass: gradient of target class w.r.t. last conv layer
3. Global average pooling: average gradients over spatial dimensions
4. Weighted combination: multiply feature maps by gradient weights
5. ReLU: only positive contributions
6. Upsample: resize to input size
7. Overlay: blend heatmap with original image

**Mathematical formulation:**
```
α_k = GlobalAvgPool(∂y^c/∂A^k)  # Importance of feature map k
L = ReLU(Σ α_k × A^k)            # Weighted combination
```

### Clinical Interpretation

**Good Grad-CAM:**
- Highlights tumor regions
- Focuses on nuclear density, architecture
- Ignores background, artifacts

**Bad Grad-CAM:**
- Highlights staining artifacts
- Focuses on image borders, tears
- Random scattered activations

**Red flag:** Model learns spurious correlations (e.g., scanner watermarks)

---

## Part 2: FastAPI for Model Serving

### Why REST API?

**Deployment scenarios:**
1. **Web application:** Pathologist uploads image → gets diagnosis
2. **Hospital integration:** EMR system calls API → stores result
3. **Batch processing:** Analyze 1000 images overnight

**FastAPI advantages:**
- Fast (async support)
- Auto-generated API docs (Swagger)
- Type validation (Pydantic)
- Easy to test

### API Design

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "image_base64": "iVBORw0KGgo...",
  "clinical_features": {
    "age": 58,
    "tumor_size_mm": 25,
    "family_history": 1,
    "magnification": "200X"
  },
  "return_heatmap": true
}
```

**Response:**
```json
{
  "prediction": "malignant",
  "probability": 0.94,
  "confidence": "high",
  "uncertainty": 0.08,
  "gradcam_heatmap_base64": "iVBORw0KGgo...",
  "segmentation_mask_base64": "iVBORw0KGgo...",
  "processing_time_ms": 187
}
```

### Production Considerations

1. **Input validation:** Check image size, format, clinical feature ranges
2. **Error handling:** Graceful failures (corrupted image → 400 error)
3. **Rate limiting:** Prevent abuse
4. **Logging:** Track all predictions for auditing
5. **Versioning:** `/v1/predict`, `/v2/predict`

---

## Part 3: Docker Containerization

### Why Docker?

**Problem:** "Works on my machine" syndrome
- Different Python versions
- Missing dependencies
- CUDA version mismatches

**Solution:** Container packages everything
- Code + dependencies + environment
- Runs identically everywhere
- Easy deployment (AWS, Azure, GCP)

### Dockerfile Structure

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

For multi-container setup:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - MODEL_PATH=/app/checkpoints/best_model.pt
  
  redis:  # For caching predictions
    image: redis:alpine
    ports:
      - "6379:6379"
```

---

## Part 4: Weights & Biases (W&B) Experiment Tracking

### Why Experiment Tracking?

**Problem:** After 50 training runs, which hyperparameters worked?
- Lost track of configs
- Can't reproduce best result
- Don't know what changed

**Solution:** Track everything
- Hyperparameters
- Metrics (loss, accuracy, AUC)
- Model checkpoints
- Code version (git commit)
- System info (GPU, CUDA)

### W&B Integration

```python
import wandb

# Initialize
wandb.init(
    project="breast-cancer-multimodal",
    config={
        "learning_rate": 1e-4,
        "architecture": "UNet-MultiModal",
        "dataset": "BreakHis",
        "epochs": 25
    }
)

# Log metrics
for epoch in range(25):
    train_loss = train_epoch(...)
    val_acc = validate(...)
    
    wandb.log({
        "train/loss": train_loss,
        "val/accuracy": val_acc,
        "val/sensitivity": sensitivity,
        "epoch": epoch
    })

# Log model
wandb.save("checkpoints/best_model.pt")
```

**Dashboard shows:**
- Training curves (real-time)
- Hyperparameter comparison
- Best checkpoint download

---

## Part 5: Model Versioning & Checkpointing

### What to Save

**Minimal checkpoint:**
```python
torch.save(model.state_dict(), "model.pt")  # Just weights
```

**Production checkpoint:**
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': 0.943,
    'training_config': config,
    'class_names': ['benign', 'malignant'],
    'normalization_stats': stats,
    'git_commit': 'a3b7f2e',
    'pytorch_version': '2.0.1',
    'timestamp': '2026-04-20T15:30:00'
}, "checkpoint_full.pt")
```

### Versioning Strategy

```
models/
  v1.0/
    model.pt
    config.json
    metrics.json
  v1.1/
    model.pt  # Added clinical features
    config.json
    metrics.json
  v2.0/
    model.pt  # Multimodal with attention
    config.json
    metrics.json
```

**Semantic versioning:**
- `1.0.0` → Initial release
- `1.1.0` → New features (clinical data)
- `2.0.0` → Breaking changes (architecture change)
- `2.0.1` → Bug fixes

---

## What We're Building Today

**File 1: `explainability/gradcam.py`**
- Grad-CAM implementation
- Heatmap generation
- Overlay visualization

**File 2: `api/main.py`**
- FastAPI application
- `/predict` endpoint
- Request/response schemas

**File 3: `api/schemas.py`**
- Pydantic models for validation
- Type checking

**File 4: `Dockerfile`**
- Container definition
- Production-ready

**File 5: `docker-compose.yml`**
- Multi-container orchestration

**File 6: `configs/experiment.yaml`**
- Hyperparameter configs
- W&B settings

---

## Expected Outcome

After today:

**1. Generate Grad-CAM:**
```python
from explainability import GradCAM

gradcam = GradCAM(model)
heatmap = gradcam(image, target_class='malignant')
# Shows tumor regions highlighted in red
```

**2. Run API:**
```bash
docker-compose up
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

**3. Track experiments:**
```bash
wandb login
python train_multimodal.py --wandb
# View at https://wandb.ai/yourname/breast-cancer-multimodal
```

---

## Time Estimate

- Grad-CAM implementation: 40 min
- FastAPI setup: 45 min
- Docker configuration: 30 min
- W&B integration: 20 min
- Testing: 30 min
- **Total: ~3 hours**

---

Ready to make this production-grade?
