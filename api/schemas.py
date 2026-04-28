from pydantic import BaseModel, Field
from typing import Optional


class ClinicalFeatures(BaseModel):
    age: int = Field(..., ge=20, le=90, description="Patient age in years")
    tumor_size_mm: float = Field(..., ge=1, le=100, description="Tumor size in mm")
    family_history: int = Field(..., ge=0, le=1, description="Family history (0=no, 1=yes)")
    magnification: str = Field(..., pattern="^(40X|100X|200X|400X)$", description="Magnification level")


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint."""
    image_base64: str = Field(..., description="Base64-encoded image")
    clinical_features: Optional[ClinicalFeatures] = Field(None, description="Optional clinical data")
    return_heatmap: bool = Field(True, description="Include Grad-CAM heatmap")
    return_segmentation: bool = Field(True, description="Include segmentation mask")


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    prediction: str = Field(..., description="Predicted class (benign/malignant)")
    probability: float = Field(..., ge=0, le=1, description="Probability of prediction")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    uncertainty: float = Field(..., description="Model uncertainty (0-1)")
    gradcam_heatmap_base64: Optional[str] = Field(None, description="Grad-CAM heatmap (base64)")
    segmentation_mask_base64: Optional[str] = Field(None, description="Segmentation mask (base64)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
