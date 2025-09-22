from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
import os

# Allow this module file to behave like a package so tests can import
# `app.models.adapters` (the project also defines a `models/adapters` package).
# Setting __path__ here tells importlib to treat this module as a package
# and search the containing directory for subpackages.
__path__ = [os.path.dirname(__file__)]


class TreatmentInfo(BaseModel):
    """Treatment information for a disease."""
    chemical: str
    cultural: str
    preventive: str


class PredictionResult(BaseModel):
    """Individual prediction result."""
    label: str
    confidence: float
    class_name: str
    plant: str
    disease: str
    severity: str
    severity_level: int
    severity_color: str
    treatment: TreatmentInfo
    symptoms: Optional[List[str]] = []
    causes: Optional[str] = ""
    urgency: Optional[str] = ""
    economic_impact: Optional[str] = ""


class PredictionResponse(BaseModel):
    """Complete prediction response."""
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[PredictionResult]
    top_prediction: PredictionResult
    confidence_level: str
    model_info: Dict[str, Any]
    inference_ms: int
    timestamp: str


class Prediction(BaseModel):
    """Legacy prediction format for compatibility."""
    label: str
    confidence: float


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    description: str
    framework: str
    input_shape: List[int]
    classes: List[str]
    supported_formats: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model: str
    timestamp: str
    model_loaded: bool
    total_classes: int


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    code: Optional[str] = None
    timestamp: Optional[str] = None