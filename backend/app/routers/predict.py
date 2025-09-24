from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiofiles

from ..models import PredictionResponse, ErrorResponse
from ..model_service import model_service
from ..config import settings


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    # Check file size
    if hasattr(file, 'size') and file.size > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size too large. Maximum size: {settings.max_image_size_bytes / (1024*1024):.1f}MB"
        )
    
    # Check content type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(..., description="Plant image file")
):
    """Predict plant disease from uploaded image."""
    try:
        print(f"[DEBUG] Received image file: {image.filename}, content_type: {image.content_type}, size: {image.size if hasattr(image, 'size') else 'unknown'}")
        
        # Validate model is loaded
        if not model_service.is_healthy():
            raise HTTPException(status_code=503, detail="Model service unavailable")
        
        # Validate image file
        validate_image_file(image)
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Validate image size after reading
        if len(image_bytes) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File size too large. Maximum size: {settings.max_image_size_bytes / (1024*1024):.1f}MB"
            )
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Make prediction
        result = model_service.predict(image_bytes)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Close the file
        if hasattr(image, 'file'):
            await image.close()