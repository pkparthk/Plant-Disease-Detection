from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Import config manager with relative import to avoid circular dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class BaseModelAdapter(ABC):
    """Base class for model adapters that provides a unified interface for different model formats."""
    
    def __init__(self, model_path: str, labels_path: str = None):
        """Initialize adapter.
        
        Args:
            model_path: Path to the model file
            labels_path: Path to labels file (optional, uses config if not provided)
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = []
        self.input_shape = (128, 128, 3)  # Default shape
        
        # Import config manager here to avoid circular imports
        try:
            from ..config_manager import config_manager
            self.config = config_manager
        except ImportError:
            self.config = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Make a prediction on the preprocessed image array."""
        pass
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess the PIL Image for model inference with ImageNet normalization."""
        # Get input shape from configuration or use default
        if self.config:
            target_size = tuple(self.config.model_config.input_shape[:2])
        else:
            target_size = self.input_shape[:2]
        
        # Resize image to model input size
        image = image.resize(target_size)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0
                
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def postprocess_predictions(self, predictions: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Convert model predictions to label-confidence pairs."""
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # Remove batch dimension
            
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.labels):
                confidence = float(predictions[idx])
                label = self.labels[idx]
                results.append((label, confidence))
        
        return results
    
    def load_labels(self) -> List[str]:
        """Load class labels from config or JSON file."""
        # Try configuration first
        if self.config:
            return self.config.get_class_names()
        
        # Fallback to labels file
        if self.labels_path and Path(self.labels_path).exists():
            try:
                with open(self.labels_path, 'r') as f:
                    labels = json.load(f)
                return labels
            except Exception as e:
                print(f"Error loading labels from file: {e}")
        
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.config:
            model_config = self.config.model_config
            return {
                "name": model_config.name,
                "version": model_config.version,
                "description": model_config.description,
                "framework": model_config.framework,
                "model_path": self.model_path,
                "input_shape": model_config.input_shape,
                "num_classes": len(self.config.get_class_names()),
                "labels": self.config.get_class_names(),
                "supported_formats": model_config.supported_formats
            }
        else:
            # Fallback for when config is not available
            return {
                "name": self.__class__.__name__,
                "model_path": self.model_path,
                "input_shape": self.input_shape,
                "num_classes": len(self.labels),
                "labels": self.labels
            }