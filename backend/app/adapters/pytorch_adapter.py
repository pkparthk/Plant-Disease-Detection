import numpy as np
from typing import List, Tuple, Optional
from .base import BaseModelAdapter


class PyTorchAdapter(BaseModelAdapter):
    """Adapter for PyTorch models."""
    
    def __init__(self, model_path: str, labels_path: str = None):
        super().__init__(model_path, labels_path)
        self.device = None
        
    def load_model(self) -> None:
        """Load PyTorch model."""
        try:
            import torch
            import torch.nn as nn
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Load labels
            self.labels = self.load_labels()
            
            # Load model
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Try to infer input shape from model (this is model-specific)
            # For now, use default shape
            self.input_shape = (128, 128, 3)
            
            print(f"PyTorch model loaded successfully")
            print(f"Input shape: {self.input_shape}")
            print(f"Number of classes: {len(self.labels)}")
            
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise
    
    def preprocess_image(self, image) -> np.ndarray:
        """Preprocess image for PyTorch model (channels first)."""
        import torch
        from PIL import Image
        
        # Resize and convert to RGB
        target_size = self.input_shape[:2]
        image = image.resize(target_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor with channels first (C, H, W)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension and convert to tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def predict(self, image_tensor) -> np.ndarray:
        """Make prediction using PyTorch model."""
        try:
            import torch
            
            if self.model is None:
                raise ValueError("Model not loaded")
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Convert to numpy
                predictions = probabilities.cpu().numpy()
                
            return predictions
            
        except Exception as e:
            print(f"Error during PyTorch prediction: {e}")
            raise