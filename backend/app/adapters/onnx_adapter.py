import numpy as np
from typing import List, Tuple
from .base import BaseModelAdapter


class ONNXAdapter(BaseModelAdapter):
    """Adapter for ONNX models."""
    
    def __init__(self, model_path: str, labels_path: str = None):
        super().__init__(model_path, labels_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        
    def load_model(self) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            # Load labels
            self.labels = self.load_labels()
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 4:  # (batch, height, width, channels) or (batch, channels, height, width)
                if input_shape[1] == 3:  # Channels first (NCHW)
                    self.input_shape = (input_shape[2], input_shape[3], input_shape[1])
                else:  # Channels last (NHWC)
                    self.input_shape = input_shape[1:4]
            
            print(f"ONNX model loaded successfully")
            print(f"Input name: {self.input_name}")
            print(f"Output name: {self.output_name}")
            print(f"Input shape: {self.input_shape}")
            print(f"Number of classes: {len(self.labels)}")
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def preprocess_image(self, image) -> np.ndarray:
        """Preprocess image for ONNX model."""
        from PIL import Image
        
        # Resize and convert to RGB
        target_size = self.input_shape[:2]
        image = image.resize(target_size)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Check if model expects channels first (NCHW) or channels last (NHWC)
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4 and input_shape[1] == 3:  # Channels first
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        else:  # Channels last
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        return image_array
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Make prediction using ONNX model."""
        try:
            if self.session is None:
                raise ValueError("Model not loaded")
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: image_array})
            predictions = outputs[0]
            
            return predictions
            
        except Exception as e:
            print(f"Error during ONNX prediction: {e}")
            raise