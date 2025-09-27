import pytest
from unittest.mock import Mock, patch
import io
from PIL import Image

from app.model_service import ModelService


class TestModelService:
    """Test model service functionality."""
    
    def test_load_model_tensorflow(self):
        """Test loading TensorFlow model."""
        service = ModelService()
        
        with patch('app.model_service.settings') as mock_settings:
            mock_settings.model_type = "tensorflow"
            mock_settings.model_path = "model.h5"
            mock_settings.model_labels_path = "labels.json"
            mock_settings.model_json_path = "model.json"
            
            with patch('app.model_service.TensorFlowAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.get_model_info.return_value = {"name": "TestModel"}
                mock_adapter_class.return_value = mock_adapter
                
                service.load_model()
                
                assert service.model_loaded is True
                mock_adapter.load_model.assert_called_once()
    
    def test_predict_success(self):
        """Test successful prediction."""
        service = ModelService()
        service.model_loaded = True
        
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.preprocess_image.return_value = "processed_image"
        mock_adapter.predict.return_value = "predictions"
        mock_adapter.postprocess_predictions.return_value = [
            ("Apple___Healthy", 0.95),
            ("Apple___Scab", 0.05)
        ]
        service.adapter = mock_adapter
        service.model_info = {"name": "TestModel"}
        
        # Create test image bytes
        img = Image.new('RGB', (128, 128), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        result = service.predict(img_bytes)
        
        assert len(result["predictions"]) == 2
        assert result["predictions"][0]["label"] == "Apple___Healthy"
        assert result["predictions"][0]["confidence"] == 0.95
        assert "treatment" in result
        assert "inference_ms" in result
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        service = ModelService()
        service.model_loaded = False
        
        with pytest.raises(ValueError, match="Model not loaded"):
            service.predict(b"fake_image_bytes")
    
    def test_is_healthy(self):
        """Test health check."""
        service = ModelService()
        
        # Not loaded
        assert service.is_healthy() is False
        
        # Loaded
        service.model_loaded = True
        service.adapter = Mock()
        assert service.is_healthy() is True
    
    def test_get_model_info(self):
        """Test getting model information."""
        service = ModelService()
        
        # Not loaded
        info = service.get_model_info()
        assert "error" in info
        
        # Loaded
        service.model_loaded = True
        service.model_info = {
            "name": "TestModel",
            "labels": ["class_0", "class_1"]
        }
        
        info = service.get_model_info()
        assert info["name"] == "TestModel"
        assert info["version"] == "1.0.0"
        assert len(info["classes"]) == 2