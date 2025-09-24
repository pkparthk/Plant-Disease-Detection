import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
from PIL import Image
import io

from app.adapters.tensorflow_adapter import TensorFlowAdapter
from app.adapters.base import BaseModelAdapter


class TestBaseAdapter:
    """Test base model adapter functionality."""
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a mock adapter
        adapter = BaseModelAdapter("fake_path", "fake_labels")
        adapter.input_shape = (128, 128, 3)
        
        # Create test image
        img = Image.new('RGB', (256, 256), color='red')
        
        # Preprocess
        processed = adapter.preprocess_image(img)
        
        # Check shape and type
        assert processed.shape == (1, 128, 128, 3)
        assert processed.dtype == np.float32
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_postprocess_predictions(self):
        """Test prediction postprocessing."""
        adapter = BaseModelAdapter("fake_path", "fake_labels")
        adapter.labels = ["class_0", "class_1", "class_2"]
        
        # Mock predictions
        predictions = np.array([0.1, 0.8, 0.1])
        
        results = adapter.postprocess_predictions(predictions, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "class_1"  # Highest confidence
        assert results[0][1] == 0.8
        assert results[1][0] == "class_0"  # Second highest
    
    @patch("builtins.open", new_callable=mock_open, read_data='["class_0", "class_1"]')
    @patch("json.load")
    def test_load_labels(self, mock_json_load, mock_file):
        """Test label loading."""
        mock_json_load.return_value = ["class_0", "class_1"]
        
        adapter = BaseModelAdapter("fake_path", "fake_labels")
        labels = adapter.load_labels()
        
        assert labels == ["class_0", "class_1"]
        mock_file.assert_called_once_with("fake_labels", 'r')


class TestTensorFlowAdapter:
    """Test TensorFlow adapter."""
    
    @patch('app.adapters.tensorflow_adapter.tf')
    @patch('app.adapters.tensorflow_adapter.model_from_json')
    def test_load_model_from_json(self, mock_model_from_json, mock_tf):
        """Test loading model from JSON + weights."""
        # Setup mocks
        mock_model = Mock()
        mock_model.input_shape = (None, 128, 128, 3)
        mock_model_from_json.return_value = mock_model
        
        # Mock file reading
        with patch("builtins.open", mock_open(read_data='{"model": "config"}')):
            with patch("json.load", return_value=["class_0", "class_1"]):
                adapter = TensorFlowAdapter("model.h5", "labels.json", "model.json")
                adapter.load_model()
        
        # Verify model loading
        mock_model_from_json.assert_called_once()
        mock_model.load_weights.assert_called_once_with("model.h5")
        assert adapter.input_shape == (128, 128, 3)
    
    def test_predict(self):
        """Test prediction with TensorFlow model."""
        adapter = TensorFlowAdapter("fake_path", "fake_labels")
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        adapter.model = mock_model
        
        # Test prediction
        input_array = np.random.random((1, 128, 128, 3))
        result = adapter.predict(input_array)
        
        mock_model.predict.assert_called_once_with(input_array, verbose=0)
        assert result.shape == (1, 3)