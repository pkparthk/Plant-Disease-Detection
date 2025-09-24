"""
Tests for model adapters.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path


class TestBaseModelAdapter:
    """Test the base model adapter."""
    
    def test_abstract_methods(self):
        """Test that base adapter cannot be instantiated."""
        from app.models.adapters import BaseModelAdapter
        
        with pytest.raises(TypeError):
            BaseModelAdapter()


class TestTensorFlowAdapter:
    """Test TensorFlow model adapter."""
    
    @patch('app.models.adapters.tf')
    def test_load_model_h5(self, mock_tf):
        """Test loading H5 model format."""
        from app.models.adapters import TensorFlowAdapter
        
        # Mock TensorFlow
        mock_model = Mock()
        mock_tf.keras.models.load_model.return_value = mock_model
        
        adapter = TensorFlowAdapter()
        result = adapter.load_model("test_model.h5")
        
        assert result == mock_model
        mock_tf.keras.models.load_model.assert_called_once_with("test_model.h5")
    
    @patch('app.models.adapters.tf')
    @patch('builtins.open')
    def test_load_model_json(self, mock_open, mock_tf):
        """Test loading JSON + weights model format."""
        from app.models.adapters import TensorFlowAdapter
        
        # Mock file reading
        mock_open.return_value.__enter__.return_value.read.return_value = '{"model": "data"}'
        
        # Mock TensorFlow
        mock_model = Mock()
        mock_tf.keras.models.model_from_json.return_value = mock_model
        
        adapter = TensorFlowAdapter()
        result = adapter.load_model("test_model.json", weights_path="test_weights.h5")
        
        assert result == mock_model
        mock_tf.keras.models.model_from_json.assert_called_once()
        mock_model.load_weights.assert_called_once_with("test_weights.h5")
    
    @patch('app.models.adapters.tf')
    def test_predict(self, mock_tf):
        """Test prediction with TensorFlow model."""
        from app.models.adapters import TensorFlowAdapter
        
        # Setup mock model
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.8, 0.1]])
        mock_model.predict.return_value = mock_predictions
        
        adapter = TensorFlowAdapter()
        adapter.model = mock_model
        
        # Mock input data
        input_data = np.random.random((1, 128, 128, 3))
        
        result = adapter.predict(input_data)
        
        assert np.array_equal(result, mock_predictions)
        mock_model.predict.assert_called_once_with(input_data)
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        from app.models.adapters import TensorFlowAdapter
        
        adapter = TensorFlowAdapter()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            adapter.predict(np.random.random((1, 128, 128, 3)))


class TestPyTorchAdapter:
    """Test PyTorch model adapter."""
    
    @patch('app.models.adapters.torch')
    def test_load_model(self, mock_torch):
        """Test loading PyTorch model."""
        from app.models.adapters import PyTorchAdapter
        
        # Mock PyTorch
        mock_model = Mock()
        mock_torch.load.return_value = mock_model
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        adapter = PyTorchAdapter()
        result = adapter.load_model("test_model.pth")
        
        assert result == mock_model
        mock_torch.load.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('app.models.adapters.torch')
    def test_predict(self, mock_torch):
        """Test prediction with PyTorch model."""
        from app.models.adapters import PyTorchAdapter
        
        # Setup mock model
        mock_model = Mock()
        mock_output = Mock()
        mock_predictions = np.array([[0.1, 0.8, 0.1]])
        mock_output.detach.return_value.cpu.return_value.numpy.return_value = mock_predictions
        mock_model.return_value = mock_output
        
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.from_numpy.return_value.float.return_value.unsqueeze.return_value = "tensor"
        
        adapter = PyTorchAdapter()
        adapter.model = mock_model
        adapter.device = "cpu"
        
        # Mock input data
        input_data = np.random.random((1, 128, 128, 3))
        
        with patch('app.models.adapters.torch.no_grad'):
            result = adapter.predict(input_data)
        
        assert np.array_equal(result, mock_predictions)


class TestONNXAdapter:
    """Test ONNX model adapter."""
    
    @patch('app.models.adapters.ort')
    def test_load_model(self, mock_ort):
        """Test loading ONNX model."""
        from app.models.adapters import ONNXAdapter
        
        # Mock ONNX Runtime
        mock_session = Mock()
        mock_ort.InferenceSession.return_value = mock_session
        
        adapter = ONNXAdapter()
        result = adapter.load_model("test_model.onnx")
        
        assert result == mock_session
        mock_ort.InferenceSession.assert_called_once_with("test_model.onnx")
    
    @patch('app.models.adapters.ort')
    def test_predict(self, mock_ort):
        """Test prediction with ONNX model.""" 
        from app.models.adapters import ONNXAdapter
        
        # Setup mock session
        mock_session = Mock()
        mock_predictions = [np.array([[0.1, 0.8, 0.1]])]
        mock_session.run.return_value = mock_predictions
        mock_session.get_inputs.return_value = [Mock(name="input")]
        
        adapter = ONNXAdapter()
        adapter.session = mock_session
        
        # Mock input data
        input_data = np.random.random((1, 128, 128, 3))
        
        result = adapter.predict(input_data)
        
        assert np.array_equal(result, mock_predictions[0])
        mock_session.run.assert_called_once()


class TestModelAdapterFactory:
    """Test model adapter factory."""
    
    def test_create_tensorflow_adapter(self):
        """Test creating TensorFlow adapter."""
        from app.models.adapters import create_model_adapter
        
        adapter = create_model_adapter("tensorflow")
        
        from app.models.adapters import TensorFlowAdapter
        assert isinstance(adapter, TensorFlowAdapter)
    
    def test_create_pytorch_adapter(self):
        """Test creating PyTorch adapter."""
        from app.models.adapters import create_model_adapter
        
        adapter = create_model_adapter("pytorch")
        
        from app.models.adapters import PyTorchAdapter
        assert isinstance(adapter, PyTorchAdapter)
    
    def test_create_onnx_adapter(self):
        """Test creating ONNX adapter."""
        from app.models.adapters import create_model_adapter
        
        adapter = create_model_adapter("onnx")
        
        from app.models.adapters import ONNXAdapter
        assert isinstance(adapter, ONNXAdapter)
    
    def test_invalid_adapter_type(self):
        """Test creating adapter with invalid type."""
        from app.models.adapters import create_model_adapter
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model_adapter("invalid_type")