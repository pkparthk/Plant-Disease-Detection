"""
Test configuration and fixtures.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
import json
import sys
import os
from PIL import Image
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing."""
    from app.core.config import Settings
    
    return Settings(
        model_path="tests/fixtures/test_model.h5",
        model_json_path="tests/fixtures/test_model.json",
        model_labels_path="tests/fixtures/test_labels.json",
        model_type="tensorflow",
        max_image_size_bytes=1024*1024,  # 1MB for testing
        rate_limit_requests=100,  # Higher limit for testing
        debug=True,
        cors_origins=["http://localhost:3000"],
        rate_limit="100/minute"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB image
    image_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    return image


@pytest.fixture
def sample_image_file(temp_dir, sample_image):
    """Create a sample image file."""
    image_path = temp_dir / "test_image.jpg"
    sample_image.save(image_path)
    return image_path


@pytest.fixture
def mock_labels():
    """Create mock labels for testing."""
    return [
        "Apple___Apple_scab",
        "Apple___Black_rot", 
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy"
    ]


@pytest.fixture
def mock_model_output():
    """Create mock model prediction output."""
    return np.array([[0.1, 0.05, 0.02, 0.8, 0.03]])


@pytest.fixture
def mock_tensorflow_model():
    """Create a mock TensorFlow model."""
    mock_model = Mock()
    mock_model.predict.return_value = mock_model_output()
    return mock_model


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model."""
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.forward.return_value = Mock()
    return mock_model


@pytest.fixture
def mock_onnx_session():
    """Create a mock ONNX session."""
    mock_session = Mock()
    mock_session.run.return_value = [mock_model_output()]
    return mock_session


@pytest.fixture
def labels_file(temp_dir, mock_labels):
    """Create a labels JSON file."""
    labels_path = temp_dir / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(mock_labels, f)
    return labels_path


class MockImageFile:
    """Mock image file for testing uploads."""
    
    def __init__(self, filename="test.jpg", content_type="image/jpeg", content=b"fake image data"):
        self.filename = filename
        self.content_type = content_type
        self.file = Mock()
        self.file.read.return_value = content
        
    def __enter__(self):
        return self.file
        
    def __exit__(self, *args):
        pass


@pytest.fixture
def mock_image_upload():
    """Create a mock uploaded image file."""
    return MockImageFile()


@pytest.fixture(autouse=True)
def reset_model_adapter():
    """Reset model adapter between tests."""
    # Clear any cached adapters
    yield