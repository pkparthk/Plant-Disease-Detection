import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
import io
from PIL import Image

from app.main import app
from app.model_service import model_service


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """Mock the model service."""
    with patch('app.routers.health.model_service') as mock:
        mock.is_healthy.return_value = True
        mock.get_model_info.return_value = {
            "name": "TestModel",
            "version": "1.0.0",
            "input_shape": [128, 128, 3],
            "classes": ["Apple___Healthy", "Apple___Scab"],
            "description": "Test model"
        }
        yield mock


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple RGB image
    img = Image.new('RGB', (128, 128), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoints:
    """Test health-related endpoints."""
    
    def test_health_check_success(self, client, mock_model_service):
        """Test successful health check."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "TestModel"
        assert "timestamp" in data
    
    def test_health_check_model_unavailable(self, client):
        """Test health check when model is unavailable."""
        with patch('app.routers.health.model_service') as mock:
            mock.is_healthy.return_value = False
            
            response = client.get("/api/health")
            assert response.status_code == 503
            assert "Model service unavailable" in response.json()["detail"]
    
    def test_model_info_success(self, client, mock_model_service):
        """Test successful model info retrieval."""
        response = client.get("/api/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "TestModel"
        assert data["version"] == "1.0.0"
        assert data["input_shape"] == [128, 128, 3]


class TestPredictionEndpoints:
    """Test prediction-related endpoints."""
    
    def test_predict_success(self, client, sample_image):
        """Test successful prediction."""
        with patch('app.routers.predict.model_service') as mock:
            mock.is_healthy.return_value = True
            mock.predict.return_value = {
                "predictions": [
                    {"label": "Apple___Healthy", "confidence": 0.95},
                    {"label": "Apple___Scab", "confidence": 0.05}
                ],
                "treatment": "Continue current care regimen.",
                "model_version": "TestModel",
                "inference_ms": 100
            }
            
            response = client.post(
                "/api/predict",
                files={"image": ("test.jpg", sample_image, "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["predictions"][0]["label"] == "Apple___Healthy"
            assert data["inference_ms"] == 100
    
    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        fake_file = io.BytesIO(b"not an image")
        
        response = client.post(
            "/api/predict",
            files={"image": ("test.txt", fake_file, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_predict_empty_file(self, client):
        """Test prediction with empty file."""
        empty_file = io.BytesIO(b"")
        
        response = client.post(
            "/api/predict",
            files={"image": ("test.jpg", empty_file, "image/jpeg")}
        )
        
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]
    
    def test_predict_model_unavailable(self, client, sample_image):
        """Test prediction when model is unavailable."""
        with patch('app.routers.predict.model_service') as mock:
            mock.is_healthy.return_value = False
            
            response = client.post(
                "/api/predict",
                files={"image": ("test.jpg", sample_image, "image/jpeg")}
            )
            
            assert response.status_code == 503
            assert "Model service unavailable" in response.json()["detail"]


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Plant Disease Detection API"
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]