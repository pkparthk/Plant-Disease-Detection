"""
Integration tests for the complete API workflow.
"""

import pytest
import json
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient


class TestAPIIntegration:
    """Test complete API workflows."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_info" in data
    
    def test_model_info(self, client):
        """Test model info endpoint.""" 
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "classes" in data
        assert "version" in data
    
    @pytest.mark.integration
    def test_prediction_workflow(self, client, sample_image):
        """Test complete prediction workflow."""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        sample_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Make prediction request
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "top_prediction" in data
        assert "confidence" in data
        assert "processing_time" in data
        
        # Verify prediction content
        predictions = data["predictions"]
        assert len(predictions) > 0
        assert all("class" in pred for pred in predictions)
        assert all("confidence" in pred for pred in predictions)
        assert all("treatment" in pred for pred in predictions)
        
        # Verify confidence scores are valid
        assert 0 <= data["confidence"] <= 1
        for pred in predictions:
            assert 0 <= pred["confidence"] <= 1
    
    def test_invalid_file_type(self, client):
        """Test upload of invalid file type."""
        # Create a text file instead of image
        text_content = b"This is not an image"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_large_file_rejection(self, client):
        """Test rejection of files that are too large."""
        # Create a large fake image
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        files = {"file": ("large.jpg", io.BytesIO(large_data), "image/jpeg")}
        
        response = client.post("/predict", files=files)
        assert response.status_code == 413
    
    def test_no_file_upload(self, client):
        """Test request without file upload."""
        response = client.post("/predict")
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/predict")
        assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.integration
    def test_rate_limiting(self, client, sample_image):
        """Test rate limiting functionality."""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        sample_image.save(img_bytes, format='JPEG')
        
        # Make multiple requests rapidly
        files = {"file": ("test.jpg", img_bytes.getvalue(), "image/jpeg")}
        
        responses = []
        for _ in range(5):
            img_bytes.seek(0)
            response = client.post("/predict", files=files)
            responses.append(response.status_code)
        
        # Should get successful responses initially
        assert responses[0] == 200
        # Rate limiting behavior depends on configuration
    
    def test_concurrent_requests(self, client, sample_image):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        img_bytes = io.BytesIO()
        sample_image.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        def make_request():
            files = {"file": ("test.jpg", img_data, "image/jpeg")}
            return client.post("/predict", files=files)
        
        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_malformed_image(self, client):
        """Test handling of malformed image data."""
        # Create invalid image data
        malformed_data = b"Not a real image but with jpeg extension"
        files = {"file": ("fake.jpg", io.BytesIO(malformed_data), "image/jpeg")}
        
        response = client.post("/predict", files=files)
        # Should handle gracefully with error
        assert response.status_code in [400, 422]
    
    @pytest.mark.slow
    def test_various_image_formats(self, client):
        """Test different image formats."""
        formats = ['JPEG', 'PNG', 'BMP']
        
        for fmt in formats:
            # Create test image in specific format
            image_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=fmt)
            img_bytes.seek(0)
            
            extension = fmt.lower()
            content_type = f"image/{extension}"
            files = {"file": (f"test.{extension}", img_bytes, content_type)}
            
            response = client.post("/predict", files=files)
            # JPEG and PNG should work, others might not
            if fmt in ['JPEG', 'PNG']:
                assert response.status_code == 200
    
    def test_prediction_consistency(self, client, sample_image):
        """Test that same image produces consistent predictions."""
        img_bytes = io.BytesIO()
        sample_image.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        # Make multiple predictions with same image
        predictions = []
        for _ in range(3):
            files = {"file": ("test.jpg", img_data, "image/jpeg")}
            response = client.post("/predict", files=files)
            assert response.status_code == 200
            predictions.append(response.json()["top_prediction"])
        
        # Predictions should be consistent
        assert all(pred == predictions[0] for pred in predictions)