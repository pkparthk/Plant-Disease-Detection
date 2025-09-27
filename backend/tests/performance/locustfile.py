"""
Performance tests using Locust for load testing the API.
"""

from locust import HttpUser, task, between
import random
import io
from PIL import Image
import numpy as np


class PlantDiseaseAPIUser(HttpUser):
    """Simulate users making requests to the plant disease detection API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        # Check if API is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"API health check failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint - lightweight check."""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint."""
        self.client.get("/model/info")
    
    @task(8)
    def predict_image(self):
        """Test image prediction endpoint - main workload."""
        # Generate a random test image
        image_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        # Make prediction request
        files = {
            "file": ("test_image.jpg", img_bytes.getvalue(), "image/jpeg")
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data and "confidence" in data:
                        response.success()
                    else:
                        response.failure(f"Invalid response format: {data}")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            elif response.status_code == 429:
                # Rate limiting is expected under load
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(2)
    def predict_with_different_sizes(self):
        """Test with different image sizes to measure performance impact."""
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        size = random.choice(sizes)
        
        # Generate test image with specific size
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            "file": (f"test_{size[0]}x{size[1]}.jpg", img_bytes.getvalue(), "image/jpeg")
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Size {size} failed: {response.status_code}")
    
    @task(1)
    def test_concurrent_predictions(self):
        """Test behavior under concurrent load."""
        # This will naturally happen with multiple users
        self.predict_image()


class HighLoadUser(HttpUser):
    """Simulate high-load scenario users."""
    
    wait_time = between(0.5, 1)  # Faster requests
    
    @task
    def rapid_predictions(self):
        """Make rapid prediction requests."""
        # Simple small image for faster processing
        image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG', quality=60)
        img_bytes.seek(0)
        
        files = {"file": ("quick_test.jpg", img_bytes.getvalue(), "image/jpeg")}
        
        self.client.post("/predict", files=files)


class ErrorTestUser(HttpUser):
    """Test error handling and edge cases."""
    
    wait_time = between(2, 5)
    
    @task(1)
    def test_invalid_file(self):
        """Test with invalid file types."""
        # Send text file as image
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        
        response = self.client.post("/predict", files=files)
        # Should get 400 or 422 error
        if response.status_code in [400, 422]:
            # Expected error response
            pass
    
    @task(1)
    def test_large_file(self):
        """Test with oversized files."""
        # Create a large fake image
        large_data = b"x" * (5 * 1024 * 1024)  # 5MB
        files = {"file": ("large.jpg", large_data, "image/jpeg")}
        
        response = self.client.post("/predict", files=files)
        # Should get 413 (Payload Too Large) or handle gracefully
    
    @task(1)
    def test_no_file(self):
        """Test request without file."""
        response = self.client.post("/predict")
        # Should get 422 (Unprocessable Entity)
    
    @task(1)
    def test_malformed_request(self):
        """Test malformed requests."""
        # Send invalid data
        response = self.client.post("/predict", data={"invalid": "data"})


# Custom test scenarios for specific load patterns
class MorningRushUser(HttpUser):
    """Simulate morning rush hour traffic."""
    wait_time = between(0.1, 0.5)
    
    @task
    def quick_check(self):
        # Farmers checking plants in the morning
        self.predict_image()


class FarmSeasonUser(HttpUser):
    """Simulate farming season with burst activity."""
    wait_time = between(0, 2)
    
    @task(5)
    def batch_analysis(self):
        # Multiple images from same user
        for _ in range(random.randint(1, 3)):
            self.predict_image()
    
    def predict_image(self):
        """Helper method for prediction."""
        image_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("farm_image.jpg", img_bytes.getvalue(), "image/jpeg")}
        self.client.post("/predict", files=files)