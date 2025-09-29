from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="HOST")
    api_port: int = Field(default=8000, alias="PORT")
    api_workers: int = Field(default=1, alias="WORKERS")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # Model Configuration
    model_path: str = Field(default="models/best_model.h5", alias="MODEL_PATH")
    model_json_path: str = Field(default="models/model.json", alias="MODEL_JSON_PATH")
    model_type: str = Field(default="tensorflow", alias="MODEL_TYPE")
    model_labels_path: str = Field(default="labels.json", alias="MODEL_LABELS_PATH")
    
    # Security Configuration
    secret_key: str = Field(default="your-super-secret-key-change-in-production", alias="SECRET_KEY")
    api_key_name: str = Field(default="X-API-Key", alias="API_KEY_NAME")
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS"
    )
    
    # File Upload & Rate Limiting
    max_image_size_bytes: int = Field(default=8 * 1024 * 1024, alias="MAX_IMAGE_SIZE_BYTES")  # 8MB
    rate_limit_requests: int = Field(default=100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, alias="RATE_LIMIT_WINDOW")  # seconds
    max_concurrent_requests: int = Field(default=10, alias="MAX_CONCURRENT_REQUESTS")
    
    # Database Configuration (Optional)
    database_url: str = Field(default="sqlite:///./plant_disease.db", alias="DATABASE_URL")
    enable_prediction_logging: bool = Field(default=True, alias="ENABLE_PREDICTION_LOGGING")
    
    # Monitoring & Observability
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_path: str = Field(default="/metrics", alias="METRICS_PATH")
    health_check_path: str = Field(default="/health", alias="HEALTH_CHECK_PATH")
    
    # Cache Configuration
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")  # 1 hour
    enable_response_cache: bool = Field(default=True, alias="ENABLE_RESPONSE_CACHE")
    
    # ML Model Performance
    model_warmup_samples: int = Field(default=5, alias="MODEL_WARMUP_SAMPLES")
    prediction_timeout: int = Field(default=30, alias="PREDICTION_TIMEOUT")  # seconds
    enable_model_validation: bool = Field(default=True, alias="ENABLE_MODEL_VALIDATION")
    

    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['tensorflow', 'pytorch', 'onnx']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list"""
        if self.cors_origins.strip() == '*':
            return ['*']
        return [origin.strip() for origin in self.cors_origins.split(',')]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ()
        populate_by_name = True


settings = Settings()