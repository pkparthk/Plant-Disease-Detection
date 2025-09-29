"""
Plant Disease Detection API
Enhanced FastAPI application with security, performance, and monitoring
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import uvicorn

from .config import settings
from .model_service import model_service
from .routers import health_router, predict_router
from .routers.predict import limiter
from .security import (
    SecurityAuditLogger,
    ResponseCache,
)
from .performance import (
    AsyncTaskManager,
    optimize_memory_usage,
)


# Setup enhanced logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log") if not settings.is_development else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize security and performance components
security_audit = SecurityAuditLogger()
response_cache = ResponseCache()
task_manager = AsyncTaskManager(max_concurrent_tasks=getattr(settings, 'max_concurrent_predictions', 10))

# Security setup
security = HTTPBearer(auto_error=False) if settings.api_key else None


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints."""
    if not settings.api_key:
        return True  # No API key required
    
    if not credentials or credentials.credentials != settings.api_key:
        await security_audit.log_event(
            event_type="authentication_failure",
            details={"reason": "invalid_api_key", "ip": "unknown"}
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    await security_audit.log_event(
        event_type="authentication_success",
        details={"ip": "unknown"}
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events with enhanced monitoring."""
    # Startup
    logger.info("🚀 Starting Plant Disease Detection API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Security features: Enabled")
    logger.info(f"Performance monitoring: Enabled")
    
    try:
        # Initialize security components
        await security_audit.initialize()
        await response_cache.initialize()
        logger.info("✅ Security components initialized")
        
        # Load model asynchronously with timeout
        await asyncio.wait_for(
            asyncio.create_task(asyncio.to_thread(model_service.load_model)),
            timeout=settings.prediction_timeout
        )
        logger.info("✅ Model loaded successfully")
        
        # Optimize memory usage
        await optimize_memory_usage()
        logger.info("✅ Memory optimization complete")
        
        # Warm up model if configured
        if settings.model_warmup_samples > 0:
            logger.info(f"🔥 Model will be warmed up on first request")
            
    except asyncio.TimeoutError:
        logger.error("⏱️ Model loading timed out")
        await security_audit.log_event(
            event_type="startup_error",
            details={"error": "model_loading_timeout"}
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        await security_audit.log_event(
            event_type="startup_error",
            details={"error": str(e)}
        )
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Plant Disease Detection API...")
    await task_manager.shutdown()
    await security_audit.close()
    await response_cache.close()
    logger.info("✅ Shutdown complete")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Plant Disease Detection API",
    description="🌱 AI-powered plant disease detection and treatment recommendation system",
    version="2.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
    debug=settings.debug
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(predict_router, prefix="/api", tags=["Prediction"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "model_info": "/api/model/info"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )