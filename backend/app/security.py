"""
Security middleware and utilities for Plant Disease Detection API
"""
import time
import logging
import hashlib
import secrets
import json
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

try:
    import redis
except ImportError:
    redis = None

from .config import settings

logger = logging.getLogger(__name__)

# Redis client (optional)
redis_client = None
if redis and hasattr(settings, 'redis_url'):
    try:
        redis_client = redis.from_url(settings.redis_url)
        # Test connection
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        redis_client = None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # HSTS header for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP header
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware with sliding window."""
    
    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_size = 60  # seconds
        
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        # Check rate limits
        if await self._is_rate_limited(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Window": str(self.window_size),
                    "Retry-After": "60"
                }
            )
        
        # Record request
        await self._record_request(client_ip)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window_size)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP with proxy support."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited using sliding window."""
        if not redis_client:
            return False  # No rate limiting without Redis
        
        try:
            current_time = int(time.time())
            window_start = current_time - self.window_size
            
            # Remove old entries
            await redis_client.zremrangebyscore(f"rate_limit:{client_ip}", 0, window_start)
            
            # Count current requests
            current_requests = await redis_client.zcard(f"rate_limit:{client_ip}")
            
            return current_requests >= self.requests_per_minute
            
        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            return False
    
    async def _record_request(self, client_ip: str):
        """Record a request for rate limiting."""
        if not redis_client:
            return
        
        try:
            current_time = time.time()
            # Add current request with unique identifier
            request_id = f"{current_time}:{secrets.token_hex(8)}"
            await redis_client.zadd(f"rate_limit:{client_ip}", {request_id: current_time})
            
            # Set expiration for cleanup
            await redis_client.expire(f"rate_limit:{client_ip}", self.window_size * 2)
            
        except Exception as e:
            logger.error(f"Failed to record request: {e}")
    
    async def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client."""
        if not redis_client:
            return self.requests_per_minute
        
        try:
            current_requests = await redis_client.zcard(f"rate_limit:{client_ip}")
            return max(0, self.requests_per_minute - current_requests)
        except Exception:
            return self.requests_per_minute


class APIKeyAuth:
    """API key authentication."""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
        # Skip authentication for public endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return True
        
        # Check if API key is required
        if not hasattr(settings, 'api_key') or not settings.api_key:
            return True  # No API key required
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Verify API key
        if not self._verify_api_key(credentials.credentials):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return True
    
    def _verify_api_key(self, provided_key: str) -> bool:
        """Verify API key using constant-time comparison."""
        if not hasattr(settings, 'api_key') or not settings.api_key:
            return True
        
        expected_key = settings.api_key
        return secrets.compare_digest(provided_key, expected_key)


class InputSanitizer:
    """Sanitize and validate inputs."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize uploaded filename."""
        import re
        import os
        
        # Remove path separators and null bytes
        filename = filename.replace('/', '').replace('\\', '').replace('\0', '')
        
        # Remove potentially dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 100:
            name = name[:100]
        
        return f"{name}{ext}".strip()
    
    @staticmethod
    def validate_image_content(image_bytes: bytes) -> bool:
        """Validate image content to prevent malicious uploads."""
        # Check file signatures
        image_signatures = {
            b'\xFF\xD8\xFF': 'jpeg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'RIFF': 'webp',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif'
        }
        
        # Check if file starts with valid image signature
        for signature in image_signatures:
            if image_bytes.startswith(signature):
                return True
        
        return False
    
    @staticmethod
    def hash_file_content(content: bytes) -> str:
        """Generate hash of file content for integrity checking."""
        return hashlib.sha256(content).hexdigest()


class SecurityAuditLogger:
    """Enhanced security audit logging with async support."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        self.redis_client = None
        self.initialized = False
        
        # Create security audit log handler
        if not self.logger.handlers:
            handler = logging.FileHandler("security_audit.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def initialize(self):
        """Initialize the security audit logger."""
        if not redis:
            self.logger.warning("Redis not available for audit logging")
            self.initialized = False
            return
            
        try:
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            self.initialized = True
            self.logger.info("Security audit logger initialized with Redis")
        except Exception as e:
            self.logger.warning(f"Redis not available for audit logging: {e}")
            self.initialized = False
    
    async def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security event with async support."""
        timestamp = datetime.utcnow().isoformat()
        event_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details
        }
        
        # Log to file
        self.logger.info(f"SECURITY_EVENT: {json.dumps(event_data)}")
        
        # Store in Redis if available
        if self.initialized and self.redis_client:
            try:
                await self.redis_client.lpush(
                    "security_events",
                    json.dumps(event_data)
                )
                # Keep only last 10000 events
                await self.redis_client.ltrim("security_events", 0, 9999)
            except Exception as e:
                self.logger.warning(f"Failed to store security event in Redis: {e}")
    
    async def close(self):
        """Close the security audit logger."""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Security audit logger closed")
    
    def log_auth_attempt(self, request: Request, success: bool, reason: str = ""):
        """Log authentication attempt."""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        self.logger.info(
            f"AUTH_ATTEMPT - IP: {client_ip} - "
            f"Success: {success} - Reason: {reason} - "
            f"User-Agent: {user_agent} - Path: {request.url.path}"
        )
    
    def log_rate_limit_exceeded(self, request: Request):
        """Log rate limit exceeded events."""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        self.logger.warning(
            f"RATE_LIMIT_EXCEEDED - IP: {client_ip} - "
            f"User-Agent: {user_agent} - Path: {request.url.path}"
        )
    
    def log_suspicious_activity(self, request: Request, activity_type: str, details: str = ""):
        """Log suspicious activity."""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        self.logger.warning(
            f"SUSPICIOUS_ACTIVITY - Type: {activity_type} - "
            f"IP: {client_ip} - Details: {details} - "
            f"User-Agent: {user_agent} - Path: {request.url.path}"
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class ResponseCache:
    """Cache responses for better performance with async support."""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.redis_client = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the response cache."""
        if not redis:
            logger.warning("Redis not available for caching")
            self.initialized = False
            return
            
        try:
            # Use synchronous Redis client for now
            self.redis_client = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.initialized = True
            logger.info("Response cache initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis not available for caching: {e}")
            self.initialized = False
    
    async def close(self):
        """Close the response cache."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Response cache closed")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        if not self.initialized or not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(f"cache:{key}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        """Set cached response."""
        if not self.initialized or not self.redis_client:
            return
        
        try:
            ttl = ttl or self.default_ttl
            self.redis_client.setex(
                f"cache:{key}",
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def generate_cache_key(self, request: Request, additional_data: str = "") -> str:
        """Generate cache key for request."""
        key_parts = [
            request.url.path,
            str(sorted(request.query_params.items())),
            additional_data
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global instances
api_key_auth = APIKeyAuth()
input_sanitizer = InputSanitizer()
security_audit_logger = SecurityAuditLogger()
response_cache = ResponseCache()


def get_security_headers() -> Dict[str, str]:
    """Get recommended security headers."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
    }


async def verify_request_integrity(request: Request) -> bool:
    """Verify request integrity and detect potential attacks."""
    # Check for common attack patterns
    suspicious_patterns = [
        "script", "javascript:", "vbscript:", "onload", "onerror",
        "../", "..\\", "<script", "</script", "eval(", "document.cookie"
    ]
    
    request_content = str(request.url) + str(request.headers)
    
    for pattern in suspicious_patterns:
        if pattern.lower() in request_content.lower():
            security_audit_logger.log_suspicious_activity(
                request, "POTENTIAL_XSS", f"Pattern: {pattern}"
            )
            return False
    
    return True