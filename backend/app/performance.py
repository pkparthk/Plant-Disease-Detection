"""
Performance optimization utilities for Plant Disease Detection API
"""
import asyncio
import time
import gc
import psutil
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import asynccontextmanager
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    requests_count: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    errors_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class PerformanceMonitor:
    """Monitor and track application performance metrics."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.request_times: List[float] = []
        self.max_stored_times = 1000  # Keep last 1000 request times
        self._lock = threading.Lock()
        
    def record_request(self, response_time: float, success: bool = True):
        """Record a request's performance metrics."""
        with self._lock:
            self.metrics.requests_count += 1
            self.metrics.total_response_time += response_time
            
            if success:
                self.request_times.append(response_time)
                if len(self.request_times) > self.max_stored_times:
                    self.request_times.pop(0)
                
                self.metrics.min_response_time = min(self.metrics.min_response_time, response_time)
                self.metrics.max_response_time = max(self.metrics.max_response_time, response_time)
                self.metrics.avg_response_time = self.metrics.total_response_time / self.metrics.requests_count
            else:
                self.metrics.errors_count += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        with self._lock:
            self.metrics.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        with self._lock:
            self.metrics.cache_misses += 1
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.request_times:
            return {}
        
        times = sorted(self.request_times)
        length = len(times)
        
        return {
            "p50": times[int(length * 0.5)],
            "p75": times[int(length * 0.75)],
            "p90": times[int(length * 0.90)],
            "p95": times[int(length * 0.95)],
            "p99": times[int(length * 0.99)],
        }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "threads_count": process.num_threads(),
                "open_files": len(process.open_files()),
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        percentiles = self.get_percentiles()
        system_metrics = self.get_system_metrics()
        
        cache_total = self.metrics.cache_hits + self.metrics.cache_misses
        cache_hit_rate = (self.metrics.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        return {
            "requests": {
                "total": self.metrics.requests_count,
                "errors": self.metrics.errors_count,
                "error_rate": (self.metrics.errors_count / self.metrics.requests_count * 100) if self.metrics.requests_count > 0 else 0,
            },
            "response_times": {
                "average": self.metrics.avg_response_time,
                "min": self.metrics.min_response_time if self.metrics.min_response_time != float('inf') else 0,
                "max": self.metrics.max_response_time,
                **percentiles
            },
            "cache": {
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "hit_rate": cache_hit_rate
            },
            "system": system_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }


class ModelOptimizer:
    """Optimize model performance and memory usage."""
    
    def __init__(self):
        self.model_cache = {}
        self.last_gc_time = time.time()
        self.gc_interval = 300  # Run garbage collection every 5 minutes
        
    @asynccontextmanager
    async def optimized_inference(self):
        """Context manager for optimized model inference."""
        start_time = time.time()
        
        try:
            # Pre-inference optimizations
            await self._pre_inference_cleanup()
            yield
        finally:
            # Post-inference optimizations
            await self._post_inference_cleanup()
            
            # Log inference time
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.3f}s")
    
    async def _pre_inference_cleanup(self):
        """Cleanup before inference."""
        # Force garbage collection if needed
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            gc.collect()
            self.last_gc_time = current_time
    
    async def _post_inference_cleanup(self):
        """Cleanup after inference."""
        # Clear any temporary variables
        pass
    
    def optimize_image_processing(self, image_array: np.ndarray) -> np.ndarray:
        """Optimize image processing for better performance."""
        # Use memory-efficient operations
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32, copy=False)
        
        return image_array
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
        }


class AsyncTaskManager:
    """Manage asynchronous tasks and prevent resource exhaustion."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks = set()
        
    async def run_task(self, coro):
        """Run a coroutine with concurrency control."""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.active_tasks.add(task)
            
            try:
                result = await task
                return result
            finally:
                self.active_tasks.discard(task)
    
    async def shutdown(self):
        """Shutdown and wait for all active tasks."""
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)


class RequestBatcher:
    """Batch requests for more efficient processing."""
    
    def __init__(self, batch_size: int = 5, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.request_queue = asyncio.Queue()
        self._processing_task = None
        
    async def add_request(self, request_data: Any) -> Any:
        """Add request to batch and wait for result."""
        future = asyncio.Future()
        await self.request_queue.put((request_data, future))
        
        # Start processing task if not running
        if not self._processing_task or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_batches())
        
        return await future
    
    async def _process_batches(self):
        """Process batched requests."""
        while True:
            batch = []
            futures = []
            
            try:
                # Collect batch
                end_time = time.time() + self.timeout
                while len(batch) < self.batch_size and time.time() < end_time:
                    try:
                        request_data, future = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=max(0.1, end_time - time.time())
                        )
                        batch.append(request_data)
                        futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    continue
                
                # Process batch
                results = await self._process_batch(batch)
                
                # Return results
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
                        
            except Exception as e:
                # Handle errors
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
    
    async def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Override this method to implement batch processing logic."""
        # Default: process each item individually
        results = []
        for item in batch:
            results.append(await self._process_single_item(item))
        return results
    
    async def _process_single_item(self, item: Any) -> Any:
        """Override this method to implement single item processing."""
        return item


def performance_timer(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class ImageProcessingOptimizer:
    """Optimize image processing operations."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    async def process_image_async(self, image_bytes: bytes, target_size: tuple) -> np.ndarray:
        """Process image asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._process_image_sync,
            image_bytes,
            target_size
        )
    
    def _process_image_sync(self, image_bytes: bytes, target_size: tuple) -> np.ndarray:
        """Synchronous image processing."""
        from PIL import Image
        import io
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Optimize: Use thumbnail for resizing (more efficient)
        if image.size != target_size:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with exact target size
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array efficiently
        return np.array(image, dtype=np.float32)
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


# Global instances
performance_monitor = PerformanceMonitor()
model_optimizer = ModelOptimizer()
image_processor = ImageProcessingOptimizer()


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return performance_monitor.get_all_metrics()


async def optimize_memory_usage():
    """Optimize memory usage across the application."""
    # Force garbage collection
    gc.collect()
    
    # Get memory stats
    memory_stats = model_optimizer.get_memory_usage()
    
    # Log memory usage
    logger.info(f"Memory optimization complete. Usage: {memory_stats['rss_mb']:.1f}MB")
    
    return memory_stats