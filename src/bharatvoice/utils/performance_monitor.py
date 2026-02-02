"""
Performance monitoring system for BharatVoice Assistant.

This module provides comprehensive performance monitoring including response time tracking,
load balancing, request queuing, and localized error handling for the voice assistant.
"""

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Deque
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.monitoring import (
    track_request_metrics,
    track_error,
    SERVICE_RESPONSE_TIME,
    REQUEST_DURATION,
    update_system_health
)


logger = structlog.get_logger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for performance monitoring."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTILINGUAL = "multilingual"


class RequestPriority(Enum):
    """Request priority levels for intelligent queuing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    response_time: float
    query_complexity: QueryComplexity
    language: LanguageCode
    success: bool
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueuedRequest:
    """Queued request data structure."""
    request_id: str
    priority: RequestPriority
    complexity: QueryComplexity
    language: LanguageCode
    timestamp: float
    callback: Callable
    args: tuple
    kwargs: dict


class PerformanceTargets:
    """Performance targets for different query types."""
    
    SIMPLE_QUERY_TARGET = 2.0  # seconds
    COMPLEX_QUERY_TARGET = 5.0  # seconds
    MULTILINGUAL_QUERY_TARGET = 5.0  # seconds
    
    # Error rate thresholds
    MAX_ERROR_RATE = 0.05  # 5%
    
    # Load thresholds
    MAX_CONCURRENT_REQUESTS = 100
    HIGH_LOAD_THRESHOLD = 80
    
    # System resource thresholds
    MAX_CPU_USAGE = 80.0  # percent
    MAX_MEMORY_USAGE = 85.0  # percent


class RequestQueue:
    """Intelligent request queue with priority handling."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues = {
            RequestPriority.CRITICAL: deque(),
            RequestPriority.HIGH: deque(),
            RequestPriority.NORMAL: deque(),
            RequestPriority.LOW: deque(),
        }
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._total_size = 0
    
    def enqueue(self, request: QueuedRequest) -> bool:
        """
        Add request to queue with priority handling.
        
        Args:
            request: Request to queue
            
        Returns:
            True if request was queued, False if queue is full
        """
        with self._condition:
            if self._total_size >= self.max_size:
                logger.warning(
                    "Request queue full, dropping request",
                    request_id=request.request_id,
                    queue_size=self._total_size
                )
                return False
            
            self._queues[request.priority].append(request)
            self._total_size += 1
            self._condition.notify()
            
            logger.debug(
                "Request queued",
                request_id=request.request_id,
                priority=request.priority.name,
                queue_size=self._total_size
            )
            return True
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[QueuedRequest]:
        """
        Get next request from queue based on priority.
        
        Args:
            timeout: Maximum time to wait for request
            
        Returns:
            Next request or None if timeout
        """
        with self._condition:
            end_time = time.time() + timeout if timeout else None
            
            while self._total_size == 0:
                if timeout is not None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            
            # Get request with highest priority
            for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH, 
                           RequestPriority.NORMAL, RequestPriority.LOW]:
                if self._queues[priority]:
                    request = self._queues[priority].popleft()
                    self._total_size -= 1
                    return request
            
            return None
    
    def size(self) -> int:
        """Get total queue size."""
        with self._lock:
            return self._total_size
    
    def clear(self) -> None:
        """Clear all queues."""
        with self._condition:
            for queue in self._queues.values():
                queue.clear()
            self._total_size = 0


class LoadBalancer:
    """Load balancer for managing concurrent requests."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self._active_requests = 0
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
    
    @asynccontextmanager
    async def acquire_slot(self):
        """Acquire a processing slot for load balancing."""
        # Try to acquire semaphore
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "Load balancer at capacity",
                active_requests=self._active_requests,
                max_concurrent=self.max_concurrent
            )
            # Wait for slot to become available
            await asyncio.get_event_loop().run_in_executor(
                None, self._semaphore.acquire
            )
        
        with self._lock:
            self._active_requests += 1
        
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
            self._semaphore.release()
    
    def get_active_count(self) -> int:
        """Get number of active requests."""
        with self._lock:
            return self._active_requests
    
    def is_high_load(self) -> bool:
        """Check if system is under high load."""
        with self._lock:
            return self._active_requests >= PerformanceTargets.HIGH_LOAD_THRESHOLD


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self._cpu_history: Deque[float] = deque(maxlen=60)  # 1 minute history
        self._memory_history: Deque[float] = deque(maxlen=60)
        self._last_update = 0
        self._update_interval = 1.0  # seconds
    
    def update_metrics(self) -> None:
        """Update system metrics if needed."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return
        
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            self._cpu_history.append(cpu_percent)
            self._memory_history.append(memory_percent)
            self._last_update = now
            
            # Update health status
            cpu_healthy = cpu_percent < PerformanceTargets.MAX_CPU_USAGE
            memory_healthy = memory_percent < PerformanceTargets.MAX_MEMORY_USAGE
            
            update_system_health("cpu", cpu_healthy)
            update_system_health("memory", memory_healthy)
            
        except Exception as e:
            logger.error("Failed to update system metrics", exc_info=e)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        self.update_metrics()
        return self._cpu_history[-1] if self._cpu_history else 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        self.update_metrics()
        return self._memory_history[-1] if self._memory_history else 0.0
    
    def get_average_cpu(self, minutes: int = 5) -> float:
        """Get average CPU usage over specified minutes."""
        self.update_metrics()
        samples = min(minutes * 60, len(self._cpu_history))
        if samples == 0:
            return 0.0
        return sum(list(self._cpu_history)[-samples:]) / samples
    
    def get_average_memory(self, minutes: int = 5) -> float:
        """Get average memory usage over specified minutes."""
        self.update_metrics()
        samples = min(minutes * 60, len(self._memory_history))
        if samples == 0:
            return 0.0
        return sum(list(self._memory_history)[-samples:]) / samples


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=10000)
        self.request_queue = RequestQueue()
        self.load_balancer = LoadBalancer()
        self.system_monitor = SystemMonitor()
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._response_times = defaultdict(list)
        self._error_counts = defaultdict(int)
        self._total_requests = defaultdict(int)
    
    async def start(self) -> None:
        """Start the performance monitoring system."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("Performance monitoring system started")
    
    async def stop(self) -> None:
        """Stop the performance monitoring system."""
        if not self._running:
            return
        
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        self._executor.shutdown(wait=True)
        logger.info("Performance monitoring system stopped")
    
    async def _process_queue(self) -> None:
        """Process queued requests."""
        while self._running:
            try:
                request = await asyncio.get_event_loop().run_in_executor(
                    None, self.request_queue.dequeue, 1.0
                )
                
                if request:
                    await self._execute_request(request)
                    
            except Exception as e:
                logger.error("Error processing queued request", exc_info=e)
    
    async def _execute_request(self, request: QueuedRequest) -> None:
        """Execute a queued request."""
        start_time = time.time()
        
        try:
            async with self.load_balancer.acquire_slot():
                result = await request.callback(*request.args, **request.kwargs)
                
                duration = time.time() - start_time
                self.record_performance(
                    response_time=duration,
                    query_complexity=request.complexity,
                    language=request.language,
                    success=True
                )
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.record_performance(
                response_time=duration,
                query_complexity=request.complexity,
                language=request.language,
                success=False,
                error_type=type(e).__name__
            )
            raise
    
    def record_performance(
        self,
        response_time: float,
        query_complexity: QueryComplexity,
        language: LanguageCode,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """
        Record performance metrics.
        
        Args:
            response_time: Response time in seconds
            query_complexity: Query complexity level
            language: Language used
            success: Whether request succeeded
            error_type: Error type if failed
        """
        metrics = PerformanceMetrics(
            response_time=response_time,
            query_complexity=query_complexity,
            language=language,
            success=success,
            error_type=error_type
        )
        
        self.metrics_history.append(metrics)
        
        # Update tracking
        key = (query_complexity, language)
        self._response_times[key].append(response_time)
        self._total_requests[key] += 1
        
        if not success:
            self._error_counts[key] += 1
            track_error(error_type or "unknown", "performance_monitor")
        
        # Check performance targets
        self._check_performance_targets(metrics)
        
        # Update Prometheus metrics
        SERVICE_RESPONSE_TIME.labels(
            service=f"{query_complexity.value}_{language.value}"
        ).observe(response_time)
    
    def _check_performance_targets(self, metrics: PerformanceMetrics) -> None:
        """Check if performance targets are being met."""
        target_time = self._get_target_time(metrics.query_complexity)
        
        if metrics.response_time > target_time:
            logger.warning(
                "Performance target exceeded",
                response_time=metrics.response_time,
                target_time=target_time,
                complexity=metrics.query_complexity.value,
                language=metrics.language.value
            )
    
    def _get_target_time(self, complexity: QueryComplexity) -> float:
        """Get target response time for query complexity."""
        if complexity == QueryComplexity.SIMPLE:
            return PerformanceTargets.SIMPLE_QUERY_TARGET
        elif complexity == QueryComplexity.COMPLEX:
            return PerformanceTargets.COMPLEX_QUERY_TARGET
        else:  # MULTILINGUAL
            return PerformanceTargets.MULTILINGUAL_QUERY_TARGET
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {
            "active_requests": self.load_balancer.get_active_count(),
            "queue_size": self.request_queue.size(),
            "cpu_usage": self.system_monitor.get_cpu_usage(),
            "memory_usage": self.system_monitor.get_memory_usage(),
            "total_metrics": len(self.metrics_history),
        }
        
        # Calculate averages by complexity
        for complexity in QueryComplexity:
            complexity_metrics = [
                m for m in self.metrics_history 
                if m.query_complexity == complexity
            ]
            
            if complexity_metrics:
                avg_response_time = sum(m.response_time for m in complexity_metrics) / len(complexity_metrics)
                error_rate = sum(1 for m in complexity_metrics if not m.success) / len(complexity_metrics)
                
                stats[f"{complexity.value}_avg_response_time"] = avg_response_time
                stats[f"{complexity.value}_error_rate"] = error_rate
                stats[f"{complexity.value}_total_requests"] = len(complexity_metrics)
        
        return stats
    
    async def queue_request(
        self,
        request_id: str,
        callback: Callable,
        complexity: QueryComplexity,
        language: LanguageCode,
        priority: RequestPriority = RequestPriority.NORMAL,
        *args,
        **kwargs
    ) -> bool:
        """
        Queue a request for processing.
        
        Args:
            request_id: Unique request identifier
            callback: Function to execute
            complexity: Query complexity level
            language: Language being processed
            priority: Request priority
            *args: Callback arguments
            **kwargs: Callback keyword arguments
            
        Returns:
            True if request was queued successfully
        """
        request = QueuedRequest(
            request_id=request_id,
            priority=priority,
            complexity=complexity,
            language=language,
            timestamp=time.time(),
            callback=callback,
            args=args,
            kwargs=kwargs
        )
        
        return self.request_queue.enqueue(request)
    
    def is_system_healthy(self) -> bool:
        """Check if system is healthy based on performance metrics."""
        cpu_usage = self.system_monitor.get_cpu_usage()
        memory_usage = self.system_monitor.get_memory_usage()
        active_requests = self.load_balancer.get_active_count()
        
        return (
            cpu_usage < PerformanceTargets.MAX_CPU_USAGE and
            memory_usage < PerformanceTargets.MAX_MEMORY_USAGE and
            active_requests < PerformanceTargets.MAX_CONCURRENT_REQUESTS
        )


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


async def monitor_performance(
    operation: str,
    complexity: QueryComplexity,
    language: LanguageCode
):
    """
    Decorator for monitoring function performance.
    
    Args:
        operation: Operation name
        complexity: Query complexity level
        language: Language being processed
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                async with monitor.load_balancer.acquire_slot():
                    result = await func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    monitor.record_performance(
                        response_time=duration,
                        query_complexity=complexity,
                        language=language,
                        success=True
                    )
                    
                    return result
                    
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_performance(
                    response_time=duration,
                    query_complexity=complexity,
                    language=language,
                    success=False,
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator