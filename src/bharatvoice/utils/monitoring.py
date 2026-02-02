"""
Monitoring and metrics collection for BharatVoice Assistant.

This module provides Prometheus metrics collection, health checks,
and performance monitoring for the voice assistant system.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    start_http_server,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
import structlog

from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)

# Global metrics registry
REGISTRY = CollectorRegistry()

# Application info
APP_INFO = Info(
    'bharatvoice_app_info',
    'BharatVoice application information',
    registry=REGISTRY
)

# Request metrics
REQUEST_COUNT = Counter(
    'bharatvoice_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'bharatvoice_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Voice processing metrics
SPEECH_RECOGNITION_COUNT = Counter(
    'bharatvoice_speech_recognition_total',
    'Total speech recognition requests',
    ['language', 'success'],
    registry=REGISTRY
)

SPEECH_RECOGNITION_DURATION = Histogram(
    'bharatvoice_speech_recognition_duration_seconds',
    'Speech recognition processing time',
    ['language'],
    registry=REGISTRY
)

SPEECH_RECOGNITION_ACCURACY = Histogram(
    'bharatvoice_speech_recognition_accuracy',
    'Speech recognition accuracy scores',
    ['language'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    registry=REGISTRY
)

TEXT_TO_SPEECH_COUNT = Counter(
    'bharatvoice_text_to_speech_total',
    'Total text-to-speech requests',
    ['language', 'accent'],
    registry=REGISTRY
)

TEXT_TO_SPEECH_DURATION = Histogram(
    'bharatvoice_text_to_speech_duration_seconds',
    'Text-to-speech processing time',
    ['language'],
    registry=REGISTRY
)

# Language processing metrics
CODE_SWITCHING_DETECTED = Counter(
    'bharatvoice_code_switching_detected_total',
    'Code-switching detection events',
    ['from_language', 'to_language'],
    registry=REGISTRY
)

TRANSLATION_COUNT = Counter(
    'bharatvoice_translation_total',
    'Translation requests',
    ['source_language', 'target_language', 'success'],
    registry=REGISTRY
)

# Session and user metrics
ACTIVE_SESSIONS = Gauge(
    'bharatvoice_active_sessions',
    'Number of active conversation sessions',
    registry=REGISTRY
)

USER_INTERACTIONS = Counter(
    'bharatvoice_user_interactions_total',
    'Total user interactions',
    ['language', 'intent'],
    registry=REGISTRY
)

# System metrics
SYSTEM_HEALTH = Gauge(
    'bharatvoice_system_health',
    'System health status (1=healthy, 0=unhealthy)',
    ['service'],
    registry=REGISTRY
)

SERVICE_RESPONSE_TIME = Histogram(
    'bharatvoice_service_response_time_seconds',
    'Service response time',
    ['service'],
    registry=REGISTRY
)

# External service metrics
EXTERNAL_SERVICE_CALLS = Counter(
    'bharatvoice_external_service_calls_total',
    'External service API calls',
    ['service', 'status'],
    registry=REGISTRY
)

EXTERNAL_SERVICE_DURATION = Histogram(
    'bharatvoice_external_service_duration_seconds',
    'External service call duration',
    ['service'],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    'bharatvoice_errors_total',
    'Total errors',
    ['error_type', 'service'],
    registry=REGISTRY
)


def setup_metrics() -> None:
    """Initialize metrics collection and set application info."""
    APP_INFO.info({
        'version': '0.1.0',
        'environment': 'development',
        'supported_languages': '11',
        'features': 'multilingual,code_switching,cultural_context'
    })
    
    logger.info("Metrics collection initialized")


def start_metrics_server(port: int = 8001) -> None:
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: Port to serve metrics on
    """
    try:
        start_http_server(port, registry=REGISTRY)
        logger.info("Metrics server started", port=port)
    except Exception as e:
        logger.error("Failed to start metrics server", exc_info=e)


def get_metrics() -> str:
    """
    Get metrics in Prometheus format.
    
    Returns:
        Metrics data in Prometheus format
    """
    return generate_latest(REGISTRY).decode('utf-8')


def track_request_metrics(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """
    Track HTTP request metrics.
    
    Args:
        method: HTTP method
        endpoint: Request endpoint
        status_code: Response status code
        duration: Request duration in seconds
    """
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_speech_recognition(
    language: LanguageCode,
    duration: float,
    accuracy: float,
    success: bool = True
) -> None:
    """
    Track speech recognition metrics.
    
    Args:
        language: Recognition language
        duration: Processing duration in seconds
        accuracy: Recognition accuracy score
        success: Whether recognition succeeded
    """
    SPEECH_RECOGNITION_COUNT.labels(
        language=language.value,
        success=str(success).lower()
    ).inc()
    
    if success:
        SPEECH_RECOGNITION_DURATION.labels(
            language=language.value
        ).observe(duration)
        
        SPEECH_RECOGNITION_ACCURACY.labels(
            language=language.value
        ).observe(accuracy)


def track_text_to_speech(
    language: LanguageCode,
    accent: str,
    duration: float
) -> None:
    """
    Track text-to-speech metrics.
    
    Args:
        language: Synthesis language
        accent: Voice accent
        duration: Processing duration in seconds
    """
    TEXT_TO_SPEECH_COUNT.labels(
        language=language.value,
        accent=accent
    ).inc()
    
    TEXT_TO_SPEECH_DURATION.labels(
        language=language.value
    ).observe(duration)


def track_code_switching(from_lang: LanguageCode, to_lang: LanguageCode) -> None:
    """
    Track code-switching detection.
    
    Args:
        from_lang: Source language
        to_lang: Target language
    """
    CODE_SWITCHING_DETECTED.labels(
        from_language=from_lang.value,
        to_language=to_lang.value
    ).inc()


def track_translation(
    source_lang: LanguageCode,
    target_lang: LanguageCode,
    success: bool = True
) -> None:
    """
    Track translation requests.
    
    Args:
        source_lang: Source language
        target_lang: Target language
        success: Whether translation succeeded
    """
    TRANSLATION_COUNT.labels(
        source_language=source_lang.value,
        target_language=target_lang.value,
        success=str(success).lower()
    ).inc()


def track_user_interaction(language: LanguageCode, intent: str) -> None:
    """
    Track user interaction.
    
    Args:
        language: Interaction language
        intent: Detected intent
    """
    USER_INTERACTIONS.labels(
        language=language.value,
        intent=intent
    ).inc()


def track_external_service_call(
    service: str,
    duration: float,
    success: bool = True
) -> None:
    """
    Track external service API call.
    
    Args:
        service: Service name
        duration: Call duration in seconds
        success: Whether call succeeded
    """
    status = "success" if success else "error"
    
    EXTERNAL_SERVICE_CALLS.labels(
        service=service,
        status=status
    ).inc()
    
    EXTERNAL_SERVICE_DURATION.labels(
        service=service
    ).observe(duration)


def track_error(error_type: str, service: str) -> None:
    """
    Track error occurrence.
    
    Args:
        error_type: Type of error
        service: Service where error occurred
    """
    ERROR_COUNT.labels(
        error_type=error_type,
        service=service
    ).inc()


def update_system_health(service: str, healthy: bool) -> None:
    """
    Update system health status.
    
    Args:
        service: Service name
        healthy: Whether service is healthy
    """
    SYSTEM_HEALTH.labels(service=service).set(1 if healthy else 0)


def update_active_sessions(count: int) -> None:
    """
    Update active sessions count.
    
    Args:
        count: Number of active sessions
    """
    ACTIVE_SESSIONS.set(count)


def monitor_performance(operation: str, service: str = "default"):
    """
    Decorator to monitor function performance.
    
    Args:
        operation: Operation name
        service: Service name
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                SERVICE_RESPONSE_TIME.labels(service=service).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                SERVICE_RESPONSE_TIME.labels(service=service).observe(duration)
                track_error(type(e).__name__, service)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                SERVICE_RESPONSE_TIME.labels(service=service).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                SERVICE_RESPONSE_TIME.labels(service=service).observe(duration)
                track_error(type(e).__name__, service)
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class MetricsCollector:
    """
    Centralized metrics collection class.
    """
    
    def __init__(self):
        self.logger = logger
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary of system metrics
        """
        # TODO: Implement actual system metrics collection
        return {
            "active_sessions": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0.0,
        }
    
    def collect_voice_metrics(self) -> Dict[str, Any]:
        """
        Collect voice processing metrics.
        
        Returns:
            Dictionary of voice processing metrics
        """
        # TODO: Implement actual voice metrics collection
        return {
            "speech_recognition_requests": 0,
            "average_recognition_accuracy": 0.0,
            "tts_requests": 0,
            "average_processing_time": 0.0,
            "code_switching_events": 0,
        }
    
    def collect_language_metrics(self) -> Dict[str, Any]:
        """
        Collect language processing metrics.
        
        Returns:
            Dictionary of language processing metrics
        """
        # TODO: Implement actual language metrics collection
        return {
            "supported_languages": 11,
            "translation_requests": 0,
            "language_detection_accuracy": 0.0,
            "cultural_context_matches": 0,
        }