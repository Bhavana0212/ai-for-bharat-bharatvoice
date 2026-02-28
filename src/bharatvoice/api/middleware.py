"""
FastAPI middleware for BharatVoice Assistant.

This module provides middleware for performance monitoring, error handling,
request tracking, and other cross-cutting concerns.
"""

import time
import uuid
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.performance_monitor import (
    get_performance_monitor,
    QueryComplexity,
    RequestPriority
)
from bharatvoice.utils.error_handler import (
    get_error_handler,
    ErrorCode,
    ErrorSeverity,
    LocalizedError
)
from bharatvoice.utils.monitoring import track_request_metrics


logger = structlog.get_logger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and request tracking.
    """
    
    def __init__(self, app, enable_monitoring: bool = True):
        super().__init__(app)
        self.enable_monitoring = enable_monitoring
        self.performance_monitor = get_performance_monitor() if enable_monitoring else None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with performance monitoring.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract language from headers or query params
        language = self._extract_language(request)
        request.state.language = language
        
        # Determine query complexity
        complexity = self._determine_complexity(request)
        request.state.complexity = complexity
        
        # Start timing
        start_time = time.time()
        request.state.start_time = start_time
        
        # Log request start
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            language=language.value,
            complexity=complexity.value,
            client_ip=request.client.host if request.client else None
        )
        
        try:
            # Check system health if monitoring enabled
            if self.performance_monitor and not self.performance_monitor.is_system_healthy():
                logger.warning(
                    "System under high load",
                    request_id=request_id,
                    active_requests=self.performance_monitor.load_balancer.get_active_count()
                )
            
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Record performance metrics
            if self.performance_monitor:
                self.performance_monitor.record_performance(
                    response_time=response_time,
                    query_complexity=complexity,
                    language=language,
                    success=response.status_code < 400
                )
            
            # Track Prometheus metrics
            track_request_metrics(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=response_time
            )
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Language"] = language.value
            response.headers["X-Complexity"] = complexity.value
            
            # Log request completion
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                response_time=response_time,
                language=language.value,
                complexity=complexity.value
            )
            
            return response
            
        except Exception as e:
            # Calculate response time for failed requests
            response_time = time.time() - start_time
            
            # Record performance metrics for errors
            if self.performance_monitor:
                self.performance_monitor.record_performance(
                    response_time=response_time,
                    query_complexity=complexity,
                    language=language,
                    success=False,
                    error_type=type(e).__name__
                )
            
            # Log error
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                response_time=response_time,
                error=str(e),
                exc_info=e
            )
            
            raise
    
    def _extract_language(self, request: Request) -> LanguageCode:
        """
        Extract language preference from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Detected or default language
        """
        # Check query parameter
        lang_param = request.query_params.get("lang")
        if lang_param:
            try:
                return LanguageCode(lang_param)
            except ValueError:
                pass
        
        # Check header
        lang_header = request.headers.get("Accept-Language")
        if lang_header:
            # Simple language detection from Accept-Language header
            lang_code = lang_header.split(",")[0].split(";")[0].strip()
            
            # Map common language codes
            lang_mapping = {
                "hi": LanguageCode.HINDI,
                "hi-IN": LanguageCode.HINDI,
                "en": LanguageCode.ENGLISH_INDIA,
                "en-IN": LanguageCode.ENGLISH_INDIA,
                "ta": LanguageCode.TAMIL,
                "ta-IN": LanguageCode.TAMIL,
                "te": LanguageCode.TELUGU,
                "te-IN": LanguageCode.TELUGU,
                "bn": LanguageCode.BENGALI,
                "bn-IN": LanguageCode.BENGALI,
                "mr": LanguageCode.MARATHI,
                "mr-IN": LanguageCode.MARATHI,
                "gu": LanguageCode.GUJARATI,
                "gu-IN": LanguageCode.GUJARATI,
                "kn": LanguageCode.KANNADA,
                "kn-IN": LanguageCode.KANNADA,
                "ml": LanguageCode.MALAYALAM,
                "ml-IN": LanguageCode.MALAYALAM,
                "pa": LanguageCode.PUNJABI,
                "pa-IN": LanguageCode.PUNJABI,
                "or": LanguageCode.ODIA,
                "or-IN": LanguageCode.ODIA,
            }
            
            if lang_code in lang_mapping:
                return lang_mapping[lang_code]
        
        # Default to English (India)
        return LanguageCode.ENGLISH_INDIA
    
    def _determine_complexity(self, request: Request) -> QueryComplexity:
        """
        Determine query complexity based on request characteristics.
        
        Args:
            request: HTTP request
            
        Returns:
            Query complexity level
        """
        path = request.url.path
        
        # Voice processing endpoints are typically complex
        if "/voice/" in path:
            if "translate" in path or "code-switch" in path:
                return QueryComplexity.MULTILINGUAL
            else:
                return QueryComplexity.COMPLEX
        
        # Context and external service calls are complex
        if "/context/" in path or "/external/" in path:
            return QueryComplexity.COMPLEX
        
        # Health checks and simple endpoints
        if path in ["/health", "/", "/metrics"]:
            return QueryComplexity.SIMPLE
        
        # Authentication endpoints are typically simple
        if "/auth/" in path:
            return QueryComplexity.SIMPLE
        
        # Default to complex for unknown endpoints
        return QueryComplexity.COMPLEX


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling with localization.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.error_handler = get_error_handler()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with error handling.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response with error handling
        """
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            language = getattr(request.state, "language", LanguageCode.ENGLISH_INDIA)
            
            # Map HTTP status codes to error codes
            error_code_mapping = {
                400: ErrorCode.INPUT_INVALID_FORMAT,
                401: ErrorCode.AUTH_INVALID_CREDENTIALS,
                403: ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
                404: ErrorCode.INPUT_UNSUPPORTED_OPERATION,
                429: ErrorCode.EXT_API_RATE_LIMITED,
                500: ErrorCode.SYS_INTERNAL_ERROR,
                502: ErrorCode.EXT_SERVICE_UNAVAILABLE,
                503: ErrorCode.SYS_RESOURCE_EXHAUSTED,
                504: ErrorCode.EXT_TIMEOUT,
            }
            
            error_code = error_code_mapping.get(e.status_code, ErrorCode.SYS_INTERNAL_ERROR)
            
            # Create localized error
            localized_error = self.error_handler.handle_error(
                error_code,
                language=language,
                severity=ErrorSeverity.MEDIUM if e.status_code < 500 else ErrorSeverity.HIGH,
                context={"original_detail": e.detail, "status_code": e.status_code}
            )
            
            # Create error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error": {
                        "code": localized_error.error_code.value,
                        "message": localized_error.message,
                        "severity": localized_error.severity.value
                    },
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except LocalizedError as e:
            # Handle already localized errors
            from fastapi.responses import JSONResponse
            
            status_code = self._get_status_code_for_error(e.error_code)
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False,
                    "error": {
                        "code": e.error_code.value,
                        "message": e.message,
                        "severity": e.severity.value
                    },
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            language = getattr(request.state, "language", LanguageCode.ENGLISH_INDIA)
            request_id = getattr(request.state, "request_id", None)
            
            # Create localized error
            localized_error = self.error_handler.handle_error(
                e,
                language=language,
                severity=ErrorSeverity.HIGH,
                context={"request_id": request_id}
            )
            
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": localized_error.error_code.value,
                        "message": localized_error.message,
                        "severity": localized_error.severity.value
                    },
                    "request_id": request_id
                }
            )
    
    def _get_status_code_for_error(self, error_code: ErrorCode) -> int:
        """
        Map error codes to HTTP status codes.
        
        Args:
            error_code: Error code
            
        Returns:
            HTTP status code
        """
        mapping = {
            # Authentication errors
            ErrorCode.AUTH_INVALID_CREDENTIALS: 401,
            ErrorCode.AUTH_TOKEN_EXPIRED: 401,
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: 403,
            ErrorCode.AUTH_MFA_REQUIRED: 401,
            
            # Input errors
            ErrorCode.INPUT_INVALID_FORMAT: 400,
            ErrorCode.INPUT_MISSING_REQUIRED: 400,
            ErrorCode.INPUT_VALUE_OUT_OF_RANGE: 400,
            ErrorCode.INPUT_UNSUPPORTED_OPERATION: 404,
            
            # External service errors
            ErrorCode.EXT_SERVICE_UNAVAILABLE: 502,
            ErrorCode.EXT_API_RATE_LIMITED: 429,
            ErrorCode.EXT_INVALID_RESPONSE: 502,
            ErrorCode.EXT_TIMEOUT: 504,
            
            # System errors
            ErrorCode.SYS_RESOURCE_EXHAUSTED: 503,
            ErrorCode.SYS_PERFORMANCE_DEGRADED: 503,
            
            # Network errors
            ErrorCode.NET_CONNECTION_FAILED: 502,
            ErrorCode.NET_TIMEOUT: 504,
        }
        
        return mapping.get(error_code, 500)


class RequestQueueMiddleware(BaseHTTPMiddleware):
    """
    Middleware for intelligent request queuing under high load.
    """
    
    def __init__(self, app, enable_queuing: bool = True):
        super().__init__(app)
        self.enable_queuing = enable_queuing
        self.performance_monitor = get_performance_monitor() if enable_queuing else None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with intelligent queuing.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        if not self.enable_queuing or not self.performance_monitor:
            return await call_next(request)
        
        # Check if system is under high load
        if not self.performance_monitor.load_balancer.is_high_load():
            # Normal processing
            return await call_next(request)
        
        # Determine request priority
        priority = self._determine_priority(request)
        
        # For high priority requests, process immediately
        if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
            return await call_next(request)
        
        # Queue lower priority requests
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        language = getattr(request.state, "language", LanguageCode.ENGLISH_INDIA)
        complexity = getattr(request.state, "complexity", QueryComplexity.COMPLEX)
        
        logger.info(
            "Queuing request due to high load",
            request_id=request_id,
            priority=priority.name,
            active_requests=self.performance_monitor.load_balancer.get_active_count()
        )
        
        # Create queued request wrapper
        async def queued_call():
            return await call_next(request)
        
        # Queue the request
        queued = await self.performance_monitor.queue_request(
            request_id=request_id,
            callback=queued_call,
            complexity=complexity,
            language=language,
            priority=priority
        )
        
        if not queued:
            # Queue is full, return error
            from fastapi.responses import JSONResponse
            error_handler = get_error_handler()
            localized_error = error_handler.handle_error(
                ErrorCode.SYS_RESOURCE_EXHAUSTED,
                language=language
            )
            
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": {
                        "code": localized_error.error_code.value,
                        "message": localized_error.message,
                        "severity": localized_error.severity.value
                    },
                    "request_id": request_id
                }
            )
        
        # Wait for request to be processed
        # In a real implementation, this would use a different mechanism
        # For now, we'll process immediately but with load balancing
        async with self.performance_monitor.load_balancer.acquire_slot():
            return await call_next(request)
    
    def _determine_priority(self, request: Request) -> RequestPriority:
        """
        Determine request priority based on characteristics.
        
        Args:
            request: HTTP request
            
        Returns:
            Request priority level
        """
        path = request.url.path
        
        # Health checks are critical
        if path in ["/health", "/metrics"]:
            return RequestPriority.CRITICAL
        
        # Authentication is high priority
        if "/auth/" in path:
            return RequestPriority.HIGH
        
        # Voice processing is normal priority
        if "/voice/" in path:
            return RequestPriority.NORMAL
        
        # Other requests are low priority
        return RequestPriority.LOW