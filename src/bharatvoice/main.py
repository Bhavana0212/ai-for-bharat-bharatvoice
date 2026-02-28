<<<<<<< HEAD
"""
Main FastAPI application for BharatVoice Assistant.

This module sets up the FastAPI application with all necessary middleware,
routers, and configuration for the BharatVoice multilingual voice assistant.
Includes intelligent load balancing, comprehensive authentication, request routing,
detailed health checks, and distributed tracing.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
import uvicorn

from bharatvoice.config import get_settings
from bharatvoice.api.health import router as health_router
from bharatvoice.api.voice import router as voice_router
from bharatvoice.api.context import router as context_router
from bharatvoice.api.auth import router as auth_router
from bharatvoice.api.accessibility import router as accessibility_router
from bharatvoice.api.gateway import router as gateway_router, initialize_default_services
from bharatvoice.api.alerts import router as alerts_router
from bharatvoice.api.middleware import (
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware,
    RequestQueueMiddleware
)
from bharatvoice.utils.logging import setup_logging
from bharatvoice.utils.monitoring import setup_metrics, start_metrics_server, track_request_metrics
from bharatvoice.utils.performance_monitor import get_performance_monitor
from bharatvoice.utils.alerting import get_alert_manager, initialize_default_alert_rules
from bharatvoice.services.auth.jwt_manager import JWTManager
from bharatvoice.services.auth.auth_service import AuthService


# Configure structured logging
logger = structlog.get_logger(__name__)

# Global instances for dependency injection
jwt_manager: Optional[JWTManager] = None
auth_service: Optional[AuthService] = None
security = HTTPBearer(auto_error=False)


class LoadBalancer:
    """
    Intelligent load balancer for request distribution and resource management.
    """
    
    def __init__(self):
        self.active_requests = 0
        self.max_concurrent_requests = 100
        self.request_queue = asyncio.Queue(maxsize=200)
        self.service_health = {}
        self.request_history = []
        self.lock = asyncio.Lock()
    
    async def can_accept_request(self) -> bool:
        """Check if system can accept new requests."""
        async with self.lock:
            return self.active_requests < self.max_concurrent_requests
    
    async def acquire_slot(self):
        """Acquire a request processing slot."""
        return RequestSlot(self)
    
    async def get_service_route(self, path: str) -> str:
        """
        Determine optimal service route based on path and load.
        
        Args:
            path: Request path
            
        Returns:
            Service route identifier
        """
        # Route based on path patterns
        if path.startswith("/voice/"):
            return "voice_processing"
        elif path.startswith("/context/"):
            return "context_management"
        elif path.startswith("/auth/"):
            return "authentication"
        elif path.startswith("/health/"):
            return "health_monitoring"
        elif path.startswith("/accessibility/"):
            return "accessibility"
        else:
            return "default"
    
    def update_service_health(self, service: str, healthy: bool, response_time: float):
        """Update service health status."""
        self.service_health[service] = {
            "healthy": healthy,
            "response_time": response_time,
            "last_check": time.time()
        }
    
    def get_load_metrics(self) -> Dict[str, Any]:
        """Get current load balancing metrics."""
        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.max_concurrent_requests,
            "queue_size": self.request_queue.qsize(),
            "service_health": self.service_health,
            "utilization": self.active_requests / self.max_concurrent_requests
        }


class RequestSlot:
    """Context manager for request slot management."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    async def __aenter__(self):
        async with self.load_balancer.lock:
            self.load_balancer.active_requests += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self.load_balancer.lock:
            self.load_balancer.active_requests -= 1


class AuthenticationMiddleware:
    """
    Comprehensive authentication and authorization middleware.
    """
    
    def __init__(self, jwt_manager: JWTManager, auth_service: AuthService):
        self.jwt_manager = jwt_manager
        self.auth_service = auth_service
        self.public_paths = {
            "/", "/health", "/health/live", "/health/ready", 
            "/docs", "/redoc", "/openapi.json", "/metrics",
            "/auth/login", "/auth/register", "/auth/refresh"
        }
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Authenticate request and return user context.
        
        Args:
            request: HTTP request
            
        Returns:
            User context if authenticated, None otherwise
        """
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return None
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        try:
            # Verify and decode token
            payload = await self.jwt_manager.verify_token(token)
            
            # Get user context
            user_id = payload.get("sub")
            if user_id:
                user_context = await self.auth_service.get_user_context(user_id)
                return user_context
            
        except Exception as e:
            logger.warning("Token verification failed", error=str(e))
        
        return None
    
    def requires_authentication(self, path: str) -> bool:
        """Check if path requires authentication."""
        return path not in self.public_paths


class DistributedTracing:
    """
    Distributed tracing for request tracking across services.
    """
    
    def __init__(self):
        self.active_traces = {}
    
    def start_trace(self, request_id: str, operation: str) -> str:
        """Start a new trace."""
        trace_id = f"{request_id}-{operation}-{int(time.time() * 1000)}"
        self.active_traces[trace_id] = {
            "request_id": request_id,
            "operation": operation,
            "start_time": time.time(),
            "spans": []
        }
        return trace_id
    
    def add_span(self, trace_id: str, service: str, operation: str, duration: float):
        """Add a span to an existing trace."""
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["spans"].append({
                "service": service,
                "operation": operation,
                "duration": duration,
                "timestamp": time.time()
            })
    
    def finish_trace(self, trace_id: str) -> Dict[str, Any]:
        """Finish a trace and return trace data."""
        if trace_id in self.active_traces:
            trace_data = self.active_traces.pop(trace_id)
            trace_data["total_duration"] = time.time() - trace_data["start_time"]
            return trace_data
        return {}


# Global instances
load_balancer = LoadBalancer()
distributed_tracing = DistributedTracing()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current authenticated user.
    
    Args:
        request: HTTP request
        credentials: Authorization credentials
        
    Returns:
        User context if authenticated
        
    Raises:
        HTTPException: If authentication is required but fails
    """
    global jwt_manager, auth_service
    
    if not jwt_manager or not auth_service:
        # Services not initialized yet
        return None
    
    auth_middleware = AuthenticationMiddleware(jwt_manager, auth_service)
    
    # Check if authentication is required
    if not auth_middleware.requires_authentication(request.url.path):
        return None
    
    # Authenticate request
    user_context = await auth_middleware.authenticate_request(request)
    
    if user_context is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_context


class GatewayMiddleware:
    """
    Main gateway middleware for request orchestration.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request through gateway."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Generate request ID and start tracing
        request_id = str(uuid.uuid4())
        trace_id = distributed_tracing.start_trace(request_id, "gateway_request")
        
        # Add to request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Check load balancer
        if not await load_balancer.can_accept_request():
            # Return 503 Service Unavailable
            response = JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "System is under high load. Please try again later.",
                    "request_id": request_id
                }
            )
            await response(scope, receive, send)
            return
        
        # Process request with load balancing
        async with load_balancer.acquire_slot():
            # Determine service route
            service_route = await load_balancer.get_service_route(request.url.path)
            request.state.service_route = service_route
            
            # Add tracing span
            start_time = time.time()
            
            try:
                # Process through FastAPI app
                await self.app(scope, receive, send)
                
                # Record successful processing
                duration = time.time() - start_time
                distributed_tracing.add_span(trace_id, service_route, "process_request", duration)
                load_balancer.update_service_health(service_route, True, duration)
                
            except Exception as e:
                # Record failed processing
                duration = time.time() - start_time
                distributed_tracing.add_span(trace_id, service_route, "process_request_error", duration)
                load_balancer.update_service_health(service_route, False, duration)
                
                logger.error(
                    "Gateway request processing failed",
                    request_id=request_id,
                    trace_id=trace_id,
                    service_route=service_route,
                    error=str(e),
                    exc_info=e
                )
                raise
            
            finally:
                # Finish trace
                trace_data = distributed_tracing.finish_trace(trace_id)
                logger.info(
                    "Request trace completed",
                    request_id=request_id,
                    trace_data=trace_data
                )
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    global jwt_manager, auth_service
    
    # Startup
    settings = get_settings()
    logger.info("Starting BharatVoice Assistant", version=settings.version)
    
    # Initialize authentication services
    try:
        jwt_manager = JWTManager(
            secret_key=settings.security.secret_key,
            algorithm=settings.security.algorithm,
            access_token_expire_minutes=settings.security.access_token_expire_minutes
        )
        auth_service = AuthService()
        logger.info("Authentication services initialized")
    except Exception as e:
        logger.error("Failed to initialize authentication services", exc_info=e)
    
    # Initialize monitoring
    if settings.monitoring.enable_metrics:
        setup_metrics()
        start_metrics_server(settings.monitoring.metrics_port)
        logger.info("Metrics collection enabled", port=settings.monitoring.metrics_port)
    
    # Initialize performance monitoring
    performance_monitor = get_performance_monitor()
    await performance_monitor.start()
    logger.info("Performance monitoring started")
    
    # Initialize load balancer
    logger.info("Load balancer initialized", max_concurrent=load_balancer.max_concurrent_requests)
    
    # Initialize distributed tracing
    logger.info("Distributed tracing initialized")
    
    # Initialize default services for gateway
    await initialize_default_services()
    logger.info("Default gateway services initialized")
    
    # Initialize alerting system
    alert_manager = get_alert_manager()
    await alert_manager.start()
    await initialize_default_alert_rules()
    logger.info("Alerting system initialized")
    
    # Initialize services
    # TODO: Initialize database connections, Redis, ML models, etc.
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BharatVoice Assistant")
    
    # Stop performance monitoring
    await performance_monitor.stop()
    logger.info("Performance monitoring stopped")
    
    # Stop alerting system
    alert_manager = get_alert_manager()
    await alert_manager.stop()
    logger.info("Alerting system stopped")
    
    # Clear active traces
    distributed_tracing.active_traces.clear()
    logger.info("Distributed tracing cleaned up")
    
    # TODO: Cleanup resources, close connections, etc.


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with intelligent load balancing,
    comprehensive authentication, request routing, and distributed tracing.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.logging)
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered multilingual voice assistant for the Indian market with intelligent load balancing and distributed tracing",
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add gateway middleware (first in chain)
    app.add_middleware(GatewayMiddleware)
    
    # Add security middleware
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.bharatvoice.ai", "localhost", "127.0.0.1"]
        )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=["*"],
    )
    
    # Add performance monitoring middleware
    if settings.monitoring.enable_metrics:
        app.add_middleware(PerformanceMonitoringMiddleware, enable_monitoring=True)
    
    # Add error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Add request queuing middleware for high load scenarios
    app.add_middleware(RequestQueueMiddleware, enable_queuing=True)
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with structured logging and tracing."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        trace_id = getattr(request.state, "trace_id", None)
        service_route = getattr(request.state, "service_route", "unknown")
        
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            method=request.method,
            url=str(request.url),
            request_id=request_id,
            trace_id=trace_id,
            service_route=service_route
        )
        
        # Add error span to trace
        if trace_id:
            distributed_tracing.add_span(trace_id, service_route, "unhandled_exception", 0.0)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request_id,
                "trace_id": trace_id
            }
        )
    
    # Include API routers with authentication dependencies
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health Check"]
    )
    
    app.include_router(
        auth_router,
        prefix="/auth",
        tags=["Authentication"]
    )
    
    app.include_router(
        voice_router,
        prefix="/voice",
        tags=["Voice Processing"],
        dependencies=[Depends(get_current_user)]
    )
    
    app.include_router(
        context_router,
        prefix="/context",
        tags=["Context Management"],
        dependencies=[Depends(get_current_user)]
    )
    
    app.include_router(
        accessibility_router,
        prefix="/accessibility",
        tags=["Accessibility"]
    )
    
    # Gateway management endpoints
    app.include_router(
        gateway_router,
        prefix="/gateway",
        tags=["Gateway Management"]
    )
    
    # Alerting endpoints
    app.include_router(
        alerts_router,
        prefix="/alerts",
        tags=["Alerting"]
    )
    
    # Gateway status and monitoring endpoints
    @app.get("/gateway/status")
    async def gateway_status():
        """Get gateway load balancing and routing status."""
        return {
            "status": "operational",
            "timestamp": time.time(),
            "load_balancer": load_balancer.get_load_metrics(),
            "active_traces": len(distributed_tracing.active_traces),
            "services": {
                "authentication": "initialized" if jwt_manager and auth_service else "not_initialized",
                "monitoring": "enabled" if settings.monitoring.enable_metrics else "disabled",
                "tracing": "enabled"
            }
        }
    
    @app.get("/gateway/routes")
    async def gateway_routes():
        """Get available service routes and their health status."""
        routes = {}
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                service_route = await load_balancer.get_service_route(route.path)
                if service_route not in routes:
                    routes[service_route] = {
                        "paths": [],
                        "health": load_balancer.service_health.get(service_route, {"healthy": True})
                    }
                routes[service_route]["paths"].append({
                    "path": route.path,
                    "methods": list(route.methods) if route.methods else ["GET"]
                })
        
        return {
            "routes": routes,
            "total_services": len(routes),
            "timestamp": time.time()
        }
    
    @app.get("/gateway/traces")
    async def gateway_traces():
        """Get active distributed traces (for debugging)."""
        if not settings.debug:
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "active_traces": len(distributed_tracing.active_traces),
            "traces": list(distributed_tracing.active_traces.keys())[:10]  # Limit for security
        }
    
    # Root endpoint with enhanced information
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """Root endpoint with comprehensive application information."""
        return {
            "name": settings.app_name,
            "version": settings.version,
            "status": "running",
            "environment": settings.environment,
            "gateway": {
                "load_balancing": "enabled",
                "authentication": "enabled",
                "distributed_tracing": "enabled",
                "request_routing": "intelligent"
            },
            "supported_languages": [
                "hi", "en-IN", "ta", "te", "bn", "mr", 
                "gu", "kn", "ml", "pa", "or"
            ],
            "features": {
                "offline_mode": settings.enable_offline_mode,
                "code_switching": settings.enable_code_switching,
                "cultural_context": settings.enable_cultural_context,
                "intelligent_routing": True,
                "load_balancing": True,
                "distributed_tracing": True
            },
            "performance": {
                "max_concurrent_requests": load_balancer.max_concurrent_requests,
                "current_load": load_balancer.active_requests,
                "queue_capacity": load_balancer.request_queue.maxsize
            }
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    """Run the application directly for development."""
    settings = get_settings()
    
    uvicorn.run(
        "bharatvoice.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.logging.level.lower(),
=======
"""
Main FastAPI application for BharatVoice Assistant.

This module sets up the FastAPI application with all necessary middleware,
routers, and configuration for the BharatVoice multilingual voice assistant.
Includes intelligent load balancing, comprehensive authentication, request routing,
detailed health checks, and distributed tracing.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
import uvicorn

from bharatvoice.config import get_settings
from bharatvoice.api.health import router as health_router
from bharatvoice.api.voice import router as voice_router
from bharatvoice.api.context import router as context_router
from bharatvoice.api.auth import router as auth_router
from bharatvoice.api.accessibility import router as accessibility_router
from bharatvoice.api.gateway import router as gateway_router, initialize_default_services
from bharatvoice.api.alerts import router as alerts_router
from bharatvoice.api.middleware import (
    PerformanceMonitoringMiddleware,
    ErrorHandlingMiddleware,
    RequestQueueMiddleware
)
from bharatvoice.utils.logging import setup_logging
from bharatvoice.utils.monitoring import setup_metrics, start_metrics_server, track_request_metrics
from bharatvoice.utils.performance_monitor import get_performance_monitor
from bharatvoice.utils.alerting import get_alert_manager, initialize_default_alert_rules
from bharatvoice.services.auth.jwt_manager import JWTManager
from bharatvoice.services.auth.auth_service import AuthService


# Configure structured logging
logger = structlog.get_logger(__name__)

# Global instances for dependency injection
jwt_manager: Optional[JWTManager] = None
auth_service: Optional[AuthService] = None
security = HTTPBearer(auto_error=False)


class LoadBalancer:
    """
    Intelligent load balancer for request distribution and resource management.
    """
    
    def __init__(self):
        self.active_requests = 0
        self.max_concurrent_requests = 100
        self.request_queue = asyncio.Queue(maxsize=200)
        self.service_health = {}
        self.request_history = []
        self.lock = asyncio.Lock()
    
    async def can_accept_request(self) -> bool:
        """Check if system can accept new requests."""
        async with self.lock:
            return self.active_requests < self.max_concurrent_requests
    
    async def acquire_slot(self):
        """Acquire a request processing slot."""
        return RequestSlot(self)
    
    async def get_service_route(self, path: str) -> str:
        """
        Determine optimal service route based on path and load.
        
        Args:
            path: Request path
            
        Returns:
            Service route identifier
        """
        # Route based on path patterns
        if path.startswith("/voice/"):
            return "voice_processing"
        elif path.startswith("/context/"):
            return "context_management"
        elif path.startswith("/auth/"):
            return "authentication"
        elif path.startswith("/health/"):
            return "health_monitoring"
        elif path.startswith("/accessibility/"):
            return "accessibility"
        else:
            return "default"
    
    def update_service_health(self, service: str, healthy: bool, response_time: float):
        """Update service health status."""
        self.service_health[service] = {
            "healthy": healthy,
            "response_time": response_time,
            "last_check": time.time()
        }
    
    def get_load_metrics(self) -> Dict[str, Any]:
        """Get current load balancing metrics."""
        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.max_concurrent_requests,
            "queue_size": self.request_queue.qsize(),
            "service_health": self.service_health,
            "utilization": self.active_requests / self.max_concurrent_requests
        }


class RequestSlot:
    """Context manager for request slot management."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    async def __aenter__(self):
        async with self.load_balancer.lock:
            self.load_balancer.active_requests += 1
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self.load_balancer.lock:
            self.load_balancer.active_requests -= 1


class AuthenticationMiddleware:
    """
    Comprehensive authentication and authorization middleware.
    """
    
    def __init__(self, jwt_manager: JWTManager, auth_service: AuthService):
        self.jwt_manager = jwt_manager
        self.auth_service = auth_service
        self.public_paths = {
            "/", "/health", "/health/live", "/health/ready", 
            "/docs", "/redoc", "/openapi.json", "/metrics",
            "/auth/login", "/auth/register", "/auth/refresh"
        }
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """
        Authenticate request and return user context.
        
        Args:
            request: HTTP request
            
        Returns:
            User context if authenticated, None otherwise
        """
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return None
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        
        try:
            # Verify and decode token
            payload = await self.jwt_manager.verify_token(token)
            
            # Get user context
            user_id = payload.get("sub")
            if user_id:
                user_context = await self.auth_service.get_user_context(user_id)
                return user_context
            
        except Exception as e:
            logger.warning("Token verification failed", error=str(e))
        
        return None
    
    def requires_authentication(self, path: str) -> bool:
        """Check if path requires authentication."""
        return path not in self.public_paths


class DistributedTracing:
    """
    Distributed tracing for request tracking across services.
    """
    
    def __init__(self):
        self.active_traces = {}
    
    def start_trace(self, request_id: str, operation: str) -> str:
        """Start a new trace."""
        trace_id = f"{request_id}-{operation}-{int(time.time() * 1000)}"
        self.active_traces[trace_id] = {
            "request_id": request_id,
            "operation": operation,
            "start_time": time.time(),
            "spans": []
        }
        return trace_id
    
    def add_span(self, trace_id: str, service: str, operation: str, duration: float):
        """Add a span to an existing trace."""
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["spans"].append({
                "service": service,
                "operation": operation,
                "duration": duration,
                "timestamp": time.time()
            })
    
    def finish_trace(self, trace_id: str) -> Dict[str, Any]:
        """Finish a trace and return trace data."""
        if trace_id in self.active_traces:
            trace_data = self.active_traces.pop(trace_id)
            trace_data["total_duration"] = time.time() - trace_data["start_time"]
            return trace_data
        return {}


# Global instances
load_balancer = LoadBalancer()
distributed_tracing = DistributedTracing()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to get current authenticated user.
    
    Args:
        request: HTTP request
        credentials: Authorization credentials
        
    Returns:
        User context if authenticated
        
    Raises:
        HTTPException: If authentication is required but fails
    """
    global jwt_manager, auth_service
    
    if not jwt_manager or not auth_service:
        # Services not initialized yet
        return None
    
    auth_middleware = AuthenticationMiddleware(jwt_manager, auth_service)
    
    # Check if authentication is required
    if not auth_middleware.requires_authentication(request.url.path):
        return None
    
    # Authenticate request
    user_context = await auth_middleware.authenticate_request(request)
    
    if user_context is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user_context


class GatewayMiddleware:
    """
    Main gateway middleware for request orchestration.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request through gateway."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Generate request ID and start tracing
        request_id = str(uuid.uuid4())
        trace_id = distributed_tracing.start_trace(request_id, "gateway_request")
        
        # Add to request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        
        # Check load balancer
        if not await load_balancer.can_accept_request():
            # Return 503 Service Unavailable
            response = JSONResponse(
                status_code=503,
                content={
                    "error": "Service temporarily unavailable",
                    "message": "System is under high load. Please try again later.",
                    "request_id": request_id
                }
            )
            await response(scope, receive, send)
            return
        
        # Process request with load balancing
        async with load_balancer.acquire_slot():
            # Determine service route
            service_route = await load_balancer.get_service_route(request.url.path)
            request.state.service_route = service_route
            
            # Add tracing span
            start_time = time.time()
            
            try:
                # Process through FastAPI app
                await self.app(scope, receive, send)
                
                # Record successful processing
                duration = time.time() - start_time
                distributed_tracing.add_span(trace_id, service_route, "process_request", duration)
                load_balancer.update_service_health(service_route, True, duration)
                
            except Exception as e:
                # Record failed processing
                duration = time.time() - start_time
                distributed_tracing.add_span(trace_id, service_route, "process_request_error", duration)
                load_balancer.update_service_health(service_route, False, duration)
                
                logger.error(
                    "Gateway request processing failed",
                    request_id=request_id,
                    trace_id=trace_id,
                    service_route=service_route,
                    error=str(e),
                    exc_info=e
                )
                raise
            
            finally:
                # Finish trace
                trace_data = distributed_tracing.finish_trace(trace_id)
                logger.info(
                    "Request trace completed",
                    request_id=request_id,
                    trace_data=trace_data
                )
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    global jwt_manager, auth_service
    
    # Startup
    settings = get_settings()
    logger.info("Starting BharatVoice Assistant", version=settings.version)
    
    # Initialize authentication services
    try:
        jwt_manager = JWTManager(
            secret_key=settings.security.secret_key,
            algorithm=settings.security.algorithm,
            access_token_expire_minutes=settings.security.access_token_expire_minutes
        )
        auth_service = AuthService()
        logger.info("Authentication services initialized")
    except Exception as e:
        logger.error("Failed to initialize authentication services", exc_info=e)
    
    # Initialize monitoring
    if settings.monitoring.enable_metrics:
        setup_metrics()
        start_metrics_server(settings.monitoring.metrics_port)
        logger.info("Metrics collection enabled", port=settings.monitoring.metrics_port)
    
    # Initialize performance monitoring
    performance_monitor = get_performance_monitor()
    await performance_monitor.start()
    logger.info("Performance monitoring started")
    
    # Initialize load balancer
    logger.info("Load balancer initialized", max_concurrent=load_balancer.max_concurrent_requests)
    
    # Initialize distributed tracing
    logger.info("Distributed tracing initialized")
    
    # Initialize default services for gateway
    await initialize_default_services()
    logger.info("Default gateway services initialized")
    
    # Initialize alerting system
    alert_manager = get_alert_manager()
    await alert_manager.start()
    await initialize_default_alert_rules()
    logger.info("Alerting system initialized")
    
    # Initialize services
    # TODO: Initialize database connections, Redis, ML models, etc.
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down BharatVoice Assistant")
    
    # Stop performance monitoring
    await performance_monitor.stop()
    logger.info("Performance monitoring stopped")
    
    # Stop alerting system
    alert_manager = get_alert_manager()
    await alert_manager.stop()
    logger.info("Alerting system stopped")
    
    # Clear active traces
    distributed_tracing.active_traces.clear()
    logger.info("Distributed tracing cleaned up")
    
    # TODO: Cleanup resources, close connections, etc.


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with intelligent load balancing,
    comprehensive authentication, request routing, and distributed tracing.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    
    # Setup logging
    setup_logging(settings.logging)
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered multilingual voice assistant for the Indian market with intelligent load balancing and distributed tracing",
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add gateway middleware (first in chain)
    app.add_middleware(GatewayMiddleware)
    
    # Add security middleware
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.bharatvoice.ai", "localhost", "127.0.0.1"]
        )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=["*"],
    )
    
    # Add performance monitoring middleware
    if settings.monitoring.enable_metrics:
        app.add_middleware(PerformanceMonitoringMiddleware, enable_monitoring=True)
    
    # Add error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Add request queuing middleware for high load scenarios
    app.add_middleware(RequestQueueMiddleware, enable_queuing=True)
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with structured logging and tracing."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        trace_id = getattr(request.state, "trace_id", None)
        service_route = getattr(request.state, "service_route", "unknown")
        
        logger.error(
            "Unhandled exception",
            exc_info=exc,
            method=request.method,
            url=str(request.url),
            request_id=request_id,
            trace_id=trace_id,
            service_route=service_route
        )
        
        # Add error span to trace
        if trace_id:
            distributed_tracing.add_span(trace_id, service_route, "unhandled_exception", 0.0)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request_id,
                "trace_id": trace_id
            }
        )
    
    # Include API routers with authentication dependencies
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health Check"]
    )
    
    app.include_router(
        auth_router,
        prefix="/auth",
        tags=["Authentication"]
    )
    
    app.include_router(
        voice_router,
        prefix="/voice",
        tags=["Voice Processing"],
        dependencies=[Depends(get_current_user)]
    )
    
    app.include_router(
        context_router,
        prefix="/context",
        tags=["Context Management"],
        dependencies=[Depends(get_current_user)]
    )
    
    app.include_router(
        accessibility_router,
        prefix="/accessibility",
        tags=["Accessibility"]
    )
    
    # Gateway management endpoints
    app.include_router(
        gateway_router,
        prefix="/gateway",
        tags=["Gateway Management"]
    )
    
    # Alerting endpoints
    app.include_router(
        alerts_router,
        prefix="/alerts",
        tags=["Alerting"]
    )
    
    # Gateway status and monitoring endpoints
    @app.get("/gateway/status")
    async def gateway_status():
        """Get gateway load balancing and routing status."""
        return {
            "status": "operational",
            "timestamp": time.time(),
            "load_balancer": load_balancer.get_load_metrics(),
            "active_traces": len(distributed_tracing.active_traces),
            "services": {
                "authentication": "initialized" if jwt_manager and auth_service else "not_initialized",
                "monitoring": "enabled" if settings.monitoring.enable_metrics else "disabled",
                "tracing": "enabled"
            }
        }
    
    @app.get("/gateway/routes")
    async def gateway_routes():
        """Get available service routes and their health status."""
        routes = {}
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                service_route = await load_balancer.get_service_route(route.path)
                if service_route not in routes:
                    routes[service_route] = {
                        "paths": [],
                        "health": load_balancer.service_health.get(service_route, {"healthy": True})
                    }
                routes[service_route]["paths"].append({
                    "path": route.path,
                    "methods": list(route.methods) if route.methods else ["GET"]
                })
        
        return {
            "routes": routes,
            "total_services": len(routes),
            "timestamp": time.time()
        }
    
    @app.get("/gateway/traces")
    async def gateway_traces():
        """Get active distributed traces (for debugging)."""
        if not settings.debug:
            raise HTTPException(status_code=404, detail="Not found")
        
        return {
            "active_traces": len(distributed_tracing.active_traces),
            "traces": list(distributed_tracing.active_traces.keys())[:10]  # Limit for security
        }
    
    # Root endpoint with enhanced information
    @app.get("/", response_model=Dict[str, Any])
    async def root():
        """Root endpoint with comprehensive application information."""
        return {
            "name": settings.app_name,
            "version": settings.version,
            "status": "running",
            "environment": settings.environment,
            "gateway": {
                "load_balancing": "enabled",
                "authentication": "enabled",
                "distributed_tracing": "enabled",
                "request_routing": "intelligent"
            },
            "supported_languages": [
                "hi", "en-IN", "ta", "te", "bn", "mr", 
                "gu", "kn", "ml", "pa", "or"
            ],
            "features": {
                "offline_mode": settings.enable_offline_mode,
                "code_switching": settings.enable_code_switching,
                "cultural_context": settings.enable_cultural_context,
                "intelligent_routing": True,
                "load_balancing": True,
                "distributed_tracing": True
            },
            "performance": {
                "max_concurrent_requests": load_balancer.max_concurrent_requests,
                "current_load": load_balancer.active_requests,
                "queue_capacity": load_balancer.request_queue.maxsize
            }
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    """Run the application directly for development."""
    settings = get_settings()
    
    uvicorn.run(
        "bharatvoice.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.logging.level.lower(),
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )