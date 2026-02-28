<<<<<<< HEAD
"""
Gateway orchestration and service routing for BharatVoice Assistant.

This module provides intelligent request routing, service discovery,
load balancing, and distributed system coordination.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
import structlog

from bharatvoice.config import get_settings, Settings
from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.monitoring import track_external_service_call


logger = structlog.get_logger(__name__)
router = APIRouter()


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Request routing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    WEIGHTED = "weighted"


class ServiceInstance(BaseModel):
    """Service instance information."""
    id: str
    name: str
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    last_health_check: datetime
    response_time_ms: float
    active_connections: int
    weight: float = 1.0
    metadata: Dict[str, Any] = {}


class ServiceRegistry:
    """
    Service registry for microservice discovery and health tracking.
    """
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.routing_strategies: Dict[str, RoutingStrategy] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.lock = asyncio.Lock()
    
    async def register_service(self, service: ServiceInstance):
        """Register a service instance."""
        async with self.lock:
            if service.name not in self.services:
                self.services[service.name] = []
            
            # Remove existing instance with same ID
            self.services[service.name] = [
                s for s in self.services[service.name] if s.id != service.id
            ]
            
            # Add new instance
            self.services[service.name].append(service)
            
            logger.info(
                "Service registered",
                service_name=service.name,
                service_id=service.id,
                host=service.host,
                port=service.port
            )
    
    async def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance."""
        async with self.lock:
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] if s.id != service_id
                ]
                
                logger.info(
                    "Service deregistered",
                    service_name=service_name,
                    service_id=service_id
                )
    
    async def get_service_instance(
        self, 
        service_name: str, 
        strategy: Optional[RoutingStrategy] = None
    ) -> Optional[ServiceInstance]:
        """Get a service instance using the specified routing strategy."""
        async with self.lock:
            if service_name not in self.services or not self.services[service_name]:
                return None
            
            healthy_instances = [
                s for s in self.services[service_name] 
                if s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            ]
            
            if not healthy_instances:
                return None
            
            # Use service-specific strategy or default
            routing_strategy = strategy or self.routing_strategies.get(
                service_name, RoutingStrategy.ROUND_ROBIN
            )
            
            return self._select_instance(healthy_instances, routing_strategy)
    
    def _select_instance(
        self, 
        instances: List[ServiceInstance], 
        strategy: RoutingStrategy
    ) -> ServiceInstance:
        """Select an instance based on routing strategy."""
        if strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin (in real implementation, would track state)
            return instances[int(time.time()) % len(instances)]
        
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.active_connections)
        
        elif strategy == RoutingStrategy.RESPONSE_TIME:
            return min(instances, key=lambda x: x.response_time_ms)
        
        elif strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection (simplified)
            total_weight = sum(i.weight for i in instances)
            if total_weight == 0:
                return instances[0]
            
            # Simple weighted selection
            return max(instances, key=lambda x: x.weight)
        
        else:
            return instances[0]
    
    async def update_service_health(
        self, 
        service_name: str, 
        service_id: str, 
        status: ServiceStatus,
        response_time_ms: float
    ):
        """Update service health status."""
        async with self.lock:
            if service_name in self.services:
                for service in self.services[service_name]:
                    if service.id == service_id:
                        service.status = status
                        service.response_time_ms = response_time_ms
                        service.last_health_check = datetime.utcnow()
                        break
    
    def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services."""
        return self.services.copy()


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.
    """
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RequestRouter:
    """
    Intelligent request router with load balancing and failover.
    """
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.route_mappings = {
            "/voice/": "voice_processing",
            "/context/": "context_management", 
            "/auth/": "authentication",
            "/accessibility/": "accessibility",
            "/external/": "external_services"
        }
    
    async def route_request(self, request: Request) -> Optional[ServiceInstance]:
        """Route request to appropriate service instance."""
        service_name = self._determine_service(request.url.path)
        
        if not service_name:
            return None
        
        # Get service instance
        instance = await self.service_registry.get_service_instance(service_name)
        
        if instance:
            logger.info(
                "Request routed",
                path=request.url.path,
                service_name=service_name,
                instance_id=instance.id,
                host=instance.host,
                port=instance.port
            )
        
        return instance
    
    def _determine_service(self, path: str) -> Optional[str]:
        """Determine service name from request path."""
        for route_prefix, service_name in self.route_mappings.items():
            if path.startswith(route_prefix):
                return service_name
        
        return None
    
    async def add_route_mapping(self, path_prefix: str, service_name: str):
        """Add new route mapping."""
        self.route_mappings[path_prefix] = service_name
        logger.info("Route mapping added", path_prefix=path_prefix, service_name=service_name)


# Global instances
service_registry = ServiceRegistry()
request_router = RequestRouter(service_registry)


@router.get("/services")
async def list_services():
    """List all registered services and their instances."""
    services = service_registry.get_all_services()
    
    service_summary = {}
    for service_name, instances in services.items():
        service_summary[service_name] = {
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
            "instances": [
                {
                    "id": instance.id,
                    "host": instance.host,
                    "port": instance.port,
                    "status": instance.status.value,
                    "response_time_ms": instance.response_time_ms,
                    "active_connections": instance.active_connections,
                    "last_health_check": instance.last_health_check.isoformat()
                }
                for instance in instances
            ]
        }
    
    return {
        "services": service_summary,
        "total_services": len(services),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/services/register")
async def register_service(service_data: Dict[str, Any]):
    """Register a new service instance."""
    try:
        service = ServiceInstance(
            id=service_data["id"],
            name=service_data["name"],
            host=service_data["host"],
            port=service_data["port"],
            status=ServiceStatus(service_data.get("status", "healthy")),
            health_check_url=service_data["health_check_url"],
            last_health_check=datetime.utcnow(),
            response_time_ms=service_data.get("response_time_ms", 0.0),
            active_connections=service_data.get("active_connections", 0),
            weight=service_data.get("weight", 1.0),
            metadata=service_data.get("metadata", {})
        )
        
        await service_registry.register_service(service)
        
        return {
            "message": "Service registered successfully",
            "service_id": service.id,
            "service_name": service.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Service registration failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@router.delete("/services/{service_name}/{service_id}")
async def deregister_service(service_name: str, service_id: str):
    """Deregister a service instance."""
    try:
        await service_registry.deregister_service(service_name, service_id)
        
        return {
            "message": "Service deregistered successfully",
            "service_name": service_name,
            "service_id": service_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Service deregistration failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Deregistration failed: {str(e)}")


@router.get("/routes")
async def list_routes():
    """List all configured route mappings."""
    return {
        "routes": request_router.route_mappings,
        "total_routes": len(request_router.route_mappings),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/routes")
async def add_route(route_data: Dict[str, str]):
    """Add a new route mapping."""
    try:
        path_prefix = route_data["path_prefix"]
        service_name = route_data["service_name"]
        
        await request_router.add_route_mapping(path_prefix, service_name)
        
        return {
            "message": "Route added successfully",
            "path_prefix": path_prefix,
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Route addition failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Route addition failed: {str(e)}")


@router.get("/circuit-breakers")
async def list_circuit_breakers():
    """List circuit breaker status for all services."""
    circuit_breakers = {}
    
    for service_name, cb in service_registry.circuit_breakers.items():
        circuit_breakers[service_name] = {
            "state": cb.state,
            "failure_count": cb.failure_count,
            "failure_threshold": cb.failure_threshold,
            "last_failure_time": cb.last_failure_time,
            "recovery_timeout": cb.recovery_timeout
        }
    
    return {
        "circuit_breakers": circuit_breakers,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str):
    """Manually reset a circuit breaker."""
    if service_name in service_registry.circuit_breakers:
        cb = service_registry.circuit_breakers[service_name]
        cb.failure_count = 0
        cb.state = "CLOSED"
        cb.last_failure_time = None
        
        return {
            "message": f"Circuit breaker reset for {service_name}",
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail="Circuit breaker not found")


@router.get("/load-balancing")
async def load_balancing_status():
    """Get load balancing status and metrics."""
    services = service_registry.get_all_services()
    
    load_balancing_info = {}
    for service_name, instances in services.items():
        total_connections = sum(i.active_connections for i in instances)
        avg_response_time = sum(i.response_time_ms for i in instances) / len(instances) if instances else 0
        
        load_balancing_info[service_name] = {
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
            "total_connections": total_connections,
            "average_response_time_ms": avg_response_time,
            "routing_strategy": service_registry.routing_strategies.get(service_name, "round_robin"),
            "load_distribution": [
                {
                    "instance_id": i.id,
                    "connections": i.active_connections,
                    "response_time_ms": i.response_time_ms,
                    "weight": i.weight
                }
                for i in instances
            ]
        }
    
    return {
        "load_balancing": load_balancing_info,
        "timestamp": datetime.utcnow().isoformat()
    }


async def initialize_default_services():
    """Initialize default service instances for development."""
    default_services = [
        {
            "id": "voice-processing-1",
            "name": "voice_processing",
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        },
        {
            "id": "context-management-1", 
            "name": "context_management",
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        },
        {
            "id": "authentication-1",
            "name": "authentication", 
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        }
    ]
    
    for service_data in default_services:
        service = ServiceInstance(
            id=service_data["id"],
            name=service_data["name"],
            host=service_data["host"],
            port=service_data["port"],
            status=ServiceStatus(service_data["status"]),
            health_check_url=service_data["health_check_url"],
            last_health_check=datetime.utcnow(),
            response_time_ms=0.0,
            active_connections=0
        )
        
        await service_registry.register_service(service)
    
=======
"""
Gateway orchestration and service routing for BharatVoice Assistant.

This module provides intelligent request routing, service discovery,
load balancing, and distributed system coordination.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
import structlog

from bharatvoice.config import get_settings, Settings
from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.monitoring import track_external_service_call


logger = structlog.get_logger(__name__)
router = APIRouter()


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Request routing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    WEIGHTED = "weighted"


class ServiceInstance(BaseModel):
    """Service instance information."""
    id: str
    name: str
    host: str
    port: int
    status: ServiceStatus
    health_check_url: str
    last_health_check: datetime
    response_time_ms: float
    active_connections: int
    weight: float = 1.0
    metadata: Dict[str, Any] = {}


class ServiceRegistry:
    """
    Service registry for microservice discovery and health tracking.
    """
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.routing_strategies: Dict[str, RoutingStrategy] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.lock = asyncio.Lock()
    
    async def register_service(self, service: ServiceInstance):
        """Register a service instance."""
        async with self.lock:
            if service.name not in self.services:
                self.services[service.name] = []
            
            # Remove existing instance with same ID
            self.services[service.name] = [
                s for s in self.services[service.name] if s.id != service.id
            ]
            
            # Add new instance
            self.services[service.name].append(service)
            
            logger.info(
                "Service registered",
                service_name=service.name,
                service_id=service.id,
                host=service.host,
                port=service.port
            )
    
    async def deregister_service(self, service_name: str, service_id: str):
        """Deregister a service instance."""
        async with self.lock:
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] if s.id != service_id
                ]
                
                logger.info(
                    "Service deregistered",
                    service_name=service_name,
                    service_id=service_id
                )
    
    async def get_service_instance(
        self, 
        service_name: str, 
        strategy: Optional[RoutingStrategy] = None
    ) -> Optional[ServiceInstance]:
        """Get a service instance using the specified routing strategy."""
        async with self.lock:
            if service_name not in self.services or not self.services[service_name]:
                return None
            
            healthy_instances = [
                s for s in self.services[service_name] 
                if s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            ]
            
            if not healthy_instances:
                return None
            
            # Use service-specific strategy or default
            routing_strategy = strategy or self.routing_strategies.get(
                service_name, RoutingStrategy.ROUND_ROBIN
            )
            
            return self._select_instance(healthy_instances, routing_strategy)
    
    def _select_instance(
        self, 
        instances: List[ServiceInstance], 
        strategy: RoutingStrategy
    ) -> ServiceInstance:
        """Select an instance based on routing strategy."""
        if strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin (in real implementation, would track state)
            return instances[int(time.time()) % len(instances)]
        
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.active_connections)
        
        elif strategy == RoutingStrategy.RESPONSE_TIME:
            return min(instances, key=lambda x: x.response_time_ms)
        
        elif strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection (simplified)
            total_weight = sum(i.weight for i in instances)
            if total_weight == 0:
                return instances[0]
            
            # Simple weighted selection
            return max(instances, key=lambda x: x.weight)
        
        else:
            return instances[0]
    
    async def update_service_health(
        self, 
        service_name: str, 
        service_id: str, 
        status: ServiceStatus,
        response_time_ms: float
    ):
        """Update service health status."""
        async with self.lock:
            if service_name in self.services:
                for service in self.services[service_name]:
                    if service.id == service_id:
                        service.status = status
                        service.response_time_ms = response_time_ms
                        service.last_health_check = datetime.utcnow()
                        break
    
    def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services."""
        return self.services.copy()


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service resilience.
    """
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RequestRouter:
    """
    Intelligent request router with load balancing and failover.
    """
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.route_mappings = {
            "/voice/": "voice_processing",
            "/context/": "context_management", 
            "/auth/": "authentication",
            "/accessibility/": "accessibility",
            "/external/": "external_services"
        }
    
    async def route_request(self, request: Request) -> Optional[ServiceInstance]:
        """Route request to appropriate service instance."""
        service_name = self._determine_service(request.url.path)
        
        if not service_name:
            return None
        
        # Get service instance
        instance = await self.service_registry.get_service_instance(service_name)
        
        if instance:
            logger.info(
                "Request routed",
                path=request.url.path,
                service_name=service_name,
                instance_id=instance.id,
                host=instance.host,
                port=instance.port
            )
        
        return instance
    
    def _determine_service(self, path: str) -> Optional[str]:
        """Determine service name from request path."""
        for route_prefix, service_name in self.route_mappings.items():
            if path.startswith(route_prefix):
                return service_name
        
        return None
    
    async def add_route_mapping(self, path_prefix: str, service_name: str):
        """Add new route mapping."""
        self.route_mappings[path_prefix] = service_name
        logger.info("Route mapping added", path_prefix=path_prefix, service_name=service_name)


# Global instances
service_registry = ServiceRegistry()
request_router = RequestRouter(service_registry)


@router.get("/services")
async def list_services():
    """List all registered services and their instances."""
    services = service_registry.get_all_services()
    
    service_summary = {}
    for service_name, instances in services.items():
        service_summary[service_name] = {
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
            "instances": [
                {
                    "id": instance.id,
                    "host": instance.host,
                    "port": instance.port,
                    "status": instance.status.value,
                    "response_time_ms": instance.response_time_ms,
                    "active_connections": instance.active_connections,
                    "last_health_check": instance.last_health_check.isoformat()
                }
                for instance in instances
            ]
        }
    
    return {
        "services": service_summary,
        "total_services": len(services),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/services/register")
async def register_service(service_data: Dict[str, Any]):
    """Register a new service instance."""
    try:
        service = ServiceInstance(
            id=service_data["id"],
            name=service_data["name"],
            host=service_data["host"],
            port=service_data["port"],
            status=ServiceStatus(service_data.get("status", "healthy")),
            health_check_url=service_data["health_check_url"],
            last_health_check=datetime.utcnow(),
            response_time_ms=service_data.get("response_time_ms", 0.0),
            active_connections=service_data.get("active_connections", 0),
            weight=service_data.get("weight", 1.0),
            metadata=service_data.get("metadata", {})
        )
        
        await service_registry.register_service(service)
        
        return {
            "message": "Service registered successfully",
            "service_id": service.id,
            "service_name": service.name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Service registration failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@router.delete("/services/{service_name}/{service_id}")
async def deregister_service(service_name: str, service_id: str):
    """Deregister a service instance."""
    try:
        await service_registry.deregister_service(service_name, service_id)
        
        return {
            "message": "Service deregistered successfully",
            "service_name": service_name,
            "service_id": service_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Service deregistration failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Deregistration failed: {str(e)}")


@router.get("/routes")
async def list_routes():
    """List all configured route mappings."""
    return {
        "routes": request_router.route_mappings,
        "total_routes": len(request_router.route_mappings),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/routes")
async def add_route(route_data: Dict[str, str]):
    """Add a new route mapping."""
    try:
        path_prefix = route_data["path_prefix"]
        service_name = route_data["service_name"]
        
        await request_router.add_route_mapping(path_prefix, service_name)
        
        return {
            "message": "Route added successfully",
            "path_prefix": path_prefix,
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error("Route addition failed", exc_info=e)
        raise HTTPException(status_code=400, detail=f"Route addition failed: {str(e)}")


@router.get("/circuit-breakers")
async def list_circuit_breakers():
    """List circuit breaker status for all services."""
    circuit_breakers = {}
    
    for service_name, cb in service_registry.circuit_breakers.items():
        circuit_breakers[service_name] = {
            "state": cb.state,
            "failure_count": cb.failure_count,
            "failure_threshold": cb.failure_threshold,
            "last_failure_time": cb.last_failure_time,
            "recovery_timeout": cb.recovery_timeout
        }
    
    return {
        "circuit_breakers": circuit_breakers,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str):
    """Manually reset a circuit breaker."""
    if service_name in service_registry.circuit_breakers:
        cb = service_registry.circuit_breakers[service_name]
        cb.failure_count = 0
        cb.state = "CLOSED"
        cb.last_failure_time = None
        
        return {
            "message": f"Circuit breaker reset for {service_name}",
            "service_name": service_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail="Circuit breaker not found")


@router.get("/load-balancing")
async def load_balancing_status():
    """Get load balancing status and metrics."""
    services = service_registry.get_all_services()
    
    load_balancing_info = {}
    for service_name, instances in services.items():
        total_connections = sum(i.active_connections for i in instances)
        avg_response_time = sum(i.response_time_ms for i in instances) / len(instances) if instances else 0
        
        load_balancing_info[service_name] = {
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
            "total_connections": total_connections,
            "average_response_time_ms": avg_response_time,
            "routing_strategy": service_registry.routing_strategies.get(service_name, "round_robin"),
            "load_distribution": [
                {
                    "instance_id": i.id,
                    "connections": i.active_connections,
                    "response_time_ms": i.response_time_ms,
                    "weight": i.weight
                }
                for i in instances
            ]
        }
    
    return {
        "load_balancing": load_balancing_info,
        "timestamp": datetime.utcnow().isoformat()
    }


async def initialize_default_services():
    """Initialize default service instances for development."""
    default_services = [
        {
            "id": "voice-processing-1",
            "name": "voice_processing",
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        },
        {
            "id": "context-management-1", 
            "name": "context_management",
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        },
        {
            "id": "authentication-1",
            "name": "authentication", 
            "host": "localhost",
            "port": 8000,
            "health_check_url": "/health",
            "status": "healthy"
        }
    ]
    
    for service_data in default_services:
        service = ServiceInstance(
            id=service_data["id"],
            name=service_data["name"],
            host=service_data["host"],
            port=service_data["port"],
            status=ServiceStatus(service_data["status"]),
            health_check_url=service_data["health_check_url"],
            last_health_check=datetime.utcnow(),
            response_time_ms=0.0,
            active_connections=0
        )
        
        await service_registry.register_service(service)
    
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    logger.info("Default services initialized")