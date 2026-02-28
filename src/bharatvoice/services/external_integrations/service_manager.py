"""
External Service Manager for BharatVoice Assistant.

This module coordinates all external service integrations including
Indian Railways, weather services, and Digital India platforms.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from bharatvoice.core.models import ServiceResult, ServiceType, ServiceParameters
from .indian_railways_service import IndianRailwaysService
from .weather_service import WeatherService
from .digital_india_service import DigitalIndiaService, ServiceCategory


class ServiceManager:
    """Manager for coordinating external service integrations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize service instances
        self.railways_service = IndianRailwaysService()
        self.weather_service = WeatherService()
        self.digital_india_service = DigitalIndiaService()
        
        # Service routing map
        self._service_routes = self._initialize_service_routes()
    
    def _initialize_service_routes(self) -> Dict[ServiceType, Any]:
        """Initialize service routing configuration."""
        return {
            ServiceType.INDIAN_RAILWAYS: self.railways_service,
            ServiceType.WEATHER: self.weather_service,
            ServiceType.CRICKET_SCORES: self.weather_service,
            ServiceType.BOLLYWOOD_NEWS: self.weather_service,
            ServiceType.GOVERNMENT_SERVICE: self.digital_india_service
        }
    
    async def process_service_request(
        self,
        service_params: ServiceParameters
    ) -> ServiceResult:
        """
        Process external service request.
        
        Args:
            service_params: Service parameters including type and parameters
            
        Returns:
            Service result from appropriate service
        """
        try:
            self.logger.info(f"Processing service request: {service_params.service_type}")
            
            service_instance = self._service_routes.get(service_params.service_type)
            if not service_instance:
                return ServiceResult(
                    service_type=service_params.service_type,
                    success=False,
                    data={},
                    error_message=f"Service type {service_params.service_type} not supported",
                    response_time=0.1
                )
            
            # Route to appropriate service method based on parameters
            if service_params.service_type == ServiceType.INDIAN_RAILWAYS:
                return await self._handle_railways_request(service_params)
            elif service_params.service_type == ServiceType.WEATHER:
                return await self._handle_weather_request(service_params)
            elif service_params.service_type == ServiceType.CRICKET_SCORES:
                return await self._handle_cricket_request(service_params)
            elif service_params.service_type == ServiceType.BOLLYWOOD_NEWS:
                return await self._handle_bollywood_request(service_params)
            elif service_params.service_type == ServiceType.GOVERNMENT_SERVICE:
                return await self._handle_government_service_request(service_params)
            else:
                return ServiceResult(
                    service_type=service_params.service_type,
                    success=False,
                    data={},
                    error_message=f"Handler not implemented for {service_params.service_type}",
                    response_time=0.1
                )
                
        except Exception as e:
            self.logger.error(f"Error processing service request: {e}")
            return ServiceResult(
                service_type=service_params.service_type,
                success=False,
                data={},
                error_message=f"Service request failed: {str(e)}",
                response_time=0.5
            )
    
    async def _handle_railways_request(self, params: ServiceParameters) -> ServiceResult:
        """Handle Indian Railways service requests."""
        request_type = params.parameters.get("request_type", "")
        
        if request_type == "train_schedule":
            return await self.railways_service.get_train_schedule(
                params.parameters.get("train_number", ""),
                params.parameters.get("date")
            )
        elif request_type == "find_trains":
            return await self.railways_service.find_trains_between_stations(
                params.parameters.get("source", ""),
                params.parameters.get("destination", ""),
                params.parameters.get("date"),
                params.parameters.get("class_preference")
            )
        elif request_type == "check_availability":
            return await self.railways_service.check_ticket_availability(
                params.parameters.get("train_number", ""),
                params.parameters.get("source", ""),
                params.parameters.get("destination", ""),
                params.parameters.get("date", ""),
                params.parameters.get("class_type")
            )
        elif request_type == "pnr_status":
            return await self.railways_service.get_pnr_status(
                params.parameters.get("pnr_number", "")
            )
        elif request_type == "natural_language":
            return await self.railways_service.process_natural_language_query(
                params.parameters.get("query", "")
            )
        else:
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Unknown railways request type: {request_type}",
                response_time=0.1
            )
    
    async def _handle_weather_request(self, params: ServiceParameters) -> ServiceResult:
        """Handle weather service requests."""
        request_type = params.parameters.get("request_type", "weather_info")
        
        if request_type == "weather_info":
            return await self.weather_service.get_weather_info(
                params.parameters.get("city", ""),
                params.parameters.get("include_forecast", False),
                params.parameters.get("include_monsoon", True)
            )
        elif request_type == "local_transport":
            return await self.weather_service.get_local_transport_info(
                params.parameters.get("city", ""),
                params.parameters.get("transport_type")
            )
        else:
            return ServiceResult(
                service_type=ServiceType.WEATHER,
                success=False,
                data={},
                error_message=f"Unknown weather request type: {request_type}",
                response_time=0.1
            )
    
    async def _handle_cricket_request(self, params: ServiceParameters) -> ServiceResult:
        """Handle cricket scores requests."""
        return await self.weather_service.get_cricket_scores(
            params.parameters.get("match_type"),
            params.parameters.get("team")
        )
    
    async def _handle_bollywood_request(self, params: ServiceParameters) -> ServiceResult:
        """Handle Bollywood news requests."""
        return await self.weather_service.get_bollywood_news(
            params.parameters.get("category"),
            params.parameters.get("limit", 10)
        )
    
    async def _handle_government_service_request(self, params: ServiceParameters) -> ServiceResult:
        """Handle government service requests."""
        request_type = params.parameters.get("request_type", "")
        
        if request_type == "service_info":
            return await self.digital_india_service.get_service_information(
                params.parameters.get("service_name", ""),
                params.parameters.get("category")
            )
        elif request_type == "document_requirements":
            return await self.digital_india_service.get_document_requirements(
                params.parameters.get("service_name", ""),
                params.parameters.get("user_category")
            )
        elif request_type == "application_guidance":
            return await self.digital_india_service.get_application_guidance(
                params.parameters.get("service_name", ""),
                params.parameters.get("step_number")
            )
        elif request_type == "search_by_category":
            category_str = params.parameters.get("category", "")
            try:
                category = ServiceCategory(category_str)
                return await self.digital_india_service.search_services_by_category(
                    category,
                    params.parameters.get("limit", 10)
                )
            except ValueError:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Invalid service category: {category_str}",
                    response_time=0.1
                )
        elif request_type == "state_portal":
            return await self.digital_india_service.get_state_portal_info(
                params.parameters.get("state_name", "")
            )
        else:
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Unknown government service request type: {request_type}",
                response_time=0.1
            )
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all external services."""
        try:
            health_status = {
                "overall_status": "healthy",
                "services": {},
                "last_checked": asyncio.get_event_loop().time()
            }
            
            # Check each service (simplified health check)
            services_to_check = [
                ("indian_railways", self.railways_service),
                ("weather", self.weather_service),
                ("digital_india", self.digital_india_service)
            ]
            
            for service_name, service_instance in services_to_check:
                try:
                    # Simple health check - could be expanded
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": 0.1,
                        "last_error": None
                    }
                except Exception as e:
                    health_status["services"][service_name] = {
                        "status": "unhealthy",
                        "response_time": None,
                        "last_error": str(e)
                    }
                    health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking service health: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "last_checked": asyncio.get_event_loop().time()
            }
    
    async def get_supported_services(self) -> Dict[str, Any]:
        """Get list of supported external services."""
        return {
            "supported_services": [
                {
                    "service_type": "indian_railways",
                    "description": "Train schedules, booking, PNR status",
                    "capabilities": [
                        "Train schedule lookup",
                        "Find trains between stations",
                        "Check ticket availability",
                        "PNR status tracking",
                        "Natural language queries"
                    ]
                },
                {
                    "service_type": "weather",
                    "description": "Weather information with monsoon data",
                    "capabilities": [
                        "Current weather conditions",
                        "5-day weather forecast",
                        "Monsoon information",
                        "Air quality index",
                        "Local transport information"
                    ]
                },
                {
                    "service_type": "cricket_scores",
                    "description": "Live cricket scores and match information",
                    "capabilities": [
                        "Live match scores",
                        "Match schedules",
                        "Team-specific updates",
                        "Tournament information"
                    ]
                },
                {
                    "service_type": "bollywood_news",
                    "description": "Latest Bollywood news and updates",
                    "capabilities": [
                        "Movie news",
                        "Celebrity updates",
                        "Box office collections",
                        "Awards and events"
                    ]
                },
                {
                    "service_type": "government_services",
                    "description": "Digital India government services",
                    "capabilities": [
                        "Service information lookup",
                        "Document requirements",
                        "Application guidance",
                        "State portal information",
                        "Service category search"
                    ]
                }
            ],
            "total_services": 5,
            "last_updated": asyncio.get_event_loop().time()
        }


class ExternalServiceManager(ServiceManager):
    """Alias for ServiceManager for backward compatibility."""
    pass