"""
External service integrations for BharatVoice Assistant.

This package provides integrations with various Indian services including
railways, weather, government services, and local platforms.
"""

from .indian_railways_service import IndianRailwaysService
from .weather_service import WeatherService
from .digital_india_service import DigitalIndiaService
from .service_manager import ExternalServiceManager

__all__ = [
    "IndianRailwaysService",
    "WeatherService", 
    "DigitalIndiaService",
    "ExternalServiceManager"
]