"""
Weather and Local Services Integration.

This module provides integration with Indian weather services, local transportation,
cricket scores, and Bollywood news with monsoon-specific information.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from bharatvoice.core.models import ServiceResult, ServiceType, WeatherData


class WeatherCondition(str, Enum):
    """Weather condition types."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    THUNDERSTORM = "thunderstorm"
    FOGGY = "foggy"
    HAZY = "hazy"
    MONSOON = "monsoon"


class MonsoonPhase(str, Enum):
    """Monsoon phases in India."""
    PRE_MONSOON = "pre_monsoon"
    SOUTHWEST_MONSOON = "southwest_monsoon"
    POST_MONSOON = "post_monsoon"
    NORTHEAST_MONSOON = "northeast_monsoon"
    WINTER = "winter"


@dataclass
class MonsoonInfo:
    """Monsoon-specific information."""
    current_phase: MonsoonPhase
    onset_date: Optional[str]
    withdrawal_date: Optional[str]
    rainfall_percentage: float  # Percentage of normal rainfall
    is_active: bool
    intensity: str  # light, moderate, heavy, very_heavy


@dataclass
class AirQualityData:
    """Air quality information."""
    aqi: int
    category: str  # Good, Satisfactory, Moderate, Poor, Very Poor, Severe
    pm25: float
    pm10: float
    dominant_pollutant: str
    health_advice: str


@dataclass
class CricketMatch:
    """Cricket match information."""
    match_id: str
    teams: List[str]
    format: str  # Test, ODI, T20
    status: str  # Live, Upcoming, Completed
    score: Optional[str]
    venue: str
    date: str
    result: Optional[str]


@dataclass
class BollywoodNews:
    """Bollywood news item."""
    title: str
    summary: str
    category: str  # Movies, Celebrity, Box Office, Awards
    published_date: str
    source: str
    image_url: Optional[str]


@dataclass
class LocalTransport:
    """Local transportation information."""
    service_type: str  # Bus, Metro, Auto, Taxi
    route: str
    fare: float
    frequency: str
    operating_hours: str
    current_status: str


class WeatherService:
    """
    Service for weather and local information integration.
    
    Provides Indian weather data with monsoon information, local transportation,
    cricket scores, and Bollywood news.
    """
    
    def __init__(self, weather_api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.weather_api_key = weather_api_key
        self.weather_base_url = "https://api.openweathermap.org/data/2.5"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Indian city coordinates for weather lookup
        self._city_coordinates = self._initialize_city_coordinates()
        self._monsoon_regions = self._initialize_monsoon_regions()
        self._transport_services = self._initialize_transport_services()
    
    def _initialize_city_coordinates(self) -> Dict[str, Dict[str, float]]:
        """Initialize major Indian city coordinates."""
        return {
            "new delhi": {"lat": 28.6139, "lon": 77.2090},
            "mumbai": {"lat": 19.0760, "lon": 72.8777},
            "kolkata": {"lat": 22.5726, "lon": 88.3639},
            "chennai": {"lat": 13.0827, "lon": 80.2707},
            "bangalore": {"lat": 12.9716, "lon": 77.5946},
            "hyderabad": {"lat": 17.3850, "lon": 78.4867},
            "pune": {"lat": 18.5204, "lon": 73.8567},
            "ahmedabad": {"lat": 23.0225, "lon": 72.5714},
            "jaipur": {"lat": 26.9124, "lon": 75.7873},
            "lucknow": {"lat": 26.8467, "lon": 80.9462},
            "kanpur": {"lat": 26.4499, "lon": 80.3319},
            "nagpur": {"lat": 21.1458, "lon": 79.0882},
            "indore": {"lat": 22.7196, "lon": 75.8577},
            "bhopal": {"lat": 23.2599, "lon": 77.4126},
            "visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
            "patna": {"lat": 25.5941, "lon": 85.1376},
            "vadodara": {"lat": 22.3072, "lon": 73.1812},
            "ghaziabad": {"lat": 28.6692, "lon": 77.4538},
            "ludhiana": {"lat": 30.9010, "lon": 75.8573},
            "agra": {"lat": 27.1767, "lon": 78.0081},
            "nashik": {"lat": 19.9975, "lon": 73.7898},
            "faridabad": {"lat": 28.4089, "lon": 77.3178},
            "meerut": {"lat": 28.9845, "lon": 77.7064},
            "rajkot": {"lat": 22.3039, "lon": 70.8022},
            "kalyan": {"lat": 19.2437, "lon": 73.1355},
            "vasai": {"lat": 19.4912, "lon": 72.8054},
            "varanasi": {"lat": 25.3176, "lon": 82.9739},
            "srinagar": {"lat": 34.0837, "lon": 74.7973},
            "aurangabad": {"lat": 19.8762, "lon": 75.3433},
            "dhanbad": {"lat": 23.7957, "lon": 86.4304},
            "amritsar": {"lat": 31.6340, "lon": 74.8723},
            "navi mumbai": {"lat": 19.0330, "lon": 73.0297},
            "allahabad": {"lat": 25.4358, "lon": 81.8463},
            "ranchi": {"lat": 23.3441, "lon": 85.3096},
            "howrah": {"lat": 22.5958, "lon": 88.2636},
            "coimbatore": {"lat": 11.0168, "lon": 76.9558},
            "jabalpur": {"lat": 23.1815, "lon": 79.9864},
            "gwalior": {"lat": 26.2183, "lon": 78.1828},
            "vijayawada": {"lat": 16.5062, "lon": 80.6480},
            "jodhpur": {"lat": 26.2389, "lon": 73.0243},
            "madurai": {"lat": 9.9252, "lon": 78.1198},
            "raipur": {"lat": 21.2514, "lon": 81.6296},
            "kota": {"lat": 25.2138, "lon": 75.8648},
            "chandigarh": {"lat": 30.7333, "lon": 76.7794},
            "guwahati": {"lat": 26.1445, "lon": 91.7362},
            "solapur": {"lat": 17.6599, "lon": 75.9064},
            "hubli": {"lat": 15.3647, "lon": 75.1240},
            "bareilly": {"lat": 28.3670, "lon": 79.4304},
            "moradabad": {"lat": 28.8386, "lon": 78.7733},
            "mysore": {"lat": 12.2958, "lon": 76.6394},
            "gurgaon": {"lat": 28.4595, "lon": 77.0266},
            "aligarh": {"lat": 27.8974, "lon": 78.0880},
            "jalandhar": {"lat": 31.3260, "lon": 75.5762},
            "tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
            "bhubaneswar": {"lat": 20.2961, "lon": 85.8245},
            "salem": {"lat": 11.6643, "lon": 78.1460},
            "warangal": {"lat": 17.9689, "lon": 79.5941},
            "mira": {"lat": 19.2952, "lon": 72.8694},
            "thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
            "bhiwandi": {"lat": 19.3002, "lon": 73.0635},
            "saharanpur": {"lat": 29.9680, "lon": 77.5552},
            "guntur": {"lat": 16.3067, "lon": 80.4365},
            "amravati": {"lat": 20.9374, "lon": 77.7796},
            "bikaner": {"lat": 28.0229, "lon": 73.3119},
            "noida": {"lat": 28.5355, "lon": 77.3910},
            "jamshedpur": {"lat": 22.8046, "lon": 86.2029},
            "bhilai nagar": {"lat": 21.1938, "lon": 81.3509},
            "cuttack": {"lat": 20.4625, "lon": 85.8828},
            "firozabad": {"lat": 27.1592, "lon": 78.3957},
            "kochi": {"lat": 9.9312, "lon": 76.2673},
            "bhavnagar": {"lat": 21.7645, "lon": 72.1519},
            "dehradun": {"lat": 30.3165, "lon": 78.0322},
            "durgapur": {"lat": 23.4800, "lon": 87.3119},
            "asansol": {"lat": 23.6739, "lon": 86.9524},
            "nanded": {"lat": 19.1383, "lon": 77.3210},
            "kolhapur": {"lat": 16.7050, "lon": 74.2433},
            "ajmer": {"lat": 26.4499, "lon": 74.6399},
            "akola": {"lat": 20.7002, "lon": 77.0082},
            "gulbarga": {"lat": 17.3297, "lon": 76.8343},
            "jamnagar": {"lat": 22.4707, "lon": 70.0577},
            "ujjain": {"lat": 23.1765, "lon": 75.7885},
            "loni": {"lat": 28.7333, "lon": 77.2833},
            "siliguri": {"lat": 26.7271, "lon": 88.3953},
            "jhansi": {"lat": 25.4484, "lon": 78.5685},
            "ulhasnagar": {"lat": 19.2215, "lon": 73.1645},
            "jammu": {"lat": 32.7266, "lon": 74.8570},
            "sangli": {"lat": 16.8524, "lon": 74.5815},
            "mangalore": {"lat": 12.9141, "lon": 74.8560},
            "erode": {"lat": 11.3410, "lon": 77.7172},
            "belgaum": {"lat": 15.8497, "lon": 74.4977},
            "ambattur": {"lat": 13.1143, "lon": 80.1548},
            "tirunelveli": {"lat": 8.7139, "lon": 77.7567},
            "malegaon": {"lat": 20.5579, "lon": 74.5287},
            "gaya": {"lat": 24.7914, "lon": 85.0002},
            "jalgaon": {"lat": 21.0077, "lon": 75.5626},
            "udaipur": {"lat": 24.5854, "lon": 73.7125},
            "maheshtala": {"lat": 22.4977, "lon": 88.2492}
        }
    
    def _initialize_monsoon_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize monsoon region information."""
        return {
            "western_ghats": {
                "states": ["Maharashtra", "Karnataka", "Kerala", "Tamil Nadu"],
                "peak_months": ["June", "July", "August", "September"],
                "average_rainfall": 2500,  # mm
                "monsoon_type": "southwest"
            },
            "northern_plains": {
                "states": ["Punjab", "Haryana", "Uttar Pradesh", "Bihar"],
                "peak_months": ["July", "August", "September"],
                "average_rainfall": 1000,
                "monsoon_type": "southwest"
            },
            "eastern_india": {
                "states": ["West Bengal", "Odisha", "Jharkhand"],
                "peak_months": ["June", "July", "August", "September"],
                "average_rainfall": 1500,
                "monsoon_type": "southwest"
            },
            "southern_peninsula": {
                "states": ["Andhra Pradesh", "Telangana", "Tamil Nadu"],
                "peak_months": ["October", "November", "December"],
                "average_rainfall": 800,
                "monsoon_type": "northeast"
            },
            "central_india": {
                "states": ["Madhya Pradesh", "Chhattisgarh"],
                "peak_months": ["July", "August", "September"],
                "average_rainfall": 1200,
                "monsoon_type": "southwest"
            },
            "rajasthan": {
                "states": ["Rajasthan"],
                "peak_months": ["July", "August"],
                "average_rainfall": 400,
                "monsoon_type": "southwest"
            }
        }
    
    def _initialize_transport_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize comprehensive local transport service information."""
        return {
            "delhi": [
                {
                    "type": "Metro",
                    "name": "Delhi Metro",
                    "routes": ["Red Line", "Blue Line", "Yellow Line", "Green Line", "Violet Line", "Pink Line"],
                    "fare_range": "10-60 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "2-5 minutes",
                    "stations": 285,
                    "accessibility": "Wheelchair accessible",
                    "payment_methods": ["Metro Card", "UPI", "Contactless Cards"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "DTC Bus",
                    "routes": ["Cluster", "Low Floor", "CNG", "Electric"],
                    "fare_range": "5-25 INR",
                    "operating_hours": "05:00-23:30",
                    "frequency": "5-15 minutes",
                    "fleet_size": "6500+ buses",
                    "accessibility": "Low floor buses available",
                    "payment_methods": ["DTC Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Auto Rickshaw",
                    "name": "Delhi Auto",
                    "fare_range": "25-200 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI", "Paytm"],
                    "booking_apps": ["Ola", "Uber", "Rapido"],
                    "current_status": "Available"
                }
            ],
            "mumbai": [
                {
                    "type": "Local Train",
                    "name": "Mumbai Local",
                    "routes": ["Western", "Central", "Harbour", "Trans-Harbour"],
                    "fare_range": "5-40 INR",
                    "operating_hours": "04:00-01:00",
                    "frequency": "3-5 minutes",
                    "daily_passengers": "7.5 million",
                    "accessibility": "Limited wheelchair access",
                    "payment_methods": ["Season Pass", "UTS App", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Metro",
                    "name": "Mumbai Metro",
                    "routes": ["Line 1 (Versova-Andheri-Ghatkopar)", "Line 2A", "Line 7"],
                    "fare_range": "10-40 INR",
                    "operating_hours": "05:30-22:30",
                    "frequency": "4-8 minutes",
                    "accessibility": "Fully wheelchair accessible",
                    "payment_methods": ["Metro Card", "UPI", "QR Code"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "BEST Bus",
                    "routes": ["AC", "Non-AC", "Electric"],
                    "fare_range": "8-50 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "10-20 minutes",
                    "fleet_size": "3200+ buses",
                    "payment_methods": ["BEST Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Taxi",
                    "name": "Mumbai Taxi",
                    "fare_range": "25-500 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI", "Card"],
                    "booking_apps": ["Ola", "Uber", "Meru"],
                    "current_status": "Available"
                }
            ],
            "bangalore": [
                {
                    "type": "Metro",
                    "name": "Namma Metro",
                    "routes": ["Purple Line", "Green Line", "Blue Line (Under Construction)"],
                    "fare_range": "10-60 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "3-10 minutes",
                    "stations": 66,
                    "accessibility": "Wheelchair accessible",
                    "payment_methods": ["Namma Metro Card", "UPI", "QR Code"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "BMTC",
                    "routes": ["Volvo", "Ordinary", "Vajra", "Big 10"],
                    "fare_range": "5-50 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "5-20 minutes",
                    "fleet_size": "6500+ buses",
                    "accessibility": "Low floor buses available",
                    "payment_methods": ["BMTC Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Auto Rickshaw",
                    "name": "Bangalore Auto",
                    "fare_range": "30-300 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI", "Paytm"],
                    "booking_apps": ["Ola", "Uber", "Rapido", "Namma Yatri"],
                    "current_status": "Available"
                }
            ],
            "chennai": [
                {
                    "type": "Metro",
                    "name": "Chennai Metro",
                    "routes": ["Blue Line", "Green Line"],
                    "fare_range": "8-50 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "5-10 minutes",
                    "accessibility": "Wheelchair accessible",
                    "payment_methods": ["Chennai Metro Card", "UPI", "Tokens"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "MTC Bus",
                    "routes": ["Ordinary", "Deluxe", "AC", "Volvo"],
                    "fare_range": "3-40 INR",
                    "operating_hours": "04:30-23:30",
                    "frequency": "5-15 minutes",
                    "fleet_size": "3500+ buses",
                    "payment_methods": ["MTC Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Auto Rickshaw",
                    "name": "Chennai Auto",
                    "fare_range": "25-250 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI"],
                    "booking_apps": ["Ola", "Uber", "Rapido"],
                    "current_status": "Available"
                }
            ],
            "kolkata": [
                {
                    "type": "Metro",
                    "name": "Kolkata Metro",
                    "routes": ["North-South Line", "East-West Line", "New Garia-Airport Line"],
                    "fare_range": "5-25 INR",
                    "operating_hours": "06:30-21:45",
                    "frequency": "5-12 minutes",
                    "accessibility": "Limited wheelchair access",
                    "payment_methods": ["Metro Card", "Tokens", "UPI"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "CTC Bus",
                    "routes": ["Ordinary", "AC", "Mini Bus"],
                    "fare_range": "6-30 INR",
                    "operating_hours": "05:00-22:00",
                    "frequency": "10-20 minutes",
                    "payment_methods": ["CTC Card", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Tram",
                    "name": "Kolkata Tram",
                    "routes": ["Heritage tram routes"],
                    "fare_range": "5-10 INR",
                    "operating_hours": "06:00-20:00",
                    "frequency": "15-30 minutes",
                    "heritage_status": "UNESCO Heritage Transport",
                    "current_status": "Limited operations"
                },
                {
                    "type": "Taxi",
                    "name": "Yellow Taxi",
                    "fare_range": "25-300 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI"],
                    "booking_apps": ["Ola", "Uber"],
                    "current_status": "Available"
                }
            ],
            "hyderabad": [
                {
                    "type": "Metro",
                    "name": "Hyderabad Metro",
                    "routes": ["Blue Line", "Green Line", "Red Line"],
                    "fare_range": "10-60 INR",
                    "operating_hours": "06:00-23:00",
                    "frequency": "4-8 minutes",
                    "accessibility": "Wheelchair accessible",
                    "payment_methods": ["Metro Card", "UPI", "QR Code"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Bus",
                    "name": "TSRTC Bus",
                    "routes": ["City Ordinary", "Metro Express", "AC"],
                    "fare_range": "5-40 INR",
                    "operating_hours": "05:00-23:00",
                    "frequency": "8-15 minutes",
                    "payment_methods": ["TSRTC Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                }
            ],
            "pune": [
                {
                    "type": "Bus",
                    "name": "PMPML Bus",
                    "routes": ["Ordinary", "AC", "BRT"],
                    "fare_range": "8-35 INR",
                    "operating_hours": "05:30-23:00",
                    "frequency": "10-20 minutes",
                    "payment_methods": ["PMPML Card", "UPI", "Cash"],
                    "current_status": "Normal operations"
                },
                {
                    "type": "Auto Rickshaw",
                    "name": "Pune Auto",
                    "fare_range": "20-200 INR",
                    "operating_hours": "24/7",
                    "payment_methods": ["Cash", "UPI"],
                    "booking_apps": ["Ola", "Uber", "Rapido"],
                    "current_status": "Available"
                }
            ]
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_weather_info(
        self,
        city: str,
        include_forecast: bool = False,
        include_monsoon: bool = True
    ) -> ServiceResult:
        """
        Get comprehensive weather information for Indian city.
        
        Args:
            city: City name
            include_forecast: Include 5-day forecast
            include_monsoon: Include monsoon-specific information
            
        Returns:
            Service result with weather data
        """
        try:
            self.logger.info(f"Getting weather info for {city}")
            
            city_lower = city.lower()
            coordinates = self._city_coordinates.get(city_lower)
            
            if not coordinates:
                return ServiceResult(
                    service_type=ServiceType.WEATHER,
                    success=False,
                    data={},
                    error_message=f"Weather data not available for {city}. Please try a major Indian city.",
                    response_time=0.3
                )
            
            # Get current weather (mock implementation)
            current_weather = await self._get_current_weather_mock(city, coordinates)
            
            weather_data = {
                "city": city,
                "current_weather": current_weather,
                "last_updated": datetime.now().isoformat()
            }
            
            # Add forecast if requested
            if include_forecast:
                forecast = await self._get_weather_forecast_mock(city, coordinates)
                weather_data["forecast"] = forecast
            
            # Add monsoon information if requested
            if include_monsoon:
                monsoon_info = await self._get_monsoon_info(city, coordinates)
                weather_data["monsoon_info"] = monsoon_info
            
            # Add air quality information
            air_quality = await self._get_air_quality_mock(city, coordinates)
            weather_data["air_quality"] = air_quality
            
            return ServiceResult(
                service_type=ServiceType.WEATHER,
                success=True,
                data=weather_data,
                error_message=None,
                response_time=1.2
            )
            
        except Exception as e:
            self.logger.error(f"Error getting weather info: {e}")
            return ServiceResult(
                service_type=ServiceType.WEATHER,
                success=False,
                data={},
                error_message=f"Failed to get weather information: {str(e)}",
                response_time=0.5
            )
    
    async def get_cricket_scores(
        self,
        match_type: Optional[str] = None,
        team: Optional[str] = None
    ) -> ServiceResult:
        """
        Get current cricket scores and match information.
        
        Args:
            match_type: Type of match (Test, ODI, T20)
            team: Specific team to filter by
            
        Returns:
            Service result with cricket scores
        """
        try:
            self.logger.info("Getting cricket scores")
            
            # Mock cricket data
            matches = await self._get_cricket_matches_mock(match_type, team)
            
            cricket_data = {
                "matches": matches,
                "last_updated": datetime.now().isoformat(),
                "total_matches": len(matches)
            }
            
            return ServiceResult(
                service_type=ServiceType.CRICKET_SCORES,
                success=True,
                data=cricket_data,
                error_message=None,
                response_time=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Error getting cricket scores: {e}")
            return ServiceResult(
                service_type=ServiceType.CRICKET_SCORES,
                success=False,
                data={},
                error_message=f"Failed to get cricket scores: {str(e)}",
                response_time=0.5
            )
    
    async def get_bollywood_news(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> ServiceResult:
        """
        Get latest Bollywood news and updates.
        
        Args:
            category: News category (Movies, Celebrity, Box Office, Awards)
            limit: Number of news items to return
            
        Returns:
            Service result with Bollywood news
        """
        try:
            self.logger.info("Getting Bollywood news")
            
            # Mock Bollywood news data
            news_items = await self._get_bollywood_news_mock(category, limit)
            
            news_data = {
                "news": news_items,
                "category": category or "All",
                "total_items": len(news_items),
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.BOLLYWOOD_NEWS,
                success=True,
                data=news_data,
                error_message=None,
                response_time=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error getting Bollywood news: {e}")
            return ServiceResult(
                service_type=ServiceType.BOLLYWOOD_NEWS,
                success=False,
                data={},
                error_message=f"Failed to get Bollywood news: {str(e)}",
                response_time=0.5
            )
    
    async def get_local_transport_info(
        self,
        city: str,
        transport_type: Optional[str] = None
    ) -> ServiceResult:
        """
        Get local transportation information.
        
        Args:
            city: City name
            transport_type: Type of transport (Metro, Bus, Train)
            
        Returns:
            Service result with transport information
        """
        try:
            self.logger.info(f"Getting transport info for {city}")
            
            city_lower = city.lower()
            transport_services = self._transport_services.get(city_lower, [])
            
            if not transport_services:
                return ServiceResult(
                    service_type=ServiceType.WEATHER,  # Using weather as general local service
                    success=False,
                    data={},
                    error_message=f"Transport information not available for {city}.",
                    response_time=0.3
                )
            
            # Filter by transport type if specified
            if transport_type:
                transport_services = [
                    service for service in transport_services
                    if service["type"].lower() == transport_type.lower()
                ]
            
            transport_data = {
                "city": city,
                "transport_services": transport_services,
                "total_services": len(transport_services),
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.WEATHER,  # Using weather as general local service
                success=True,
                data=transport_data,
                error_message=None,
                response_time=0.6
            )
            
        except Exception as e:
            self.logger.error(f"Error getting transport info: {e}")
            return ServiceResult(
                service_type=ServiceType.WEATHER,
                success=False,
                data={},
                error_message=f"Failed to get transport information: {str(e)}",
                response_time=0.5
            )
    
    # Mock API methods
    
    async def _get_current_weather_mock(
        self,
        city: str,
        coordinates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Enhanced mock current weather API call with improved Indian context."""
        await asyncio.sleep(0.5)
        
        import random
        
        # Enhanced weather data with better Indian context
        current_month = datetime.now().month
        lat = coordinates["lat"]
        
        # Temperature ranges based on season and location
        if current_month in [6, 7, 8, 9]:  # Monsoon season
            temp_celsius = random.randint(22, 35)
            conditions = ["rainy", "cloudy", "thunderstorm", "monsoon"]
            precipitation = random.randint(5, 80)
        elif current_month in [12, 1, 2]:  # Winter
            if lat > 25:  # Northern India - colder
                temp_celsius = random.randint(8, 25)
            else:  # Southern India - milder
                temp_celsius = random.randint(18, 30)
            conditions = ["clear", "foggy", "hazy"]
            precipitation = random.randint(0, 2)
        elif current_month in [3, 4, 5]:  # Summer/Pre-monsoon
            temp_celsius = random.randint(28, 45)
            conditions = ["clear", "hazy", "cloudy"]
            precipitation = random.randint(0, 10)
        else:  # Post-monsoon
            temp_celsius = random.randint(20, 35)
            conditions = ["clear", "cloudy", "hazy"]
            precipitation = random.randint(0, 15)
        
        condition = random.choice(conditions)
        humidity = random.randint(40, 90) if condition in ["rainy", "monsoon"] else random.randint(30, 70)
        
        # Enhanced weather data with Indian-specific information
        return {
            "temperature_celsius": temp_celsius,
            "feels_like_celsius": temp_celsius + random.randint(-3, 8),
            "humidity": humidity,
            "condition": condition,
            "description": self._get_enhanced_weather_description(condition, temp_celsius, current_month),
            "wind_speed_kmh": random.randint(5, 35),
            "precipitation_mm": precipitation,
            "visibility_km": random.randint(2, 8) if condition == "foggy" else random.randint(8, 15),
            "uv_index": random.randint(1, 11),
            "sunrise": self._get_sunrise_time(lat, current_month),
            "sunset": self._get_sunset_time(lat, current_month),
            "heat_index": self._calculate_heat_index(temp_celsius, humidity),
            "seasonal_info": self._get_seasonal_info(current_month, lat)
        }
    
    async def _get_weather_forecast_mock(
        self,
        city: str,
        coordinates: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Mock weather forecast API call."""
        await asyncio.sleep(0.3)
        
        forecast = []
        base_date = datetime.now()
        
        for i in range(5):
            date = base_date + timedelta(days=i+1)
            temp_high = random.randint(25, 40)
            temp_low = temp_high - random.randint(5, 15)
            
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "temperature_high_celsius": temp_high,
                "temperature_low_celsius": temp_low,
                "condition": random.choice(["clear", "cloudy", "rainy", "thunderstorm"]),
                "humidity": random.randint(40, 85),
                "precipitation_chance": random.randint(0, 80),
                "wind_speed_kmh": random.randint(8, 20)
            })
        
        return forecast
    
    async def _get_monsoon_info(
        self,
        city: str,
        coordinates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get monsoon-specific information."""
        await asyncio.sleep(0.2)
        
        # Determine monsoon region
        lat = coordinates["lat"]
        
        if lat > 28:  # Northern India
            region = "northern_plains"
        elif lat < 15:  # Southern India
            region = "southern_peninsula"
        elif coordinates["lon"] < 75:  # Western India
            region = "western_ghats"
        else:  # Central/Eastern India
            region = "central_india"
        
        region_info = self._monsoon_regions.get(region, self._monsoon_regions["central_india"])
        
        # Determine current monsoon phase
        current_month = datetime.now().month
        if current_month in [3, 4, 5]:
            phase = MonsoonPhase.PRE_MONSOON
        elif current_month in [6, 7, 8, 9]:
            phase = MonsoonPhase.SOUTHWEST_MONSOON
        elif current_month in [10, 11]:
            phase = MonsoonPhase.POST_MONSOON
        else:
            phase = MonsoonPhase.WINTER
        
        return {
            "current_phase": phase.value,
            "region": region,
            "is_monsoon_active": phase in [MonsoonPhase.SOUTHWEST_MONSOON, MonsoonPhase.NORTHEAST_MONSOON],
            "peak_months": region_info["peak_months"],
            "average_rainfall_mm": region_info["average_rainfall"],
            "monsoon_type": region_info["monsoon_type"],
            "rainfall_percentage_of_normal": random.randint(70, 130),
            "onset_prediction": "First week of June" if region != "southern_peninsula" else "Mid October",
            "withdrawal_prediction": "End of September" if region != "southern_peninsula" else "End of December"
        }
    
    async def _get_air_quality_mock(
        self,
        city: str,
        coordinates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Mock air quality API call."""
        await asyncio.sleep(0.3)
        
        import random
        
        # Mock AQI data - higher for major cities
        major_cities = ["new delhi", "mumbai", "kolkata", "chennai", "bangalore"]
        if city.lower() in major_cities:
            aqi = random.randint(100, 300)  # Moderate to Poor
        else:
            aqi = random.randint(50, 150)   # Good to Moderate
        
        # Determine category
        if aqi <= 50:
            category = "Good"
            health_advice = "Air quality is satisfactory for most people."
        elif aqi <= 100:
            category = "Satisfactory"
            health_advice = "Air quality is acceptable for most people."
        elif aqi <= 200:
            category = "Moderate"
            health_advice = "Sensitive individuals should consider limiting outdoor activities."
        elif aqi <= 300:
            category = "Poor"
            health_advice = "Everyone should limit outdoor activities."
        else:
            category = "Very Poor"
            health_advice = "Avoid outdoor activities. Use air purifiers indoors."
        
        return {
            "aqi": aqi,
            "category": category,
            "pm25": random.randint(20, 150),
            "pm10": random.randint(30, 200),
            "dominant_pollutant": random.choice(["PM2.5", "PM10", "NO2", "O3"]),
            "health_advice": health_advice,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _get_cricket_matches_mock(
        self,
        match_type: Optional[str],
        team: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Enhanced mock cricket matches API call with comprehensive Indian cricket data."""
        await asyncio.sleep(0.4)
        
        # Enhanced cricket matches with more realistic data
        matches = [
            {
                "match_id": "IND_vs_AUS_2024_T20_1",
                "teams": ["India", "Australia"],
                "format": "T20",
                "status": "Live",
                "score": "India 165/4 (18.2 ov) vs Australia 160/8 (20 ov)",
                "venue": "Wankhede Stadium, Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "result": None,
                "current_over": "18.2",
                "target": "161 runs",
                "required_rate": "8.5 RPO",
                "key_players": {
                    "india": ["Virat Kohli", "Rohit Sharma", "Jasprit Bumrah"],
                    "australia": ["Steve Smith", "David Warner", "Pat Cummins"]
                },
                "match_situation": "India needs 6 runs from 10 balls",
                "weather_impact": "No weather interruptions expected"
            },
            {
                "match_id": "IPL_2024_MI_vs_CSK",
                "teams": ["Mumbai Indians", "Chennai Super Kings"],
                "format": "T20",
                "status": "Upcoming",
                "score": None,
                "venue": "M.A. Chidambaram Stadium, Chennai",
                "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "result": None,
                "match_time": "19:30 IST",
                "key_players": {
                    "mumbai_indians": ["Rohit Sharma", "Jasprit Bumrah", "Suryakumar Yadav"],
                    "chennai_super_kings": ["MS Dhoni", "Ravindra Jadeja", "Ruturaj Gaikwad"]
                },
                "head_to_head": "CSK leads 19-16 in IPL matches",
                "pitch_conditions": "Spin-friendly surface expected"
            },
            {
                "match_id": "IND_vs_ENG_2024_Test_1",
                "teams": ["India", "England"],
                "format": "Test",
                "status": "Completed",
                "score": "India 445 & 255/3d vs England 218 & 292",
                "venue": "Rajiv Gandhi International Stadium, Hyderabad",
                "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "result": "India won by 190 runs",
                "key_players": {
                    "india": ["Yashasvi Jaiswal", "Ravichandran Ashwin", "Jasprit Bumrah"],
                    "england": ["Joe Root", "Ben Stokes", "James Anderson"]
                },
                "match_highlights": [
                    "Jaiswal scored brilliant 209 in first innings",
                    "Ashwin took 9 wickets in the match",
                    "England collapsed in second innings"
                ]
            },
            {
                "match_id": "WI_vs_IND_2024_ODI_2",
                "teams": ["West Indies", "India"],
                "format": "ODI",
                "status": "Live",
                "score": "West Indies 245/8 (45.3 ov) vs India 180/3 (32 ov)",
                "venue": "Kensington Oval, Barbados",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "result": None,
                "current_over": "32.0",
                "target": "246 runs",
                "required_rate": "3.67 RPO",
                "key_players": {
                    "west_indies": ["Shai Hope", "Nicholas Pooran", "Alzarri Joseph"],
                    "india": ["Shubman Gill", "Virat Kohli", "Mohammed Siraj"]
                },
                "match_situation": "India cruising towards target",
                "partnership": "Kohli and Gill - 85 runs for 3rd wicket"
            },
            {
                "match_id": "RCB_vs_KKR_2024_IPL",
                "teams": ["Royal Challengers Bangalore", "Kolkata Knight Riders"],
                "format": "T20",
                "status": "Upcoming",
                "score": None,
                "venue": "M. Chinnaswamy Stadium, Bangalore",
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "result": None,
                "match_time": "15:30 IST",
                "key_players": {
                    "rcb": ["Virat Kohli", "Glenn Maxwell", "Mohammed Siraj"],
                    "kkr": ["Shreyas Iyer", "Andre Russell", "Sunil Narine"]
                },
                "venue_stats": "High-scoring ground, average first innings score: 180",
                "weather_forecast": "Clear skies, no rain expected"
            },
            {
                "match_id": "IND_W_vs_AUS_W_2024_T20",
                "teams": ["India Women", "Australia Women"],
                "format": "T20",
                "status": "Completed",
                "score": "India Women 151/7 (20 ov) vs Australia Women 155/4 (19.2 ov)",
                "venue": "DY Patil Stadium, Mumbai",
                "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "result": "Australia Women won by 6 wickets",
                "key_players": {
                    "india_women": ["Smriti Mandhana", "Harmanpreet Kaur", "Renuka Singh"],
                    "australia_women": ["Alyssa Healy", "Beth Mooney", "Megan Schutt"]
                },
                "player_of_match": "Alyssa Healy (65* off 42 balls)"
            }
        ]
        
        # Filter by match type if specified
        if match_type:
            matches = [m for m in matches if m["format"].lower() == match_type.lower()]
        
        # Filter by team if specified
        if team:
            team_lower = team.lower()
            matches = [
                m for m in matches 
                if any(team_lower in t.lower() for t in m["teams"])
            ]
        
        return matches
    
    async def _get_bollywood_news_mock(
        self,
        category: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Enhanced mock Bollywood news API call with comprehensive Indian entertainment content."""
        await asyncio.sleep(0.6)
        
        # Enhanced Bollywood news with more diverse and current content
        news_items = [
            {
                "title": "Shah Rukh Khan's 'Dunki' Crosses ₹400 Crores Worldwide",
                "summary": "Rajkumar Hirani's directorial featuring Shah Rukh Khan achieves massive box office success, becoming one of the highest-grossing films of 2024.",
                "category": "Box Office",
                "published_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "Bollywood Hungama",
                "image_url": None,
                "tags": ["Shah Rukh Khan", "Dunki", "Box Office", "Rajkumar Hirani"],
                "read_time": "3 min read",
                "trending_score": 95
            },
            {
                "title": "Deepika Padukone Announces Production House's Next Project",
                "summary": "The actress-producer reveals her production company's upcoming film focusing on women's empowerment in rural India.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d"),
                "source": "Film Companion",
                "image_url": None,
                "tags": ["Deepika Padukone", "Production", "Women Empowerment"],
                "read_time": "4 min read",
                "trending_score": 88
            },
            {
                "title": "Ranbir Kapoor and Alia Bhatt's 'Brahmastra 2' Gets Release Date",
                "summary": "The highly anticipated sequel to the superhero fantasy film gets an official release date of December 2025.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d"),
                "source": "Variety India",
                "image_url": None,
                "tags": ["Ranbir Kapoor", "Alia Bhatt", "Brahmastra", "Sequel"],
                "read_time": "2 min read",
                "trending_score": 92
            },
            {
                "title": "Priyanka Chopra Wins International Recognition at Cannes",
                "summary": "Global icon Priyanka Chopra receives special recognition at Cannes Film Festival for her contribution to international cinema.",
                "category": "Awards",
                "published_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "source": "Entertainment Weekly",
                "image_url": None,
                "tags": ["Priyanka Chopra", "Cannes", "International", "Awards"],
                "read_time": "5 min read",
                "trending_score": 85
            },
            {
                "title": "Rajkummar Rao's 'Stree 3' Officially Announced",
                "summary": "The horror-comedy franchise continues with Rajkummar Rao and Shraddha Kapoor returning for the third installment.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "source": "Pinkvilla",
                "image_url": None,
                "tags": ["Rajkummar Rao", "Stree", "Horror Comedy", "Franchise"],
                "read_time": "3 min read",
                "trending_score": 90
            },
            {
                "title": "Kareena Kapoor Khan Launches Digital Platform for New Mothers",
                "summary": "The actress launches an innovative digital platform providing support and resources for new mothers across India.",
                "category": "Celebrity",
                "published_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "source": "Mumbai Mirror",
                "image_url": None,
                "tags": ["Kareena Kapoor", "Digital Platform", "Motherhood", "Social Impact"],
                "read_time": "4 min read",
                "trending_score": 78
            },
            {
                "title": "Ranveer Singh's '83' Sequel in Development",
                "summary": "Following the success of '83', makers announce a sequel focusing on India's 2011 World Cup victory.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "source": "Times of India",
                "image_url": None,
                "tags": ["Ranveer Singh", "83", "Cricket", "World Cup", "Sequel"],
                "read_time": "3 min read",
                "trending_score": 87
            },
            {
                "title": "Aamir Khan's Next Film to Address Climate Change",
                "summary": "The perfectionist actor announces his next project will focus on environmental issues and climate change awareness.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "source": "Indian Express",
                "image_url": None,
                "tags": ["Aamir Khan", "Climate Change", "Environment", "Social Message"],
                "read_time": "4 min read",
                "trending_score": 82
            },
            {
                "title": "Katrina Kaif's Production Debut with Women-Centric Thriller",
                "summary": "Katrina Kaif steps into production with a female-led thriller exploring cybercrime in modern India.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "source": "Filmfare",
                "image_url": None,
                "tags": ["Katrina Kaif", "Production", "Thriller", "Women-Centric"],
                "read_time": "3 min read",
                "trending_score": 80
            },
            {
                "title": "Hrithik Roshan's 'Krrish 4' Gets New Director",
                "summary": "The superhero franchise gets a fresh perspective with acclaimed director taking over the fourth installment.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
                "source": "DNA India",
                "image_url": None,
                "tags": ["Hrithik Roshan", "Krrish", "Superhero", "Director Change"],
                "read_time": "2 min read",
                "trending_score": 89
            },
            {
                "title": "Anushka Sharma Returns to Acting After 3-Year Break",
                "summary": "The actress announces her comeback with a Netflix original series focusing on women entrepreneurs.",
                "category": "Celebrity",
                "published_date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
                "source": "Hindustan Times",
                "image_url": None,
                "tags": ["Anushka Sharma", "Comeback", "Netflix", "Entrepreneurs"],
                "read_time": "4 min read",
                "trending_score": 86
            },
            {
                "title": "South Indian Cinema Dominates National Awards 2024",
                "summary": "Regional films from Tamil, Telugu, and Malayalam cinema sweep major categories at the National Film Awards.",
                "category": "Awards",
                "published_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "source": "The Hindu",
                "image_url": None,
                "tags": ["National Awards", "South Cinema", "Regional Films"],
                "read_time": "5 min read",
                "trending_score": 84
            },
            {
                "title": "Bollywood Embraces AI Technology for Film Production",
                "summary": "Major production houses announce integration of AI technology for script writing, editing, and visual effects.",
                "category": "Movies",
                "published_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "source": "Business Standard",
                "image_url": None,
                "tags": ["AI Technology", "Film Production", "Innovation", "VFX"],
                "read_time": "6 min read",
                "trending_score": 75
            }
        ]
        
        # Filter by category if specified
        if category:
            category_lower = category.lower()
            news_items = [
                item for item in news_items 
                if item["category"].lower() == category_lower
            ]
        
        # Sort by trending score and published date
        news_items.sort(key=lambda x: (x["trending_score"], x["published_date"]), reverse=True)
        
        return news_items[:limit]
    
    def _get_weather_description(self, condition: str, temperature: int) -> str:
        """Get weather description based on condition and temperature."""
        descriptions = {
            "clear": f"Clear skies with temperature {temperature}°C",
            "cloudy": f"Cloudy weather with temperature {temperature}°C",
            "rainy": f"Rainy conditions with temperature {temperature}°C",
            "thunderstorm": f"Thunderstorms expected with temperature {temperature}°C",
            "foggy": f"Foggy conditions with temperature {temperature}°C",
            "hazy": f"Hazy weather with temperature {temperature}°C",
            "monsoon": f"Monsoon conditions with temperature {temperature}°C"
        }
        
        return descriptions.get(condition, f"Weather condition: {condition}, temperature {temperature}°C")
    
    def _get_enhanced_weather_description(self, condition: str, temperature: int, month: int) -> str:
        """Get enhanced weather description with Indian context."""
        seasonal_context = ""
        if month in [6, 7, 8, 9]:
            seasonal_context = " during monsoon season"
        elif month in [12, 1, 2]:
            seasonal_context = " in winter"
        elif month in [3, 4, 5]:
            seasonal_context = " in summer"
        else:
            seasonal_context = " in post-monsoon period"
        
        base_descriptions = {
            "clear": f"Clear and sunny skies{seasonal_context}",
            "cloudy": f"Cloudy conditions{seasonal_context}",
            "rainy": f"Rainy weather{seasonal_context}",
            "thunderstorm": f"Thunderstorms with heavy rain{seasonal_context}",
            "foggy": f"Dense fog reducing visibility{seasonal_context}",
            "hazy": f"Hazy conditions with reduced air quality{seasonal_context}",
            "monsoon": f"Active monsoon with intermittent showers"
        }
        
        description = base_descriptions.get(condition, f"{condition.title()} weather{seasonal_context}")
        
        # Add temperature context
        if temperature > 40:
            description += " - Very hot conditions"
        elif temperature > 35:
            description += " - Hot weather"
        elif temperature < 15:
            description += " - Cool conditions"
        elif temperature < 10:
            description += " - Cold weather"
        
        return f"{description}. Temperature: {temperature}°C"
    
    def _get_sunrise_time(self, lat: float, month: int) -> str:
        """Calculate approximate sunrise time based on latitude and month."""
        # Simplified sunrise calculation for Indian latitudes
        base_sunrise = 6.0  # 6:00 AM base
        
        # Adjust for latitude (northern cities have more variation)
        if lat > 28:  # Northern India
            if month in [12, 1]:
                base_sunrise += 0.5  # Later sunrise in winter
            elif month in [6, 7]:
                base_sunrise -= 0.3  # Earlier sunrise in summer
        
        # Convert to time string
        hours = int(base_sunrise)
        minutes = int((base_sunrise - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def _get_sunset_time(self, lat: float, month: int) -> str:
        """Calculate approximate sunset time based on latitude and month."""
        # Simplified sunset calculation for Indian latitudes
        base_sunset = 18.5  # 6:30 PM base
        
        # Adjust for latitude and season
        if lat > 28:  # Northern India
            if month in [12, 1]:
                base_sunset -= 0.5  # Earlier sunset in winter
            elif month in [6, 7]:
                base_sunset += 0.5  # Later sunset in summer
        
        # Convert to time string
        hours = int(base_sunset)
        minutes = int((base_sunset - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def _calculate_heat_index(self, temp_celsius: int, humidity: int) -> int:
        """Calculate heat index (feels like temperature) for Indian conditions."""
        # Simplified heat index calculation
        if temp_celsius < 27:
            return temp_celsius
        
        # Heat index becomes significant above 27°C with high humidity
        heat_index = temp_celsius
        if humidity > 60:
            heat_index += int((humidity - 60) * 0.1 * (temp_celsius - 27))
        
        return min(heat_index, temp_celsius + 10)  # Cap the increase
    
    def _get_seasonal_info(self, month: int, lat: float) -> Dict[str, str]:
        """Get seasonal information for Indian context."""
        if month in [6, 7, 8, 9]:
            return {
                "season": "Monsoon",
                "advice": "Carry umbrella, expect traffic delays due to rain",
                "clothing": "Light waterproof clothing recommended"
            }
        elif month in [12, 1, 2]:
            if lat > 25:
                return {
                    "season": "Winter",
                    "advice": "Cool weather, good for outdoor activities",
                    "clothing": "Light jacket or sweater recommended"
                }
            else:
                return {
                    "season": "Winter",
                    "advice": "Pleasant weather, ideal for sightseeing",
                    "clothing": "Light cotton clothing sufficient"
                }
        elif month in [3, 4, 5]:
            return {
                "season": "Summer",
                "advice": "Stay hydrated, avoid midday sun",
                "clothing": "Light cotton clothing, sun protection recommended"
            }
        else:
            return {
                "season": "Post-Monsoon",
                "advice": "Pleasant weather with occasional showers",
                "clothing": "Light clothing with light rain protection"
            }