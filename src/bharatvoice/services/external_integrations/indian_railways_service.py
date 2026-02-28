"""
Indian Railways API Integration Service.

This module provides integration with Indian Railways APIs for train schedules,
route planning, ticket availability, and booking information.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re

import httpx
from bharatvoice.core.models import ServiceResult, ServiceType
from bharatvoice.config.settings import get_settings


class TrainClass(str, Enum):
    """Train class types."""
    SLEEPER = "SL"
    AC_3_TIER = "3A"
    AC_2_TIER = "2A"
    AC_1_TIER = "1A"
    CHAIR_CAR = "CC"
    EXECUTIVE_CHAIR = "EC"
    GENERAL = "GN"


class TrainStatus(str, Enum):
    """Train status types."""
    ON_TIME = "on_time"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


@dataclass
class TrainStation:
    """Train station information."""
    code: str
    name: str
    city: str
    state: str
    arrival_time: Optional[str] = None
    departure_time: Optional[str] = None
    platform: Optional[str] = None
    distance: Optional[float] = None


@dataclass
class TrainRoute:
    """Train route information."""
    train_number: str
    train_name: str
    source_station: TrainStation
    destination_station: TrainStation
    departure_time: str
    arrival_time: str
    duration: str
    distance: float
    days_of_operation: List[str]
    intermediate_stations: List[TrainStation]


@dataclass
class TicketAvailability:
    """Ticket availability information."""
    train_number: str
    date: str
    class_type: TrainClass
    available_seats: int
    waiting_list: int
    current_status: str
    fare: float


@dataclass
class TrainSchedule:
    """Train schedule information."""
    train_number: str
    train_name: str
    current_status: TrainStatus
    scheduled_departure: str
    actual_departure: Optional[str]
    scheduled_arrival: str
    actual_arrival: Optional[str]
    delay_minutes: int
    platform: Optional[str]
    last_updated: datetime


class IndianRailwaysService:
    """
    Service for integrating with Indian Railways APIs.
    
    Provides train schedule information, route planning, ticket availability,
    and booking assistance with comprehensive error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.api_key = api_key or self.settings.external_services.indian_railways_api_key
        
        # Use real API endpoints when API key is available, otherwise fallback to mock
        if self.api_key:
            self.base_url = "https://indianrailapi.com/api/v2"  # Real API endpoint
            self.use_real_api = True
            self.logger.info("Using real Indian Railways API")
        else:
            self.base_url = "https://api.railwayapi.site"  # Mock API endpoint
            self.use_real_api = False
            self.logger.warning("No API key provided, using mock Indian Railways API")
        
        self.session: Optional[httpx.AsyncClient] = None
        self.max_retries = 3
        self.timeout = 30.0
        
        # Station code mappings for major Indian cities
        self._station_codes = self._initialize_station_codes()
        self._train_classes = self._initialize_train_classes()
        
        # API endpoint mappings
        self._api_endpoints = {
            "train_schedule": "/trainSchedule",
            "trains_between_stations": "/trainsBetweenStations", 
            "seat_availability": "/seatAvailability",
            "pnr_status": "/pnrStatus",
            "train_info": "/trainInfo"
        }
    
    def _initialize_station_codes(self) -> Dict[str, str]:
        """Initialize major Indian railway station codes."""
        return {
            # Major cities
            "new delhi": "NDLS",
            "delhi": "DLI", 
            "mumbai": "CSTM",
            "mumbai central": "BCT",
            "kolkata": "HWH",
            "chennai": "MAS",
            "bangalore": "SBC",
            "hyderabad": "HYB",
            "pune": "PUNE",
            "ahmedabad": "ADI",
            "jaipur": "JP",
            "lucknow": "LJN",
            "kanpur": "CNB",
            "nagpur": "NGP",
            "bhopal": "BPL",
            "indore": "INDB",
            "surat": "ST",
            "vadodara": "BRC",
            "rajkot": "RJT",
            "jodhpur": "JU",
            "udaipur": "UDZ",
            "agra": "AGC",
            "varanasi": "BSB",
            "patna": "PNBE",
            "gaya": "GAYA",
            "ranchi": "RNC",
            "bhubaneswar": "BBS",
            "cuttack": "CTC",
            "visakhapatnam": "VSKP",
            "vijayawada": "BZA",
            "tirupati": "TPTY",
            "coimbatore": "CBE",
            "madurai": "MDU",
            "trivandrum": "TVC",
            "kochi": "ERS",
            "mangalore": "MAQ",
            "mysore": "MYS",
            "hubli": "UBL",
            "goa": "MAO",
            "panaji": "PNJI",
            "amritsar": "ASR",
            "chandigarh": "CDG",
            "jammu": "JAT",
            "srinagar": "SINA",
            "dehradun": "DDN",
            "haridwar": "HW",
            "rishikesh": "RKSH"
        }
    
    def _initialize_train_classes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize train class information."""
        return {
            TrainClass.SLEEPER.value: {
                "name": "Sleeper Class",
                "description": "Basic sleeper accommodation",
                "amenities": ["Berth", "Fan", "Charging point"]
            },
            TrainClass.AC_3_TIER.value: {
                "name": "AC 3 Tier",
                "description": "Air conditioned 3-tier sleeper",
                "amenities": ["AC", "Berth", "Blanket", "Charging point", "Reading light"]
            },
            TrainClass.AC_2_TIER.value: {
                "name": "AC 2 Tier", 
                "description": "Air conditioned 2-tier sleeper",
                "amenities": ["AC", "Berth", "Blanket", "Charging point", "Reading light", "Curtains"]
            },
            TrainClass.AC_1_TIER.value: {
                "name": "AC 1 Tier",
                "description": "First class air conditioned",
                "amenities": ["AC", "Coupe", "Blanket", "Charging point", "Reading light", "Curtains", "Meals"]
            },
            TrainClass.CHAIR_CAR.value: {
                "name": "Chair Car",
                "description": "Reserved seating",
                "amenities": ["Reserved seat", "Fan/AC", "Charging point"]
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    async def _make_api_call(
        self,
        endpoint: str,
        params: Dict[str, Any],
        fallback_method: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Make API call with comprehensive error handling and fallback.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            fallback_method: Fallback method to call if API fails
            
        Returns:
            API response data
            
        Raises:
            Exception: If both API and fallback fail
        """
        if not self.session:
            self.session = httpx.AsyncClient(timeout=self.timeout)
        
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to parameters if using real API
        if self.use_real_api and self.api_key:
            params["key"] = self.api_key
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}: {url}")
                
                response = await self.session.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for API-specific error responses
                    if self.use_real_api:
                        if data.get("status") == "error":
                            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
                    
                    return data
                
                elif response.status_code == 401:
                    self.logger.error("API authentication failed - invalid API key")
                    if fallback_method:
                        self.logger.info("Falling back to mock API")
                        return await fallback_method()
                    raise Exception("Invalid API key for Indian Railways service")
                
                elif response.status_code == 429:
                    # Rate limit exceeded - wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Rate limit exceeded, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                
                elif response.status_code >= 500:
                    # Server error - retry
                    self.logger.warning(f"Server error {response.status_code}, retrying...")
                    await asyncio.sleep(1)
                    continue
                
                else:
                    error_text = response.text
                    self.logger.error(f"API call failed with status {response.status_code}: {error_text}")
                    
                    if fallback_method and attempt == self.max_retries - 1:
                        self.logger.info("Falling back to mock API after all retries failed")
                        return await fallback_method()
                    
                    if attempt == self.max_retries - 1:
                        raise Exception(f"API call failed with status {response.status_code}")
            
            except httpx.RequestError as e:
                self.logger.error(f"Network error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    if fallback_method:
                        self.logger.info("Falling back to mock API due to network error")
                        return await fallback_method()
                    raise Exception(f"Network error: {str(e)}")
                await asyncio.sleep(1)
            
            except httpx.TimeoutException:
                self.logger.error(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    if fallback_method:
                        self.logger.info("Falling back to mock API due to timeout")
                        return await fallback_method()
                    raise Exception("API request timed out")
                await asyncio.sleep(1)
            
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    if fallback_method:
                        self.logger.info("Falling back to mock API due to unexpected error")
                        return await fallback_method()
                    raise
                await asyncio.sleep(1)
        
        # This should never be reached, but just in case
        if fallback_method:
            return await fallback_method()
        raise Exception("All API call attempts failed")
    
    async def get_train_schedule(
        self,
        train_number: str,
        date: Optional[str] = None
    ) -> ServiceResult:
        """
        Get train schedule information.
        
        Args:
            train_number: Train number to query
            date: Date in YYYY-MM-DD format (optional, defaults to today)
            
        Returns:
            Service result with train schedule information
        """
        try:
            self.logger.info(f"Getting train schedule for {train_number}")
            
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            # Validate train number format
            if not self._validate_train_number(train_number):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid train number format. Please provide a valid 5-digit train number.",
                    response_time=0.1
                )
            
            # Validate date format
            if not self._validate_date_format(date):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid date format. Please use YYYY-MM-DD format.",
                    response_time=0.1
                )
            
            start_time = asyncio.get_event_loop().time()
            
            # Make API call with fallback to mock
            api_params = {
                "trainNumber": train_number,
                "date": date
            }
            
            schedule_data = await self._make_api_call(
                self._api_endpoints["train_schedule"],
                api_params,
                fallback_method=lambda: self._mock_train_schedule_api(train_number, date)
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            if schedule_data:
                # Process and standardize the response data
                processed_data = self._process_train_schedule_response(schedule_data, train_number, date)
                
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=True,
                    data=processed_data,
                    error_message=None,
                    response_time=response_time
                )
            else:
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message=f"Train {train_number} not found or no schedule available for {date}",
                    response_time=response_time
                )
                
        except Exception as e:
            self.logger.error(f"Error getting train schedule: {e}")
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Failed to retrieve train schedule: {str(e)}",
                response_time=0.5
            )
    
    async def find_trains_between_stations(
        self,
        source: str,
        destination: str,
        date: Optional[str] = None,
        class_preference: Optional[TrainClass] = None
    ) -> ServiceResult:
        """
        Find trains between two stations.
        
        Args:
            source: Source station name or code
            destination: Destination station name or code
            date: Travel date in YYYY-MM-DD format
            class_preference: Preferred train class
            
        Returns:
            Service result with available trains
        """
        try:
            self.logger.info(f"Finding trains from {source} to {destination}")
            
            # Convert station names to codes
            source_code = self._get_station_code(source)
            dest_code = self._get_station_code(destination)
            
            if not source_code or not dest_code:
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid station name. Please provide valid station names or codes.",
                    response_time=0.3
                )
            
            if source_code == dest_code:
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Source and destination stations cannot be the same.",
                    response_time=0.2
                )
            
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            
            # Validate date format
            if not self._validate_date_format(date):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid date format. Please use YYYY-MM-DD format.",
                    response_time=0.1
                )
            
            start_time = asyncio.get_event_loop().time()
            
            # Make API call with fallback to mock
            api_params = {
                "fromStationCode": source_code,
                "toStationCode": dest_code,
                "dateOfJourney": date
            }
            
            if class_preference:
                api_params["classType"] = class_preference.value
            
            trains_data = await self._make_api_call(
                self._api_endpoints["trains_between_stations"],
                api_params,
                fallback_method=lambda: self._mock_trains_between_stations_api(
                    source_code, dest_code, date, class_preference
                )
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Process and enhance the response data
            processed_data = self._process_trains_between_stations_response(
                trains_data, source, destination, source_code, dest_code, date, class_preference
            )
            
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=True,
                data=processed_data,
                error_message=None,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Error finding trains: {e}")
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Failed to find trains: {str(e)}",
                response_time=0.5
            )
    
    async def check_ticket_availability(
        self,
        train_number: str,
        source: str,
        destination: str,
        date: str,
        class_type: TrainClass
    ) -> ServiceResult:
        """
        Check ticket availability for a specific train.
        
        Args:
            train_number: Train number
            source: Source station
            destination: Destination station  
            date: Travel date in YYYY-MM-DD format
            class_type: Train class
            
        Returns:
            Service result with availability information
        """
        try:
            self.logger.info(f"Checking availability for train {train_number}")
            
            # Validate inputs
            if not self._validate_train_number(train_number):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid train number format. Please provide a valid 5-digit train number.",
                    response_time=0.1
                )
            
            if not self._validate_date_format(date):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid date format. Please use YYYY-MM-DD format.",
                    response_time=0.1
                )
            
            # Convert station names to codes
            source_code = self._get_station_code(source)
            dest_code = self._get_station_code(destination)
            
            if not source_code or not dest_code:
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid station name. Please provide valid station names or codes.",
                    response_time=0.2
                )
            
            start_time = asyncio.get_event_loop().time()
            
            # Make API call with fallback to mock
            api_params = {
                "trainNumber": train_number,
                "fromStationCode": source_code,
                "toStationCode": dest_code,
                "dateOfJourney": date,
                "classType": class_type.value
            }
            
            availability_data = await self._make_api_call(
                self._api_endpoints["seat_availability"],
                api_params,
                fallback_method=lambda: self._mock_ticket_availability_api(
                    train_number, source, destination, date, class_type
                )
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Process and enhance the response data
            processed_data = self._process_availability_response(
                availability_data, train_number, source, destination, date, class_type
            )
            
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=True,
                data=processed_data,
                error_message=None,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Error checking availability: {e}")
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Failed to check availability: {str(e)}",
                response_time=0.5
            )
    
    async def get_pnr_status(self, pnr_number: str) -> ServiceResult:
        """
        Get PNR status information.
        
        Args:
            pnr_number: 10-digit PNR number
            
        Returns:
            Service result with PNR status
        """
        try:
            self.logger.info(f"Getting PNR status for {pnr_number}")
            
            if not self._validate_pnr(pnr_number):
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={},
                    error_message="Invalid PNR number. Please provide a valid 10-digit PNR.",
                    response_time=0.2
                )
            
            start_time = asyncio.get_event_loop().time()
            
            # Make API call with fallback to mock
            api_params = {
                "pnrNumber": pnr_number
            }
            
            pnr_data = await self._make_api_call(
                self._api_endpoints["pnr_status"],
                api_params,
                fallback_method=lambda: self._mock_pnr_status_api(pnr_number)
            )
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Process and enhance the response data
            processed_data = self._process_pnr_response(pnr_data, pnr_number)
            
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=True,
                data=processed_data,
                error_message=None,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"Error getting PNR status: {e}")
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Failed to get PNR status: {str(e)}",
                response_time=0.5
            )
    
    async def process_natural_language_query(self, query: str) -> ServiceResult:
        """
        Process natural language query for train information.
        
        Args:
            query: Natural language query about trains
            
        Returns:
            Service result with processed query response
        """
        try:
            self.logger.info(f"Processing NL query: {query[:50]}...")
            
            # Extract intent and entities from query
            query_analysis = await self._analyze_train_query(query)
            
            if query_analysis["intent"] == "train_schedule":
                return await self.get_train_schedule(
                    query_analysis["train_number"],
                    query_analysis.get("date")
                )
            elif query_analysis["intent"] == "find_trains":
                return await self.find_trains_between_stations(
                    query_analysis["source"],
                    query_analysis["destination"],
                    query_analysis.get("date"),
                    query_analysis.get("class_preference")
                )
            elif query_analysis["intent"] == "check_availability":
                return await self.check_ticket_availability(
                    query_analysis["train_number"],
                    query_analysis["source"],
                    query_analysis["destination"],
                    query_analysis["date"],
                    query_analysis["class_type"]
                )
            elif query_analysis["intent"] == "pnr_status":
                return await self.get_pnr_status(query_analysis["pnr_number"])
            else:
                return ServiceResult(
                    service_type=ServiceType.INDIAN_RAILWAYS,
                    success=False,
                    data={"query_analysis": query_analysis},
                    error_message="Could not understand the train-related query. Please be more specific.",
                    response_time=0.3
                )
                
        except Exception as e:
            self.logger.error(f"Error processing NL query: {e}")
            return ServiceResult(
                service_type=ServiceType.INDIAN_RAILWAYS,
                success=False,
                data={},
                error_message=f"Failed to process query: {str(e)}",
                response_time=0.5
            )
    
    def _get_station_code(self, station_name: str) -> Optional[str]:
        """Get station code from station name."""
        station_lower = station_name.lower().strip()
        
        # First check if it's already a station code (3-4 uppercase letters)
        if re.match(r'^[A-Z]{3,4}$', station_name.upper()):
            return station_name.upper()
        
        # Then check our mapping
        return self._station_codes.get(station_lower)
    
    def _validate_pnr(self, pnr: str) -> bool:
        """Validate PNR number format."""
        return len(pnr) == 10 and pnr.isdigit()
    
    def _validate_train_number(self, train_number: str) -> bool:
        """Validate train number format."""
        return len(train_number) == 5 and train_number.isdigit()
    
    def _validate_date_format(self, date: str) -> bool:
        """Validate date format (YYYY-MM-DD)."""
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def _process_train_schedule_response(
        self, 
        raw_data: Dict[str, Any], 
        train_number: str, 
        date: str
    ) -> Dict[str, Any]:
        """Process and standardize train schedule response."""
        if self.use_real_api:
            # Process real API response format
            train_data = raw_data.get("train", {})
            return {
                "train_schedule": {
                    "train_number": train_data.get("number", train_number),
                    "train_name": train_data.get("name", "Unknown"),
                    "source": train_data.get("source", {}),
                    "destination": train_data.get("destination", {}),
                    "status": train_data.get("status", "unknown"),
                    "platform": train_data.get("platform"),
                    "distance": train_data.get("distance", 0),
                    "duration": train_data.get("duration", "Unknown")
                },
                "query_date": date,
                "last_updated": datetime.now().isoformat(),
                "data_source": "real_api"
            }
        else:
            # Process mock API response format
            return {
                "train_schedule": raw_data,
                "query_date": date,
                "last_updated": datetime.now().isoformat(),
                "data_source": "mock_api"
            }
    
    def _process_trains_between_stations_response(
        self,
        raw_data: Dict[str, Any],
        source: str,
        destination: str,
        source_code: str,
        dest_code: str,
        date: str,
        class_preference: Optional[TrainClass]
    ) -> Dict[str, Any]:
        """Process and standardize trains between stations response."""
        if self.use_real_api:
            trains = raw_data.get("trains", [])
            processed_trains = []
            
            for train in trains:
                processed_train = {
                    "train_number": train.get("number"),
                    "train_name": train.get("name"),
                    "departure_time": train.get("departureTime"),
                    "arrival_time": train.get("arrivalTime"),
                    "duration": train.get("duration"),
                    "distance": train.get("distance", 0),
                    "days": train.get("runsOn", []),
                    "classes_available": train.get("classes", []),
                    "fare": train.get("fare", {})
                }
                processed_trains.append(processed_train)
            
            return {
                "available_trains": processed_trains,
                "source_station": {"name": source, "code": source_code},
                "destination_station": {"name": destination, "code": dest_code},
                "travel_date": date,
                "total_trains": len(processed_trains),
                "class_preference": class_preference.value if class_preference else None,
                "data_source": "real_api"
            }
        else:
            # Process mock API response format
            return {
                "available_trains": raw_data,
                "source_station": {"name": source, "code": source_code},
                "destination_station": {"name": destination, "code": dest_code},
                "travel_date": date,
                "total_trains": len(raw_data),
                "class_preference": class_preference.value if class_preference else None,
                "data_source": "mock_api"
            }
    
    def _process_availability_response(
        self,
        raw_data: Dict[str, Any],
        train_number: str,
        source: str,
        destination: str,
        date: str,
        class_type: TrainClass
    ) -> Dict[str, Any]:
        """Process and standardize availability response."""
        if self.use_real_api:
            availability = raw_data.get("availability", {})
            processed_availability = {
                "train_number": train_number,
                "date": date,
                "class_type": class_type.value,
                "available_seats": availability.get("availableSeats", 0),
                "waiting_list": availability.get("waitingList", 0),
                "current_status": availability.get("status", "Unknown"),
                "fare": availability.get("fare", 0),
                "last_updated": datetime.now().isoformat()
            }
        else:
            # Process mock API response format
            processed_availability = raw_data
        
        return {
            "availability": processed_availability,
            "booking_status": self._get_booking_status(processed_availability),
            "alternative_dates": self._get_alternative_dates_sync(
                train_number, source, destination, date, class_type
            ),
            "data_source": "real_api" if self.use_real_api else "mock_api"
        }
    
    def _process_pnr_response(
        self,
        raw_data: Dict[str, Any],
        pnr_number: str
    ) -> Dict[str, Any]:
        """Process and standardize PNR response."""
        if self.use_real_api:
            pnr_info = raw_data.get("pnrInfo", {})
            processed_pnr = {
                "pnr_number": pnr_number,
                "train_number": pnr_info.get("trainNumber"),
                "train_name": pnr_info.get("trainName"),
                "date_of_journey": pnr_info.get("dateOfJourney"),
                "from_station": pnr_info.get("fromStation", {}),
                "to_station": pnr_info.get("toStation", {}),
                "class": pnr_info.get("class"),
                "passengers": pnr_info.get("passengers", []),
                "chart_status": pnr_info.get("chartStatus"),
                "total_fare": pnr_info.get("totalFare", 0)
            }
        else:
            # Process mock API response format
            processed_pnr = raw_data
        
        return {
            "pnr_status": processed_pnr,
            "last_updated": datetime.now().isoformat(),
            "data_source": "real_api" if self.use_real_api else "mock_api"
        }
    
    def _get_booking_status(self, availability: Dict[str, Any]) -> str:
        """Get booking status based on availability."""
        available_seats = availability.get("available_seats", 0)
        waiting_list = availability.get("waiting_list", 0)
        
        if available_seats > 0:
            return "Available for booking"
        elif waiting_list < 50:
            return f"Waiting list available (WL {waiting_list})"
        else:
            return "Not available for booking"
    
    def _get_alternative_dates_sync(
        self,
        train_number: str,
        source: str,
        destination: str,
        date: str,
        class_type: TrainClass
    ) -> List[Dict[str, Any]]:
        """Get alternative dates with availability (synchronous version)."""
        alternatives = []
        base_date = datetime.strptime(date, "%Y-%m-%d")
        
        for i in range(1, 4):  # Check next 3 days
            alt_date = base_date + timedelta(days=i)
            alt_date_str = alt_date.strftime("%Y-%m-%d")
            
            # Mock availability check
            alternatives.append({
                "date": alt_date_str,
                "day_name": alt_date.strftime("%A"),
                "available_seats": max(0, 50 - (i * 15)),  # Mock decreasing availability
                "fare": 1200 + (i * 50)  # Mock increasing fare
            })
        
        return alternatives
    
    async def _get_alternative_dates(
        self,
        train_number: str,
        source: str,
        destination: str,
        date: str,
        class_type: TrainClass
    ) -> List[Dict[str, Any]]:
        """Get alternative dates with availability."""
        alternatives = []
        base_date = datetime.strptime(date, "%Y-%m-%d")
        
        for i in range(1, 4):  # Check next 3 days
            alt_date = base_date + timedelta(days=i)
            alt_date_str = alt_date.strftime("%Y-%m-%d")
            
            # Mock availability check
            alternatives.append({
                "date": alt_date_str,
                "day_name": alt_date.strftime("%A"),
                "available_seats": max(0, 50 - (i * 15)),  # Mock decreasing availability
                "fare": 1200 + (i * 50)  # Mock increasing fare
            })
        
        return alternatives
    
    async def _analyze_train_query(self, query: str) -> Dict[str, Any]:
        """Analyze natural language query for train information."""
        query_lower = query.lower()
        analysis = {"intent": "unknown", "confidence": 0.5}
        
        # Enhanced pattern matching with better entity extraction
        
        # Train schedule queries
        if any(word in query_lower for word in ["schedule", "timing", "time", "when does", "departure", "arrival"]):
            analysis["intent"] = "train_schedule"
            analysis["confidence"] = 0.8
            
            # Extract train number with better patterns
            train_patterns = [
                r'\b(\d{5})\b',  # 5-digit train number
                r'train\s+(?:number\s+)?(\d{5})',  # "train 12345" or "train number 12345"
                r'(\d{5})\s+train'  # "12345 train"
            ]
            
            for pattern in train_patterns:
                train_match = re.search(pattern, query)
                if train_match:
                    analysis["train_number"] = train_match.group(1)
                    break
            
            # Extract date patterns
            date_patterns = [
                r'on\s+(\d{4}-\d{2}-\d{2})',  # "on 2024-01-15"
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # "15/01/2024" or "15-01-2024"
                r'(today|tomorrow|yesterday)',
                r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)'
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, query_lower)
                if date_match:
                    analysis["date_text"] = date_match.group(0)
                    # Convert to standard format if needed
                    analysis["date"] = self._parse_date_from_text(date_match.group(0))
                    break
        
        # Train search queries
        elif any(word in query_lower for word in ["from", "to", "between", "trains from", "travel from"]):
            analysis["intent"] = "find_trains"
            analysis["confidence"] = 0.9
            
            # Enhanced station extraction
            stations = self._extract_stations_from_query(query_lower)
            if len(stations) >= 2:
                analysis["source"] = stations[0]
                analysis["destination"] = stations[1]
            elif len(stations) == 1:
                # Try to find "from X to Y" pattern
                from_to_pattern = r'from\s+([^to]+)\s+to\s+(.+?)(?:\s+on|\s+at|\s*$)'
                match = re.search(from_to_pattern, query_lower)
                if match:
                    analysis["source"] = match.group(1).strip()
                    analysis["destination"] = match.group(2).strip()
            
            # Extract class preference
            class_keywords = {
                "sleeper": TrainClass.SLEEPER,
                "sl": TrainClass.SLEEPER,
                "3a": TrainClass.AC_3_TIER,
                "3ac": TrainClass.AC_3_TIER,
                "ac 3": TrainClass.AC_3_TIER,
                "2a": TrainClass.AC_2_TIER,
                "2ac": TrainClass.AC_2_TIER,
                "ac 2": TrainClass.AC_2_TIER,
                "1a": TrainClass.AC_1_TIER,
                "1ac": TrainClass.AC_1_TIER,
                "ac 1": TrainClass.AC_1_TIER,
                "chair car": TrainClass.CHAIR_CAR,
                "cc": TrainClass.CHAIR_CAR
            }
            
            for keyword, train_class in class_keywords.items():
                if keyword in query_lower:
                    analysis["class_preference"] = train_class
                    break
        
        # Availability queries
        elif any(word in query_lower for word in ["availability", "available", "book", "seats", "ticket"]):
            analysis["intent"] = "check_availability"
            analysis["confidence"] = 0.8
            
            # Extract train number, stations, date, and class
            train_match = re.search(r'\b(\d{5})\b', query)
            if train_match:
                analysis["train_number"] = train_match.group(1)
            
            stations = self._extract_stations_from_query(query_lower)
            if len(stations) >= 2:
                analysis["source"] = stations[0]
                analysis["destination"] = stations[1]
        
        # PNR status queries
        elif any(word in query_lower for word in ["pnr", "status", "booking status"]):
            analysis["intent"] = "pnr_status"
            analysis["confidence"] = 0.9
            
            # Extract PNR number with better patterns
            pnr_patterns = [
                r'\b(\d{10})\b',  # 10-digit PNR
                r'pnr\s+(?:number\s+)?(\d{10})',  # "pnr 1234567890"
                r'(\d{10})\s+pnr'  # "1234567890 pnr"
            ]
            
            for pattern in pnr_patterns:
                pnr_match = re.search(pattern, query)
                if pnr_match:
                    analysis["pnr_number"] = pnr_match.group(1)
                    break
        
        return analysis
    
    def _extract_stations_from_query(self, query: str) -> List[str]:
        """Extract station names from query text."""
        stations = []
        
        # Check for known station names in our mapping
        for station_name in self._station_codes.keys():
            if station_name in query:
                stations.append(station_name)
        
        # Also check for station codes
        station_code_pattern = r'\b([A-Z]{3,4})\b'
        codes = re.findall(station_code_pattern, query.upper())
        for code in codes:
            # Convert code back to name if possible
            for name, mapped_code in self._station_codes.items():
                if mapped_code == code:
                    stations.append(name)
                    break
            else:
                stations.append(code.lower())
        
        return list(set(stations))  # Remove duplicates
    
    def _parse_date_from_text(self, date_text: str) -> str:
        """Parse date from natural language text."""
        date_text = date_text.lower().strip()
        
        if date_text == "today":
            return datetime.now().strftime("%Y-%m-%d")
        elif date_text == "tomorrow":
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_text == "yesterday":
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Try to parse DD/MM/YYYY or DD-MM-YYYY format
        date_match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_text)
        if date_match:
            day, month, year = date_match.groups()
            try:
                parsed_date = datetime(int(year), int(month), int(day))
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
        
        # For now, return today's date if parsing fails
        return datetime.now().strftime("%Y-%m-%d")
    
    # Mock API methods (in real implementation, these would call actual APIs)
    
    async def _mock_train_schedule_api(self, train_number: str, date: str) -> Optional[Dict[str, Any]]:
        """Mock train schedule API call."""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        # Mock data for common trains
        mock_schedules = {
            "12002": {
                "train_number": "12002",
                "train_name": "New Delhi Shatabdi Express",
                "source": {"name": "New Delhi", "code": "NDLS", "departure": "06:00"},
                "destination": {"name": "Bhopal", "code": "BPL", "arrival": "14:30"},
                "status": "on_time",
                "platform": "16",
                "distance": 707,
                "duration": "8h 30m"
            },
            "12951": {
                "train_number": "12951", 
                "train_name": "Mumbai Rajdhani Express",
                "source": {"name": "Mumbai Central", "code": "BCT", "departure": "17:05"},
                "destination": {"name": "New Delhi", "code": "NDLS", "arrival": "08:35"},
                "status": "delayed",
                "delay_minutes": 15,
                "platform": "1",
                "distance": 1384,
                "duration": "15h 30m"
            }
        }
        
        return mock_schedules.get(train_number)
    
    async def _mock_trains_between_stations_api(
        self,
        source_code: str,
        dest_code: str,
        date: str,
        class_preference: Optional[TrainClass]
    ) -> List[Dict[str, Any]]:
        """Mock trains between stations API call."""
        await asyncio.sleep(0.8)  # Simulate API delay
        
        # Mock train data
        mock_trains = [
            {
                "train_number": "12002",
                "train_name": "Shatabdi Express",
                "departure_time": "06:00",
                "arrival_time": "14:30",
                "duration": "8h 30m",
                "distance": 707,
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                "classes_available": ["CC", "EC"],
                "fare": {"CC": 1200, "EC": 2400}
            },
            {
                "train_number": "12626",
                "train_name": "Kerala Express",
                "departure_time": "22:00",
                "arrival_time": "04:30+1",
                "duration": "6h 30m",
                "distance": 650,
                "days": ["Daily"],
                "classes_available": ["SL", "3A", "2A", "1A"],
                "fare": {"SL": 400, "3A": 1100, "2A": 1600, "1A": 2800}
            }
        ]
        
        return mock_trains
    
    async def _mock_ticket_availability_api(
        self,
        train_number: str,
        source: str,
        destination: str,
        date: str,
        class_type: TrainClass
    ) -> Dict[str, Any]:
        """Mock ticket availability API call."""
        await asyncio.sleep(0.6)  # Simulate API delay
        
        # Mock availability data
        import random
        available_seats = random.randint(0, 100)
        waiting_list = random.randint(0, 200) if available_seats == 0 else 0
        
        return {
            "train_number": train_number,
            "date": date,
            "class_type": class_type.value,
            "available_seats": available_seats,
            "waiting_list": waiting_list,
            "current_status": "Available" if available_seats > 0 else f"WL {waiting_list}",
            "fare": random.randint(500, 3000),
            "last_updated": datetime.now().isoformat()
        }
    
    async def _mock_pnr_status_api(self, pnr_number: str) -> Dict[str, Any]:
        """Mock PNR status API call."""
        await asyncio.sleep(0.4)  # Simulate API delay
        
        # Mock PNR data
        return {
            "pnr_number": pnr_number,
            "train_number": "12002",
            "train_name": "Shatabdi Express",
            "date_of_journey": "2024-01-15",
            "from_station": {"name": "New Delhi", "code": "NDLS"},
            "to_station": {"name": "Bhopal", "code": "BPL"},
            "class": "CC",
            "passengers": [
                {
                    "name": "PASSENGER 1",
                    "age": 35,
                    "gender": "M",
                    "current_status": "CNF/B1/25",
                    "booking_status": "CNF/B1/25"
                }
            ],
            "chart_status": "Chart Not Prepared",
            "total_fare": 1200
        }