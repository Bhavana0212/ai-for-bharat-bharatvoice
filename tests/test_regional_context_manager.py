"""
Tests for Regional Context Management system.

This module contains comprehensive tests for the regional context management
functionality, including unit tests and property-based tests for Indian
cultural context, weather services, and local service integration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from bharatvoice.core.models import (
    LocationData,
    RegionalContextData,
    CulturalEvent,
    WeatherData,
    LocalService,
    TransportService,
    GovernmentService,
    LanguageCode
)
from bharatvoice.services.context_management.regional_context_manager import (
    RegionalContextManager,
    IndianFestivalCalendar,
    WeatherService,
    LocalServiceDirectory,
    TransportationService,
    GovernmentServiceDirectory
)


class TestIndianFestivalCalendar:
    """Test cases for Indian Festival Calendar."""
    
    @pytest.fixture
    def festival_calendar(self):
        return IndianFestivalCalendar()
    
    @pytest.fixture
    def sample_location(self):
        return LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
    
    @pytest.mark.asyncio
    async def test_get_upcoming_festivals_returns_events(self, festival_calendar, sample_location):
        """Test that upcoming festivals are returned for a location."""
        festivals = await festival_calendar.get_upcoming_festivals(sample_location)
        
        assert isinstance(festivals, list)
        assert len(festivals) > 0
        
        for festival in festivals:
            assert isinstance(festival, CulturalEvent)
            assert festival.name
            assert festival.date
            assert festival.significance
            assert sample_location.state in festival.regional_relevance
    
    @pytest.mark.asyncio
    async def test_get_upcoming_festivals_includes_regional_events(self, festival_calendar, sample_location):
        """Test that regional festivals are included for Maharashtra."""
        festivals = await festival_calendar.get_upcoming_festivals(sample_location)
        
        # Check for Maharashtra-specific festivals
        festival_names = [f.name for f in festivals]
        assert any("Gudi Padwa" in name or "Ganpati" in name for name in festival_names)
    
    @pytest.mark.asyncio
    async def test_get_upcoming_festivals_with_custom_days(self, festival_calendar, sample_location):
        """Test getting festivals with custom day range."""
        festivals = await festival_calendar.get_upcoming_festivals(sample_location, days_ahead=60)
        
        assert isinstance(festivals, list)
        # Should return festivals within the specified range
        for festival in festivals:
            days_diff = (festival.date - datetime.now()).days
            assert days_diff <= 60


class TestWeatherService:
    """Test cases for Weather Service."""
    
    @pytest.fixture
    def weather_service(self):
        return WeatherService()
    
    @pytest.fixture
    def desert_location(self):
        return LocationData(
            latitude=26.9124,
            longitude=75.7873,
            city="Jaipur",
            state="Rajasthan",
            country="India"
        )
    
    @pytest.fixture
    def coastal_location(self):
        return LocationData(
            latitude=15.2993,
            longitude=74.1240,
            city="Panaji",
            state="Goa",
            country="India"
        )
    
    @pytest.mark.asyncio
    async def test_get_weather_data_returns_valid_data(self, weather_service, desert_location):
        """Test that weather data is returned with valid values."""
        weather = await weather_service.get_weather_data(desert_location)
        
        assert isinstance(weather, WeatherData)
        assert weather.temperature_celsius > 0
        assert 0 <= weather.humidity <= 100
        assert weather.description
        assert weather.wind_speed_kmh >= 0
        assert weather.precipitation_mm >= 0
    
    @pytest.mark.asyncio
    async def test_weather_varies_by_region(self, weather_service, desert_location, coastal_location):
        """Test that weather varies appropriately by region."""
        desert_weather = await weather_service.get_weather_data(desert_location)
        coastal_weather = await weather_service.get_weather_data(coastal_location)
        
        # Coastal areas should generally be more humid
        assert coastal_weather.humidity > desert_weather.humidity
    
    @pytest.mark.asyncio
    async def test_monsoon_season_detection(self, weather_service, desert_location):
        """Test that monsoon season is properly detected."""
        with patch('bharatvoice.services.context_management.regional_context_manager.datetime') as mock_datetime:
            # Mock July (monsoon month)
            mock_datetime.now.return_value = datetime(2024, 7, 15)
            
            weather = await weather_service.get_weather_data(desert_location)
            assert weather.is_monsoon_season is True
            assert weather.precipitation_mm > 0


class TestLocalServiceDirectory:
    """Test cases for Local Service Directory."""
    
    @pytest.fixture
    def service_directory(self):
        return LocalServiceDirectory()
    
    @pytest.fixture
    def sample_location(self):
        return LocationData(
            latitude=12.9716,
            longitude=77.5946,
            city="Bangalore",
            state="Karnataka",
            country="India"
        )
    
    @pytest.mark.asyncio
    async def test_find_local_services_returns_services(self, service_directory, sample_location):
        """Test that local services are returned."""
        services = await service_directory.find_local_services(sample_location)
        
        assert isinstance(services, list)
        assert len(services) > 0
        
        for service in services:
            assert isinstance(service, LocalService)
            assert service.name
            assert service.category
            assert service.address
            assert sample_location.city in service.name or sample_location.city in service.address
    
    @pytest.mark.asyncio
    async def test_find_services_by_category(self, service_directory, sample_location):
        """Test finding services by specific category."""
        healthcare_services = await service_directory.find_local_services(
            sample_location, 
            category="healthcare"
        )
        
        assert all(service.category == "healthcare" for service in healthcare_services)
    
    @pytest.mark.asyncio
    async def test_find_services_with_radius(self, service_directory, sample_location):
        """Test finding services within specified radius."""
        radius = 3.0
        services = await service_directory.find_local_services(
            sample_location, 
            radius_km=radius
        )
        
        for service in services:
            if service.distance_km:
                assert service.distance_km <= radius


class TestTransportationService:
    """Test cases for Transportation Service."""
    
    @pytest.fixture
    def transport_service(self):
        return TransportationService()
    
    @pytest.fixture
    def metro_city_location(self):
        return LocationData(
            latitude=28.7041,
            longitude=77.1025,
            city="Delhi",
            state="Delhi",
            country="India"
        )
    
    @pytest.fixture
    def small_city_location(self):
        return LocationData(
            latitude=23.2599,
            longitude=77.4126,
            city="Bhopal",
            state="Madhya Pradesh",
            country="India"
        )
    
    @pytest.mark.asyncio
    async def test_get_transport_options_returns_basic_services(self, transport_service, small_city_location):
        """Test that basic transport services are always available."""
        transport_options = await transport_service.get_transport_options(small_city_location)
        
        assert isinstance(transport_options, list)
        assert len(transport_options) >= 3  # Railway, Bus, Auto should always be available
        
        service_types = [option.service_type for option in transport_options]
        assert "Railway" in service_types
        assert "Bus" in service_types
        assert "Auto Rickshaw" in service_types
    
    @pytest.mark.asyncio
    async def test_metro_available_in_major_cities(self, transport_service, metro_city_location):
        """Test that metro service is available in major cities."""
        transport_options = await transport_service.get_transport_options(metro_city_location)
        
        service_types = [option.service_type for option in transport_options]
        assert "Metro" in service_types
    
    @pytest.mark.asyncio
    async def test_transport_services_have_valid_data(self, transport_service, metro_city_location):
        """Test that transport services have valid data."""
        transport_options = await transport_service.get_transport_options(metro_city_location)
        
        for option in transport_options:
            assert isinstance(option, TransportService)
            assert option.service_type
            assert option.name
            assert option.route
            assert isinstance(option.availability, bool)


class TestGovernmentServiceDirectory:
    """Test cases for Government Service Directory."""
    
    @pytest.fixture
    def gov_service_directory(self):
        return GovernmentServiceDirectory()
    
    @pytest.fixture
    def sample_location(self):
        return LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India"
        )
    
    @pytest.mark.asyncio
    async def test_get_government_services_returns_services(self, gov_service_directory, sample_location):
        """Test that government services are returned."""
        services = await gov_service_directory.get_government_services(sample_location)
        
        assert isinstance(services, list)
        assert len(services) > 0
        
        for service in services:
            assert isinstance(service, GovernmentService)
            assert service.service_name
            assert service.department
            assert service.description
            assert isinstance(service.required_documents, list)
    
    @pytest.mark.asyncio
    async def test_essential_services_included(self, gov_service_directory, sample_location):
        """Test that essential government services are included."""
        services = await gov_service_directory.get_government_services(sample_location)
        
        service_names = [service.service_name for service in services]
        essential_services = ["Aadhaar Card", "PAN Card", "Passport", "Driving License"]
        
        for essential in essential_services:
            assert essential in service_names


class TestRegionalContextManager:
    """Test cases for Regional Context Manager."""
    
    @pytest.fixture
    def context_manager(self):
        return RegionalContextManager()
    
    @pytest.fixture
    def sample_location(self):
        return LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
    
    @pytest.mark.asyncio
    async def test_get_regional_context_returns_complete_data(self, context_manager, sample_location):
        """Test that complete regional context is returned."""
        context = await context_manager.get_regional_context(sample_location)
        
        assert isinstance(context, RegionalContextData)
        assert context.location == sample_location
        assert isinstance(context.local_services, list)
        assert isinstance(context.cultural_events, list)
        assert isinstance(context.transport_options, list)
        assert isinstance(context.government_services, list)
        assert context.local_language in LanguageCode
    
    @pytest.mark.asyncio
    async def test_language_mapping_by_state(self, context_manager):
        """Test that local language is correctly mapped by state."""
        tamil_location = LocationData(
            latitude=13.0827,
            longitude=80.2707,
            city="Chennai",
            state="Tamil Nadu",
            country="India"
        )
        
        context = await context_manager.get_regional_context(tamil_location)
        assert context.local_language == LanguageCode.TAMIL
    
    @pytest.mark.asyncio
    async def test_dialect_info_for_major_cities(self, context_manager, sample_location):
        """Test that dialect information is provided for major cities."""
        context = await context_manager.get_regional_context(sample_location)
        
        # Mumbai should have dialect info
        assert context.dialect_info is not None
        assert "Mumbai" in context.dialect_info or "Marathi" in context.dialect_info
    
    @pytest.mark.asyncio
    async def test_get_cultural_events_by_type(self, context_manager, sample_location):
        """Test filtering cultural events by type."""
        hindu_events = await context_manager.get_cultural_events_by_type(
            sample_location, 
            "Hindu Festival"
        )
        
        for event in hindu_events:
            assert event.celebration_type == "Hindu Festival"
    
    @pytest.mark.asyncio
    async def test_search_local_services(self, context_manager, sample_location):
        """Test searching local services by query."""
        hospital_services = await context_manager.search_local_services(
            sample_location, 
            "hospital"
        )
        
        for service in hospital_services:
            assert ("hospital" in service.name.lower() or 
                   "hospital" in service.category.lower())
    
    @pytest.mark.asyncio
    async def test_weather_forecast(self, context_manager, sample_location):
        """Test getting weather forecast."""
        forecast = await context_manager.get_weather_forecast(sample_location, days=3)
        
        assert isinstance(forecast, list)
        assert len(forecast) <= 3
        
        for weather in forecast:
            assert isinstance(weather, WeatherData)
    
    @pytest.mark.asyncio
    async def test_update_location_context(self, context_manager, sample_location):
        """Test updating location context."""
        user_id = uuid4()
        
        new_context = await context_manager.update_location_context(user_id, sample_location)
        
        assert isinstance(new_context, RegionalContextData)
        assert new_context.location == sample_location
    
    @pytest.mark.asyncio
    async def test_error_handling_returns_minimal_context(self, context_manager, sample_location):
        """Test that error handling returns minimal context."""
        # Mock all services to raise exceptions
        with patch.object(context_manager.festival_calendar, 'get_upcoming_festivals', side_effect=Exception("Test error")):
            with patch.object(context_manager.weather_service, 'get_weather_data', side_effect=Exception("Test error")):
                with patch.object(context_manager.local_service_directory, 'find_local_services', side_effect=Exception("Test error")):
                    with patch.object(context_manager.transportation_service, 'get_transport_options', side_effect=Exception("Test error")):
                        with patch.object(context_manager.government_service_directory, 'get_government_services', side_effect=Exception("Test error")):
                            
                            context = await context_manager.get_regional_context(sample_location)
                            
                            # Should still return a valid context object
                            assert isinstance(context, RegionalContextData)
                            assert context.location == sample_location
                            assert context.local_services == []
                            assert context.cultural_events == []
                            assert context.transport_options == []
                            assert context.government_services == []


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st
    import hypothesis.strategies as st
    
    class TestRegionalContextProperties:
        """Property-based tests for regional context system."""
        
        @given(
            latitude=st.floats(min_value=-90.0, max_value=90.0),
            longitude=st.floats(min_value=-180.0, max_value=180.0),
            city=st.text(min_size=1, max_size=50),
            state=st.text(min_size=1, max_size=50)
        )
        @pytest.mark.asyncio
        async def test_regional_context_always_returns_valid_structure(self, latitude, longitude, city, state):
            """Property: Regional context always returns valid structure for any location."""
            location = LocationData(
                latitude=latitude,
                longitude=longitude,
                city=city,
                state=state,
                country="India"
            )
            
            context_manager = RegionalContextManager()
            context = await context_manager.get_regional_context(location)
            
            # Properties that should always hold
            assert isinstance(context, RegionalContextData)
            assert context.location == location
            assert isinstance(context.local_services, list)
            assert isinstance(context.cultural_events, list)
            assert isinstance(context.transport_options, list)
            assert isinstance(context.government_services, list)
            assert context.local_language in LanguageCode
        
        @given(
            days_ahead=st.integers(min_value=1, max_value=365)
        )
        @pytest.mark.asyncio
        async def test_festival_calendar_respects_date_range(self, days_ahead):
            """Property: Festival calendar respects the specified date range."""
            location = LocationData(
                latitude=19.0760,
                longitude=72.8777,
                city="Mumbai",
                state="Maharashtra",
                country="India"
            )
            
            calendar = IndianFestivalCalendar()
            festivals = await calendar.get_upcoming_festivals(location, days_ahead=days_ahead)
            
            current_date = datetime.now()
            max_date = current_date + timedelta(days=days_ahead)
            
            for festival in festivals:
                assert current_date <= festival.date <= max_date
        
        @given(
            radius_km=st.floats(min_value=0.1, max_value=50.0)
        )
        @pytest.mark.asyncio
        async def test_local_services_respect_radius(self, radius_km):
            """Property: Local services respect the specified radius."""
            location = LocationData(
                latitude=19.0760,
                longitude=72.8777,
                city="Mumbai",
                state="Maharashtra",
                country="India"
            )
            
            directory = LocalServiceDirectory()
            services = await directory.find_local_services(location, radius_km=radius_km)
            
            for service in services:
                if service.distance_km is not None:
                    assert service.distance_km <= radius_km

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__])