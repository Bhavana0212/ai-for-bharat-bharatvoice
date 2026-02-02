"""
Property-Based Tests for Indian Service Integration.

**Property 13: Indian Service Integration**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

This module tests the integration with Indian services including railways,
weather, government services, cricket scores, and Bollywood news.
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Dict, Any, List

from bharatvoice.core.models import ServiceType, ServiceParameters, ServiceResult
from bharatvoice.services.external_integrations.service_manager import ExternalServiceManager
from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService, TrainClass
from bharatvoice.services.external_integrations.weather_service import WeatherService
from bharatvoice.services.external_integrations.digital_india_service import DigitalIndiaService, ServiceCategory


class TestIndianServiceIntegrationProperties:
    """Property-based tests for Indian service integration."""
    
    @pytest.fixture
    def service_manager(self):
        """Create service manager instance."""
        return ExternalServiceManager()
    
    @pytest.fixture
    def railways_service(self):
        """Create railways service instance."""
        return IndianRailwaysService()
    
    @pytest.fixture
    def weather_service(self):
        """Create weather service instance."""
        return WeatherService()
    
    @pytest.fixture
    def digital_india_service(self):
        """Create digital India service instance."""
        return DigitalIndiaService()
    
    # Property 13.1: Service Response Consistency
    @given(
        service_type=st.sampled_from([
            ServiceType.INDIAN_RAILWAYS,
            ServiceType.WEATHER,
            ServiceType.CRICKET_SCORES,
            ServiceType.BOLLYWOOD_NEWS,
            ServiceType.GOVERNMENT_SERVICE
        ]),
        timeout=st.floats(min_value=1.0, max_value=30.0)
    )
    @settings(max_examples=20, deadline=10000)
    @pytest.mark.asyncio
    async def test_service_response_consistency_property(
        self,
        service_manager,
        service_type,
        timeout
    ):
        """
        **Property 13.1: Service Response Consistency**
        
        All service responses must have consistent structure with required fields:
        - service_type matches request
        - success field is boolean
        - data field is present (dict)
        - error_message field is present (string or None)
        - response_time field is present (float)
        """
        # Create service parameters based on type
        params = self._create_service_parameters(service_type, timeout)
        
        # Execute service request
        result = await service_manager.process_service_request(params)
        
        # Verify response structure consistency
        assert isinstance(result, ServiceResult)
        assert result.service_type == service_type
        assert isinstance(result.success, bool)
        assert isinstance(result.data, dict)
        assert result.error_message is None or isinstance(result.error_message, str)
        assert isinstance(result.response_time, (int, float))
        assert result.response_time >= 0
        
        # If successful, data should not be empty for most services
        if result.success and service_type != ServiceType.GOVERNMENT_SERVICE:
            assert len(result.data) > 0
        
        # If failed, error message should be provided
        if not result.success:
            assert result.error_message is not None
            assert len(result.error_message) > 0
    
    # Property 13.2: Railways Service Reliability
    @given(
        train_number=st.text(min_size=5, max_size=5).filter(lambda x: x.isdigit()),
        date=st.dates().map(lambda d: d.strftime("%Y-%m-%d"))
    )
    @settings(max_examples=15, deadline=8000)
    @pytest.mark.asyncio
    async def test_railways_service_reliability_property(
        self,
        railways_service,
        train_number,
        date
    ):
        """
        **Property 13.2: Railways Service Reliability**
        
        Railways service must handle all valid train queries consistently:
        - Valid train numbers return structured data or appropriate error
        - Response time is reasonable (< 5 seconds)
        - Error messages are informative for invalid inputs
        """
        result = await railways_service.get_train_schedule(train_number, date)
        
        # Verify response structure
        assert isinstance(result, ServiceResult)
        assert result.service_type == ServiceType.INDIAN_RAILWAYS
        assert result.response_time < 5.0
        
        if result.success:
            # Successful response should have train schedule data
            assert "train_schedule" in result.data or "query_date" in result.data
            if "train_schedule" in result.data:
                schedule = result.data["train_schedule"]
                assert "train_number" in schedule
                assert "train_name" in schedule
        else:
            # Failed response should have informative error
            assert result.error_message is not None
            assert len(result.error_message) > 10  # Reasonably informative
    
    # Property 13.3: Weather Service Data Validity
    @given(
        city=st.sampled_from([
            "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
            "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
        ]),
        include_forecast=st.booleans(),
        include_monsoon=st.booleans()
    )
    @settings(max_examples=15, deadline=8000)
    @pytest.mark.asyncio
    async def test_weather_service_data_validity_property(
        self,
        weather_service,
        city,
        include_forecast,
        include_monsoon
    ):
        """
        **Property 13.3: Weather Service Data Validity**
        
        Weather service must return valid meteorological data:
        - Temperature values are within reasonable range (-10°C to 50°C)
        - Humidity is between 0-100%
        - Forecast data is properly structured when requested
        - Monsoon information is included for Indian context
        """
        result = await weather_service.get_weather_info(
            city, include_forecast, include_monsoon
        )
        
        assert isinstance(result, ServiceResult)
        assert result.service_type == ServiceType.WEATHER
        
        if result.success:
            assert "current_weather" in result.data
            weather = result.data["current_weather"]
            
            # Validate temperature range (reasonable for India)
            temp = weather["temperature_celsius"]
            assert -10 <= temp <= 50
            
            # Validate humidity
            humidity = weather["humidity"]
            assert 0 <= humidity <= 100
            
            # Validate wind speed
            wind_speed = weather["wind_speed_kmh"]
            assert 0 <= wind_speed <= 200  # Reasonable wind speed range
            
            # Check forecast structure if requested
            if include_forecast and "forecast" in result.data:
                forecast = result.data["forecast"]
                assert isinstance(forecast, list)
                assert len(forecast) <= 7  # Reasonable forecast length
                
                for day in forecast:
                    assert "temperature_high_celsius" in day
                    assert "temperature_low_celsius" in day
                    assert day["temperature_high_celsius"] >= day["temperature_low_celsius"]
            
            # Check monsoon information if requested
            if include_monsoon and "monsoon_info" in result.data:
                monsoon = result.data["monsoon_info"]
                assert "current_phase" in monsoon
                assert "is_monsoon_active" in monsoon
                assert isinstance(monsoon["is_monsoon_active"], bool)
    
    # Property 13.4: Government Service Information Completeness
    @given(
        service_name=st.sampled_from([
            "aadhaar", "pan", "passport", "driving license", "pm kisan"
        ])
    )
    @settings(max_examples=10, deadline=6000)
    @pytest.mark.asyncio
    async def test_government_service_completeness_property(
        self,
        digital_india_service,
        service_name
    ):
        """
        **Property 13.4: Government Service Information Completeness**
        
        Government service information must be comprehensive and accurate:
        - Service information includes all required fields
        - Document requirements are clearly specified
        - Application steps are logically ordered
        - Processing times and fees are provided
        """
        result = await digital_india_service.get_service_information(service_name)
        
        assert isinstance(result, ServiceResult)
        assert result.service_type == ServiceType.GOVERNMENT_SERVICE
        
        if result.success:
            service_info = result.data["service_info"]
            
            # Verify essential service information fields
            required_fields = [
                "service_id", "service_name", "department", "category",
                "description", "processing_time", "fees"
            ]
            for field in required_fields:
                assert field in service_info
                assert service_info[field] is not None
            
            # Verify document requirements structure
            assert "required_documents" in result.data
            documents = result.data["required_documents"]
            assert isinstance(documents, list)
            
            for doc in documents:
                assert "name" in doc
                assert "mandatory" in doc
                assert "description" in doc
                assert isinstance(doc["mandatory"], bool)
            
            # Verify application steps structure
            assert "application_steps" in result.data
            steps = result.data["application_steps"]
            assert isinstance(steps, list)
            assert len(steps) > 0
            
            # Steps should be properly ordered
            for i, step in enumerate(steps):
                assert "step" in step
                assert step["step"] == i + 1
                assert "title" in step
                assert "description" in step
                assert "online_available" in step
                assert isinstance(step["online_available"], bool)
    
    # Property 13.5: Service Error Handling Robustness
    @given(
        invalid_input=st.one_of(
            st.text(max_size=3),  # Too short
            st.text(min_size=100),  # Too long
            st.just(""),  # Empty
            st.just("invalid_service_name_12345"),  # Non-existent
            st.text().filter(lambda x: any(c in x for c in "!@#$%^&*()"))  # Special chars
        ),
        service_type=st.sampled_from([
            ServiceType.INDIAN_RAILWAYS,
            ServiceType.WEATHER,
            ServiceType.GOVERNMENT_SERVICE
        ])
    )
    @settings(max_examples=20, deadline=8000)
    @pytest.mark.asyncio
    async def test_service_error_handling_robustness_property(
        self,
        service_manager,
        invalid_input,
        service_type
    ):
        """
        **Property 13.5: Service Error Handling Robustness**
        
        Services must handle invalid inputs gracefully:
        - Invalid inputs return failure with informative error messages
        - No exceptions are raised for invalid inputs
        - Response structure remains consistent even for errors
        - Error messages are user-friendly and actionable
        """
        # Create parameters with invalid input
        params = ServiceParameters(
            service_type=service_type,
            parameters=self._create_invalid_parameters(service_type, invalid_input),
            timeout=10.0
        )
        
        # Service should handle invalid input gracefully
        result = await service_manager.process_service_request(params)
        
        # Verify error handling
        assert isinstance(result, ServiceResult)
        assert result.service_type == service_type
        
        # For clearly invalid inputs, service should fail gracefully
        if len(invalid_input.strip()) == 0 or len(invalid_input) > 50:
            assert not result.success
            assert result.error_message is not None
            assert len(result.error_message) > 5  # Reasonably informative
            
            # Error message should be user-friendly (no technical jargon)
            error_msg = result.error_message.lower()
            technical_terms = ["exception", "traceback", "null", "undefined", "error code"]
            assert not any(term in error_msg for term in technical_terms)
    
    # Property 13.6: Service Performance Consistency
    @given(
        service_type=st.sampled_from([
            ServiceType.WEATHER,
            ServiceType.CRICKET_SCORES,
            ServiceType.BOLLYWOOD_NEWS
        ]),
        num_requests=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=10, deadline=15000)
    @pytest.mark.asyncio
    async def test_service_performance_consistency_property(
        self,
        service_manager,
        service_type,
        num_requests
    ):
        """
        **Property 13.6: Service Performance Consistency**
        
        Service performance must be consistent across multiple requests:
        - Response times are within reasonable bounds
        - Multiple requests to same service show consistent performance
        - No significant performance degradation over time
        """
        response_times = []
        
        for _ in range(num_requests):
            params = self._create_service_parameters(service_type, 10.0)
            result = await service_manager.process_service_request(params)
            
            assert isinstance(result, ServiceResult)
            response_times.append(result.response_time)
        
        # Verify performance consistency
        assert len(response_times) == num_requests
        assert all(0 <= rt <= 10.0 for rt in response_times)  # Reasonable response times
        
        # Performance should be relatively consistent (no outliers > 3x average)
        if len(response_times) > 1:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            assert max_time <= avg_time * 3  # No response more than 3x average
    
    # Property 13.7: Data Format Standardization
    @given(
        service_type=st.sampled_from([
            ServiceType.WEATHER,
            ServiceType.CRICKET_SCORES,
            ServiceType.BOLLYWOOD_NEWS
        ])
    )
    @settings(max_examples=15, deadline=8000)
    @pytest.mark.asyncio
    async def test_data_format_standardization_property(
        self,
        service_manager,
        service_type
    ):
        """
        **Property 13.7: Data Format Standardization**
        
        All services must return data in standardized formats:
        - Dates in ISO format (YYYY-MM-DD)
        - Times in 24-hour format or with timezone
        - Consistent field naming conventions
        - Proper data types for numeric values
        """
        params = self._create_service_parameters(service_type, 10.0)
        result = await service_manager.process_service_request(params)
        
        assert isinstance(result, ServiceResult)
        
        if result.success and result.data:
            # Check for date format standardization
            self._verify_date_formats(result.data)
            
            # Check for consistent field naming (snake_case)
            self._verify_field_naming(result.data)
            
            # Check for proper data types
            self._verify_data_types(result.data)
    
    # Helper methods
    
    def _create_service_parameters(self, service_type: ServiceType, timeout: float) -> ServiceParameters:
        """Create service parameters for testing."""
        if service_type == ServiceType.INDIAN_RAILWAYS:
            return ServiceParameters(
                service_type=service_type,
                parameters={
                    "request_type": "train_schedule",
                    "train_number": "12002",
                    "date": "2024-01-15"
                },
                timeout=timeout
            )
        elif service_type == ServiceType.WEATHER:
            return ServiceParameters(
                service_type=service_type,
                parameters={
                    "request_type": "weather_info",
                    "city": "Delhi",
                    "include_forecast": True,
                    "include_monsoon": True
                },
                timeout=timeout
            )
        elif service_type == ServiceType.CRICKET_SCORES:
            return ServiceParameters(
                service_type=service_type,
                parameters={
                    "match_type": "T20",
                    "team": "India"
                },
                timeout=timeout
            )
        elif service_type == ServiceType.BOLLYWOOD_NEWS:
            return ServiceParameters(
                service_type=service_type,
                parameters={
                    "category": "Movies",
                    "limit": 5
                },
                timeout=timeout
            )
        elif service_type == ServiceType.GOVERNMENT_SERVICE:
            return ServiceParameters(
                service_type=service_type,
                parameters={
                    "request_type": "service_info",
                    "service_name": "aadhaar"
                },
                timeout=timeout
            )
        else:
            return ServiceParameters(
                service_type=service_type,
                parameters={},
                timeout=timeout
            )
    
    def _create_invalid_parameters(self, service_type: ServiceType, invalid_input: str) -> Dict[str, Any]:
        """Create invalid parameters for error testing."""
        if service_type == ServiceType.INDIAN_RAILWAYS:
            return {
                "request_type": "train_schedule",
                "train_number": invalid_input,
                "date": "invalid-date"
            }
        elif service_type == ServiceType.WEATHER:
            return {
                "request_type": "weather_info",
                "city": invalid_input
            }
        elif service_type == ServiceType.GOVERNMENT_SERVICE:
            return {
                "request_type": "service_info",
                "service_name": invalid_input
            }
        else:
            return {"invalid_param": invalid_input}
    
    def _verify_date_formats(self, data: Dict[str, Any]) -> None:
        """Verify date formats in response data."""
        import re
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        datetime_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
        
        def check_dates(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and ('date' in key.lower() or 'time' in key.lower()):
                        if 'date' in key.lower() and not key.lower().endswith('time'):
                            # Should be date format
                            if len(value) == 10:  # YYYY-MM-DD
                                assert date_pattern.match(value), f"Invalid date format at {path}.{key}: {value}"
                        elif 'updated' in key.lower() or 'timestamp' in key.lower():
                            # Should be datetime format
                            assert datetime_pattern.match(value), f"Invalid datetime format at {path}.{key}: {value}"
                    elif isinstance(value, (dict, list)):
                        check_dates(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_dates(item, f"{path}[{i}]")
        
        check_dates(data)
    
    def _verify_field_naming(self, data: Dict[str, Any]) -> None:
        """Verify consistent field naming conventions."""
        import re
        snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*[a-z0-9]$|^[a-z]$')
        
        def check_naming(obj, path=""):
            if isinstance(obj, dict):
                for key in obj.keys():
                    # Allow some exceptions for external API compatibility
                    exceptions = ['id', 'url', 'api', 'pnr', 'aqi', 'pm25', 'pm10']
                    if key.lower() not in exceptions:
                        assert snake_case_pattern.match(key) or key.islower(), f"Non-snake_case field at {path}: {key}"
                    
                    if isinstance(obj[key], (dict, list)):
                        check_naming(obj[key], f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        check_naming(item, f"{path}[{i}]")
        
        check_naming(data)
    
    def _verify_data_types(self, data: Dict[str, Any]) -> None:
        """Verify proper data types for numeric and boolean values."""
        def check_types(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check numeric fields
                    if any(term in key.lower() for term in ['temperature', 'humidity', 'speed', 'time', 'fare', 'price', 'cost']):
                        if value is not None and not isinstance(value, str):
                            assert isinstance(value, (int, float)), f"Non-numeric value for {path}.{key}: {value} ({type(value)})"
                    
                    # Check boolean fields
                    if any(term in key.lower() for term in ['is_', 'has_', 'available', 'active', 'success', 'mandatory']):
                        if value is not None:
                            assert isinstance(value, bool), f"Non-boolean value for {path}.{key}: {value} ({type(value)})"
                    
                    if isinstance(value, (dict, list)):
                        check_types(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        check_types(item, f"{path}[{i}]")
        
        check_types(data)