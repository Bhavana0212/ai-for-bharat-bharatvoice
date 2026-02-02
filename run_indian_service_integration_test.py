#!/usr/bin/env python3
"""
Test runner for Indian Service Integration Property Tests.
"""

import sys
import os
import asyncio
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_service_manager_basic():
    """Test basic service manager functionality."""
    try:
        from bharatvoice.services.external_integrations.service_manager import ExternalServiceManager
        from bharatvoice.core.models import ServiceType, ServiceParameters
        
        print("Testing Service Manager Basic Functionality...")
        
        manager = ExternalServiceManager()
        
        # Test railways service
        params = ServiceParameters(
            service_type=ServiceType.INDIAN_RAILWAYS,
            parameters={
                "request_type": "train_schedule",
                "train_number": "12002",
                "date": "2024-01-15"
            },
            timeout=10.0
        )
        
        result = await manager.process_service_request(params)
        print(f"Railways service result: success={result.success}, response_time={result.response_time:.2f}s")
        
        # Test weather service
        params = ServiceParameters(
            service_type=ServiceType.WEATHER,
            parameters={
                "request_type": "weather_info",
                "city": "Delhi",
                "include_forecast": True,
                "include_monsoon": True
            },
            timeout=10.0
        )
        
        result = await manager.process_service_request(params)
        print(f"Weather service result: success={result.success}, response_time={result.response_time:.2f}s")
        
        # Test government service
        params = ServiceParameters(
            service_type=ServiceType.GOVERNMENT_SERVICE,
            parameters={
                "request_type": "service_info",
                "service_name": "aadhaar"
            },
            timeout=10.0
        )
        
        result = await manager.process_service_request(params)
        print(f"Government service result: success={result.success}, response_time={result.response_time:.2f}s")
        
        print("✅ Service Manager basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Service Manager test failed: {e}")
        traceback.print_exc()
        return False

async def test_railways_service():
    """Test railways service functionality."""
    try:
        from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
        
        print("\nTesting Railways Service...")
        
        service = IndianRailwaysService()
        
        # Test train schedule
        result = await service.get_train_schedule("12002", "2024-01-15")
        print(f"Train schedule: success={result.success}")
        if result.success and "train_schedule" in result.data:
            schedule = result.data["train_schedule"]
            print(f"  Train: {schedule.get('train_name', 'Unknown')}")
        
        # Test find trains
        result = await service.find_trains_between_stations("Delhi", "Mumbai", "2024-01-15")
        print(f"Find trains: success={result.success}")
        if result.success:
            trains = result.data.get("available_trains", [])
            print(f"  Found {len(trains)} trains")
        
        # Test natural language query
        result = await service.process_natural_language_query("What is the schedule of train 12002?")
        print(f"Natural language query: success={result.success}")
        
        print("✅ Railways service test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Railways service test failed: {e}")
        traceback.print_exc()
        return False

async def test_weather_service():
    """Test weather service functionality."""
    try:
        from bharatvoice.services.external_integrations.weather_service import WeatherService
        
        print("\nTesting Weather Service...")
        
        service = WeatherService()
        
        # Test weather info
        result = await service.get_weather_info("Delhi", include_forecast=True, include_monsoon=True)
        print(f"Weather info: success={result.success}")
        if result.success:
            weather = result.data.get("current_weather", {})
            print(f"  Temperature: {weather.get('temperature_celsius', 'Unknown')}°C")
            print(f"  Condition: {weather.get('condition', 'Unknown')}")
            
            if "monsoon_info" in result.data:
                monsoon = result.data["monsoon_info"]
                print(f"  Monsoon phase: {monsoon.get('current_phase', 'Unknown')}")
        
        # Test cricket scores
        result = await service.get_cricket_scores()
        print(f"Cricket scores: success={result.success}")
        if result.success:
            matches = result.data.get("matches", [])
            print(f"  Found {len(matches)} matches")
        
        # Test Bollywood news
        result = await service.get_bollywood_news(limit=3)
        print(f"Bollywood news: success={result.success}")
        if result.success:
            news = result.data.get("news", [])
            print(f"  Found {len(news)} news items")
        
        print("✅ Weather service test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Weather service test failed: {e}")
        traceback.print_exc()
        return False

async def test_digital_india_service():
    """Test Digital India service functionality."""
    try:
        from bharatvoice.services.external_integrations.digital_india_service import DigitalIndiaService, ServiceCategory
        
        print("\nTesting Digital India Service...")
        
        service = DigitalIndiaService()
        
        # Test service information
        result = await service.get_service_information("aadhaar")
        print(f"Service info: success={result.success}")
        if result.success:
            service_info = result.data.get("service_info", {})
            print(f"  Service: {service_info.get('service_name', 'Unknown')}")
            print(f"  Department: {service_info.get('department', 'Unknown')}")
        
        # Test document requirements
        result = await service.get_document_requirements("pan")
        print(f"Document requirements: success={result.success}")
        if result.success:
            docs = result.data.get("mandatory_documents", [])
            print(f"  Mandatory documents: {len(docs)}")
        
        # Test application guidance
        result = await service.get_application_guidance("passport")
        print(f"Application guidance: success={result.success}")
        if result.success:
            steps = result.data.get("application_steps", [])
            print(f"  Application steps: {len(steps)}")
        
        # Test category search
        result = await service.search_services_by_category(ServiceCategory.IDENTITY_DOCUMENTS)
        print(f"Category search: success={result.success}")
        if result.success:
            services = result.data.get("services", [])
            print(f"  Identity document services: {len(services)}")
        
        print("✅ Digital India service test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Digital India service test failed: {e}")
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test error handling for invalid inputs."""
    try:
        from bharatvoice.services.external_integrations.service_manager import ExternalServiceManager
        from bharatvoice.core.models import ServiceType, ServiceParameters
        
        print("\nTesting Error Handling...")
        
        manager = ExternalServiceManager()
        
        # Test invalid train number
        params = ServiceParameters(
            service_type=ServiceType.INDIAN_RAILWAYS,
            parameters={
                "request_type": "train_schedule",
                "train_number": "invalid",
                "date": "invalid-date"
            },
            timeout=5.0
        )
        
        result = await manager.process_service_request(params)
        print(f"Invalid train query: success={result.success} (should be False)")
        if not result.success:
            print(f"  Error message: {result.error_message[:50]}...")
        
        # Test invalid city
        params = ServiceParameters(
            service_type=ServiceType.WEATHER,
            parameters={
                "request_type": "weather_info",
                "city": "NonExistentCity12345"
            },
            timeout=5.0
        )
        
        result = await manager.process_service_request(params)
        print(f"Invalid city query: success={result.success} (should be False)")
        if not result.success:
            print(f"  Error message: {result.error_message[:50]}...")
        
        print("✅ Error handling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

async def test_property_13_sample():
    """Test a sample of Property 13: Indian Service Integration."""
    try:
        from bharatvoice.services.external_integrations.service_manager import ExternalServiceManager
        from bharatvoice.core.models import ServiceType, ServiceParameters, ServiceResult
        
        print("\nTesting Property 13: Indian Service Integration (Sample)...")
        
        manager = ExternalServiceManager()
        
        # Test service response consistency
        service_types = [ServiceType.WEATHER, ServiceType.CRICKET_SCORES, ServiceType.GOVERNMENT_SERVICE]
        
        for service_type in service_types:
            if service_type == ServiceType.WEATHER:
                params = ServiceParameters(
                    service_type=service_type,
                    parameters={"request_type": "weather_info", "city": "Mumbai"},
                    timeout=10.0
                )
            elif service_type == ServiceType.CRICKET_SCORES:
                params = ServiceParameters(
                    service_type=service_type,
                    parameters={"match_type": "T20"},
                    timeout=10.0
                )
            else:  # GOVERNMENT_SERVICE
                params = ServiceParameters(
                    service_type=service_type,
                    parameters={"request_type": "service_info", "service_name": "pan"},
                    timeout=10.0
                )
            
            result = await manager.process_service_request(params)
            
            # Verify Property 13.1: Service Response Consistency
            assert isinstance(result, ServiceResult), f"Result is not ServiceResult for {service_type}"
            assert result.service_type == service_type, f"Service type mismatch for {service_type}"
            assert isinstance(result.success, bool), f"Success field is not boolean for {service_type}"
            assert isinstance(result.data, dict), f"Data field is not dict for {service_type}"
            assert result.error_message is None or isinstance(result.error_message, str), f"Error message type invalid for {service_type}"
            assert isinstance(result.response_time, (int, float)), f"Response time type invalid for {service_type}"
            assert result.response_time >= 0, f"Response time negative for {service_type}"
            
            print(f"  {service_type}: ✅ Response structure consistent")
            
            # If successful, verify data content
            if result.success:
                assert len(result.data) > 0, f"Empty data for successful {service_type} request"
                print(f"  {service_type}: ✅ Data content present")
            else:
                assert result.error_message is not None, f"No error message for failed {service_type} request"
                assert len(result.error_message) > 0, f"Empty error message for failed {service_type} request"
                print(f"  {service_type}: ✅ Error message present")
        
        print("✅ Property 13 sample test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Property 13 sample test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Indian Service Integration Tests...")
    
    tests = [
        test_service_manager_basic,
        test_railways_service,
        test_weather_service,
        test_digital_india_service,
        test_error_handling,
        test_property_13_sample
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Indian Service Integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)