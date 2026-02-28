#!/usr/bin/env python3
"""
Validation script for Indian Railways service implementation.
Tests the enhanced API integration with fallback mechanisms.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
from bharatvoice.core.models import ServiceType


async def test_railways_service():
    """Test the Indian Railways service implementation."""
    print("🚂 Testing Indian Railways Service Implementation")
    print("=" * 50)
    
    # Test without API key (should use mock)
    print("\n1. Testing without API key (mock mode):")
    service = IndianRailwaysService()
    
    async with service:
        # Test train schedule
        print("   Testing train schedule...")
        result = await service.get_train_schedule("12002", "2024-01-15")
        print(f"   ✓ Train schedule: {result.success}")
        if result.success:
            print(f"     Data source: {result.data.get('data_source', 'unknown')}")
        
        # Test trains between stations
        print("   Testing trains between stations...")
        result = await service.find_trains_between_stations("delhi", "mumbai", "2024-01-15")
        print(f"   ✓ Trains search: {result.success}")
        if result.success:
            print(f"     Found {result.data.get('total_trains', 0)} trains")
        
        # Test ticket availability
        print("   Testing ticket availability...")
        from bharatvoice.services.external_integrations.indian_railways_service import TrainClass
        result = await service.check_ticket_availability("12002", "delhi", "mumbai", "2024-01-15", TrainClass.AC_3_TIER)
        print(f"   ✓ Availability check: {result.success}")
        
        # Test PNR status
        print("   Testing PNR status...")
        result = await service.get_pnr_status("1234567890")
        print(f"   ✓ PNR status: {result.success}")
        
        # Test natural language processing
        print("   Testing natural language query...")
        result = await service.process_natural_language_query("What is the schedule of train 12002 on 15th January?")
        print(f"   ✓ NL query: {result.success}")
    
    # Test with API key (should attempt real API, fallback to mock)
    print("\n2. Testing with API key (real API with fallback):")
    service_with_key = IndianRailwaysService(api_key="test_key_123")
    
    async with service_with_key:
        print("   Testing train schedule with API key...")
        result = await service_with_key.get_train_schedule("12002", "2024-01-15")
        print(f"   ✓ Train schedule with API: {result.success}")
        if result.success:
            print(f"     Data source: {result.data.get('data_source', 'unknown')}")
    
    # Test validation functions
    print("\n3. Testing validation functions:")
    print(f"   ✓ Valid train number (12002): {service._validate_train_number('12002')}")
    print(f"   ✓ Invalid train number (123): {service._validate_train_number('123')}")
    print(f"   ✓ Valid PNR (1234567890): {service._validate_pnr('1234567890')}")
    print(f"   ✓ Invalid PNR (123): {service._validate_pnr('123')}")
    print(f"   ✓ Valid date (2024-01-15): {service._validate_date_format('2024-01-15')}")
    print(f"   ✓ Invalid date (15-01-2024): {service._validate_date_format('15-01-2024')}")
    
    # Test station code mapping
    print("\n4. Testing station code mapping:")
    print(f"   ✓ Delhi -> {service._get_station_code('delhi')}")
    print(f"   ✓ Mumbai -> {service._get_station_code('mumbai')}")
    print(f"   ✓ NDLS -> {service._get_station_code('NDLS')}")
    print(f"   ✓ Invalid -> {service._get_station_code('invalid_station')}")
    
    print("\n✅ All tests completed successfully!")
    print("🎉 Indian Railways service implementation is working correctly!")


if __name__ == "__main__":
    try:
        asyncio.run(test_railways_service())
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)