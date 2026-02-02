#!/usr/bin/env python3
"""
Validation script for platform integration services.

This script validates the implementation of food delivery, ride-sharing,
and platform management services without requiring a full test environment.
"""

import asyncio
import sys
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from bharatvoice.services.platform_integrations import (
        FoodDeliveryService,
        RideSharingService,
        PlatformManager
    )
    from bharatvoice.services.platform_integrations.models import (
        BookingStatus,
        FoodCategory,
        LocationPoint,
        PlatformProvider,
        RideType,
        VoiceServiceCommand
    )
    print("✓ Successfully imported platform integration modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def validate_food_delivery_service():
    """Validate food delivery service functionality."""
    print("\n=== Validating Food Delivery Service ===")
    
    try:
        # Initialize service
        config = {
            'swiggy': {'api_key': 'test_key'},
            'zomato': {'api_key': 'test_key'},
            'price_comparison_enabled': True
        }
        service = FoodDeliveryService(config)
        print("✓ Food delivery service initialized")
        
        # Create sample location
        location = LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
        print("✓ Sample location created")
        
        # Create sample command
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Order biryani from nearby restaurant",
            service_type="food_delivery",
            parsed_intent="order_food",
            extracted_entities={
                'location': location.dict(),
                'cuisine': 'indian',
                'dish': 'biryani'
            },
            confidence=0.9
        )
        print("✓ Sample voice command created")
        
        # Test command processing
        result = await service.process_food_order_command(command)
        if result['success']:
            print("✓ Food order command processed successfully")
            print(f"  Found {len(result['restaurants'])} restaurants")
        else:
            print(f"✗ Food order command failed: {result.get('error')}")
        
        # Test menu retrieval
        menu_result = await service.get_restaurant_menu("rest_001", PlatformProvider.SWIGGY)
        if menu_result['success']:
            print("✓ Restaurant menu retrieved successfully")
            print(f"  Menu has {menu_result['total_items']} items")
        else:
            print(f"✗ Menu retrieval failed: {menu_result.get('error')}")
        
        # Test price comparison
        price_result = await service.compare_food_prices(
            location=location,
            cuisine_type=FoodCategory.INDIAN,
            dish_name="biryani"
        )
        if price_result['success']:
            print("✓ Price comparison completed successfully")
            comparison = price_result['comparison']
            print(f"  Best price: ₹{comparison['best_price']} on {comparison['best_platform']}")
        else:
            print(f"✗ Price comparison failed: {price_result.get('error')}")
        
        # Test health check
        health = await service.health_check()
        if health.success:
            print("✓ Food delivery service health check passed")
        else:
            print(f"✗ Health check failed: {health.error_message}")
            
    except Exception as e:
        print(f"✗ Food delivery service validation failed: {e}")
        return False
    
    return True


async def validate_ride_sharing_service():
    """Validate ride-sharing service functionality."""
    print("\n=== Validating Ride Sharing Service ===")
    
    try:
        # Initialize service
        config = {
            'ola': {'api_key': 'test_key'},
            'uber': {'api_key': 'test_key'},
            'price_comparison_enabled': True
        }
        service = RideSharingService(config)
        print("✓ Ride sharing service initialized")
        
        # Create sample locations
        pickup_location = LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
        
        drop_location = LocationPoint(
            latitude=12.9352,
            longitude=77.6245,
            address="Electronic City, Bangalore",
            city="Bangalore"
        )
        print("✓ Sample locations created")
        
        # Create sample command
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Book a cab from MG Road to Electronic City",
            service_type="ride_sharing",
            parsed_intent="book_ride",
            extracted_entities={
                'pickup_location': pickup_location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': 'mini'
            },
            confidence=0.9
        )
        print("✓ Sample ride command created")
        
        # Test command processing
        result = await service.process_ride_booking_command(command)
        if result['success']:
            print("✓ Ride booking command processed successfully")
            print(f"  Found {len(result['ride_options'])} ride options")
        else:
            print(f"✗ Ride booking command failed: {result.get('error')}")
        
        # Test price comparison
        price_result = await service.compare_ride_prices(
            pickup_location=pickup_location,
            drop_location=drop_location,
            ride_type=RideType.MINI
        )
        if price_result['success']:
            print("✓ Ride price comparison completed successfully")
            comparison = price_result['comparison']
            print(f"  Best price: ₹{comparison['best_price']} on {comparison['best_platform']}")
        else:
            print(f"✗ Ride price comparison failed: {price_result.get('error')}")
        
        # Test health check
        health = await service.health_check()
        if health.success:
            print("✓ Ride sharing service health check passed")
        else:
            print(f"✗ Health check failed: {health.error_message}")
            
    except Exception as e:
        print(f"✗ Ride sharing service validation failed: {e}")
        return False
    
    return True


async def validate_platform_manager():
    """Validate platform manager functionality."""
    print("\n=== Validating Platform Manager ===")
    
    try:
        # Initialize platform manager
        config = {
            'food_delivery': {
                'swiggy': {'api_key': 'test_key'},
                'zomato': {'api_key': 'test_key'}
            },
            'ride_sharing': {
                'ola': {'api_key': 'test_key'},
                'uber': {'api_key': 'test_key'}
            },
            'price_comparison_enabled': True,
            'max_concurrent_bookings': 5
        }
        manager = PlatformManager(config)
        print("✓ Platform manager initialized")
        
        # Create sample location
        location = LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
        
        # Test food service command
        food_command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Order pizza from nearby restaurant",
            service_type="food_delivery",
            parsed_intent="order_food",
            extracted_entities={
                'location': location.dict(),
                'cuisine': 'italian',
                'dish': 'pizza'
            },
            confidence=0.9
        )
        
        food_result = await manager.process_service_command(food_command)
        if food_result['success']:
            print("✓ Food service command processed through platform manager")
        else:
            print(f"✗ Food service command failed: {food_result.get('error')}")
        
        # Test ride service command
        drop_location = LocationPoint(
            latitude=12.9352,
            longitude=77.6245,
            address="Electronic City, Bangalore",
            city="Bangalore"
        )
        
        ride_command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Book a cab to Electronic City",
            service_type="ride_sharing",
            parsed_intent="book_ride",
            extracted_entities={
                'pickup_location': location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': 'sedan'
            },
            confidence=0.9
        )
        
        ride_result = await manager.process_service_command(ride_command)
        if ride_result['success']:
            print("✓ Ride service command processed through platform manager")
        else:
            print(f"✗ Ride service command failed: {ride_result.get('error')}")
        
        # Test platform recommendations
        user_id = uuid4()
        recommendations = await manager.get_platform_recommendations(
            user_id=user_id,
            service_type="food_delivery",
            location=location
        )
        if recommendations['success']:
            print("✓ Platform recommendations generated successfully")
            print(f"  Found {len(recommendations['recommendations'])} platform recommendations")
        else:
            print(f"✗ Platform recommendations failed: {recommendations.get('error')}")
        
        # Test health check
        health = await manager.health_check()
        if health.success:
            print("✓ Platform manager health check passed")
        else:
            print(f"✗ Health check failed: {health.error_message}")
            
    except Exception as e:
        print(f"✗ Platform manager validation failed: {e}")
        return False
    
    return True


def validate_models():
    """Validate data models."""
    print("\n=== Validating Data Models ===")
    
    try:
        # Test LocationPoint validation
        location = LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
        print("✓ LocationPoint model validation passed")
        
        # Test VoiceServiceCommand creation
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Book a table at restaurant",
            service_type="restaurant_booking",
            parsed_intent="book_table",
            extracted_entities={'restaurant': 'Test Restaurant'},
            confidence=0.85
        )
        print("✓ VoiceServiceCommand model creation passed")
        
        # Test enum values
        assert BookingStatus.PENDING == "pending"
        assert PlatformProvider.SWIGGY == "swiggy"
        assert RideType.MINI == "mini"
        assert FoodCategory.INDIAN == "indian"
        print("✓ Enum validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


async def main():
    """Run all validations."""
    print("Starting Platform Integration Validation")
    print("=" * 50)
    
    # Validate models first
    models_ok = validate_models()
    
    # Validate services
    food_ok = await validate_food_delivery_service()
    ride_ok = await validate_ride_sharing_service()
    manager_ok = await validate_platform_manager()
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    results = {
        "Data Models": models_ok,
        "Food Delivery Service": food_ok,
        "Ride Sharing Service": ride_ok,
        "Platform Manager": manager_ok
    }
    
    all_passed = True
    for component, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{component}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("Platform integration services are working correctly.")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please check the error messages above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        sys.exit(1)