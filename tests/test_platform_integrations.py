"""
Tests for platform integration services.

This module tests the food delivery, ride-sharing, and platform management
services for booking, tracking, and price comparison functionality.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

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


class TestFoodDeliveryService:
    """Test food delivery service functionality."""
    
    @pytest.fixture
    def food_service(self):
        """Create food delivery service instance."""
        config = {
            'swiggy': {'api_key': 'test_key'},
            'zomato': {'api_key': 'test_key'},
            'uber_eats': {'api_key': 'test_key'},
            'price_comparison_enabled': True
        }
        return FoodDeliveryService(config)
    
    @pytest.fixture
    def sample_location(self):
        """Create sample location."""
        return LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
    
    @pytest.fixture
    def sample_command(self, sample_location):
        """Create sample voice command."""
        return VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Order biryani from nearby restaurant",
            service_type="food_delivery",
            parsed_intent="order_food",
            extracted_entities={
                'location': sample_location.dict(),
                'cuisine': 'indian',
                'dish': 'biryani'
            },
            confidence=0.9
        )
    
    @pytest.mark.asyncio
    async def test_process_food_order_command(self, food_service, sample_command):
        """Test processing food order command."""
        result = await food_service.process_food_order_command(sample_command)
        
        assert result['success'] is True
        assert 'restaurants' in result
        assert 'voice_response' in result
        assert len(result['restaurants']) > 0
        
        # Check restaurant data structure
        restaurant = result['restaurants'][0]
        assert 'restaurant_id' in restaurant
        assert 'name' in restaurant
        assert 'rating' in restaurant
        assert 'delivery_time' in restaurant
    
    @pytest.mark.asyncio
    async def test_get_restaurant_menu(self, food_service):
        """Test getting restaurant menu."""
        result = await food_service.get_restaurant_menu("rest_001", PlatformProvider.SWIGGY)
        
        assert result['success'] is True
        assert 'menu_categories' in result
        assert 'total_items' in result
        assert 'voice_response' in result
        
        # Check menu structure
        menu_categories = result['menu_categories']
        assert isinstance(menu_categories, dict)
        assert len(menu_categories) > 0
    
    @pytest.mark.asyncio
    async def test_place_food_order(self, food_service, sample_location):
        """Test placing food order."""
        order_details = {
            'platform': 'swiggy',
            'restaurant': {
                'restaurant_id': 'rest_001',
                'name': 'Test Restaurant',
                'cuisine_types': ['indian'],
                'location': sample_location.dict(),
                'rating': 4.2,
                'delivery_time': 30,
                'minimum_order': 150,
                'delivery_fee': 25,
                'is_open': True
            },
            'items': [
                {'item_id': 'item_001', 'name': 'Biryani', 'quantity': 1, 'price': 280}
            ],
            'delivery_address': sample_location.dict(),
            'total_amount': 280,
            'delivery_fee': 25,
            'taxes': 28,
            'final_amount': 333,
            'contact_number': '+91-9876543210',
            'payment_method': 'UPI'
        }
        
        result = await food_service.place_food_order(order_details, uuid4())
        
        assert result['success'] is True
        assert 'order_id' in result
        assert 'platform_order_id' in result
        assert 'estimated_delivery_time' in result
        assert 'voice_response' in result
    
    @pytest.mark.asyncio
    async def test_compare_food_prices(self, food_service, sample_location):
        """Test food price comparison."""
        result = await food_service.compare_food_prices(
            location=sample_location,
            cuisine_type=FoodCategory.INDIAN,
            dish_name="biryani"
        )
        
        assert result['success'] is True
        assert 'comparison' in result
        assert 'voice_response' in result
        
        comparison = result['comparison']
        assert 'best_price' in comparison
        assert 'best_platform' in comparison
        assert 'platforms' in comparison
        assert len(comparison['platforms']) > 0


class TestRideSharingService:
    """Test ride-sharing service functionality."""
    
    @pytest.fixture
    def ride_service(self):
        """Create ride-sharing service instance."""
        config = {
            'ola': {'api_key': 'test_key'},
            'uber': {'api_key': 'test_key'},
            'rapido': {'api_key': 'test_key'},
            'price_comparison_enabled': True
        }
        return RideSharingService(config)
    
    @pytest.fixture
    def pickup_location(self):
        """Create pickup location."""
        return LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
    
    @pytest.fixture
    def drop_location(self):
        """Create drop location."""
        return LocationPoint(
            latitude=12.9352,
            longitude=77.6245,
            address="Electronic City, Bangalore",
            city="Bangalore"
        )
    
    @pytest.fixture
    def ride_command(self, pickup_location, drop_location):
        """Create sample ride booking command."""
        return VoiceServiceCommand(
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
    
    @pytest.mark.asyncio
    async def test_process_ride_booking_command(self, ride_service, ride_command):
        """Test processing ride booking command."""
        result = await ride_service.process_ride_booking_command(ride_command)
        
        assert result['success'] is True
        assert 'ride_options' in result
        assert 'pickup_location' in result
        assert 'drop_location' in result
        assert 'voice_response' in result
        assert len(result['ride_options']) > 0
        
        # Check ride option structure
        ride_option = result['ride_options'][0]
        assert 'platform' in ride_option
        assert 'estimated_fare' in ride_option
        assert 'estimated_duration' in ride_option
        assert 'recommendation_score' in ride_option
    
    @pytest.mark.asyncio
    async def test_book_ride(self, ride_service, pickup_location, drop_location):
        """Test booking a ride."""
        booking_details = {
            'platform': 'ola',
            'ride_type': 'mini',
            'pickup_location': pickup_location.dict(),
            'drop_location': drop_location.dict(),
            'estimated_fare': 120,
            'estimated_distance': 8.5,
            'estimated_duration': 25,
            'contact_number': '+91-9876543210',
            'payment_method': 'UPI'
        }
        
        result = await ride_service.book_ride(booking_details, uuid4())
        
        assert result['success'] is True
        assert 'booking_id' in result
        assert 'platform_booking_id' in result
        assert 'otp' in result
        assert 'driver_details' in result
        assert 'vehicle_details' in result
        assert 'voice_response' in result
    
    @pytest.mark.asyncio
    async def test_compare_ride_prices(self, ride_service, pickup_location, drop_location):
        """Test ride price comparison."""
        result = await ride_service.compare_ride_prices(
            pickup_location=pickup_location,
            drop_location=drop_location,
            ride_type=RideType.MINI
        )
        
        assert result['success'] is True
        assert 'comparison' in result
        assert 'voice_response' in result
        
        comparison = result['comparison']
        assert 'best_price' in comparison
        assert 'best_platform' in comparison
        assert 'platforms' in comparison
        assert len(comparison['platforms']) > 0


class TestPlatformManager:
    """Test platform manager functionality."""
    
    @pytest.fixture
    def platform_manager(self):
        """Create platform manager instance."""
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
        return PlatformManager(config)
    
    @pytest.fixture
    def sample_location(self):
        """Create sample location."""
        return LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
    
    @pytest.mark.asyncio
    async def test_process_food_service_command(self, platform_manager, sample_location):
        """Test processing food service command through platform manager."""
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Order pizza from nearby restaurant",
            service_type="food_delivery",
            parsed_intent="order_food",
            extracted_entities={
                'location': sample_location.dict(),
                'cuisine': 'italian',
                'dish': 'pizza'
            },
            confidence=0.9
        )
        
        result = await platform_manager.process_service_command(command)
        
        assert result['success'] is True
        assert 'restaurants' in result
        assert 'voice_response' in result
    
    @pytest.mark.asyncio
    async def test_process_ride_service_command(self, platform_manager, sample_location):
        """Test processing ride service command through platform manager."""
        drop_location = LocationPoint(
            latitude=12.9352,
            longitude=77.6245,
            address="Electronic City, Bangalore",
            city="Bangalore"
        )
        
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Book a cab to Electronic City",
            service_type="ride_sharing",
            parsed_intent="book_ride",
            extracted_entities={
                'pickup_location': sample_location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': 'sedan'
            },
            confidence=0.9
        )
        
        result = await platform_manager.process_service_command(command)
        
        assert result['success'] is True
        assert 'ride_options' in result
        assert 'voice_response' in result
    
    @pytest.mark.asyncio
    async def test_get_platform_recommendations(self, platform_manager, sample_location):
        """Test getting platform recommendations."""
        user_id = uuid4()
        
        result = await platform_manager.get_platform_recommendations(
            user_id=user_id,
            service_type="food_delivery",
            location=sample_location
        )
        
        assert result['success'] is True
        assert 'recommendations' in result
        assert 'voice_response' in result
        
        recommendations = result['recommendations']
        assert len(recommendations) > 0
        
        # Check recommendation structure
        recommendation = recommendations[0]
        assert 'platform' in recommendation
        assert 'recommendation_score' in recommendation
        assert 'reasons' in recommendation
        assert 'estimated_availability' in recommendation
    
    @pytest.mark.asyncio
    async def test_compare_service_prices(self, platform_manager, sample_location):
        """Test comparing service prices through platform manager."""
        # Test food delivery price comparison
        food_result = await platform_manager.compare_service_prices(
            service_type="food_delivery",
            service_details={
                'location': sample_location.dict(),
                'cuisine_type': 'indian',
                'dish_name': 'biryani'
            }
        )
        
        assert food_result['success'] is True
        assert 'comparison' in food_result
        
        # Test ride sharing price comparison
        drop_location = LocationPoint(
            latitude=12.9352,
            longitude=77.6245,
            address="Electronic City, Bangalore",
            city="Bangalore"
        )
        
        ride_result = await platform_manager.compare_service_prices(
            service_type="ride_sharing",
            service_details={
                'pickup_location': sample_location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': 'mini'
            }
        )
        
        assert ride_result['success'] is True
        assert 'comparison' in ride_result
    
    @pytest.mark.asyncio
    async def test_health_check(self, platform_manager):
        """Test platform manager health check."""
        result = await platform_manager.health_check()
        
        assert result.success is True
        assert 'status' in result.data
        assert 'services' in result.data
        assert 'food_delivery' in result.data['services']
        assert 'ride_sharing' in result.data['services']


class TestPlatformIntegrationModels:
    """Test platform integration data models."""
    
    def test_location_point_validation(self):
        """Test LocationPoint model validation."""
        # Valid location
        location = LocationPoint(
            latitude=12.9716,
            longitude=77.5946,
            address="MG Road, Bangalore",
            city="Bangalore"
        )
        assert location.latitude == 12.9716
        assert location.longitude == 77.5946
        
        # Invalid latitude
        with pytest.raises(ValueError):
            LocationPoint(
                latitude=91.0,  # Invalid latitude
                longitude=77.5946,
                address="Test Address",
                city="Test City"
            )
        
        # Invalid longitude
        with pytest.raises(ValueError):
            LocationPoint(
                latitude=12.9716,
                longitude=181.0,  # Invalid longitude
                address="Test Address",
                city="Test City"
            )
    
    def test_voice_service_command_creation(self):
        """Test VoiceServiceCommand model creation."""
        command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Book a table at restaurant",
            service_type="restaurant_booking",
            parsed_intent="book_table",
            extracted_entities={'restaurant': 'Test Restaurant'},
            confidence=0.85
        )
        
        assert command.voice_input == "Book a table at restaurant"
        assert command.service_type == "restaurant_booking"
        assert command.confidence == 0.85
        assert command.requires_clarification is False
    
    def test_booking_status_enum(self):
        """Test BookingStatus enum values."""
        assert BookingStatus.PENDING == "pending"
        assert BookingStatus.CONFIRMED == "confirmed"
        assert BookingStatus.IN_PROGRESS == "in_progress"
        assert BookingStatus.COMPLETED == "completed"
        assert BookingStatus.CANCELLED == "cancelled"
        assert BookingStatus.FAILED == "failed"
    
    def test_platform_provider_enum(self):
        """Test PlatformProvider enum values."""
        # Food delivery platforms
        assert PlatformProvider.SWIGGY == "swiggy"
        assert PlatformProvider.ZOMATO == "zomato"
        assert PlatformProvider.UBER_EATS == "uber_eats"
        
        # Ride-sharing platforms
        assert PlatformProvider.OLA == "ola"
        assert PlatformProvider.UBER == "uber"
        assert PlatformProvider.RAPIDO == "rapido"
        
        # Other platforms
        assert PlatformProvider.URBAN_COMPANY == "urban_company"
        assert PlatformProvider.JUSTDIAL == "justdial"


if __name__ == "__main__":
    pytest.main([__file__])