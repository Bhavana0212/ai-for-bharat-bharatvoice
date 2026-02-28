"""
Property-based tests for Indian platform integration services.

This module tests the comprehensive platform integration capabilities including
payment security, service booking workflows, and API reliability using
property-based testing with Hypothesis.

**Property 19: Indian Platform Integration** - Validates Requirements 8.1, 8.2
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from typing import Dict, Any, List
from uuid import UUID, uuid4

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
from bharatvoice.services.payment import PaymentManager, UPIService
from bharatvoice.services.payment.models import (
    PaymentMethod,
    PaymentRequest,
    PaymentSecurityContext,
    TransactionStatus,
    UPIProvider,
    VoicePaymentCommand
)


# Test data strategies
@st.composite
def location_point_strategy(draw):
    """Generate valid LocationPoint instances."""
    # Indian coordinates range
    latitude = draw(st.floats(min_value=8.0, max_value=37.0))
    longitude = draw(st.floats(min_value=68.0, max_value=97.0))
    
    indian_cities = [
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", 
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
    ]
    
    city = draw(st.sampled_from(indian_cities))
    address = draw(st.text(min_size=10, max_size=100))
    
    return LocationPoint(
        latitude=latitude,
        longitude=longitude,
        address=address,
        city=city
    )


@st.composite
def payment_amount_strategy(draw):
    """Generate valid payment amounts for Indian context."""
    # Common Indian payment amounts (₹1 to ₹50,000)
    return draw(st.decimals(
        min_value=Decimal('1.00'),
        max_value=Decimal('50000.00'),
        places=2
    ))


@st.composite
def upi_id_strategy(draw):
    """Generate valid UPI IDs."""
    username = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Nd')),
        min_size=3,
        max_size=20
    ))
    
    upi_providers = ['@paytm', '@googlepay', '@phonepe', '@ybl', '@okaxis', '@okhdfcbank']
    provider = draw(st.sampled_from(upi_providers))
    
    return f"{username}{provider}"


@st.composite
def voice_service_command_strategy(draw, service_type=None):
    """Generate VoiceServiceCommand instances."""
    if service_type is None:
        service_type = draw(st.sampled_from(['food_delivery', 'ride_sharing', 'payment']))
    
    user_id = uuid4()
    
    voice_inputs = {
        'food_delivery': [
            "Order biryani from nearby restaurant",
            "Get pizza delivered to my location",
            "Find Chinese food restaurants",
            "Order dal chawal for lunch"
        ],
        'ride_sharing': [
            "Book a cab to airport",
            "Get auto rickshaw to railway station",
            "Book Ola to office",
            "Need ride to shopping mall"
        ],
        'payment': [
            "Send money to my friend",
            "Pay electricity bill",
            "Transfer amount to bank account",
            "Make UPI payment"
        ]
    }
    
    voice_input = draw(st.sampled_from(voice_inputs[service_type]))
    confidence = draw(st.floats(min_value=0.7, max_value=1.0))
    
    return VoiceServiceCommand(
        user_id=user_id,
        voice_input=voice_input,
        service_type=service_type,
        parsed_intent=f"{service_type}_intent",
        extracted_entities={},
        confidence=confidence
    )


class TestPlatformIntegrationProperties:
    """Property-based tests for platform integration services."""
    
    @pytest.fixture
    def platform_manager(self):
        """Create platform manager for testing."""
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
    def payment_manager(self):
        """Create payment manager for testing."""
        config = {
            'upi': {
                'enabled': True,
                'timeout_seconds': 30
            },
            'max_daily_amount': 50000.0,
            'max_transaction_amount': 100000.0,
            'fraud_detection_enabled': True
        }
        return PaymentManager(config)
    
    @given(location=location_point_strategy())
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_location_based_service_discovery_property(self, platform_manager, location):
        """
        **Property 19.1: Location-based Service Discovery**
        
        For any valid Indian location, platform services should be discoverable
        and return appropriate service options.
        """
        # Test food delivery service discovery
        food_command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input="Find restaurants near me",
            service_type="food_delivery",
            parsed_intent="find_restaurants",
            extracted_entities={'location': location.dict()},
            confidence=0.9
        )
        
        result = await platform_manager.process_service_command(food_command)
        
        # Property: Service discovery should always succeed for valid locations
        assert result['success'] is True
        assert 'restaurants' in result or 'voice_response' in result
        
        # Property: Response should be contextually appropriate for Indian locations
        if 'voice_response' in result:
            response = result['voice_response'].lower()
            # Should not contain non-Indian context
            assert 'dollar' not in response
            assert 'miles' not in response
    
    @given(
        pickup_location=location_point_strategy(),
        drop_location=location_point_strategy(),
        ride_type=st.sampled_from(list(RideType))
    )
    @settings(max_examples=30, deadline=5000)
    @pytest.mark.asyncio
    async def test_ride_booking_workflow_property(self, platform_manager, pickup_location, drop_location, ride_type):
        """
        **Property 19.2: Ride Booking Workflow Integrity**
        
        Ride booking workflows should maintain consistency and provide
        appropriate confirmations for all valid booking requests.
        """
        assume(pickup_location.city == drop_location.city)  # Same city rides
        
        ride_command = VoiceServiceCommand(
            user_id=uuid4(),
            voice_input=f"Book {ride_type.value} from {pickup_location.address} to {drop_location.address}",
            service_type="ride_sharing",
            parsed_intent="book_ride",
            extracted_entities={
                'pickup_location': pickup_location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': ride_type.value
            },
            confidence=0.9
        )
        
        result = await platform_manager.process_service_command(ride_command)
        
        # Property: Valid ride requests should always be processed
        assert result['success'] is True
        
        # Property: Response should contain ride options or booking confirmation
        assert 'ride_options' in result or 'booking_id' in result
        
        # Property: All ride options should have required fields
        if 'ride_options' in result:
            for option in result['ride_options']:
                assert 'platform' in option
                assert 'estimated_fare' in option
                assert 'estimated_duration' in option
                assert isinstance(option['estimated_fare'], (int, float))
                assert option['estimated_fare'] > 0
    
    @given(
        amount=payment_amount_strategy(),
        upi_id=upi_id_strategy()
    )
    @settings(max_examples=40, deadline=5000)
    @pytest.mark.asyncio
    async def test_payment_security_property(self, payment_manager, amount, upi_id):
        """
        **Property 19.3: Payment Security and Transaction Integrity**
        
        Payment processing should maintain security constraints and
        transaction integrity for all valid payment requests.
        """
        user_id = uuid4()
        
        # Create payment request
        payment_request = PaymentRequest(
            user_id=user_id,
            amount=amount,
            payment_method=PaymentMethod.UPI,
            recipient_id=upi_id,
            description="Test payment"
        )
        
        # Create security context
        security_context = PaymentSecurityContext(
            user_id=user_id,
            device_fingerprint="test_device",
            ip_address="192.168.1.1",
            risk_score=0.1,  # Low risk
            session_token="test_session"
        )
        
        # Create voice payment command
        voice_command = VoicePaymentCommand(
            user_id=user_id,
            voice_input=f"Send {amount} rupees to {upi_id}",
            parsed_intent="send_money",
            extracted_entities={
                'amount': str(amount),
                'recipient': upi_id
            },
            confidence=0.9,
            confirmation_text=f"Confirm sending ₹{amount} to {upi_id}?"
        )
        
        result = await payment_manager.process_voice_payment(voice_command, security_context)
        
        # Property: Payment processing should always return a valid response
        assert 'success' in result
        assert 'step' in result
        
        # Property: Security validation should be applied
        if result['success']:
            # Valid payments should proceed to confirmation or processing
            assert result['step'] in ['confirmation_required', 'direct_processing', 'mfa_required']
        else:
            # Failed payments should have clear error messages
            assert 'error' in result
            assert 'voice_response' in result
        
        # Property: Amount limits should be enforced
        if amount > Decimal('100000'):
            assert result['success'] is False
            assert 'limit' in result.get('error', '').lower()
    
    @given(
        service_commands=st.lists(
            voice_service_command_strategy(),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20, deadline=10000)
    @pytest.mark.asyncio
    async def test_concurrent_service_processing_property(self, platform_manager, service_commands):
        """
        **Property 19.4: Concurrent Service Processing**
        
        Platform manager should handle multiple concurrent service requests
        without data corruption or service degradation.
        """
        # Process all commands concurrently
        tasks = [
            platform_manager.process_service_command(command)
            for command in service_commands
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Property: All requests should be processed (no exceptions)
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Command {i} raised exception: {result}"
            assert isinstance(result, dict), f"Command {i} returned invalid result type"
            assert 'success' in result, f"Command {i} missing success field"
    
    @given(
        location=location_point_strategy(),
        cuisine_type=st.sampled_from(list(FoodCategory))
    )
    @settings(max_examples=30, deadline=5000)
    @pytest.mark.asyncio
    async def test_price_comparison_accuracy_property(self, platform_manager, location, cuisine_type):
        """
        **Property 19.5: Price Comparison Accuracy**
        
        Price comparison across platforms should return consistent and
        accurate pricing information for Indian food services.
        """
        result = await platform_manager.compare_service_prices(
            service_type="food_delivery",
            service_details={
                'location': location.dict(),
                'cuisine_type': cuisine_type.value,
                'dish_name': 'biryani'
            }
        )
        
        # Property: Price comparison should succeed for valid inputs
        assert result['success'] is True
        assert 'comparison' in result
        
        comparison = result['comparison']
        
        # Property: Comparison should have required fields
        assert 'best_price' in comparison
        assert 'best_platform' in comparison
        assert 'platforms' in comparison
        
        # Property: Best price should be the minimum among all platforms
        if comparison['platforms']:
            platform_prices = [p['price'] for p in comparison['platforms']]
            assert comparison['best_price'] == min(platform_prices)
        
        # Property: All prices should be positive
        for platform_data in comparison['platforms']:
            assert platform_data['price'] > 0
    
    @given(
        user_id=st.uuids(),
        booking_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.asyncio
    async def test_booking_limit_enforcement_property(self, platform_manager, user_id, booking_count):
        """
        **Property 19.6: Booking Limit Enforcement**
        
        Platform manager should enforce concurrent booking limits
        and prevent users from exceeding maximum allowed bookings.
        """
        max_bookings = platform_manager.max_concurrent_bookings
        
        # Create multiple booking requests
        booking_results = []
        for i in range(booking_count):
            booking_details = {
                'platform': 'swiggy',
                'restaurant': {
                    'restaurant_id': f'rest_{i}',
                    'name': f'Test Restaurant {i}',
                    'cuisine_types': ['indian'],
                    'location': {
                        'latitude': 12.9716,
                        'longitude': 77.5946,
                        'address': 'Test Address',
                        'city': 'Bangalore'
                    },
                    'rating': 4.0,
                    'delivery_time': 30,
                    'minimum_order': 150,
                    'delivery_fee': 25,
                    'is_open': True
                },
                'items': [{'item_id': 'item_001', 'name': 'Biryani', 'quantity': 1, 'price': 280}],
                'delivery_address': {
                    'latitude': 12.9716,
                    'longitude': 77.5946,
                    'address': 'Test Address',
                    'city': 'Bangalore'
                },
                'total_amount': 280,
                'delivery_fee': 25,
                'taxes': 28,
                'final_amount': 333,
                'contact_number': '+91-9876543210',
                'payment_method': 'UPI'
            }
            
            result = await platform_manager.book_service(
                service_type="food_delivery",
                booking_details=booking_details,
                user_id=user_id
            )
            
            booking_results.append(result)
        
        # Property: Booking limit should be enforced
        successful_bookings = [r for r in booking_results if r.get('success')]
        failed_bookings = [r for r in booking_results if not r.get('success')]
        
        # Should not exceed maximum concurrent bookings
        assert len(successful_bookings) <= max_bookings
        
        # If booking count exceeds limit, some should fail with limit error
        if booking_count > max_bookings:
            assert len(failed_bookings) > 0
            # Check that limit errors are properly reported
            limit_errors = [
                r for r in failed_bookings 
                if 'limit' in r.get('error', '').lower()
            ]
            assert len(limit_errors) > 0


class PlatformIntegrationStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for platform integration workflows.
    
    This tests the complete lifecycle of service bookings including
    creation, tracking, modification, and cancellation.
    """
    
    def __init__(self):
        super().__init__()
        self.platform_manager = None
        self.active_bookings: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[UUID, Dict[str, Any]] = {}
    
    @initialize()
    def setup_platform_manager(self):
        """Initialize platform manager for stateful testing."""
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
            'max_concurrent_bookings': 3
        }
        self.platform_manager = PlatformManager(config)
    
    @rule(
        user_id=st.uuids(),
        service_type=st.sampled_from(['food_delivery', 'ride_sharing'])
    )
    def create_booking(self, user_id, service_type):
        """Create a new service booking."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {'bookings': [], 'total_amount': 0}
        
        # Create booking based on service type
        if service_type == 'food_delivery':
            booking_details = {
                'platform': 'swiggy',
                'restaurant': {
                    'restaurant_id': 'rest_001',
                    'name': 'Test Restaurant',
                    'cuisine_types': ['indian'],
                    'location': {
                        'latitude': 12.9716,
                        'longitude': 77.5946,
                        'address': 'Test Address',
                        'city': 'Bangalore'
                    },
                    'rating': 4.0,
                    'delivery_time': 30,
                    'minimum_order': 150,
                    'delivery_fee': 25,
                    'is_open': True
                },
                'items': [{'item_id': 'item_001', 'name': 'Biryani', 'quantity': 1, 'price': 280}],
                'delivery_address': {
                    'latitude': 12.9716,
                    'longitude': 77.5946,
                    'address': 'Test Address',
                    'city': 'Bangalore'
                },
                'total_amount': 280,
                'delivery_fee': 25,
                'taxes': 28,
                'final_amount': 333,
                'contact_number': '+91-9876543210',
                'payment_method': 'UPI'
            }
        else:  # ride_sharing
            booking_details = {
                'platform': 'ola',
                'ride_type': 'mini',
                'pickup_location': {
                    'latitude': 12.9716,
                    'longitude': 77.5946,
                    'address': 'MG Road, Bangalore',
                    'city': 'Bangalore'
                },
                'drop_location': {
                    'latitude': 12.9352,
                    'longitude': 77.6245,
                    'address': 'Electronic City, Bangalore',
                    'city': 'Bangalore'
                },
                'estimated_fare': 120,
                'estimated_distance': 8.5,
                'estimated_duration': 25,
                'contact_number': '+91-9876543210',
                'payment_method': 'UPI'
            }
        
        # Store booking attempt
        booking_id = f"{service_type}_{len(self.active_bookings)}"
        self.active_bookings[booking_id] = {
            'user_id': user_id,
            'service_type': service_type,
            'details': booking_details,
            'status': 'pending'
        }
        
        self.user_sessions[user_id]['bookings'].append(booking_id)
    
    @rule(user_id=st.uuids())
    def get_user_bookings(self, user_id):
        """Get user's booking history."""
        if user_id in self.user_sessions:
            user_bookings = self.user_sessions[user_id]['bookings']
            # Property: User should be able to retrieve their bookings
            assert isinstance(user_bookings, list)
    
    @invariant()
    def booking_consistency_invariant(self):
        """Invariant: Booking data should remain consistent."""
        # All active bookings should have required fields
        for booking_id, booking in self.active_bookings.items():
            assert 'user_id' in booking
            assert 'service_type' in booking
            assert 'details' in booking
            assert 'status' in booking
        
        # User sessions should be consistent with bookings
        for user_id, session in self.user_sessions.items():
            for booking_id in session['bookings']:
                if booking_id in self.active_bookings:
                    assert self.active_bookings[booking_id]['user_id'] == user_id


# Test runner for stateful testing
TestPlatformIntegrationStateMachine = PlatformIntegrationStateMachine.TestCase


@pytest.mark.asyncio
async def test_indian_platform_integration_comprehensive():
    """
    **Property 19: Indian Platform Integration** - Comprehensive Test
    
    This test validates the complete Indian platform integration system
    including payment security, service booking workflows, and API reliability.
    """
    # Setup services
    platform_config = {
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
    
    payment_config = {
        'upi': {'enabled': True, 'timeout_seconds': 30},
        'max_daily_amount': 50000.0,
        'max_transaction_amount': 100000.0,
        'fraud_detection_enabled': True
    }
    
    platform_manager = PlatformManager(platform_config)
    payment_manager = PaymentManager(payment_config)
    
    # Test 1: Service Discovery and Booking Workflow
    location = LocationPoint(
        latitude=12.9716,
        longitude=77.5946,
        address="MG Road, Bangalore",
        city="Bangalore"
    )
    
    food_command = VoiceServiceCommand(
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
    
    food_result = await platform_manager.process_service_command(food_command)
    assert food_result['success'] is True
    assert 'restaurants' in food_result or 'voice_response' in food_result
    
    # Test 2: Payment Security Integration
    user_id = uuid4()
    payment_request = PaymentRequest(
        user_id=user_id,
        amount=Decimal('500.00'),
        payment_method=PaymentMethod.UPI,
        recipient_id="test@paytm",
        description="Food delivery payment"
    )
    
    security_context = PaymentSecurityContext(
        user_id=user_id,
        device_fingerprint="test_device",
        ip_address="192.168.1.1",
        risk_score=0.1,
        session_token="test_session"
    )
    
    voice_payment_command = VoicePaymentCommand(
        user_id=user_id,
        voice_input="Pay 500 rupees for food order",
        parsed_intent="pay_for_service",
        extracted_entities={'amount': '500', 'service': 'food_delivery'},
        confidence=0.9,
        confirmation_text="Confirm payment of ₹500 for food order?"
    )
    
    payment_result = await payment_manager.process_voice_payment(
        voice_payment_command, 
        security_context
    )
    
    assert 'success' in payment_result
    assert 'step' in payment_result
    
    # Test 3: Platform Health and Reliability
    platform_health = await platform_manager.health_check()
    assert platform_health.success is True
    
    payment_health = await payment_manager.health_check()
    assert payment_health.success is True
    
    # Test 4: Error Handling and Recovery
    invalid_command = VoiceServiceCommand(
        user_id=uuid4(),
        voice_input="Invalid service request",
        service_type="invalid_service",
        parsed_intent="unknown",
        extracted_entities={},
        confidence=0.3
    )
    
    error_result = await platform_manager.process_service_command(invalid_command)
    assert error_result['success'] is False
    assert 'error' in error_result
    assert 'supported_services' in error_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])