"""
Ride-sharing service integration for BharatVoice Assistant.

This module provides integration with ride-sharing platforms like Ola,
Uber, Rapido, and other popular Indian ride-sharing services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp

from ...core.interfaces import BaseService
from ...core.models import ServiceResult, ServiceType
from .models import (
    BookingStatus,
    LocationPoint,
    PlatformProvider,
    PriceComparison,
    RideBooking,
    RideType,
    VoiceServiceCommand
)

logger = logging.getLogger(__name__)


class RideSharingService(BaseService):
    """
    Ride-sharing service integration for booking rides through voice commands.
    
    This service handles ride booking, fare estimation, driver tracking,
    and ride management across multiple ride-sharing platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ride-sharing service with configuration.
        
        Args:
            config: Service configuration including API keys and endpoints
        """
        super().__init__(config)
        
        # Platform configurations
        self.ola_config = config.get('ola', {})
        self.uber_config = config.get('uber', {})
        self.rapido_config = config.get('rapido', {})
        
        # Service settings
        self.default_search_radius = config.get('search_radius_km', 5.0)
        self.max_wait_time = config.get('max_wait_time_minutes', 15)
        self.price_comparison_enabled = config.get('price_comparison_enabled', True)
        
        # In-memory storage for demo (replace with database in production)
        self._active_bookings: Dict[UUID, RideBooking] = {}
        self._driver_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Ride-sharing service initialized")
    
    async def process_ride_booking_command(
        self, 
        command: VoiceServiceCommand
    ) -> Dict[str, Any]:
        """
        Process a voice command for ride booking.
        
        Args:
            command: Voice service command with parsed intent and entities
            
        Returns:
            Dict containing processed booking details and next steps
        """
        try:
            logger.info(f"Processing ride booking command: {command.command_id}")
            
            # Extract booking details from voice command
            pickup_location = self._extract_pickup_location(command.extracted_entities)
            drop_location = self._extract_drop_location(command.extracted_entities)
            ride_type = self._extract_ride_type(command.extracted_entities)
            scheduled_time = self._extract_scheduled_time(command.extracted_entities)
            
            if not pickup_location:
                return {
                    'success': False,
                    'error': 'Could not determine pickup location',
                    'requires_clarification': True,
                    'clarification_questions': ['Where would you like to be picked up from?']
                }
            
            if not drop_location:
                return {
                    'success': False,
                    'error': 'Could not determine drop location',
                    'requires_clarification': True,
                    'clarification_questions': ['Where would you like to go?']
                }
            
            # Get fare estimates from different platforms
            fare_estimates = await self._get_fare_estimates(
                pickup_location=pickup_location,
                drop_location=drop_location,
                ride_type=ride_type
            )
            
            if not fare_estimates:
                return {
                    'success': False,
                    'error': 'No rides available in your area',
                    'suggestions': ['Try again in a few minutes', 'Consider different ride types']
                }
            
            # Generate ride recommendations
            recommendations = self._generate_ride_recommendations(fare_estimates, command.voice_input)
            
            return {
                'success': True,
                'ride_options': recommendations,
                'pickup_location': pickup_location.dict(),
                'drop_location': drop_location.dict(),
                'ride_type': ride_type.value if ride_type else None,
                'scheduled_time': scheduled_time.isoformat() if scheduled_time else None,
                'next_action': 'select_ride',
                'voice_response': self._generate_ride_options_response(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error processing ride booking command: {e}")
            return {
                'success': False,
                'error': f'Failed to process ride booking: {str(e)}',
                'requires_retry': True
            }
    
    async def book_ride(
        self, 
        booking_details: Dict[str, Any],
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Book a ride on the selected platform.
        
        Args:
            booking_details: Booking details including locations, ride type, etc.
            user_id: User identifier
            
        Returns:
            Dict containing booking confirmation
        """
        try:
            logger.info(f"Booking ride for user: {user_id}")
            
            # Create ride booking object
            ride_booking = RideBooking(
                user_id=user_id,
                platform=PlatformProvider(booking_details['platform']),
                ride_type=RideType(booking_details['ride_type']),
                pickup_location=LocationPoint(**booking_details['pickup_location']),
                drop_location=LocationPoint(**booking_details['drop_location']),
                estimated_fare=Decimal(str(booking_details['estimated_fare'])),
                estimated_distance=booking_details['estimated_distance'],
                estimated_duration=booking_details['estimated_duration'],
                contact_number=booking_details['contact_number'],
                payment_method=booking_details['payment_method'],
                special_requests=booking_details.get('special_requests'),
                scheduled_time=datetime.fromisoformat(booking_details['scheduled_time']) if booking_details.get('scheduled_time') else None
            )
            
            # Submit booking to platform
            platform_response = await self._submit_booking_to_platform(ride_booking)
            
            if platform_response['success']:
                ride_booking.platform_booking_id = platform_response['platform_booking_id']
                ride_booking.status = BookingStatus.CONFIRMED
                ride_booking.otp = platform_response.get('otp')
                ride_booking.driver_details = platform_response.get('driver_details')
                ride_booking.vehicle_details = platform_response.get('vehicle_details')
                
                # Store booking
                self._active_bookings[ride_booking.booking_id] = ride_booking
                
                return {
                    'success': True,
                    'booking_id': str(ride_booking.booking_id),
                    'platform_booking_id': ride_booking.platform_booking_id,
                    'otp': ride_booking.otp,
                    'driver_details': ride_booking.driver_details,
                    'vehicle_details': ride_booking.vehicle_details,
                    'estimated_arrival': platform_response.get('estimated_arrival'),
                    'voice_response': self._generate_booking_confirmation_response(ride_booking, platform_response)
                }
            else:
                return {
                    'success': False,
                    'error': platform_response['error'],
                    'voice_response': f"Sorry, I couldn't book your ride: {platform_response['error']}"
                }
                
        except Exception as e:
            logger.error(f"Error booking ride: {e}")
            return {
                'success': False,
                'error': f'Failed to book ride: {str(e)}',
                'voice_response': "There was an error booking your ride. Please try again."
            }
    
    async def track_ride(self, booking_id: UUID) -> Dict[str, Any]:
        """
        Track the status of a ride booking.
        
        Args:
            booking_id: Booking identifier
            
        Returns:
            Dict containing ride tracking information
        """
        try:
            booking = self._active_bookings.get(booking_id)
            if not booking:
                return {
                    'success': False,
                    'error': 'Booking not found',
                    'voice_response': "I couldn't find that booking. Please check the booking ID."
                }
            
            # Get latest status from platform
            tracking_info = await self._get_ride_tracking(booking)
            
            # Update booking status
            booking.status = BookingStatus(tracking_info.get('status', booking.status.value))
            booking.updated_at = datetime.utcnow()
            
            # Update driver location if available
            if tracking_info.get('driver_location'):
                if not booking.driver_details:
                    booking.driver_details = {}
                booking.driver_details['current_location'] = tracking_info['driver_location']
            
            return {
                'success': True,
                'booking_id': str(booking_id),
                'status': booking.status.value,
                'driver_details': booking.driver_details,
                'vehicle_details': booking.vehicle_details,
                'tracking_details': tracking_info,
                'voice_response': self._generate_tracking_response(booking, tracking_info)
            }
            
        except Exception as e:
            logger.error(f"Error tracking ride: {e}")
            return {
                'success': False,
                'error': f'Failed to track ride: {str(e)}',
                'voice_response': "There was an error tracking your ride. Please try again."
            }
    
    async def cancel_ride(self, booking_id: UUID, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a ride booking.
        
        Args:
            booking_id: Booking identifier
            reason: Cancellation reason
            
        Returns:
            Dict containing cancellation confirmation
        """
        try:
            booking = self._active_bookings.get(booking_id)
            if not booking:
                return {
                    'success': False,
                    'error': 'Booking not found'
                }
            
            # Cancel booking on platform
            cancellation_response = await self._cancel_booking_on_platform(booking, reason)
            
            if cancellation_response['success']:
                booking.status = BookingStatus.CANCELLED
                booking.updated_at = datetime.utcnow()
                
                return {
                    'success': True,
                    'booking_id': str(booking_id),
                    'cancellation_fee': cancellation_response.get('cancellation_fee', 0),
                    'refund_amount': cancellation_response.get('refund_amount', 0),
                    'voice_response': self._generate_cancellation_response(booking, cancellation_response)
                }
            else:
                return {
                    'success': False,
                    'error': cancellation_response['error'],
                    'voice_response': f"Could not cancel your ride: {cancellation_response['error']}"
                }
                
        except Exception as e:
            logger.error(f"Error cancelling ride: {e}")
            return {
                'success': False,
                'error': f'Failed to cancel ride: {str(e)}'
            }
    
    async def compare_ride_prices(
        self, 
        pickup_location: LocationPoint,
        drop_location: LocationPoint,
        ride_type: Optional[RideType] = None
    ) -> Dict[str, Any]:
        """
        Compare ride prices across different platforms.
        
        Args:
            pickup_location: Pickup location
            drop_location: Drop location
            ride_type: Type of ride
            
        Returns:
            Dict containing price comparison
        """
        try:
            if not self.price_comparison_enabled:
                return {
                    'success': False,
                    'error': 'Price comparison is not enabled'
                }
            
            # Get prices from different platforms
            platform_prices = []
            
            for platform in [PlatformProvider.OLA, PlatformProvider.UBER, PlatformProvider.RAPIDO]:
                try:
                    price_data = await self._get_platform_ride_prices(
                        platform, pickup_location, drop_location, ride_type
                    )
                    if price_data:
                        platform_prices.append(price_data)
                except Exception as e:
                    logger.warning(f"Failed to get prices from {platform}: {e}")
            
            if not platform_prices:
                return {
                    'success': False,
                    'error': 'Could not get price information from any platform'
                }
            
            # Create price comparison
            best_price = min(p['estimated_fare'] for p in platform_prices)
            best_platform = next(p['platform'] for p in platform_prices if p['estimated_fare'] == best_price)
            worst_price = max(p['estimated_fare'] for p in platform_prices)
            
            comparison = PriceComparison(
                service_type='ride_sharing',
                location=pickup_location,
                platforms=platform_prices,
                best_price=best_price,
                best_platform=PlatformProvider(best_platform),
                price_difference=worst_price - best_price
            )
            
            return {
                'success': True,
                'comparison': comparison.dict(),
                'voice_response': self._generate_price_comparison_response(comparison)
            }
            
        except Exception as e:
            logger.error(f"Error comparing ride prices: {e}")
            return {
                'success': False,
                'error': f'Failed to compare prices: {str(e)}'
            }
    
    def _extract_pickup_location(self, entities: Dict[str, Any]) -> Optional[LocationPoint]:
        """Extract pickup location from voice command entities."""
        if 'pickup_location' in entities:
            loc_data = entities['pickup_location']
            if isinstance(loc_data, dict):
                return LocationPoint(**loc_data)
        
        # Try current location or common pickup terms
        if 'current_location' in entities:
            return LocationPoint(**entities['current_location'])
        
        return None
    
    def _extract_drop_location(self, entities: Dict[str, Any]) -> Optional[LocationPoint]:
        """Extract drop location from voice command entities."""
        if 'drop_location' in entities:
            loc_data = entities['drop_location']
            if isinstance(loc_data, dict):
                return LocationPoint(**loc_data)
        
        # Try to construct from destination entities
        if 'destination' in entities:
            dest = entities['destination']
            if isinstance(dest, str):
                return LocationPoint(
                    latitude=0.0,  # Would be geocoded in production
                    longitude=0.0,
                    address=dest,
                    city='Unknown'
                )
        
        return None
    
    def _extract_ride_type(self, entities: Dict[str, Any]) -> Optional[RideType]:
        """Extract ride type from voice command entities."""
        ride_keywords = {
            'mini': RideType.MINI,
            'sedan': RideType.SEDAN,
            'suv': RideType.SUV,
            'auto': RideType.AUTO,
            'bike': RideType.BIKE,
            'shared': RideType.SHARED,
            'premium': RideType.PREMIUM,
            'rickshaw': RideType.AUTO
        }
        
        for key, value in entities.items():
            if isinstance(value, str):
                for keyword, ride_type in ride_keywords.items():
                    if keyword in value.lower():
                        return ride_type
        
        return RideType.MINI  # Default to mini
    
    def _extract_scheduled_time(self, entities: Dict[str, Any]) -> Optional[datetime]:
        """Extract scheduled time from voice command entities."""
        if 'scheduled_time' in entities:
            try:
                return datetime.fromisoformat(entities['scheduled_time'])
            except (ValueError, TypeError):
                pass
        
        # Try to parse relative time expressions
        if 'time' in entities:
            time_str = str(entities['time']).lower()
            if 'now' in time_str or 'immediately' in time_str:
                return datetime.utcnow()
            # Add more time parsing logic here
        
        return None
    
    async def _get_fare_estimates(
        self,
        pickup_location: LocationPoint,
        drop_location: LocationPoint,
        ride_type: Optional[RideType] = None
    ) -> List[Dict[str, Any]]:
        """Get fare estimates from different platforms."""
        estimates = []
        
        # Mock implementation - replace with actual API calls
        platforms = [PlatformProvider.OLA, PlatformProvider.UBER, PlatformProvider.RAPIDO]
        base_fares = {
            PlatformProvider.OLA: 120,
            PlatformProvider.UBER: 135,
            PlatformProvider.RAPIDO: 100
        }
        
        for platform in platforms:
            try:
                # Calculate mock distance and duration
                distance = 8.5  # Mock distance in km
                duration = 25   # Mock duration in minutes
                base_fare = base_fares.get(platform, 120)
                
                # Adjust fare based on ride type
                type_multiplier = {
                    RideType.MINI: 1.0,
                    RideType.SEDAN: 1.3,
                    RideType.SUV: 1.6,
                    RideType.AUTO: 0.7,
                    RideType.BIKE: 0.5,
                    RideType.SHARED: 0.6,
                    RideType.PREMIUM: 2.0
                }.get(ride_type or RideType.MINI, 1.0)
                
                estimated_fare = Decimal(str(base_fare * type_multiplier))
                
                estimates.append({
                    'platform': platform.value,
                    'ride_type': (ride_type or RideType.MINI).value,
                    'estimated_fare': estimated_fare,
                    'estimated_distance': distance,
                    'estimated_duration': duration,
                    'surge_multiplier': 1.0,
                    'available_drivers': 5
                })
                
            except Exception as e:
                logger.warning(f"Failed to get estimate from {platform}: {e}")
        
        return estimates
    
    def _generate_ride_recommendations(
        self, 
        fare_estimates: List[Dict[str, Any]], 
        voice_input: str
    ) -> List[Dict[str, Any]]:
        """Generate ride recommendations with scoring."""
        recommendations = []
        
        for estimate in fare_estimates:
            score = self._calculate_ride_score(estimate, voice_input)
            
            recommendation = {
                **estimate,
                'recommendation_score': score,
                'is_recommended': score > 0.7,
                'estimated_arrival': (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }
            
            recommendations.append(recommendation)
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations
    
    def _calculate_ride_score(self, estimate: Dict[str, Any], voice_input: str) -> float:
        """Calculate recommendation score for a ride option."""
        score = 0.0
        
        # Price score (lower price is better)
        max_fare = 300  # Assumed max fare for normalization
        price_score = max(0, (max_fare - float(estimate['estimated_fare'])) / max_fare)
        score += price_score * 0.4
        
        # Availability score
        available_drivers = estimate.get('available_drivers', 0)
        availability_score = min(available_drivers / 10, 1.0)
        score += availability_score * 0.3
        
        # Platform preference (could be based on user history)
        platform_scores = {
            'ola': 0.8,
            'uber': 0.9,
            'rapido': 0.7
        }
        platform_score = platform_scores.get(estimate['platform'], 0.5)
        score += platform_score * 0.3
        
        return min(score, 1.0)
    
    async def _submit_booking_to_platform(self, booking: RideBooking) -> Dict[str, Any]:
        """Submit booking to the selected platform."""
        # Mock implementation - replace with actual platform API calls
        await asyncio.sleep(2)  # Simulate API call
        
        # Mock success response
        return {
            'success': True,
            'platform_booking_id': f"{booking.platform.value.upper()}{booking.booking_id.hex[:8]}",
            'otp': '1234',
            'driver_details': {
                'name': 'Rajesh Kumar',
                'phone': '+91-9876543210',
                'rating': 4.5,
                'photo_url': 'https://example.com/driver.jpg'
            },
            'vehicle_details': {
                'make': 'Maruti',
                'model': 'Swift',
                'color': 'White',
                'number': 'KA01AB1234'
            },
            'estimated_arrival': (datetime.utcnow() + timedelta(minutes=8)).isoformat()
        }
    
    async def _get_ride_tracking(self, booking: RideBooking) -> Dict[str, Any]:
        """Get ride tracking information from platform."""
        # Mock implementation - replace with actual platform API calls
        statuses = ['confirmed', 'driver_assigned', 'driver_arriving', 'in_progress', 'completed']
        current_status_index = min(
            len(statuses) - 1,
            int((datetime.utcnow() - booking.created_at).total_seconds() / 300)  # Progress every 5 minutes
        )
        
        return {
            'status': statuses[current_status_index],
            'status_message': f'Ride is {statuses[current_status_index].replace("_", " ")}',
            'driver_location': {
                'latitude': 12.9716,
                'longitude': 77.5946,
                'address': 'MG Road, Bangalore'
            } if current_status_index >= 1 else None,
            'estimated_arrival': (datetime.utcnow() + timedelta(minutes=max(0, 8 - current_status_index * 2))).isoformat()
        }
    
    async def _cancel_booking_on_platform(
        self, 
        booking: RideBooking, 
        reason: Optional[str]
    ) -> Dict[str, Any]:
        """Cancel booking on the platform."""
        # Mock implementation - replace with actual platform API calls
        await asyncio.sleep(1)  # Simulate API call
        
        # Calculate cancellation fee based on booking status
        cancellation_fee = Decimal('0')
        if booking.status in [BookingStatus.CONFIRMED, BookingStatus.IN_PROGRESS]:
            cancellation_fee = Decimal('20')  # Mock cancellation fee
        
        refund_amount = booking.estimated_fare - cancellation_fee
        
        return {
            'success': True,
            'cancellation_fee': cancellation_fee,
            'refund_amount': refund_amount,
            'refund_timeline': '3-5 business days'
        }
    
    async def _get_platform_ride_prices(
        self,
        platform: PlatformProvider,
        pickup_location: LocationPoint,
        drop_location: LocationPoint,
        ride_type: Optional[RideType]
    ) -> Optional[Dict[str, Any]]:
        """Get price information from a specific platform."""
        # Mock implementation - replace with actual platform API calls
        base_fares = {
            PlatformProvider.OLA: 120,
            PlatformProvider.UBER: 135,
            PlatformProvider.RAPIDO: 100
        }
        
        base_fare = base_fares.get(platform, 120)
        type_multiplier = {
            RideType.MINI: 1.0,
            RideType.SEDAN: 1.3,
            RideType.SUV: 1.6,
            RideType.AUTO: 0.7,
            RideType.BIKE: 0.5,
            RideType.SHARED: 0.6,
            RideType.PREMIUM: 2.0
        }.get(ride_type or RideType.MINI, 1.0)
        
        return {
            'platform': platform.value,
            'estimated_fare': Decimal(str(base_fare * type_multiplier)),
            'surge_multiplier': 1.0,
            'estimated_duration': 25,
            'estimated_distance': 8.5
        }
    
    def _generate_ride_options_response(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate voice response for ride options."""
        if not recommendations:
            return "I couldn't find any available rides in your area."
        
        top_ride = recommendations[0]
        response = f"I found {len(recommendations)} ride options for you. "
        response += f"The best option is {top_ride['platform'].title()} {top_ride['ride_type']} "
        response += f"for ₹{top_ride['estimated_fare']} with {top_ride['estimated_duration']} minutes travel time. "
        response += "Would you like to book this ride or hear about other options?"
        
        return response
    
    def _generate_booking_confirmation_response(
        self, 
        booking: RideBooking, 
        platform_response: Dict[str, Any]
    ) -> str:
        """Generate voice response for booking confirmation."""
        response = f"Great! Your {booking.ride_type.value} ride has been booked with {booking.platform.value.title()}. "
        
        if booking.driver_details:
            driver_name = booking.driver_details.get('name', 'your driver')
            response += f"Your driver is {driver_name}. "
        
        if booking.vehicle_details:
            vehicle = booking.vehicle_details
            response += f"Vehicle details: {vehicle.get('color', '')} {vehicle.get('make', '')} {vehicle.get('model', '')} "
            response += f"with number {vehicle.get('number', '')}. "
        
        if booking.otp:
            response += f"Your OTP is {booking.otp}. "
        
        response += f"Booking ID is {booking.platform_booking_id}. "
        response += "I'll keep you updated on the driver's arrival."
        
        return response
    
    def _generate_tracking_response(self, booking: RideBooking, tracking_info: Dict[str, Any]) -> str:
        """Generate voice response for ride tracking."""
        status_message = tracking_info.get('status_message', 'Ride status updated')
        response = f"Your {booking.ride_type.value} ride with {booking.platform.value.title()} is {status_message}. "
        
        if tracking_info.get('driver_location'):
            response += "Your driver is on the way. "
        
        if tracking_info.get('estimated_arrival'):
            try:
                arrival_time = datetime.fromisoformat(tracking_info['estimated_arrival'])
                response += f"Estimated arrival time is {arrival_time.strftime('%I:%M %p')}. "
            except:
                pass
        
        return response
    
    def _generate_cancellation_response(
        self, 
        booking: RideBooking, 
        cancellation_response: Dict[str, Any]
    ) -> str:
        """Generate voice response for ride cancellation."""
        response = f"Your {booking.ride_type.value} ride has been cancelled. "
        
        cancellation_fee = cancellation_response.get('cancellation_fee', 0)
        if cancellation_fee > 0:
            response += f"A cancellation fee of ₹{cancellation_fee} will be charged. "
        
        refund_amount = cancellation_response.get('refund_amount', 0)
        if refund_amount > 0:
            response += f"₹{refund_amount} will be refunded to your account "
            timeline = cancellation_response.get('refund_timeline', 'within 3-5 business days')
            response += f"{timeline}. "
        
        return response
    
    def _generate_price_comparison_response(self, comparison: PriceComparison) -> str:
        """Generate voice response for price comparison."""
        response = f"I compared prices across {len(comparison.platforms)} platforms. "
        response += f"The best price is ₹{comparison.best_price} on {comparison.best_platform.value.title()}. "
        
        if comparison.price_difference > 0:
            response += f"You can save up to ₹{comparison.price_difference} by choosing the best option. "
        
        response += "Would you like to proceed with the best option or hear about other platforms?"
        
        return response
    
    async def health_check(self) -> ServiceResult:
        """Check ride-sharing service health."""
        try:
            return ServiceResult(
                service_type=ServiceType.RIDE_SHARING,
                success=True,
                data={'status': 'healthy', 'platforms': ['ola', 'uber', 'rapido']},
                response_time=0.1
            )
            
        except Exception as e:
            return ServiceResult(
                service_type=ServiceType.RIDE_SHARING,
                success=False,
                error_message=str(e),
                response_time=0.1
            )