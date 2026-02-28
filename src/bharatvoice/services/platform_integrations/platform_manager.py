"""
Platform manager for coordinating service platform integrations.

This module provides a unified interface for managing multiple service platforms
including food delivery, ride-sharing, and other booking services.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...core.interfaces import BaseService
from ...core.models import ServiceResult, ServiceType
from .food_delivery_service import FoodDeliveryService
from .ride_sharing_service import RideSharingService
from .models import (
    BookingConfirmation,
    BookingStatus,
    LocationPoint,
    PlatformProvider,
    PriceComparison,
    ServiceBooking,
    VoiceServiceCommand
)

logger = logging.getLogger(__name__)


class PlatformManager(BaseService):
    """
    Unified platform manager for coordinating service integrations.
    
    This service provides a single interface for managing bookings across
    multiple platforms and service types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize platform manager with service configurations.
        
        Args:
            config: Configuration for all platform services
        """
        super().__init__(config)
        
        # Initialize individual services
        self.food_delivery_service = FoodDeliveryService(config.get('food_delivery', {}))
        self.ride_sharing_service = RideSharingService(config.get('ride_sharing', {}))
        
        # Service settings
        self.price_comparison_enabled = config.get('price_comparison_enabled', True)
        self.booking_timeout_minutes = config.get('booking_timeout_minutes', 30)
        self.max_concurrent_bookings = config.get('max_concurrent_bookings', 5)
        
        # In-memory storage for demo (replace with database in production)
        self._active_bookings: Dict[UUID, ServiceBooking] = {}
        self._booking_confirmations: Dict[UUID, BookingConfirmation] = {}
        self._user_preferences: Dict[UUID, Dict[str, Any]] = {}
        
        logger.info("Platform manager initialized")
    
    async def process_service_command(
        self, 
        command: VoiceServiceCommand
    ) -> Dict[str, Any]:
        """
        Process a voice command for any service platform.
        
        Args:
            command: Voice service command with parsed intent and entities
            
        Returns:
            Dict containing processed service details and next steps
        """
        try:
            logger.info(f"Processing service command: {command.service_type}")
            
            # Route to appropriate service based on service type
            if command.service_type == 'food_delivery':
                return await self.food_delivery_service.process_food_order_command(command)
            elif command.service_type == 'ride_sharing':
                return await self.ride_sharing_service.process_ride_booking_command(command)
            elif command.service_type == 'general_booking':
                return await self._process_general_booking_command(command)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported service type: {command.service_type}',
                    'supported_services': ['food_delivery', 'ride_sharing', 'general_booking']
                }
                
        except Exception as e:
            logger.error(f"Error processing service command: {e}")
            return {
                'success': False,
                'error': f'Failed to process service command: {str(e)}',
                'requires_retry': True
            }
    
    async def book_service(
        self, 
        service_type: str,
        booking_details: Dict[str, Any],
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Book a service on the appropriate platform.
        
        Args:
            service_type: Type of service to book
            booking_details: Service-specific booking details
            user_id: User identifier
            
        Returns:
            Dict containing booking confirmation
        """
        try:
            logger.info(f"Booking {service_type} service for user: {user_id}")
            
            # Check user's concurrent booking limit
            user_active_bookings = [
                b for b in self._active_bookings.values() 
                if b.user_id == user_id and b.status in [BookingStatus.PENDING, BookingStatus.CONFIRMED, BookingStatus.IN_PROGRESS]
            ]
            
            if len(user_active_bookings) >= self.max_concurrent_bookings:
                return {
                    'success': False,
                    'error': f'Maximum concurrent bookings limit ({self.max_concurrent_bookings}) reached',
                    'voice_response': 'You have reached the maximum number of active bookings. Please complete or cancel existing bookings first.'
                }
            
            # Route to appropriate service
            if service_type == 'food_delivery':
                result = await self.food_delivery_service.place_food_order(booking_details, user_id)
            elif service_type == 'ride_sharing':
                result = await self.ride_sharing_service.book_ride(booking_details, user_id)
            else:
                result = await self._book_general_service(service_type, booking_details, user_id)
            
            # Store booking confirmation if successful
            if result.get('success'):
                confirmation = BookingConfirmation(
                    booking_id=UUID(result['booking_id']) if isinstance(result['booking_id'], str) else result['booking_id'],
                    platform=PlatformProvider(booking_details['platform']),
                    confirmation_code=result.get('platform_booking_id', result.get('booking_id')),
                    estimated_time=datetime.fromisoformat(result['estimated_time']) if result.get('estimated_time') else None,
                    contact_details=self._extract_contact_details(result),
                    cancellation_policy=self._get_cancellation_policy(booking_details['platform']),
                    tracking_url=result.get('tracking_url'),
                    qr_code=result.get('qr_code')
                )
                
                self._booking_confirmations[confirmation.booking_id] = confirmation
                
                # Update user preferences based on successful booking
                await self._update_user_preferences(user_id, service_type, booking_details)
            
            return result
            
        except Exception as e:
            logger.error(f"Error booking service: {e}")
            return {
                'success': False,
                'error': f'Failed to book service: {str(e)}',
                'voice_response': "There was an error booking your service. Please try again."
            }
    
    async def track_booking(self, booking_id: UUID, service_type: str) -> Dict[str, Any]:
        """
        Track the status of any service booking.
        
        Args:
            booking_id: Booking identifier
            service_type: Type of service
            
        Returns:
            Dict containing booking tracking information
        """
        try:
            # Route to appropriate service
            if service_type == 'food_delivery':
                return await self.food_delivery_service.track_food_order(booking_id)
            elif service_type == 'ride_sharing':
                return await self.ride_sharing_service.track_ride(booking_id)
            else:
                return await self._track_general_booking(booking_id)
                
        except Exception as e:
            logger.error(f"Error tracking booking: {e}")
            return {
                'success': False,
                'error': f'Failed to track booking: {str(e)}',
                'voice_response': "There was an error tracking your booking. Please try again."
            }
    
    async def cancel_booking(
        self, 
        booking_id: UUID, 
        service_type: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel any service booking.
        
        Args:
            booking_id: Booking identifier
            service_type: Type of service
            reason: Cancellation reason
            
        Returns:
            Dict containing cancellation confirmation
        """
        try:
            # Route to appropriate service
            if service_type == 'ride_sharing':
                result = await self.ride_sharing_service.cancel_ride(booking_id, reason)
            else:
                result = await self._cancel_general_booking(booking_id, reason)
            
            # Update booking confirmation if successful
            if result.get('success') and booking_id in self._booking_confirmations:
                del self._booking_confirmations[booking_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Error cancelling booking: {e}")
            return {
                'success': False,
                'error': f'Failed to cancel booking: {str(e)}'
            }
    
    async def compare_service_prices(
        self, 
        service_type: str,
        service_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare prices across platforms for a service.
        
        Args:
            service_type: Type of service
            service_details: Service-specific details for comparison
            
        Returns:
            Dict containing price comparison
        """
        try:
            if not self.price_comparison_enabled:
                return {
                    'success': False,
                    'error': 'Price comparison is not enabled'
                }
            
            # Route to appropriate service
            if service_type == 'food_delivery':
                location = LocationPoint(**service_details['location'])
                return await self.food_delivery_service.compare_food_prices(
                    location=location,
                    cuisine_type=service_details.get('cuisine_type'),
                    dish_name=service_details.get('dish_name')
                )
            elif service_type == 'ride_sharing':
                pickup_location = LocationPoint(**service_details['pickup_location'])
                drop_location = LocationPoint(**service_details['drop_location'])
                return await self.ride_sharing_service.compare_ride_prices(
                    pickup_location=pickup_location,
                    drop_location=drop_location,
                    ride_type=service_details.get('ride_type')
                )
            else:
                return await self._compare_general_service_prices(service_type, service_details)
                
        except Exception as e:
            logger.error(f"Error comparing service prices: {e}")
            return {
                'success': False,
                'error': f'Failed to compare prices: {str(e)}'
            }
    
    async def get_user_bookings(
        self, 
        user_id: UUID,
        status_filter: Optional[BookingStatus] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get user's booking history.
        
        Args:
            user_id: User identifier
            status_filter: Filter by booking status
            limit: Maximum number of bookings to return
            
        Returns:
            Dict containing user's bookings
        """
        try:
            user_bookings = [
                booking for booking in self._active_bookings.values()
                if booking.user_id == user_id
            ]
            
            if status_filter:
                user_bookings = [b for b in user_bookings if b.status == status_filter]
            
            # Sort by creation time (most recent first)
            user_bookings.sort(key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            user_bookings = user_bookings[:limit]
            
            return {
                'success': True,
                'bookings': [booking.dict() for booking in user_bookings],
                'total_count': len(user_bookings),
                'voice_response': self._generate_booking_history_response(user_bookings)
            }
            
        except Exception as e:
            logger.error(f"Error getting user bookings: {e}")
            return {
                'success': False,
                'error': f'Failed to get bookings: {str(e)}'
            }
    
    async def get_platform_recommendations(
        self, 
        user_id: UUID,
        service_type: str,
        location: LocationPoint
    ) -> Dict[str, Any]:
        """
        Get personalized platform recommendations for a user.
        
        Args:
            user_id: User identifier
            service_type: Type of service
            location: User's location
            
        Returns:
            Dict containing platform recommendations
        """
        try:
            user_prefs = self._user_preferences.get(user_id, {})
            service_prefs = user_prefs.get(service_type, {})
            
            # Get available platforms for the service type
            available_platforms = self._get_available_platforms(service_type, location)
            
            # Score platforms based on user preferences and historical data
            recommendations = []
            for platform in available_platforms:
                score = self._calculate_platform_score(platform, service_prefs, location)
                
                recommendations.append({
                    'platform': platform.value,
                    'recommendation_score': score,
                    'reasons': self._get_recommendation_reasons(platform, service_prefs),
                    'estimated_availability': self._get_platform_availability(platform, location)
                })
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return {
                'success': True,
                'recommendations': recommendations,
                'voice_response': self._generate_recommendations_response(recommendations, service_type)
            }
            
        except Exception as e:
            logger.error(f"Error getting platform recommendations: {e}")
            return {
                'success': False,
                'error': f'Failed to get recommendations: {str(e)}'
            }
    
    async def _process_general_booking_command(
        self, 
        command: VoiceServiceCommand
    ) -> Dict[str, Any]:
        """Process a general service booking command."""
        # Mock implementation for other services
        return {
            'success': True,
            'service_type': 'general_booking',
            'available_services': ['home_services', 'entertainment', 'healthcare'],
            'voice_response': 'I can help you book various services. What specific service are you looking for?'
        }
    
    async def _book_general_service(
        self, 
        service_type: str,
        booking_details: Dict[str, Any],
        user_id: UUID
    ) -> Dict[str, Any]:
        """Book a general service."""
        # Mock implementation for other services
        booking = ServiceBooking(
            user_id=user_id,
            platform=PlatformProvider(booking_details['platform']),
            service_type=service_type,
            service_details=booking_details,
            location=LocationPoint(**booking_details['location']),
            scheduled_time=datetime.fromisoformat(booking_details['scheduled_time']),
            estimated_cost=booking_details['estimated_cost'],
            contact_number=booking_details['contact_number'],
            payment_method=booking_details['payment_method']
        )
        
        self._active_bookings[booking.booking_id] = booking
        
        return {
            'success': True,
            'booking_id': str(booking.booking_id),
            'platform_booking_id': f"GEN{booking.booking_id.hex[:8]}",
            'estimated_time': booking.scheduled_time.isoformat(),
            'voice_response': f"Your {service_type} service has been booked successfully."
        }
    
    async def _track_general_booking(self, booking_id: UUID) -> Dict[str, Any]:
        """Track a general service booking."""
        booking = self._active_bookings.get(booking_id)
        if not booking:
            return {
                'success': False,
                'error': 'Booking not found'
            }
        
        return {
            'success': True,
            'booking_id': str(booking_id),
            'status': booking.status.value,
            'scheduled_time': booking.scheduled_time.isoformat(),
            'voice_response': f"Your {booking.service_type} service is {booking.status.value}."
        }
    
    async def _cancel_general_booking(
        self, 
        booking_id: UUID, 
        reason: Optional[str]
    ) -> Dict[str, Any]:
        """Cancel a general service booking."""
        booking = self._active_bookings.get(booking_id)
        if not booking:
            return {
                'success': False,
                'error': 'Booking not found'
            }
        
        booking.status = BookingStatus.CANCELLED
        booking.updated_at = datetime.utcnow()
        
        return {
            'success': True,
            'booking_id': str(booking_id),
            'cancellation_fee': 0,
            'refund_amount': float(booking.estimated_cost),
            'voice_response': f"Your {booking.service_type} service has been cancelled."
        }
    
    async def _compare_general_service_prices(
        self, 
        service_type: str,
        service_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare prices for general services."""
        # Mock implementation
        return {
            'success': True,
            'comparison': {
                'service_type': service_type,
                'platforms': [
                    {'platform': 'urban_company', 'price': 500},
                    {'platform': 'justdial', 'price': 450}
                ],
                'best_price': 450,
                'best_platform': 'justdial'
            },
            'voice_response': f"I compared prices for {service_type}. The best price is ₹450 on JustDial."
        }
    
    def _extract_contact_details(self, booking_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract contact details from booking result."""
        contact_details = {}
        
        if 'driver_details' in booking_result:
            driver = booking_result['driver_details']
            contact_details['driver_name'] = driver.get('name', '')
            contact_details['driver_phone'] = driver.get('phone', '')
        
        if 'restaurant' in booking_result:
            restaurant = booking_result['restaurant']
            contact_details['restaurant_name'] = restaurant.get('name', '')
            contact_details['restaurant_phone'] = restaurant.get('phone', '')
        
        return contact_details
    
    def _get_cancellation_policy(self, platform: str) -> str:
        """Get cancellation policy for a platform."""
        policies = {
            'ola': 'Free cancellation within 5 minutes, ₹20 fee after that',
            'uber': 'Free cancellation within 2 minutes, ₹25 fee after that',
            'swiggy': 'Free cancellation before restaurant confirms order',
            'zomato': 'Cancellation charges may apply based on order status'
        }
        
        return policies.get(platform, 'Cancellation charges may apply')
    
    async def _update_user_preferences(
        self, 
        user_id: UUID,
        service_type: str,
        booking_details: Dict[str, Any]
    ) -> None:
        """Update user preferences based on successful booking."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
        
        if service_type not in self._user_preferences[user_id]:
            self._user_preferences[user_id][service_type] = {}
        
        prefs = self._user_preferences[user_id][service_type]
        
        # Update platform preference
        platform = booking_details.get('platform')
        if platform:
            prefs['preferred_platform'] = platform
        
        # Update other preferences based on service type
        if service_type == 'ride_sharing':
            ride_type = booking_details.get('ride_type')
            if ride_type:
                prefs['preferred_ride_type'] = ride_type
        elif service_type == 'food_delivery':
            cuisine_type = booking_details.get('cuisine_type')
            if cuisine_type:
                prefs['preferred_cuisine'] = cuisine_type
    
    def _get_available_platforms(
        self, 
        service_type: str,
        location: LocationPoint
    ) -> List[PlatformProvider]:
        """Get available platforms for a service type and location."""
        if service_type == 'food_delivery':
            return [PlatformProvider.SWIGGY, PlatformProvider.ZOMATO, PlatformProvider.UBER_EATS]
        elif service_type == 'ride_sharing':
            return [PlatformProvider.OLA, PlatformProvider.UBER, PlatformProvider.RAPIDO]
        else:
            return [PlatformProvider.URBAN_COMPANY, PlatformProvider.JUSTDIAL]
    
    def _calculate_platform_score(
        self, 
        platform: PlatformProvider,
        user_prefs: Dict[str, Any],
        location: LocationPoint
    ) -> float:
        """Calculate recommendation score for a platform."""
        score = 0.5  # Base score
        
        # User preference score
        if user_prefs.get('preferred_platform') == platform.value:
            score += 0.3
        
        # Platform reliability score (mock data)
        reliability_scores = {
            PlatformProvider.SWIGGY: 0.85,
            PlatformProvider.ZOMATO: 0.80,
            PlatformProvider.OLA: 0.82,
            PlatformProvider.UBER: 0.88,
            PlatformProvider.RAPIDO: 0.75
        }
        
        score += reliability_scores.get(platform, 0.7) * 0.2
        
        return min(score, 1.0)
    
    def _get_recommendation_reasons(
        self, 
        platform: PlatformProvider,
        user_prefs: Dict[str, Any]
    ) -> List[str]:
        """Get reasons for platform recommendation."""
        reasons = []
        
        if user_prefs.get('preferred_platform') == platform.value:
            reasons.append('Your preferred platform')
        
        # Add platform-specific reasons
        platform_reasons = {
            PlatformProvider.SWIGGY: ['Fast delivery', 'Wide restaurant selection'],
            PlatformProvider.ZOMATO: ['Good ratings', 'Premium restaurants'],
            PlatformProvider.OLA: ['Reliable service', 'Local drivers'],
            PlatformProvider.UBER: ['Global platform', 'Premium vehicles'],
            PlatformProvider.RAPIDO: ['Budget-friendly', 'Quick rides']
        }
        
        reasons.extend(platform_reasons.get(platform, ['Good service']))
        
        return reasons
    
    def _get_platform_availability(
        self, 
        platform: PlatformProvider,
        location: LocationPoint
    ) -> str:
        """Get platform availability status."""
        # Mock implementation
        return 'Available'
    
    def _generate_booking_history_response(self, bookings: List[ServiceBooking]) -> str:
        """Generate voice response for booking history."""
        if not bookings:
            return "You don't have any recent bookings."
        
        response = f"You have {len(bookings)} recent bookings. "
        
        latest_booking = bookings[0]
        response += f"Your most recent booking was for {latest_booking.service_type} "
        response += f"on {latest_booking.platform.value.title()} "
        response += f"with status {latest_booking.status.value}. "
        
        if len(bookings) > 1:
            response += "Would you like to hear about your other bookings?"
        
        return response
    
    def _generate_recommendations_response(
        self, 
        recommendations: List[Dict[str, Any]],
        service_type: str
    ) -> str:
        """Generate voice response for platform recommendations."""
        if not recommendations:
            return f"I couldn't find any platforms for {service_type} in your area."
        
        top_recommendation = recommendations[0]
        response = f"For {service_type}, I recommend {top_recommendation['platform'].title()}. "
        
        reasons = top_recommendation.get('reasons', [])
        if reasons:
            response += f"Reasons: {', '.join(reasons[:2])}. "
        
        if len(recommendations) > 1:
            response += f"I also found {len(recommendations) - 1} other options. "
        
        response += "Would you like to proceed with this recommendation?"
        
        return response
    
    async def health_check(self) -> ServiceResult:
        """Check platform manager health."""
        try:
            # Check individual services
            food_health = await self.food_delivery_service.health_check()
            ride_health = await self.ride_sharing_service.health_check()
            
            all_healthy = food_health.success and ride_health.success
            
            return ServiceResult(
                service_type=ServiceType.PLATFORM_INTEGRATION,
                success=all_healthy,
                data={
                    'status': 'healthy' if all_healthy else 'degraded',
                    'services': {
                        'food_delivery': food_health.success,
                        'ride_sharing': ride_health.success
                    }
                },
                response_time=0.1
            )
            
        except Exception as e:
            return ServiceResult(
                service_type=ServiceType.PLATFORM_INTEGRATION,
                success=False,
                error_message=str(e),
                response_time=0.1
            )