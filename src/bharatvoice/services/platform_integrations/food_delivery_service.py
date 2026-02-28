"""
Food delivery service integration for BharatVoice Assistant.

This module provides integration with food delivery platforms like Swiggy,
Zomato, and other popular Indian food delivery services.
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
    BookingConfirmation,
    BookingStatus,
    FoodCategory,
    FoodOrder,
    LocationPoint,
    MenuItem,
    PlatformProvider,
    PriceComparison,
    Restaurant,
    VoiceServiceCommand
)

logger = logging.getLogger(__name__)


class FoodDeliveryService(BaseService):
    """
    Food delivery service integration for ordering food through voice commands.
    
    This service handles restaurant discovery, menu browsing, order placement,
    and order tracking across multiple food delivery platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize food delivery service with configuration.
        
        Args:
            config: Service configuration including API keys and endpoints
        """
        super().__init__(config)
        
        # Platform configurations
        self.swiggy_config = config.get('swiggy', {})
        self.zomato_config = config.get('zomato', {})
        self.uber_eats_config = config.get('uber_eats', {})
        
        # Service settings
        self.default_delivery_radius = config.get('delivery_radius_km', 10.0)
        self.max_delivery_time = config.get('max_delivery_time_minutes', 60)
        self.price_comparison_enabled = config.get('price_comparison_enabled', True)
        
        # In-memory storage for demo (replace with database in production)
        self._active_orders: Dict[UUID, FoodOrder] = {}
        self._restaurant_cache: Dict[str, List[Restaurant]] = {}
        self._menu_cache: Dict[str, List[MenuItem]] = {}
        
        logger.info("Food delivery service initialized")
    
    async def process_food_order_command(
        self, 
        command: VoiceServiceCommand
    ) -> Dict[str, Any]:
        """
        Process a voice command for food ordering.
        
        Args:
            command: Voice service command with parsed intent and entities
            
        Returns:
            Dict containing processed order details and next steps
        """
        try:
            logger.info(f"Processing food order command: {command.command_id}")
            
            # Extract order details from voice command
            cuisine_type = self._extract_cuisine_type(command.extracted_entities)
            location = self._extract_location(command.extracted_entities)
            budget = self._extract_budget(command.extracted_entities)
            dietary_preferences = self._extract_dietary_preferences(command.extracted_entities)
            
            if not location:
                return {
                    'success': False,
                    'error': 'Could not determine delivery location',
                    'requires_clarification': True,
                    'clarification_questions': ['What is your delivery address?']
                }
            
            # Search for restaurants
            restaurants = await self._search_restaurants(
                location=location,
                cuisine_type=cuisine_type,
                budget=budget,
                dietary_preferences=dietary_preferences
            )
            
            if not restaurants:
                return {
                    'success': False,
                    'error': 'No restaurants found matching your criteria',
                    'suggestions': ['Try expanding your search radius', 'Consider different cuisine types']
                }
            
            # Generate restaurant recommendations
            recommendations = self._generate_restaurant_recommendations(restaurants, command.voice_input)
            
            return {
                'success': True,
                'restaurants': recommendations,
                'location': location.dict(),
                'search_criteria': {
                    'cuisine_type': cuisine_type,
                    'budget': float(budget) if budget else None,
                    'dietary_preferences': dietary_preferences
                },
                'next_action': 'select_restaurant',
                'voice_response': self._generate_restaurant_list_response(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error processing food order command: {e}")
            return {
                'success': False,
                'error': f'Failed to process food order: {str(e)}',
                'requires_retry': True
            }
    
    async def get_restaurant_menu(
        self, 
        restaurant_id: str, 
        platform: PlatformProvider
    ) -> Dict[str, Any]:
        """
        Get menu for a specific restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            platform: Platform provider
            
        Returns:
            Dict containing restaurant menu
        """
        try:
            # Check cache first
            cache_key = f"{platform}_{restaurant_id}"
            if cache_key in self._menu_cache:
                menu_items = self._menu_cache[cache_key]
            else:
                # Fetch menu from platform API
                menu_items = await self._fetch_restaurant_menu(restaurant_id, platform)
                self._menu_cache[cache_key] = menu_items
            
            # Categorize menu items
            categorized_menu = self._categorize_menu_items(menu_items)
            
            return {
                'success': True,
                'restaurant_id': restaurant_id,
                'platform': platform.value,
                'menu_categories': categorized_menu,
                'total_items': len(menu_items),
                'voice_response': self._generate_menu_response(categorized_menu)
            }
            
        except Exception as e:
            logger.error(f"Error getting restaurant menu: {e}")
            return {
                'success': False,
                'error': f'Failed to get menu: {str(e)}'
            }
    
    async def place_food_order(
        self, 
        order_details: Dict[str, Any],
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Place a food order on the selected platform.
        
        Args:
            order_details: Order details including items, restaurant, etc.
            user_id: User identifier
            
        Returns:
            Dict containing order confirmation
        """
        try:
            logger.info(f"Placing food order for user: {user_id}")
            
            # Create food order object
            food_order = FoodOrder(
                user_id=user_id,
                platform=PlatformProvider(order_details['platform']),
                restaurant=Restaurant(**order_details['restaurant']),
                items=order_details['items'],
                delivery_address=LocationPoint(**order_details['delivery_address']),
                total_amount=Decimal(str(order_details['total_amount'])),
                delivery_fee=Decimal(str(order_details.get('delivery_fee', 0))),
                taxes=Decimal(str(order_details.get('taxes', 0))),
                discount=Decimal(str(order_details.get('discount', 0))),
                final_amount=Decimal(str(order_details['final_amount'])),
                contact_number=order_details['contact_number'],
                payment_method=order_details['payment_method'],
                special_instructions=order_details.get('special_instructions')
            )
            
            # Submit order to platform
            platform_response = await self._submit_order_to_platform(food_order)
            
            if platform_response['success']:
                food_order.platform_order_id = platform_response['platform_order_id']
                food_order.status = BookingStatus.CONFIRMED
                food_order.estimated_delivery_time = datetime.utcnow() + timedelta(
                    minutes=platform_response.get('estimated_delivery_minutes', 45)
                )
                
                # Store order
                self._active_orders[food_order.order_id] = food_order
                
                return {
                    'success': True,
                    'order_id': str(food_order.order_id),
                    'platform_order_id': food_order.platform_order_id,
                    'estimated_delivery_time': food_order.estimated_delivery_time.isoformat(),
                    'total_amount': float(food_order.final_amount),
                    'voice_response': self._generate_order_confirmation_response(food_order)
                }
            else:
                return {
                    'success': False,
                    'error': platform_response['error'],
                    'voice_response': f"Sorry, I couldn't place your order: {platform_response['error']}"
                }
                
        except Exception as e:
            logger.error(f"Error placing food order: {e}")
            return {
                'success': False,
                'error': f'Failed to place order: {str(e)}',
                'voice_response': "There was an error placing your order. Please try again."
            }
    
    async def track_food_order(self, order_id: UUID) -> Dict[str, Any]:
        """
        Track the status of a food order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict containing order tracking information
        """
        try:
            order = self._active_orders.get(order_id)
            if not order:
                return {
                    'success': False,
                    'error': 'Order not found',
                    'voice_response': "I couldn't find that order. Please check the order ID."
                }
            
            # Get latest status from platform
            tracking_info = await self._get_order_tracking(order)
            
            # Update order status
            order.status = BookingStatus(tracking_info.get('status', order.status.value))
            order.updated_at = datetime.utcnow()
            
            return {
                'success': True,
                'order_id': str(order_id),
                'status': order.status.value,
                'estimated_delivery_time': order.estimated_delivery_time.isoformat() if order.estimated_delivery_time else None,
                'tracking_details': tracking_info,
                'voice_response': self._generate_tracking_response(order, tracking_info)
            }
            
        except Exception as e:
            logger.error(f"Error tracking food order: {e}")
            return {
                'success': False,
                'error': f'Failed to track order: {str(e)}',
                'voice_response': "There was an error tracking your order. Please try again."
            }
    
    async def compare_food_prices(
        self, 
        location: LocationPoint,
        cuisine_type: Optional[FoodCategory] = None,
        dish_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare food prices across different platforms.
        
        Args:
            location: Delivery location
            cuisine_type: Type of cuisine
            dish_name: Specific dish name
            
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
            
            for platform in [PlatformProvider.SWIGGY, PlatformProvider.ZOMATO, PlatformProvider.UBER_EATS]:
                try:
                    price_data = await self._get_platform_prices(platform, location, cuisine_type, dish_name)
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
            best_price = min(p['average_price'] for p in platform_prices)
            best_platform = next(p['platform'] for p in platform_prices if p['average_price'] == best_price)
            worst_price = max(p['average_price'] for p in platform_prices)
            
            comparison = PriceComparison(
                service_type='food_delivery',
                location=location,
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
            logger.error(f"Error comparing food prices: {e}")
            return {
                'success': False,
                'error': f'Failed to compare prices: {str(e)}'
            }
    
    def _extract_cuisine_type(self, entities: Dict[str, Any]) -> Optional[FoodCategory]:
        """Extract cuisine type from voice command entities."""
        cuisine_keywords = {
            'indian': FoodCategory.INDIAN,
            'chinese': FoodCategory.CHINESE,
            'italian': FoodCategory.ITALIAN,
            'fast food': FoodCategory.FAST_FOOD,
            'pizza': FoodCategory.ITALIAN,
            'burger': FoodCategory.FAST_FOOD,
            'biryani': FoodCategory.INDIAN,
            'noodles': FoodCategory.CHINESE
        }
        
        for key, value in entities.items():
            if isinstance(value, str):
                for keyword, category in cuisine_keywords.items():
                    if keyword in value.lower():
                        return category
        
        return None
    
    def _extract_location(self, entities: Dict[str, Any]) -> Optional[LocationPoint]:
        """Extract delivery location from voice command entities."""
        if 'location' in entities:
            loc_data = entities['location']
            if isinstance(loc_data, dict):
                return LocationPoint(**loc_data)
        
        # Try to construct from address components
        address_parts = []
        for key in ['address', 'street', 'area', 'city']:
            if key in entities:
                address_parts.append(str(entities[key]))
        
        if address_parts:
            return LocationPoint(
                latitude=0.0,  # Would be geocoded in production
                longitude=0.0,
                address=' '.join(address_parts),
                city=entities.get('city', 'Unknown')
            )
        
        return None
    
    def _extract_budget(self, entities: Dict[str, Any]) -> Optional[Decimal]:
        """Extract budget from voice command entities."""
        for key, value in entities.items():
            if 'budget' in key.lower() or 'price' in key.lower():
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    continue
        return None
    
    def _extract_dietary_preferences(self, entities: Dict[str, Any]) -> List[str]:
        """Extract dietary preferences from voice command entities."""
        preferences = []
        
        dietary_keywords = {
            'vegetarian': 'vegetarian',
            'vegan': 'vegan',
            'non-veg': 'non_vegetarian',
            'jain': 'jain',
            'halal': 'halal'
        }
        
        for key, value in entities.items():
            if isinstance(value, str):
                for keyword, preference in dietary_keywords.items():
                    if keyword in value.lower():
                        preferences.append(preference)
        
        return preferences
    
    async def _search_restaurants(
        self,
        location: LocationPoint,
        cuisine_type: Optional[FoodCategory] = None,
        budget: Optional[Decimal] = None,
        dietary_preferences: List[str] = None
    ) -> List[Restaurant]:
        """Search for restaurants based on criteria."""
        # Mock implementation - replace with actual API calls
        mock_restaurants = [
            Restaurant(
                restaurant_id="rest_001",
                name="Spice Garden",
                cuisine_types=[FoodCategory.INDIAN],
                location=location,
                rating=4.2,
                delivery_time=30,
                minimum_order=Decimal('150'),
                delivery_fee=Decimal('25'),
                is_open=True
            ),
            Restaurant(
                restaurant_id="rest_002", 
                name="Dragon Palace",
                cuisine_types=[FoodCategory.CHINESE],
                location=location,
                rating=4.0,
                delivery_time=35,
                minimum_order=Decimal('200'),
                delivery_fee=Decimal('30'),
                is_open=True
            ),
            Restaurant(
                restaurant_id="rest_003",
                name="Pizza Corner",
                cuisine_types=[FoodCategory.ITALIAN],
                location=location,
                rating=3.8,
                delivery_time=25,
                minimum_order=Decimal('300'),
                delivery_fee=Decimal('20'),
                is_open=True
            )
        ]
        
        # Filter by cuisine type
        if cuisine_type:
            mock_restaurants = [r for r in mock_restaurants if cuisine_type in r.cuisine_types]
        
        # Filter by budget (approximate)
        if budget:
            mock_restaurants = [r for r in mock_restaurants if r.minimum_order <= budget]
        
        return mock_restaurants
    
    def _generate_restaurant_recommendations(
        self, 
        restaurants: List[Restaurant], 
        voice_input: str
    ) -> List[Dict[str, Any]]:
        """Generate restaurant recommendations with scoring."""
        recommendations = []
        
        for restaurant in restaurants[:5]:  # Top 5 recommendations
            score = self._calculate_restaurant_score(restaurant, voice_input)
            
            recommendations.append({
                'restaurant_id': restaurant.restaurant_id,
                'name': restaurant.name,
                'cuisine_types': [c.value for c in restaurant.cuisine_types],
                'rating': restaurant.rating,
                'delivery_time': restaurant.delivery_time,
                'minimum_order': float(restaurant.minimum_order) if restaurant.minimum_order else None,
                'delivery_fee': float(restaurant.delivery_fee) if restaurant.delivery_fee else None,
                'recommendation_score': score,
                'is_recommended': score > 0.7
            })
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations
    
    def _calculate_restaurant_score(self, restaurant: Restaurant, voice_input: str) -> float:
        """Calculate recommendation score for a restaurant."""
        score = 0.0
        
        # Rating score (0-1)
        if restaurant.rating:
            score += (restaurant.rating / 5.0) * 0.4
        
        # Delivery time score (faster is better)
        if restaurant.delivery_time:
            time_score = max(0, (60 - restaurant.delivery_time) / 60)
            score += time_score * 0.3
        
        # Keyword matching score
        restaurant_text = f"{restaurant.name} {' '.join([c.value for c in restaurant.cuisine_types])}"
        keyword_matches = sum(1 for word in voice_input.lower().split() 
                            if word in restaurant_text.lower())
        score += min(keyword_matches * 0.1, 0.3)
        
        return min(score, 1.0)
    
    async def _fetch_restaurant_menu(
        self, 
        restaurant_id: str, 
        platform: PlatformProvider
    ) -> List[MenuItem]:
        """Fetch restaurant menu from platform API."""
        # Mock implementation - replace with actual API calls
        mock_menu = [
            MenuItem(
                item_id="item_001",
                name="Butter Chicken",
                description="Creamy tomato-based chicken curry",
                price=Decimal('280'),
                category=FoodCategory.INDIAN,
                is_vegetarian=False,
                rating=4.5,
                preparation_time=20
            ),
            MenuItem(
                item_id="item_002",
                name="Paneer Tikka",
                description="Grilled cottage cheese with spices",
                price=Decimal('220'),
                category=FoodCategory.INDIAN,
                is_vegetarian=True,
                rating=4.2,
                preparation_time=15
            ),
            MenuItem(
                item_id="item_003",
                name="Chicken Biryani",
                description="Aromatic basmati rice with chicken",
                price=Decimal('320'),
                category=FoodCategory.INDIAN,
                is_vegetarian=False,
                rating=4.7,
                preparation_time=25
            )
        ]
        
        return mock_menu
    
    def _categorize_menu_items(self, menu_items: List[MenuItem]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize menu items by food category."""
        categorized = {}
        
        for item in menu_items:
            category = item.category.value
            if category not in categorized:
                categorized[category] = []
            
            categorized[category].append({
                'item_id': item.item_id,
                'name': item.name,
                'description': item.description,
                'price': float(item.price),
                'is_vegetarian': item.is_vegetarian,
                'rating': item.rating,
                'preparation_time': item.preparation_time
            })
        
        return categorized
    
    async def _submit_order_to_platform(self, order: FoodOrder) -> Dict[str, Any]:
        """Submit order to the selected platform."""
        # Mock implementation - replace with actual platform API calls
        await asyncio.sleep(2)  # Simulate API call
        
        # Mock success response
        return {
            'success': True,
            'platform_order_id': f"{order.platform.value.upper()}{order.order_id.hex[:8]}",
            'estimated_delivery_minutes': 35,
            'confirmation_message': 'Order placed successfully'
        }
    
    async def _get_order_tracking(self, order: FoodOrder) -> Dict[str, Any]:
        """Get order tracking information from platform."""
        # Mock implementation - replace with actual platform API calls
        statuses = ['confirmed', 'preparing', 'out_for_delivery', 'delivered']
        current_status_index = min(
            len(statuses) - 1,
            int((datetime.utcnow() - order.created_at).total_seconds() / 600)  # Progress every 10 minutes
        )
        
        return {
            'status': statuses[current_status_index],
            'status_message': f'Your order is {statuses[current_status_index].replace("_", " ")}',
            'estimated_delivery_time': order.estimated_delivery_time.isoformat() if order.estimated_delivery_time else None,
            'delivery_person': {
                'name': 'Raj Kumar',
                'phone': '+91-9876543210'
            } if current_status_index >= 2 else None
        }
    
    async def _get_platform_prices(
        self,
        platform: PlatformProvider,
        location: LocationPoint,
        cuisine_type: Optional[FoodCategory],
        dish_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get price information from a specific platform."""
        # Mock implementation - replace with actual platform API calls
        base_prices = {
            PlatformProvider.SWIGGY: 250,
            PlatformProvider.ZOMATO: 280,
            PlatformProvider.UBER_EATS: 270
        }
        
        return {
            'platform': platform.value,
            'average_price': Decimal(str(base_prices.get(platform, 250))),
            'delivery_fee': Decimal('25'),
            'service_fee': Decimal('15'),
            'total_estimated': Decimal(str(base_prices.get(platform, 250) + 40))
        }
    
    def _generate_restaurant_list_response(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate voice response for restaurant recommendations."""
        if not recommendations:
            return "I couldn't find any restaurants matching your criteria."
        
        top_restaurant = recommendations[0]
        response = f"I found {len(recommendations)} restaurants for you. "
        response += f"The top recommendation is {top_restaurant['name']} "
        response += f"with a {top_restaurant['rating']} star rating and "
        response += f"{top_restaurant['delivery_time']} minutes delivery time. "
        response += "Would you like to see their menu or hear about other options?"
        
        return response
    
    def _generate_menu_response(self, categorized_menu: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate voice response for restaurant menu."""
        categories = list(categorized_menu.keys())
        total_items = sum(len(items) for items in categorized_menu.values())
        
        response = f"This restaurant has {total_items} items across {len(categories)} categories: "
        response += ", ".join(categories) + ". "
        response += "Which category would you like to explore, or would you like me to recommend popular items?"
        
        return response
    
    def _generate_order_confirmation_response(self, order: FoodOrder) -> str:
        """Generate voice response for order confirmation."""
        response = f"Great! Your order has been placed with {order.restaurant.name}. "
        response += f"Order total is ₹{order.final_amount}. "
        if order.estimated_delivery_time:
            delivery_time = order.estimated_delivery_time.strftime("%I:%M %p")
            response += f"Expected delivery time is {delivery_time}. "
        response += f"Your order ID is {order.platform_order_id}. "
        response += "I'll keep you updated on the delivery status."
        
        return response
    
    def _generate_tracking_response(self, order: FoodOrder, tracking_info: Dict[str, Any]) -> str:
        """Generate voice response for order tracking."""
        status_message = tracking_info.get('status_message', 'Order status updated')
        response = f"Your order from {order.restaurant.name} is {status_message}. "
        
        if tracking_info.get('delivery_person'):
            delivery_person = tracking_info['delivery_person']
            response += f"Your delivery person is {delivery_person['name']}. "
        
        if order.estimated_delivery_time:
            delivery_time = order.estimated_delivery_time.strftime("%I:%M %p")
            response += f"Expected delivery time is {delivery_time}."
        
        return response
    
    async def get_booking_confirmation(self, order_id: UUID) -> Dict[str, Any]:
        """
        Get detailed booking confirmation for a food order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict containing detailed booking confirmation
        """
        try:
            order = self._active_orders.get(order_id)
            if not order:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            confirmation = BookingConfirmation(
                booking_id=order.order_id,
                platform=order.platform,
                confirmation_code=order.platform_order_id or str(order.order_id),
                estimated_time=order.estimated_delivery_time,
                contact_details={
                    'restaurant_name': order.restaurant.name,
                    'customer_phone': order.contact_number,
                    'delivery_phone': '+91-9876543210'  # Mock delivery contact
                },
                cancellation_policy=self._get_cancellation_policy(order.platform),
                tracking_url=f"https://{order.platform.value}.com/track/{order.platform_order_id}",
                qr_code=f"QR_{order.platform_order_id}"
            )
            
            return {
                'success': True,
                'confirmation': confirmation.dict(),
                'voice_response': self._generate_detailed_confirmation_response(order, confirmation)
            }
            
        except Exception as e:
            logger.error(f"Error getting booking confirmation: {e}")
            return {
                'success': False,
                'error': f'Failed to get confirmation: {str(e)}'
            }
    
    async def get_real_time_status(self, order_id: UUID) -> Dict[str, Any]:
        """
        Get real-time status updates for a food order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dict containing real-time status information
        """
        try:
            order = self._active_orders.get(order_id)
            if not order:
                return {
                    'success': False,
                    'error': 'Order not found'
                }
            
            # Get real-time tracking from platform
            real_time_info = await self._get_real_time_tracking(order)
            
            # Update order with latest information
            order.status = BookingStatus(real_time_info.get('status', order.status.value))
            order.updated_at = datetime.utcnow()
            
            return {
                'success': True,
                'order_id': str(order_id),
                'real_time_status': real_time_info,
                'estimated_delivery': order.estimated_delivery_time.isoformat() if order.estimated_delivery_time else None,
                'live_tracking': real_time_info.get('live_tracking', {}),
                'voice_response': self._generate_real_time_status_response(order, real_time_info)
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time status: {e}")
            return {
                'success': False,
                'error': f'Failed to get real-time status: {str(e)}'
            }
    
    def _get_cancellation_policy(self, platform: PlatformProvider) -> str:
        """Get cancellation policy for a platform."""
        policies = {
            PlatformProvider.SWIGGY: 'Free cancellation before restaurant confirms order. Partial refund after confirmation.',
            PlatformProvider.ZOMATO: 'Cancellation charges may apply based on order status and restaurant policy.',
            PlatformProvider.UBER_EATS: 'Free cancellation within 5 minutes. Full charges apply after preparation starts.',
            PlatformProvider.FOOD_PANDA: 'Cancellation policy varies by restaurant. Check order details for specifics.'
        }
        
        return policies.get(platform, 'Cancellation charges may apply based on order status')
    
    async def _get_real_time_tracking(self, order: FoodOrder) -> Dict[str, Any]:
        """Get real-time tracking information from platform."""
        # Mock implementation with more detailed real-time data
        statuses = ['confirmed', 'preparing', 'ready_for_pickup', 'out_for_delivery', 'delivered']
        current_status_index = min(
            len(statuses) - 1,
            int((datetime.utcnow() - order.created_at).total_seconds() / 600)  # Progress every 10 minutes
        )
        
        current_status = statuses[current_status_index]
        
        # Mock live tracking data
        live_tracking = {}
        if current_status_index >= 3:  # Out for delivery
            live_tracking = {
                'delivery_person': {
                    'name': 'Amit Sharma',
                    'phone': '+91-9876543210',
                    'rating': 4.6,
                    'vehicle_type': 'Bike'
                },
                'current_location': {
                    'latitude': 12.9716,
                    'longitude': 77.5946,
                    'address': 'Near MG Road Metro Station'
                },
                'estimated_arrival_minutes': max(0, 15 - (current_status_index - 3) * 5)
            }
        
        return {
            'status': current_status,
            'status_message': self._get_status_message(current_status),
            'progress_percentage': min(100, (current_status_index + 1) * 20),
            'live_tracking': live_tracking,
            'restaurant_contact': {
                'name': order.restaurant.name,
                'phone': '+91-8765432109'
            },
            'order_timeline': self._generate_order_timeline(order, current_status_index)
        }
    
    def _get_status_message(self, status: str) -> str:
        """Get user-friendly status message."""
        messages = {
            'confirmed': 'Your order has been confirmed and sent to the restaurant',
            'preparing': 'The restaurant is preparing your order',
            'ready_for_pickup': 'Your order is ready and waiting for pickup',
            'out_for_delivery': 'Your order is on the way to you',
            'delivered': 'Your order has been delivered'
        }
        
        return messages.get(status, 'Order status updated')
    
    def _generate_order_timeline(self, order: FoodOrder, current_index: int) -> List[Dict[str, Any]]:
        """Generate order timeline with timestamps."""
        timeline = []
        base_time = order.created_at
        
        statuses = [
            ('Order Placed', 'confirmed'),
            ('Restaurant Confirmed', 'preparing'),
            ('Food Ready', 'ready_for_pickup'),
            ('Out for Delivery', 'out_for_delivery'),
            ('Delivered', 'delivered')
        ]
        
        for i, (title, status) in enumerate(statuses):
            timestamp = base_time + timedelta(minutes=i * 10)
            timeline.append({
                'title': title,
                'status': status,
                'timestamp': timestamp.isoformat(),
                'completed': i <= current_index,
                'estimated': i > current_index
            })
        
        return timeline
    
    def _generate_detailed_confirmation_response(
        self, 
        order: FoodOrder, 
        confirmation: BookingConfirmation
    ) -> str:
        """Generate detailed voice response for booking confirmation."""
        response = f"Your order confirmation is ready. "
        response += f"Order from {order.restaurant.name} for ₹{order.final_amount} "
        response += f"has been confirmed with booking ID {confirmation.confirmation_code}. "
        
        if order.estimated_delivery_time:
            delivery_time = order.estimated_delivery_time.strftime("%I:%M %p")
            response += f"Expected delivery at {delivery_time}. "
        
        response += f"You can track your order using the confirmation code. "
        response += "I'll keep you updated on the delivery progress."
        
        return response
    
    def _generate_real_time_status_response(
        self, 
        order: FoodOrder, 
        real_time_info: Dict[str, Any]
    ) -> str:
        """Generate voice response for real-time status."""
        status_message = real_time_info.get('status_message', 'Order status updated')
        response = f"{status_message}. "
        
        progress = real_time_info.get('progress_percentage', 0)
        response += f"Your order is {progress}% complete. "
        
        live_tracking = real_time_info.get('live_tracking', {})
        if live_tracking and live_tracking.get('delivery_person'):
            delivery_person = live_tracking['delivery_person']
            response += f"Your delivery person {delivery_person['name']} is on the way. "
            
            arrival_minutes = live_tracking.get('estimated_arrival_minutes', 0)
            if arrival_minutes > 0:
                response += f"Estimated arrival in {arrival_minutes} minutes. "
        
        return response
    
    def _generate_price_comparison_response(self, comparison: PriceComparison) -> str:
        """Generate voice response for price comparison."""
        response = f"I compared prices across {len(comparison.platforms)} platforms. "
        response += f"The best price is ₹{comparison.best_price} on {comparison.best_platform.value}. "
        
        if comparison.price_difference > 0:
            response += f"You can save up to ₹{comparison.price_difference} by choosing the best option. "
        
        response += "Would you like to proceed with the best option or hear about other platforms?"
        
        return response
    
    async def health_check(self) -> ServiceResult:
        """Check food delivery service health."""
        try:
            return ServiceResult(
                service_type=ServiceType.FOOD_DELIVERY,
                success=True,
                data={'status': 'healthy', 'platforms': ['swiggy', 'zomato', 'uber_eats']},
                response_time=0.1
            )
            
        except Exception as e:
            return ServiceResult(
                service_type=ServiceType.FOOD_DELIVERY,
                success=False,
                error_message=str(e),
                response_time=0.1
            )