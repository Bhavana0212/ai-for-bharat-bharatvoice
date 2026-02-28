"""
Platform integration data models for BharatVoice Assistant.

This module defines data structures for service platform integrations
including food delivery, ride-sharing, and booking services.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class PlatformProvider(str, Enum):
    """Supported platform providers."""
    
    # Food delivery platforms
    SWIGGY = "swiggy"
    ZOMATO = "zomato"
    UBER_EATS = "uber_eats"
    FOOD_PANDA = "food_panda"
    
    # Ride-sharing platforms
    OLA = "ola"
    UBER = "uber"
    RAPIDO = "rapido"
    AUTO_RICKSHAW = "auto_rickshaw"
    
    # Other service platforms
    URBAN_COMPANY = "urban_company"
    JUSTDIAL = "justdial"
    BOOKMYSHOW = "bookmyshow"


class BookingStatus(str, Enum):
    """Booking status values."""
    
    PENDING = "pending"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FoodCategory(str, Enum):
    """Food categories for delivery."""
    
    INDIAN = "indian"
    CHINESE = "chinese"
    ITALIAN = "italian"
    FAST_FOOD = "fast_food"
    DESSERTS = "desserts"
    BEVERAGES = "beverages"
    HEALTHY = "healthy"
    VEGETARIAN = "vegetarian"
    NON_VEGETARIAN = "non_vegetarian"


class RideType(str, Enum):
    """Types of ride services."""
    
    MINI = "mini"
    SEDAN = "sedan"
    SUV = "suv"
    AUTO = "auto"
    BIKE = "bike"
    SHARED = "shared"
    PREMIUM = "premium"


class LocationPoint(BaseModel):
    """Geographic location point."""
    
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    address: str = Field(..., description="Human-readable address")
    landmark: Optional[str] = Field(None, description="Nearby landmark")
    city: str = Field(..., description="City name")
    postal_code: Optional[str] = Field(None, description="Postal code")


class MenuItem(BaseModel):
    """Food menu item."""
    
    item_id: str = Field(..., description="Item identifier")
    name: str = Field(..., description="Item name")
    description: Optional[str] = Field(None, description="Item description")
    price: Decimal = Field(..., gt=0, description="Item price")
    category: FoodCategory = Field(..., description="Food category")
    is_vegetarian: bool = Field(default=True, description="Whether item is vegetarian")
    is_available: bool = Field(default=True, description="Whether item is available")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Item rating")
    preparation_time: Optional[int] = Field(None, description="Preparation time in minutes")
    image_url: Optional[str] = Field(None, description="Item image URL")


class Restaurant(BaseModel):
    """Restaurant information."""
    
    restaurant_id: str = Field(..., description="Restaurant identifier")
    name: str = Field(..., description="Restaurant name")
    cuisine_types: List[FoodCategory] = Field(..., description="Cuisine types")
    location: LocationPoint = Field(..., description="Restaurant location")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Restaurant rating")
    delivery_time: Optional[int] = Field(None, description="Estimated delivery time in minutes")
    minimum_order: Optional[Decimal] = Field(None, description="Minimum order amount")
    delivery_fee: Optional[Decimal] = Field(None, description="Delivery fee")
    is_open: bool = Field(default=True, description="Whether restaurant is open")
    menu_items: List[MenuItem] = Field(default_factory=list, description="Available menu items")


class FoodOrder(BaseModel):
    """Food delivery order."""
    
    order_id: UUID = Field(default_factory=uuid4, description="Unique order ID")
    user_id: UUID = Field(..., description="User identifier")
    platform: PlatformProvider = Field(..., description="Delivery platform")
    restaurant: Restaurant = Field(..., description="Restaurant details")
    items: List[Dict[str, Any]] = Field(..., description="Ordered items with quantities")
    delivery_address: LocationPoint = Field(..., description="Delivery address")
    total_amount: Decimal = Field(..., description="Total order amount")
    delivery_fee: Decimal = Field(default=Decimal('0'), description="Delivery fee")
    taxes: Decimal = Field(default=Decimal('0'), description="Taxes and charges")
    discount: Decimal = Field(default=Decimal('0'), description="Discount amount")
    final_amount: Decimal = Field(..., description="Final payable amount")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="Order status")
    estimated_delivery_time: Optional[datetime] = Field(None, description="Estimated delivery time")
    special_instructions: Optional[str] = Field(None, description="Special delivery instructions")
    contact_number: str = Field(..., description="Contact number for delivery")
    payment_method: str = Field(..., description="Payment method")
    platform_order_id: Optional[str] = Field(None, description="Platform-specific order ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Order creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class RideBooking(BaseModel):
    """Ride booking details."""
    
    booking_id: UUID = Field(default_factory=uuid4, description="Unique booking ID")
    user_id: UUID = Field(..., description="User identifier")
    platform: PlatformProvider = Field(..., description="Ride platform")
    ride_type: RideType = Field(..., description="Type of ride")
    pickup_location: LocationPoint = Field(..., description="Pickup location")
    drop_location: LocationPoint = Field(..., description="Drop location")
    estimated_fare: Decimal = Field(..., description="Estimated fare")
    estimated_distance: float = Field(..., description="Estimated distance in km")
    estimated_duration: int = Field(..., description="Estimated duration in minutes")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="Booking status")
    driver_details: Optional[Dict[str, Any]] = Field(None, description="Driver information")
    vehicle_details: Optional[Dict[str, Any]] = Field(None, description="Vehicle information")
    otp: Optional[str] = Field(None, description="OTP for ride verification")
    special_requests: Optional[str] = Field(None, description="Special requests")
    contact_number: str = Field(..., description="Contact number")
    payment_method: str = Field(..., description="Payment method")
    platform_booking_id: Optional[str] = Field(None, description="Platform-specific booking ID")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled ride time")
    actual_pickup_time: Optional[datetime] = Field(None, description="Actual pickup time")
    actual_drop_time: Optional[datetime] = Field(None, description="Actual drop time")
    final_fare: Optional[Decimal] = Field(None, description="Final fare charged")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Booking creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class ServiceBooking(BaseModel):
    """Generic service booking."""
    
    booking_id: UUID = Field(default_factory=uuid4, description="Unique booking ID")
    user_id: UUID = Field(..., description="User identifier")
    platform: PlatformProvider = Field(..., description="Service platform")
    service_type: str = Field(..., description="Type of service")
    service_details: Dict[str, Any] = Field(..., description="Service-specific details")
    location: LocationPoint = Field(..., description="Service location")
    scheduled_time: datetime = Field(..., description="Scheduled service time")
    estimated_cost: Decimal = Field(..., description="Estimated service cost")
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="Booking status")
    provider_details: Optional[Dict[str, Any]] = Field(None, description="Service provider details")
    special_instructions: Optional[str] = Field(None, description="Special instructions")
    contact_number: str = Field(..., description="Contact number")
    payment_method: str = Field(..., description="Payment method")
    platform_booking_id: Optional[str] = Field(None, description="Platform-specific booking ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Booking creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class PriceComparison(BaseModel):
    """Price comparison across platforms."""
    
    service_type: str = Field(..., description="Type of service being compared")
    location: LocationPoint = Field(..., description="Service location")
    platforms: List[Dict[str, Any]] = Field(..., description="Platform pricing details")
    best_price: Decimal = Field(..., description="Best available price")
    best_platform: PlatformProvider = Field(..., description="Platform with best price")
    price_difference: Decimal = Field(..., description="Difference between best and worst price")
    comparison_time: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")
    
    def get_savings(self, platform: PlatformProvider) -> Optional[Decimal]:
        """Get potential savings by choosing best platform over specified platform."""
        platform_data = next((p for p in self.platforms if p['platform'] == platform), None)
        if platform_data:
            return platform_data['price'] - self.best_price
        return None


class VoiceServiceCommand(BaseModel):
    """Voice command for service booking."""
    
    command_id: UUID = Field(default_factory=uuid4, description="Command ID")
    user_id: UUID = Field(..., description="User identifier")
    voice_input: str = Field(..., description="Transcribed voice input")
    service_type: str = Field(..., description="Detected service type")
    parsed_intent: str = Field(..., description="Parsed service intent")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Command confidence")
    requires_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions for clarification")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Command timestamp")


class BookingConfirmation(BaseModel):
    """Booking confirmation details."""
    
    confirmation_id: UUID = Field(default_factory=uuid4, description="Confirmation ID")
    booking_id: UUID = Field(..., description="Associated booking ID")
    platform: PlatformProvider = Field(..., description="Service platform")
    confirmation_code: str = Field(..., description="Platform confirmation code")
    estimated_time: Optional[datetime] = Field(None, description="Estimated service time")
    contact_details: Dict[str, str] = Field(..., description="Contact information")
    cancellation_policy: str = Field(..., description="Cancellation policy")
    tracking_url: Optional[str] = Field(None, description="Tracking URL")
    qr_code: Optional[str] = Field(None, description="QR code for verification")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Confirmation timestamp")