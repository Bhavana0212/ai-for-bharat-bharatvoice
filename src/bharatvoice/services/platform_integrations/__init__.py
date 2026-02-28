<<<<<<< HEAD
"""
Platform integrations module for BharatVoice Assistant.

This module provides integrations with popular Indian service platforms
including food delivery, ride-sharing, and other booking services.
"""

from .food_delivery_service import FoodDeliveryService
from .ride_sharing_service import RideSharingService
from .platform_manager import PlatformManager
from .models import (
    FoodOrder,
    RideBooking,
    ServiceBooking,
    BookingStatus,
    PlatformProvider,
    PriceComparison
)

__all__ = [
    "FoodDeliveryService",
    "RideSharingService", 
    "PlatformManager",
    "FoodOrder",
    "RideBooking",
    "ServiceBooking",
    "BookingStatus",
    "PlatformProvider",
    "PriceComparison"
=======
"""
Platform integrations module for BharatVoice Assistant.

This module provides integrations with popular Indian service platforms
including food delivery, ride-sharing, and other booking services.
"""

from .food_delivery_service import FoodDeliveryService
from .ride_sharing_service import RideSharingService
from .platform_manager import PlatformManager
from .models import (
    FoodOrder,
    RideBooking,
    ServiceBooking,
    BookingStatus,
    PlatformProvider,
    PriceComparison
)

__all__ = [
    "FoodDeliveryService",
    "RideSharingService", 
    "PlatformManager",
    "FoodOrder",
    "RideBooking",
    "ServiceBooking",
    "BookingStatus",
    "PlatformProvider",
    "PriceComparison"
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
]