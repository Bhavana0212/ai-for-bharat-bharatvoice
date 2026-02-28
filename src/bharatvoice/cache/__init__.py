"""
Caching package for BharatVoice Assistant.

This package provides Redis-based caching with fallback to database storage.
"""

from .redis_cache import RedisCache, get_redis_cache
from .cache_manager import CacheManager, get_cache_manager
from .strategies import (
    CacheStrategy,
    TTLStrategy,
    LRUStrategy,
    TagBasedStrategy
)

__all__ = [
    "RedisCache",
    "get_redis_cache",
    "CacheManager", 
    "get_cache_manager",
    "CacheStrategy",
    "TTLStrategy",
    "LRUStrategy",
    "TagBasedStrategy"
]