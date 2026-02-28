<<<<<<< HEAD
"""
Redis cache implementation for BharatVoice Assistant.
"""

import asyncio
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching system with connection pooling."""
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False
    
    async def initialize(self):
        """Initialize Redis connection and pool."""
        try:
            # Parse Redis URL
            redis_url = self.settings.redis.url
            
            # Create connection pool
            self._connection_pool = ConnectionPool.from_url(
                redis_url,
                max_connections=self.settings.redis.max_connections,
                socket_timeout=self.settings.redis.socket_timeout,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Create Redis client
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            await self._redis.ping()
            self._is_healthy = True
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._is_healthy = False
            # Don't raise - allow fallback to database cache
    
    async def close(self):
        """Close Redis connections."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._redis:
                await self._redis.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Redis cache closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis cache: {e}")
    
    async def _periodic_health_check(self):
        """Periodic health check for Redis connection."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self._redis:
                    await self._redis.ping()
                    self._is_healthy = True
                    logger.debug("Redis health check passed")
                else:
                    self._is_healthy = False
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                self._is_healthy = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if Redis is healthy."""
        return self._is_healthy and self._redis is not None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first (for simple types)
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_healthy:
            return None
        
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Failed to get cache key '{key}': {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            serialized_value = self._serialize_value(value)
            
            result = await self._redis.set(
                key, 
                serialized_value, 
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check cache key '{key}': {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set expiration for key '{key}': {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get time to live for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        if not self.is_healthy:
            return -2
        
        try:
            return await self._redis.ttl(key)
            
        except Exception as e:
            logger.error(f"Failed to get TTL for key '{key}': {e}")
            return -2
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        if not self.is_healthy:
            return []
        
        try:
            keys = await self._redis.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern '{pattern}': {e}")
            return []
    
    async def flush_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            await self._redis.flushall()
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False
    
    async def flush_db(self) -> bool:
        """
        Clear current database cache entries.
        
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            await self._redis.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush database cache: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment numeric value.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None if failed
        """
        if not self.is_healthy:
            return None
        
        try:
            return await self._redis.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to increment key '{key}': {e}")
            return None
    
    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement numeric value.
        
        Args:
            key: Cache key
            amount: Amount to decrement
            
        Returns:
            New value or None if failed
        """
        if not self.is_healthy:
            return None
        
        try:
            return await self._redis.decrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to decrement key '{key}': {e}")
            return None
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Server information dictionary
        """
        if not self.is_healthy:
            return {"status": "unhealthy"}
        
        try:
            info = await self._redis.info()
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"status": "error", "error": str(e)}


# Global Redis cache instance
_redis_cache: Optional[RedisCache] = None


async def get_redis_cache() -> RedisCache:
    """Get global Redis cache instance."""
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache()
        await _redis_cache.initialize()
    
    return _redis_cache


async def close_redis_cache():
    """Close global Redis cache."""
    global _redis_cache
    
    if _redis_cache:
        await _redis_cache.close()
=======
"""
Redis cache implementation for BharatVoice Assistant.
"""

import asyncio
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching system with connection pooling."""
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[redis.Redis] = None
        self._connection_pool: Optional[ConnectionPool] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False
    
    async def initialize(self):
        """Initialize Redis connection and pool."""
        try:
            # Parse Redis URL
            redis_url = self.settings.redis.url
            
            # Create connection pool
            self._connection_pool = ConnectionPool.from_url(
                redis_url,
                max_connections=self.settings.redis.max_connections,
                socket_timeout=self.settings.redis.socket_timeout,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Create Redis client
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Test connection
            await self._redis.ping()
            self._is_healthy = True
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._is_healthy = False
            # Don't raise - allow fallback to database cache
    
    async def close(self):
        """Close Redis connections."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._redis:
                await self._redis.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Redis cache closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis cache: {e}")
    
    async def _periodic_health_check(self):
        """Periodic health check for Redis connection."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self._redis:
                    await self._redis.ping()
                    self._is_healthy = True
                    logger.debug("Redis health check passed")
                else:
                    self._is_healthy = False
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
                self._is_healthy = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if Redis is healthy."""
        return self._is_healthy and self._redis is not None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first (for simple types)
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_healthy:
            return None
        
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Failed to get cache key '{key}': {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            serialized_value = self._serialize_value(value)
            
            result = await self._redis.set(
                key, 
                serialized_value, 
                ex=ttl,
                nx=nx,
                xx=xx
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check cache key '{key}': {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            result = await self._redis.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set expiration for key '{key}': {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get time to live for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        if not self.is_healthy:
            return -2
        
        try:
            return await self._redis.ttl(key)
            
        except Exception as e:
            logger.error(f"Failed to get TTL for key '{key}': {e}")
            return -2
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern
            
        Returns:
            List of matching keys
        """
        if not self.is_healthy:
            return []
        
        try:
            keys = await self._redis.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern '{pattern}': {e}")
            return []
    
    async def flush_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            await self._redis.flushall()
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            return False
    
    async def flush_db(self) -> bool:
        """
        Clear current database cache entries.
        
        Returns:
            True if successful
        """
        if not self.is_healthy:
            return False
        
        try:
            await self._redis.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Failed to flush database cache: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment numeric value.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None if failed
        """
        if not self.is_healthy:
            return None
        
        try:
            return await self._redis.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to increment key '{key}': {e}")
            return None
    
    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement numeric value.
        
        Args:
            key: Cache key
            amount: Amount to decrement
            
        Returns:
            New value or None if failed
        """
        if not self.is_healthy:
            return None
        
        try:
            return await self._redis.decrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to decrement key '{key}': {e}")
            return None
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Server information dictionary
        """
        if not self.is_healthy:
            return {"status": "unhealthy"}
        
        try:
            info = await self._redis.info()
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {"status": "error", "error": str(e)}


# Global Redis cache instance
_redis_cache: Optional[RedisCache] = None


async def get_redis_cache() -> RedisCache:
    """Get global Redis cache instance."""
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache()
        await _redis_cache.initialize()
    
    return _redis_cache


async def close_redis_cache():
    """Close global Redis cache."""
    global _redis_cache
    
    if _redis_cache:
        await _redis_cache.close()
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        _redis_cache = None