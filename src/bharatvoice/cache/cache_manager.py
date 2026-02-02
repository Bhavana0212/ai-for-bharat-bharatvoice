"""
Cache manager with Redis and database fallback.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from .redis_cache import RedisCache, get_redis_cache
from .strategies import CacheStrategy, TTLStrategy, TagBasedStrategy, CompositeStrategy
from bharatvoice.database.base import get_db_session
from bharatvoice.database.models import CacheEntry
from sqlalchemy import select, delete
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager with Redis primary and database fallback.
    """
    
    def __init__(self):
        self._redis_cache: Optional[RedisCache] = None
        self._strategy: CacheStrategy = CompositeStrategy([
            TTLStrategy(default_ttl=3600),  # 1 hour default
            TagBasedStrategy()
        ])
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize cache manager."""
        try:
            # Initialize Redis cache
            self._redis_cache = await get_redis_cache()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            self._initialized = True
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            # Continue without Redis - database fallback will be used
            self._initialized = True
    
    async def close(self):
        """Close cache manager."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self._redis_cache:
                await self._redis_cache.close()
            
            logger.info("Cache manager closed")
            
        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")
    
    async def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache entry
            
        Returns:
            Cached value or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        # Try Redis first
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                value = await self._redis_cache.get(key)
                if value is not None:
                    # Update access metadata
                    await self._update_access_metadata(key, cache_type)
                    return value
            except Exception as e:
                logger.warning(f"Redis get failed for key '{key}': {e}")
        
        # Fallback to database
        try:
            async with get_db_session() as session:
                stmt = select(CacheEntry).where(CacheEntry.cache_key == key)
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                
                if entry:
                    # Check if entry should be evicted
                    metadata = {
                        "ttl": entry.ttl,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                        "created_at": entry.created_at.isoformat(),
                        "tags": entry.tags or [],
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None
                    }
                    
                    if await self._strategy.should_evict(key, metadata):
                        await self._delete_from_database(key)
                        return None
                    
                    # Update access metadata
                    updated_metadata = await self._strategy.on_access(key, metadata)
                    await self._update_database_metadata(entry, updated_metadata)
                    
                    # Return data
                    return entry.data if entry.data is not None else entry.binary_data
                
                return None
                
        except Exception as e:
            logger.error(f"Database get failed for key '{key}': {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = "default",
        tags: Optional[List[str]] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_type: Type of cache entry
            tags: Tags for invalidation
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        # Prepare metadata
        metadata = {
            "ttl": ttl,
            "tags": tags or [],
            "cache_type": cache_type
        }
        
        # Apply strategy
        metadata = await self._strategy.on_set(key, value, metadata)
        
        success = False
        
        # Try Redis first
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                success = await self._redis_cache.set(key, value, ttl=ttl, nx=nx, xx=xx)
                if success:
                    logger.debug(f"Cached key '{key}' in Redis")
            except Exception as e:
                logger.warning(f"Redis set failed for key '{key}': {e}")
        
        # Always store in database as backup
        try:
            await self._store_in_database(key, value, metadata, cache_type)
            success = True
            logger.debug(f"Cached key '{key}' in database")
            
        except Exception as e:
            logger.error(f"Database set failed for key '{key}': {e}")
        
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        if not self._initialized:
            await self.initialize()
        
        success = False
        
        # Delete from Redis
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                redis_success = await self._redis_cache.delete(key)
                if redis_success:
                    success = True
            except Exception as e:
                logger.warning(f"Redis delete failed for key '{key}': {e}")
        
        # Delete from database
        try:
            db_success = await self._delete_from_database(key)
            if db_success:
                success = True
        except Exception as e:
            logger.error(f"Database delete failed for key '{key}': {e}")
        
        return success
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        if not self._initialized:
            await self.initialize()
        
        # Check Redis first
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                if await self._redis_cache.exists(key):
                    return True
            except Exception as e:
                logger.warning(f"Redis exists check failed for key '{key}': {e}")
        
        # Check database
        try:
            async with get_db_session() as session:
                stmt = select(CacheEntry.id).where(CacheEntry.cache_key == key)
                result = await session.execute(stmt)
                return result.scalar_one_or_none() is not None
                
        except Exception as e:
            logger.error(f"Database exists check failed for key '{key}': {e}")
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with the given tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if not self._initialized:
            await self.initialize()
        
        count = 0
        
        # Get strategy if it supports tags
        tag_strategy = None
        if hasattr(self._strategy, 'strategies'):
            for strategy in self._strategy.strategies:
                if isinstance(strategy, TagBasedStrategy):
                    tag_strategy = strategy
                    break
        elif isinstance(self._strategy, TagBasedStrategy):
            tag_strategy = self._strategy
        
        if tag_strategy:
            keys = await tag_strategy.invalidate_tag(tag)
            for key in keys:
                if await self.delete(key):
                    count += 1
        
        # Also invalidate from database
        try:
            async with get_db_session() as session:
                # Find entries with the tag
                stmt = select(CacheEntry).where(CacheEntry.tags.contains([tag]))
                result = await session.execute(stmt)
                entries = result.scalars().all()
                
                # Delete entries
                for entry in entries:
                    await session.delete(entry)
                    count += 1
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Database tag invalidation failed for tag '{tag}': {e}")
        
        return count
    
    async def clear(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache entries.
        
        Args:
            cache_type: Specific cache type to clear, or None for all
            
        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()
        
        success = True
        
        # Clear Redis
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                if cache_type:
                    # Clear specific type (would need pattern matching)
                    keys = await self._redis_cache.keys(f"{cache_type}:*")
                    for key in keys:
                        await self._redis_cache.delete(key)
                else:
                    await self._redis_cache.flush_db()
            except Exception as e:
                logger.error(f"Redis clear failed: {e}")
                success = False
        
        # Clear database
        try:
            async with get_db_session() as session:
                if cache_type:
                    stmt = delete(CacheEntry).where(CacheEntry.cache_type == cache_type)
                else:
                    stmt = delete(CacheEntry)
                
                await session.execute(stmt)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Database clear failed: {e}")
            success = False
        
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        stats = {
            "redis": {"status": "unavailable"},
            "database": {"status": "unavailable"}
        }
        
        # Redis stats
        if self._redis_cache and self._redis_cache.is_healthy:
            try:
                stats["redis"] = await self._redis_cache.get_info()
            except Exception as e:
                stats["redis"] = {"status": "error", "error": str(e)}
        
        # Database stats
        try:
            async with get_db_session() as session:
                from sqlalchemy import func
                
                # Count entries by type
                stmt = select(
                    CacheEntry.cache_type,
                    func.count(CacheEntry.id).label('count'),
                    func.sum(CacheEntry.data_size).label('total_size')
                ).group_by(CacheEntry.cache_type)
                
                result = await session.execute(stmt)
                entries = result.all()
                
                stats["database"] = {
                    "status": "healthy",
                    "entries_by_type": {
                        entry.cache_type: {
                            "count": entry.count,
                            "total_size": entry.total_size or 0
                        }
                        for entry in entries
                    }
                }
                
        except Exception as e:
            stats["database"] = {"status": "error", "error": str(e)}
        
        return stats
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired_entries()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            async with get_db_session() as session:
                from datetime import datetime
                
                # Delete expired entries
                stmt = delete(CacheEntry).where(
                    CacheEntry.expires_at < datetime.now()
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Cleaned up {result.rowcount} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    async def _store_in_database(
        self, 
        key: str, 
        value: Any, 
        metadata: Dict[str, Any], 
        cache_type: str
    ):
        """Store cache entry in database."""
        async with get_db_session() as session:
            # Check if entry exists
            stmt = select(CacheEntry).where(CacheEntry.cache_key == key)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            
            if entry:
                # Update existing entry
                entry.data = value if not isinstance(value, bytes) else None
                entry.binary_data = value if isinstance(value, bytes) else None
                entry.cache_type = cache_type
                entry.ttl = metadata.get("ttl")
                entry.tags = metadata.get("tags")
                entry.data_size = len(str(value)) if value else 0
                
                if metadata.get("expires_at"):
                    from datetime import datetime
                    entry.expires_at = datetime.fromisoformat(metadata["expires_at"])
            else:
                # Create new entry
                entry = CacheEntry(
                    cache_key=key,
                    cache_type=cache_type,
                    data=value if not isinstance(value, bytes) else None,
                    binary_data=value if isinstance(value, bytes) else None,
                    ttl=metadata.get("ttl"),
                    tags=metadata.get("tags"),
                    data_size=len(str(value)) if value else 0
                )
                
                if metadata.get("expires_at"):
                    from datetime import datetime
                    entry.expires_at = datetime.fromisoformat(metadata["expires_at"])
                
                session.add(entry)
            
            await session.commit()
    
    async def _delete_from_database(self, key: str) -> bool:
        """Delete cache entry from database."""
        async with get_db_session() as session:
            stmt = delete(CacheEntry).where(CacheEntry.cache_key == key)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0
    
    async def _update_access_metadata(self, key: str, cache_type: str):
        """Update access metadata for cache entry."""
        try:
            async with get_db_session() as session:
                stmt = select(CacheEntry).where(CacheEntry.cache_key == key)
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                
                if entry:
                    from datetime import datetime
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    await session.commit()
                    
        except Exception as e:
            logger.warning(f"Failed to update access metadata for key '{key}': {e}")
    
    async def _update_database_metadata(self, entry: CacheEntry, metadata: Dict[str, Any]):
        """Update database entry metadata."""
        try:
            if "access_count" in metadata:
                entry.access_count = metadata["access_count"]
            
            if "last_accessed" in metadata and metadata["last_accessed"]:
                from datetime import datetime
                entry.last_accessed = datetime.fromisoformat(metadata["last_accessed"])
                
        except Exception as e:
            logger.warning(f"Failed to update database metadata: {e}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache_manager():
    """Close global cache manager."""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None