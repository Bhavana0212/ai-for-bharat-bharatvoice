<<<<<<< HEAD
"""
Cache invalidation and management strategies.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Base class for cache strategies."""
    
    @abstractmethod
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be evicted.
        
        Args:
            key: Cache key
            metadata: Cache entry metadata
            
        Returns:
            True if entry should be evicted
        """
        pass
    
    @abstractmethod
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache access event.
        
        Args:
            key: Cache key
            metadata: Cache entry metadata
            
        Returns:
            Updated metadata
        """
        pass
    
    @abstractmethod
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache set event.
        
        Args:
            key: Cache key
            value: Cache value
            metadata: Cache entry metadata
            
        Returns:
            Updated metadata
        """
        pass


class TTLStrategy(CacheStrategy):
    """Time-to-live based cache strategy."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize TTL strategy.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry has expired."""
        expires_at = metadata.get("expires_at")
        if not expires_at:
            return False
        
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        
        return datetime.now() > expires_at
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update last accessed time."""
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Set expiration time."""
        ttl = metadata.get("ttl", self.default_ttl)
        metadata["expires_at"] = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata


class LRUStrategy(CacheStrategy):
    """Least Recently Used cache strategy."""
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize LRU strategy.
        
        Args:
            max_entries: Maximum number of cache entries
        """
        self.max_entries = max_entries
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry should be evicted based on LRU."""
        async with self._lock:
            if len(self._access_order) <= self.max_entries:
                return False
            
            # Evict least recently used entries
            return key in self._access_order[:-self.max_entries]
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update access order."""
        async with self._lock:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add to access order."""
        async with self._lock:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata
    
    async def get_eviction_candidates(self) -> List[str]:
        """Get keys that should be evicted."""
        async with self._lock:
            if len(self._access_order) <= self.max_entries:
                return []
            
            return self._access_order[:-self.max_entries]


class TagBasedStrategy(CacheStrategy):
    """Tag-based cache invalidation strategy."""
    
    def __init__(self):
        """Initialize tag-based strategy."""
        self._tag_to_keys: Dict[str, Set[str]] = {}
        self._key_to_tags: Dict[str, Set[str]] = {}
        self._invalidated_tags: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry should be evicted based on tags."""
        async with self._lock:
            key_tags = self._key_to_tags.get(key, set())
            return bool(key_tags.intersection(self._invalidated_tags))
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update access metadata."""
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register tags for key."""
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        
        async with self._lock:
            # Remove old tag associations
            if key in self._key_to_tags:
                for old_tag in self._key_to_tags[key]:
                    if old_tag in self._tag_to_keys:
                        self._tag_to_keys[old_tag].discard(key)
            
            # Add new tag associations
            self._key_to_tags[key] = set(tags)
            for tag in tags:
                if tag not in self._tag_to_keys:
                    self._tag_to_keys[tag] = set()
                self._tag_to_keys[tag].add(key)
        
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata
    
    async def invalidate_tag(self, tag: str) -> List[str]:
        """
        Invalidate all cache entries with the given tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            List of keys that were invalidated
        """
        async with self._lock:
            self._invalidated_tags.add(tag)
            keys = list(self._tag_to_keys.get(tag, set()))
            
            # Clean up tag associations
            if tag in self._tag_to_keys:
                for key in self._tag_to_keys[tag]:
                    if key in self._key_to_tags:
                        self._key_to_tags[key].discard(tag)
                del self._tag_to_keys[tag]
            
            return keys
    
    async def clear_invalidated_tags(self):
        """Clear the set of invalidated tags."""
        async with self._lock:
            self._invalidated_tags.clear()
    
    async def get_keys_by_tag(self, tag: str) -> List[str]:
        """
        Get all keys associated with a tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of keys with the tag
        """
        async with self._lock:
            return list(self._tag_to_keys.get(tag, set()))
    
    async def get_tags_by_key(self, key: str) -> List[str]:
        """
        Get all tags associated with a key.
        
        Args:
            key: Key to search for
            
        Returns:
            List of tags for the key
        """
        async with self._lock:
            return list(self._key_to_tags.get(key, set()))


class CompositeStrategy(CacheStrategy):
    """Composite strategy that combines multiple strategies."""
    
    def __init__(self, strategies: List[CacheStrategy]):
        """
        Initialize composite strategy.
        
        Args:
            strategies: List of strategies to combine
        """
        self.strategies = strategies
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if any strategy says to evict."""
        for strategy in self.strategies:
            if await strategy.should_evict(key, metadata):
                return True
        return False
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all strategies on access."""
        for strategy in self.strategies:
            metadata = await strategy.on_access(key, metadata)
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all strategies on set."""
        for strategy in self.strategies:
            metadata = await strategy.on_set(key, value, metadata)
=======
"""
Cache invalidation and management strategies.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Base class for cache strategies."""
    
    @abstractmethod
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if a cache entry should be evicted.
        
        Args:
            key: Cache key
            metadata: Cache entry metadata
            
        Returns:
            True if entry should be evicted
        """
        pass
    
    @abstractmethod
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache access event.
        
        Args:
            key: Cache key
            metadata: Cache entry metadata
            
        Returns:
            Updated metadata
        """
        pass
    
    @abstractmethod
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle cache set event.
        
        Args:
            key: Cache key
            value: Cache value
            metadata: Cache entry metadata
            
        Returns:
            Updated metadata
        """
        pass


class TTLStrategy(CacheStrategy):
    """Time-to-live based cache strategy."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize TTL strategy.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry has expired."""
        expires_at = metadata.get("expires_at")
        if not expires_at:
            return False
        
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        
        return datetime.now() > expires_at
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update last accessed time."""
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Set expiration time."""
        ttl = metadata.get("ttl", self.default_ttl)
        metadata["expires_at"] = (datetime.now() + timedelta(seconds=ttl)).isoformat()
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata


class LRUStrategy(CacheStrategy):
    """Least Recently Used cache strategy."""
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize LRU strategy.
        
        Args:
            max_entries: Maximum number of cache entries
        """
        self.max_entries = max_entries
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry should be evicted based on LRU."""
        async with self._lock:
            if len(self._access_order) <= self.max_entries:
                return False
            
            # Evict least recently used entries
            return key in self._access_order[:-self.max_entries]
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update access order."""
        async with self._lock:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add to access order."""
        async with self._lock:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata
    
    async def get_eviction_candidates(self) -> List[str]:
        """Get keys that should be evicted."""
        async with self._lock:
            if len(self._access_order) <= self.max_entries:
                return []
            
            return self._access_order[:-self.max_entries]


class TagBasedStrategy(CacheStrategy):
    """Tag-based cache invalidation strategy."""
    
    def __init__(self):
        """Initialize tag-based strategy."""
        self._tag_to_keys: Dict[str, Set[str]] = {}
        self._key_to_tags: Dict[str, Set[str]] = {}
        self._invalidated_tags: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if entry should be evicted based on tags."""
        async with self._lock:
            key_tags = self._key_to_tags.get(key, set())
            return bool(key_tags.intersection(self._invalidated_tags))
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update access metadata."""
        metadata["last_accessed"] = datetime.now().isoformat()
        metadata["access_count"] = metadata.get("access_count", 0) + 1
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Register tags for key."""
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        
        async with self._lock:
            # Remove old tag associations
            if key in self._key_to_tags:
                for old_tag in self._key_to_tags[key]:
                    if old_tag in self._tag_to_keys:
                        self._tag_to_keys[old_tag].discard(key)
            
            # Add new tag associations
            self._key_to_tags[key] = set(tags)
            for tag in tags:
                if tag not in self._tag_to_keys:
                    self._tag_to_keys[tag] = set()
                self._tag_to_keys[tag].add(key)
        
        metadata["created_at"] = datetime.now().isoformat()
        metadata["access_count"] = 0
        return metadata
    
    async def invalidate_tag(self, tag: str) -> List[str]:
        """
        Invalidate all cache entries with the given tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            List of keys that were invalidated
        """
        async with self._lock:
            self._invalidated_tags.add(tag)
            keys = list(self._tag_to_keys.get(tag, set()))
            
            # Clean up tag associations
            if tag in self._tag_to_keys:
                for key in self._tag_to_keys[tag]:
                    if key in self._key_to_tags:
                        self._key_to_tags[key].discard(tag)
                del self._tag_to_keys[tag]
            
            return keys
    
    async def clear_invalidated_tags(self):
        """Clear the set of invalidated tags."""
        async with self._lock:
            self._invalidated_tags.clear()
    
    async def get_keys_by_tag(self, tag: str) -> List[str]:
        """
        Get all keys associated with a tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of keys with the tag
        """
        async with self._lock:
            return list(self._tag_to_keys.get(tag, set()))
    
    async def get_tags_by_key(self, key: str) -> List[str]:
        """
        Get all tags associated with a key.
        
        Args:
            key: Key to search for
            
        Returns:
            List of tags for the key
        """
        async with self._lock:
            return list(self._key_to_tags.get(key, set()))


class CompositeStrategy(CacheStrategy):
    """Composite strategy that combines multiple strategies."""
    
    def __init__(self, strategies: List[CacheStrategy]):
        """
        Initialize composite strategy.
        
        Args:
            strategies: List of strategies to combine
        """
        self.strategies = strategies
    
    async def should_evict(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if any strategy says to evict."""
        for strategy in self.strategies:
            if await strategy.should_evict(key, metadata):
                return True
        return False
    
    async def on_access(self, key: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all strategies on access."""
        for strategy in self.strategies:
            metadata = await strategy.on_access(key, metadata)
        return metadata
    
    async def on_set(self, key: str, value: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all strategies on set."""
        for strategy in self.strategies:
            metadata = await strategy.on_set(key, value, metadata)
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        return metadata