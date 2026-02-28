<<<<<<< HEAD
"""
Session management for BharatVoice Assistant.

This module provides secure session management with Redis backend,
session timeout handling, and concurrent session control.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings


logger = structlog.get_logger(__name__)


class SessionData(BaseModel):
    """Session data model."""
    
    session_id: str
    user_id: str
    username: str
    device_id: Optional[str] = None
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = {}


class SessionManager:
    """Session manager with Redis backend."""
    
    def __init__(self, settings: Settings, redis_client=None):
        """
        Initialize session manager.
        
        Args:
            settings: Application settings
            redis_client: Redis client (optional, will create if None)
        """
        self.settings = settings
        self.redis_client = redis_client
        self.session_timeout = timedelta(minutes=settings.security.access_token_expire_minutes)
        self.max_sessions_per_user = 5  # Configurable limit
        
        # Session key prefixes
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"
        
        logger.info("Session manager initialized", timeout_minutes=self.session_timeout.total_seconds() / 60)
    
    def create_session(
        self,
        user_id: UUID,
        username: str,
        device_id: Optional[str] = None,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create new user session.
        
        Args:
            user_id: User identifier
            username: Username
            device_id: Device identifier
            device_info: Device information
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Additional session metadata
            
        Returns:
            Created session data
        """
        try:
            session_id = str(uuid4())
            now = datetime.utcnow()
            expires_at = now + self.session_timeout
            
            session_data = SessionData(
                session_id=session_id,
                user_id=str(user_id),
                username=username,
                device_id=device_id,
                device_info=device_info,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                is_active=True,
                metadata=metadata or {}
            )
            
            # Store session in Redis if available
            if self.redis_client:
                self._store_session_redis(session_data)
                self._add_to_user_sessions(str(user_id), session_id)
                self._cleanup_old_sessions(str(user_id))
            
            logger.info(
                "Session created",
                session_id=session_id,
                user_id=str(user_id),
                device_id=device_id,
                expires_at=expires_at.isoformat()
            )
            
            return session_data
            
        except Exception as e:
            logger.error("Failed to create session", exc_info=e)
            raise
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if found and valid, None otherwise
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available for session retrieval")
                return None
            
            session_key = f"{self.session_prefix}{session_id}"
            session_json = self.redis_client.get(session_key)
            
            if not session_json:
                logger.debug("Session not found", session_id=session_id)
                return None
            
            session_dict = json.loads(session_json)
            
            # Convert datetime strings back to datetime objects
            session_dict["created_at"] = datetime.fromisoformat(session_dict["created_at"])
            session_dict["last_accessed"] = datetime.fromisoformat(session_dict["last_accessed"])
            session_dict["expires_at"] = datetime.fromisoformat(session_dict["expires_at"])
            
            session_data = SessionData(**session_dict)
            
            # Check if session is expired
            if datetime.utcnow() > session_data.expires_at:
                logger.info("Session expired", session_id=session_id)
                self.invalidate_session(session_id)
                return None
            
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            self._store_session_redis(session_data)
            
            logger.debug("Session retrieved", session_id=session_id)
            return session_data
            
        except Exception as e:
            logger.error("Failed to get session", session_id=session_id, exc_info=e)
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if session was updated successfully
        """
        try:
            session_data = self.get_session(session_id)
            if not session_data:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(session_data, key):
                    setattr(session_data, key, value)
            
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            
            # Store updated session
            if self.redis_client:
                self._store_session_redis(session_data)
            
            logger.info("Session updated", session_id=session_id, updates=list(updates.keys()))
            return True
            
        except Exception as e:
            logger.error("Failed to update session", session_id=session_id, exc_info=e)
            return False
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was invalidated successfully
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available for session invalidation")
                return False
            
            # Get session to find user_id
            session_data = self.get_session(session_id)
            if session_data:
                self._remove_from_user_sessions(session_data.user_id, session_id)
            
            # Remove session from Redis
            session_key = f"{self.session_prefix}{session_id}"
            result = self.redis_client.delete(session_key)
            
            logger.info("Session invalidated", session_id=session_id, found=bool(result))
            return bool(result)
            
        except Exception as e:
            logger.error("Failed to invalidate session", session_id=session_id, exc_info=e)
            return False
    
    def invalidate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            user_id: User identifier
            except_session: Session ID to keep active (optional)
            
        Returns:
            Number of sessions invalidated
        """
        try:
            if not self.redis_client:
                return 0
            
            user_sessions = self._get_user_sessions(user_id)
            invalidated_count = 0
            
            for session_id in user_sessions:
                if session_id != except_session:
                    if self.invalidate_session(session_id):
                        invalidated_count += 1
            
            logger.info(
                "User sessions invalidated",
                user_id=user_id,
                count=invalidated_count,
                except_session=except_session
            )
            
            return invalidated_count
            
        except Exception as e:
            logger.error("Failed to invalidate user sessions", user_id=user_id, exc_info=e)
            return 0
    
    def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active session data
        """
        try:
            if not self.redis_client:
                return []
            
            session_ids = self._get_user_sessions(user_id)
            sessions = []
            
            for session_id in session_ids:
                session_data = self.get_session(session_id)
                if session_data and session_data.is_active:
                    sessions.append(session_data)
            
            logger.debug("Retrieved user sessions", user_id=user_id, count=len(sessions))
            return sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions", user_id=user_id, exc_info=e)
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            if not self.redis_client:
                return 0
            
            # This would require scanning all session keys
            # For now, we rely on Redis TTL and lazy cleanup
            logger.info("Session cleanup completed")
            return 0
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", exc_info=e)
            return 0
    
    def _store_session_redis(self, session_data: SessionData) -> None:
        """Store session data in Redis."""
        if not self.redis_client:
            return
        
        session_key = f"{self.session_prefix}{session_data.session_id}"
        
        # Convert datetime objects to ISO strings for JSON serialization
        session_dict = session_data.dict()
        session_dict["created_at"] = session_data.created_at.isoformat()
        session_dict["last_accessed"] = session_data.last_accessed.isoformat()
        session_dict["expires_at"] = session_data.expires_at.isoformat()
        
        session_json = json.dumps(session_dict)
        
        # Set with TTL based on expiration
        ttl_seconds = int((session_data.expires_at - datetime.utcnow()).total_seconds())
        if ttl_seconds > 0:
            self.redis_client.setex(session_key, ttl_seconds, session_json)
    
    def _get_user_sessions(self, user_id: str) -> List[str]:
        """Get list of session IDs for a user."""
        if not self.redis_client:
            return []
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = self.redis_client.smembers(user_sessions_key)
        return [sid.decode() if isinstance(sid, bytes) else sid for sid in session_ids]
    
    def _add_to_user_sessions(self, user_id: str, session_id: str) -> None:
        """Add session ID to user's session set."""
        if not self.redis_client:
            return
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.sadd(user_sessions_key, session_id)
        
        # Set TTL for user sessions set
        self.redis_client.expire(user_sessions_key, int(self.session_timeout.total_seconds()) * 2)
    
    def _remove_from_user_sessions(self, user_id: str, session_id: str) -> None:
        """Remove session ID from user's session set."""
        if not self.redis_client:
            return
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.srem(user_sessions_key, session_id)
    
    def _cleanup_old_sessions(self, user_id: str) -> None:
        """Clean up old sessions if user has too many."""
        if not self.redis_client:
            return
        
        sessions = self.get_user_sessions(user_id)
        if len(sessions) > self.max_sessions_per_user:
            # Sort by last accessed time and remove oldest
            sessions.sort(key=lambda s: s.last_accessed)
            sessions_to_remove = sessions[:-self.max_sessions_per_user]
            
            for session in sessions_to_remove:
                self.invalidate_session(session.session_id)
            
            logger.info(
                "Cleaned up old sessions",
                user_id=user_id,
                removed_count=len(sessions_to_remove)
=======
"""
Session management for BharatVoice Assistant.

This module provides secure session management with Redis backend,
session timeout handling, and concurrent session control.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings


logger = structlog.get_logger(__name__)


class SessionData(BaseModel):
    """Session data model."""
    
    session_id: str
    user_id: str
    username: str
    device_id: Optional[str] = None
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = {}


class SessionManager:
    """Session manager with Redis backend."""
    
    def __init__(self, settings: Settings, redis_client=None):
        """
        Initialize session manager.
        
        Args:
            settings: Application settings
            redis_client: Redis client (optional, will create if None)
        """
        self.settings = settings
        self.redis_client = redis_client
        self.session_timeout = timedelta(minutes=settings.security.access_token_expire_minutes)
        self.max_sessions_per_user = 5  # Configurable limit
        
        # Session key prefixes
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"
        
        logger.info("Session manager initialized", timeout_minutes=self.session_timeout.total_seconds() / 60)
    
    def create_session(
        self,
        user_id: UUID,
        username: str,
        device_id: Optional[str] = None,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionData:
        """
        Create new user session.
        
        Args:
            user_id: User identifier
            username: Username
            device_id: Device identifier
            device_info: Device information
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Additional session metadata
            
        Returns:
            Created session data
        """
        try:
            session_id = str(uuid4())
            now = datetime.utcnow()
            expires_at = now + self.session_timeout
            
            session_data = SessionData(
                session_id=session_id,
                user_id=str(user_id),
                username=username,
                device_id=device_id,
                device_info=device_info,
                ip_address=ip_address,
                user_agent=user_agent,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                is_active=True,
                metadata=metadata or {}
            )
            
            # Store session in Redis if available
            if self.redis_client:
                self._store_session_redis(session_data)
                self._add_to_user_sessions(str(user_id), session_id)
                self._cleanup_old_sessions(str(user_id))
            
            logger.info(
                "Session created",
                session_id=session_id,
                user_id=str(user_id),
                device_id=device_id,
                expires_at=expires_at.isoformat()
            )
            
            return session_data
            
        except Exception as e:
            logger.error("Failed to create session", exc_info=e)
            raise
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if found and valid, None otherwise
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available for session retrieval")
                return None
            
            session_key = f"{self.session_prefix}{session_id}"
            session_json = self.redis_client.get(session_key)
            
            if not session_json:
                logger.debug("Session not found", session_id=session_id)
                return None
            
            session_dict = json.loads(session_json)
            
            # Convert datetime strings back to datetime objects
            session_dict["created_at"] = datetime.fromisoformat(session_dict["created_at"])
            session_dict["last_accessed"] = datetime.fromisoformat(session_dict["last_accessed"])
            session_dict["expires_at"] = datetime.fromisoformat(session_dict["expires_at"])
            
            session_data = SessionData(**session_dict)
            
            # Check if session is expired
            if datetime.utcnow() > session_data.expires_at:
                logger.info("Session expired", session_id=session_id)
                self.invalidate_session(session_id)
                return None
            
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            self._store_session_redis(session_data)
            
            logger.debug("Session retrieved", session_id=session_id)
            return session_data
            
        except Exception as e:
            logger.error("Failed to get session", session_id=session_id, exc_info=e)
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if session was updated successfully
        """
        try:
            session_data = self.get_session(session_id)
            if not session_data:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(session_data, key):
                    setattr(session_data, key, value)
            
            # Update last accessed time
            session_data.last_accessed = datetime.utcnow()
            
            # Store updated session
            if self.redis_client:
                self._store_session_redis(session_data)
            
            logger.info("Session updated", session_id=session_id, updates=list(updates.keys()))
            return True
            
        except Exception as e:
            logger.error("Failed to update session", session_id=session_id, exc_info=e)
            return False
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was invalidated successfully
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available for session invalidation")
                return False
            
            # Get session to find user_id
            session_data = self.get_session(session_id)
            if session_data:
                self._remove_from_user_sessions(session_data.user_id, session_id)
            
            # Remove session from Redis
            session_key = f"{self.session_prefix}{session_id}"
            result = self.redis_client.delete(session_key)
            
            logger.info("Session invalidated", session_id=session_id, found=bool(result))
            return bool(result)
            
        except Exception as e:
            logger.error("Failed to invalidate session", session_id=session_id, exc_info=e)
            return False
    
    def invalidate_user_sessions(self, user_id: str, except_session: Optional[str] = None) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            user_id: User identifier
            except_session: Session ID to keep active (optional)
            
        Returns:
            Number of sessions invalidated
        """
        try:
            if not self.redis_client:
                return 0
            
            user_sessions = self._get_user_sessions(user_id)
            invalidated_count = 0
            
            for session_id in user_sessions:
                if session_id != except_session:
                    if self.invalidate_session(session_id):
                        invalidated_count += 1
            
            logger.info(
                "User sessions invalidated",
                user_id=user_id,
                count=invalidated_count,
                except_session=except_session
            )
            
            return invalidated_count
            
        except Exception as e:
            logger.error("Failed to invalidate user sessions", user_id=user_id, exc_info=e)
            return 0
    
    def get_user_sessions(self, user_id: str) -> List[SessionData]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active session data
        """
        try:
            if not self.redis_client:
                return []
            
            session_ids = self._get_user_sessions(user_id)
            sessions = []
            
            for session_id in session_ids:
                session_data = self.get_session(session_id)
                if session_data and session_data.is_active:
                    sessions.append(session_data)
            
            logger.debug("Retrieved user sessions", user_id=user_id, count=len(sessions))
            return sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions", user_id=user_id, exc_info=e)
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            if not self.redis_client:
                return 0
            
            # This would require scanning all session keys
            # For now, we rely on Redis TTL and lazy cleanup
            logger.info("Session cleanup completed")
            return 0
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", exc_info=e)
            return 0
    
    def _store_session_redis(self, session_data: SessionData) -> None:
        """Store session data in Redis."""
        if not self.redis_client:
            return
        
        session_key = f"{self.session_prefix}{session_data.session_id}"
        
        # Convert datetime objects to ISO strings for JSON serialization
        session_dict = session_data.dict()
        session_dict["created_at"] = session_data.created_at.isoformat()
        session_dict["last_accessed"] = session_data.last_accessed.isoformat()
        session_dict["expires_at"] = session_data.expires_at.isoformat()
        
        session_json = json.dumps(session_dict)
        
        # Set with TTL based on expiration
        ttl_seconds = int((session_data.expires_at - datetime.utcnow()).total_seconds())
        if ttl_seconds > 0:
            self.redis_client.setex(session_key, ttl_seconds, session_json)
    
    def _get_user_sessions(self, user_id: str) -> List[str]:
        """Get list of session IDs for a user."""
        if not self.redis_client:
            return []
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        session_ids = self.redis_client.smembers(user_sessions_key)
        return [sid.decode() if isinstance(sid, bytes) else sid for sid in session_ids]
    
    def _add_to_user_sessions(self, user_id: str, session_id: str) -> None:
        """Add session ID to user's session set."""
        if not self.redis_client:
            return
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.sadd(user_sessions_key, session_id)
        
        # Set TTL for user sessions set
        self.redis_client.expire(user_sessions_key, int(self.session_timeout.total_seconds()) * 2)
    
    def _remove_from_user_sessions(self, user_id: str, session_id: str) -> None:
        """Remove session ID from user's session set."""
        if not self.redis_client:
            return
        
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        self.redis_client.srem(user_sessions_key, session_id)
    
    def _cleanup_old_sessions(self, user_id: str) -> None:
        """Clean up old sessions if user has too many."""
        if not self.redis_client:
            return
        
        sessions = self.get_user_sessions(user_id)
        if len(sessions) > self.max_sessions_per_user:
            # Sort by last accessed time and remove oldest
            sessions.sort(key=lambda s: s.last_accessed)
            sessions_to_remove = sessions[:-self.max_sessions_per_user]
            
            for session in sessions_to_remove:
                self.invalidate_session(session.session_id)
            
            logger.info(
                "Cleaned up old sessions",
                user_id=user_id,
                removed_count=len(sessions_to_remove)
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
            )