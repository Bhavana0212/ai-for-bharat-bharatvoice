"""
JWT token management for BharatVoice Assistant.

This module provides JWT token creation, validation, and refresh functionality
with secure key management and token expiration handling.
"""

import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from uuid import UUID

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings


logger = structlog.get_logger(__name__)


class TokenPayload(BaseModel):
    """JWT token payload model."""
    
    user_id: str
    username: str
    session_id: str
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token tracking
    device_id: Optional[str] = None


class JWTManager:
    """JWT token manager for authentication."""
    
    def __init__(self, settings: Settings):
        """
        Initialize JWT manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.secret_key = settings.security.secret_key
        self.algorithm = settings.security.algorithm
        self.access_token_expire_minutes = settings.security.access_token_expire_minutes
        
        # Validate secret key
        if self.secret_key == "your-secret-key-change-in-production":
            logger.warning("Using default secret key - change in production!")
    
    def create_access_token(
        self,
        user_id: UUID,
        username: str,
        session_id: str,
        device_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User identifier
            username: Username
            session_id: Session identifier
            device_id: Device identifier
            expires_delta: Custom expiration time
            
        Returns:
            JWT access token
        """
        try:
            now = datetime.utcnow()
            
            if expires_delta:
                expire = now + expires_delta
            else:
                expire = now + timedelta(minutes=self.access_token_expire_minutes)
            
            # Create unique JWT ID
            jti = f"{user_id}_{session_id}_{int(now.timestamp())}"
            
            payload = {
                "user_id": str(user_id),
                "username": username,
                "session_id": session_id,
                "exp": expire,
                "iat": now,
                "jti": jti,
                "device_id": device_id
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            logger.info(
                "Access token created",
                user_id=str(user_id),
                session_id=session_id,
                expires_at=expire.isoformat()
            )
            
            return token
            
        except Exception as e:
            logger.error("Failed to create access token", exc_info=e)
            raise
    
    def verify_token(self, token: str) -> TokenPayload:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            jwt.ExpiredSignatureError: Token has expired
            jwt.InvalidTokenError: Token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Convert timestamps back to datetime objects
            payload["exp"] = datetime.fromtimestamp(payload["exp"])
            payload["iat"] = datetime.fromtimestamp(payload["iat"])
            
            token_payload = TokenPayload(**payload)
            
            logger.debug(
                "Token verified successfully",
                user_id=token_payload.user_id,
                session_id=token_payload.session_id
            )
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired", token=token[:20] + "...")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e), token=token[:20] + "...")
            raise
        except Exception as e:
            logger.error("Token verification failed", exc_info=e)
            raise
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh JWT token with new expiration.
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token
            
        Raises:
            jwt.ExpiredSignatureError: Token has expired
            jwt.InvalidTokenError: Token is invalid
        """
        try:
            # Verify current token (allow expired for refresh)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens for refresh
            )
            
            # Check if token is too old to refresh (more than 7 days)
            iat = datetime.fromtimestamp(payload["iat"])
            if datetime.utcnow() - iat > timedelta(days=7):
                logger.warning("Token too old to refresh", iat=iat.isoformat())
                raise jwt.InvalidTokenError("Token too old to refresh")
            
            # Create new token with same payload but new expiration
            new_token = self.create_access_token(
                user_id=UUID(payload["user_id"]),
                username=payload["username"],
                session_id=payload["session_id"],
                device_id=payload.get("device_id")
            )
            
            logger.info(
                "Token refreshed",
                user_id=payload["user_id"],
                session_id=payload["session_id"]
            )
            
            return new_token
            
        except jwt.InvalidTokenError:
            raise
        except Exception as e:
            logger.error("Token refresh failed", exc_info=e)
            raise
    
    def decode_token_unsafe(self, token: str) -> Dict[str, Any]:
        """
        Decode token without verification (for debugging/logging).
        
        Args:
            token: JWT token to decode
            
        Returns:
            Decoded payload (unverified)
        """
        try:
            return jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False}
            )
        except Exception:
            return {}
    
    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired without full verification.
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is expired
        """
        try:
            payload = self.decode_token_unsafe(token)
            if "exp" in payload:
                exp = datetime.fromtimestamp(payload["exp"])
                return datetime.utcnow() > exp
            return True
        except Exception:
            return True