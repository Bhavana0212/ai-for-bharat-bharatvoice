"""
Main authentication service for BharatVoice Assistant.

This module provides the main authentication service that coordinates
JWT tokens, password management, sessions, and MFA for secure user authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel
from fastapi import HTTPException, status

from bharatvoice.config import Settings
from bharatvoice.core.models import UserProfile
from .jwt_manager import JWTManager, TokenPayload
from .password_manager import PasswordManager, PasswordStrength
from .session_manager import SessionManager, SessionData
from .mfa_manager import MFAManager, MFASecret, MFAVerification


logger = structlog.get_logger(__name__)


class UserCredentials(BaseModel):
    """User credentials for authentication."""
    
    user_id: UUID
    username: str
    email: str
    password_hash: str
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    mfa_backup_codes: List[str] = []
    phone_number: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


class LoginRequest(BaseModel):
    """Login request model."""
    
    username: str
    password: str
    mfa_token: Optional[str] = None
    device_id: Optional[str] = None
    device_info: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response model."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    session_id: str
    requires_mfa: bool = False
    mfa_methods: List[str] = []


class RegisterRequest(BaseModel):
    """User registration request."""
    
    username: str
    email: str
    password: str
    phone_number: Optional[str] = None
    preferred_languages: List[str] = ["hi", "en-IN"]
    primary_language: str = "hi"


class RegisterResponse(BaseModel):
    """User registration response."""
    
    user_id: str
    username: str
    email: str
    is_verified: bool
    message: str


class AuthService:
    """Main authentication service."""
    
    def __init__(
        self,
        settings: Settings,
        redis_client=None,
        database=None,
        sms_service=None
    ):
        """
        Initialize authentication service.
        
        Args:
            settings: Application settings
            redis_client: Redis client for sessions and caching
            database: Database connection for user storage
            sms_service: SMS service for MFA
        """
        self.settings = settings
        self.redis_client = redis_client
        self.database = database
        
        # Initialize managers
        self.jwt_manager = JWTManager(settings)
        self.password_manager = PasswordManager()
        self.session_manager = SessionManager(settings, redis_client)
        self.mfa_manager = MFAManager(settings, redis_client, sms_service)
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
        logger.info("Authentication service initialized")
    
    async def register_user(self, request: RegisterRequest) -> RegisterResponse:
        """
        Register new user.
        
        Args:
            request: Registration request
            
        Returns:
            Registration response
            
        Raises:
            HTTPException: If registration fails
        """
        try:
            # Check if username/email already exists
            if await self._user_exists(request.username, request.email):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username or email already exists"
                )
            
            # Validate and hash password
            password_hash = self.password_manager.hash_password(request.password)
            
            # Create user credentials
            user_id = uuid4()
            user_credentials = UserCredentials(
                user_id=user_id,
                username=request.username,
                email=request.email,
                password_hash=password_hash,
                phone_number=request.phone_number,
                created_at=datetime.utcnow()
            )
            
            # Store user in database (mock implementation)
            await self._store_user_credentials(user_credentials)
            
            # Create user profile
            user_profile = UserProfile(
                user_id=user_id,
                preferred_languages=request.preferred_languages,
                primary_language=request.primary_language
            )
            
            # Store user profile (mock implementation)
            await self._store_user_profile(user_profile)
            
            logger.info(
                "User registered successfully",
                user_id=str(user_id),
                username=request.username
            )
            
            return RegisterResponse(
                user_id=str(user_id),
                username=request.username,
                email=request.email,
                is_verified=False,
                message="Registration successful. Please verify your email."
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("User registration failed", exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def authenticate_user(self, request: LoginRequest) -> LoginResponse:
        """
        Authenticate user and create session.
        
        Args:
            request: Login request
            
        Returns:
            Login response with token
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Get user credentials
            user_credentials = await self._get_user_credentials(request.username)
            if not user_credentials:
                logger.warning("Login attempt with unknown username", username=request.username)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if account is locked
            if self._is_account_locked(user_credentials):
                logger.warning("Login attempt on locked account", user_id=str(user_credentials.user_id))
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail="Account temporarily locked due to failed login attempts"
                )
            
            # Verify password
            if not self.password_manager.verify_password(request.password, user_credentials.password_hash):
                await self._handle_failed_login(user_credentials)
                logger.warning("Invalid password", user_id=str(user_credentials.user_id))
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if MFA is required
            if user_credentials.mfa_enabled:
                if not request.mfa_token:
                    logger.info("MFA required", user_id=str(user_credentials.user_id))
                    return LoginResponse(
                        access_token="",
                        expires_in=0,
                        user_id=str(user_credentials.user_id),
                        session_id="",
                        requires_mfa=True,
                        mfa_methods=["totp", "backup_code", "sms"] if user_credentials.phone_number else ["totp", "backup_code"]
                    )
                
                # Verify MFA token
                mfa_result = await self._verify_mfa_token(user_credentials, request.mfa_token)
                if not mfa_result.is_valid:
                    logger.warning("MFA verification failed", user_id=str(user_credentials.user_id))
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid MFA token"
                    )
            
            # Create session
            session_data = self.session_manager.create_session(
                user_id=user_credentials.user_id,
                username=user_credentials.username,
                device_id=request.device_id,
                device_info=request.device_info,
                ip_address=request.ip_address,
                user_agent=request.user_agent
            )
            
            # Create JWT token
            access_token = self.jwt_manager.create_access_token(
                user_id=user_credentials.user_id,
                username=user_credentials.username,
                session_id=session_data.session_id,
                device_id=request.device_id
            )
            
            # Update last login
            await self._update_last_login(user_credentials.user_id)
            
            logger.info(
                "User authenticated successfully",
                user_id=str(user_credentials.user_id),
                session_id=session_data.session_id
            )
            
            return LoginResponse(
                access_token=access_token,
                expires_in=self.settings.security.access_token_expire_minutes * 60,
                user_id=str(user_credentials.user_id),
                session_id=session_data.session_id
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authentication failed", exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service unavailable"
            )
    
    async def verify_token(self, token: str) -> TokenPayload:
        """
        Verify JWT token and return payload.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Verify JWT token
            token_payload = self.jwt_manager.verify_token(token)
            
            # Verify session is still active
            session_data = self.session_manager.get_session(token_payload.session_id)
            if not session_data or not session_data.is_active:
                logger.warning("Session not found or inactive", session_id=token_payload.session_id)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired"
                )
            
            return token_payload
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Token verification failed", exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def refresh_token(self, token: str) -> str:
        """
        Refresh JWT token.
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token
            
        Raises:
            HTTPException: If refresh fails
        """
        try:
            new_token = self.jwt_manager.refresh_token(token)
            logger.info("Token refreshed successfully")
            return new_token
            
        except Exception as e:
            logger.error("Token refresh failed", exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token refresh failed"
            )
    
    async def logout_user(self, token: str) -> bool:
        """
        Logout user and invalidate session.
        
        Args:
            token: JWT token
            
        Returns:
            True if logout successful
        """
        try:
            # Get token payload
            token_payload = self.jwt_manager.verify_token(token)
            
            # Invalidate session
            success = self.session_manager.invalidate_session(token_payload.session_id)
            
            logger.info(
                "User logged out",
                user_id=token_payload.user_id,
                session_id=token_payload.session_id
            )
            
            return success
            
        except Exception as e:
            logger.error("Logout failed", exc_info=e)
            return False
    
    async def change_password(self, user_id: UUID, current_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User identifier
            current_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
            
        Raises:
            HTTPException: If password change fails
        """
        try:
            # Get user credentials
            user_credentials = await self._get_user_credentials_by_id(user_id)
            if not user_credentials:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not self.password_manager.verify_password(current_password, user_credentials.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Current password is incorrect"
                )
            
            # Hash new password
            new_password_hash = self.password_manager.hash_password(new_password)
            
            # Update password in database
            await self._update_password(user_id, new_password_hash)
            
            # Invalidate all sessions except current one
            self.session_manager.invalidate_user_sessions(str(user_id))
            
            logger.info("Password changed successfully", user_id=str(user_id))
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Password change failed", user_id=str(user_id), exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed"
            )
    
    async def setup_mfa(self, user_id: UUID) -> Dict[str, Any]:
        """
        Setup MFA for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            MFA setup data including QR code
        """
        try:
            user_credentials = await self._get_user_credentials_by_id(user_id)
            if not user_credentials:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Generate MFA secret
            mfa_secret = self.mfa_manager.generate_secret(str(user_id), user_credentials.username)
            
            # Generate QR code
            qr_code = self.mfa_manager.get_qr_code(mfa_secret, user_credentials.username)
            
            # Store MFA secret (temporarily until verified)
            await self._store_temp_mfa_secret(user_id, mfa_secret)
            
            logger.info("MFA setup initiated", user_id=str(user_id))
            
            return {
                "secret": mfa_secret.secret,
                "qr_code": qr_code,
                "backup_codes": mfa_secret.backup_codes
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("MFA setup failed", user_id=str(user_id), exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MFA setup failed"
            )
    
    async def verify_mfa_setup(self, user_id: UUID, token: str) -> bool:
        """
        Verify MFA setup with TOTP token.
        
        Args:
            user_id: User identifier
            token: TOTP token
            
        Returns:
            True if MFA setup verified
        """
        try:
            # Get temporary MFA secret
            mfa_secret = await self._get_temp_mfa_secret(user_id)
            if not mfa_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No MFA setup in progress"
                )
            
            # Verify TOTP token
            is_valid = self.mfa_manager.verify_totp_setup(mfa_secret, token)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid TOTP token"
                )
            
            # Enable MFA for user
            await self._enable_mfa(user_id, mfa_secret)
            
            # Remove temporary secret
            await self._remove_temp_mfa_secret(user_id)
            
            logger.info("MFA setup completed", user_id=str(user_id))
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("MFA setup verification failed", user_id=str(user_id), exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MFA verification failed"
            )
    
    # Mock database operations (replace with actual database implementation)
    
    async def _user_exists(self, username: str, email: str) -> bool:
        """Check if user exists (mock implementation)."""
        # In production, query actual database
        return username == "existing_user" or email == "existing@example.com"
    
    async def _store_user_credentials(self, user_credentials: UserCredentials) -> None:
        """Store user credentials (mock implementation)."""
        # In production, store in actual database
        logger.info("User credentials stored (mock)", user_id=str(user_credentials.user_id))
    
    async def _store_user_profile(self, user_profile: UserProfile) -> None:
        """Store user profile (mock implementation)."""
        # In production, store in actual database
        logger.info("User profile stored (mock)", user_id=str(user_profile.user_id))
    
    async def _get_user_credentials(self, username: str) -> Optional[UserCredentials]:
        """Get user credentials by username (mock implementation)."""
        # In production, query actual database
        if username == "demo":
            return UserCredentials(
                user_id=uuid4(),
                username="demo",
                email="demo@example.com",
                password_hash=self.password_manager.hash_password("demo"),
                created_at=datetime.utcnow()
            )
        return None
    
    async def _get_user_credentials_by_id(self, user_id: UUID) -> Optional[UserCredentials]:
        """Get user credentials by ID (mock implementation)."""
        # In production, query actual database
        return UserCredentials(
            user_id=user_id,
            username="demo",
            email="demo@example.com",
            password_hash=self.password_manager.hash_password("demo"),
            created_at=datetime.utcnow()
        )
    
    async def _handle_failed_login(self, user_credentials: UserCredentials) -> None:
        """Handle failed login attempt (mock implementation)."""
        # In production, update database with failed attempt count
        logger.info("Failed login recorded (mock)", user_id=str(user_credentials.user_id))
    
    async def _update_last_login(self, user_id: UUID) -> None:
        """Update last login time (mock implementation)."""
        # In production, update database
        logger.info("Last login updated (mock)", user_id=str(user_id))
    
    async def _update_password(self, user_id: UUID, password_hash: str) -> None:
        """Update user password (mock implementation)."""
        # In production, update database
        logger.info("Password updated (mock)", user_id=str(user_id))
    
    async def _store_temp_mfa_secret(self, user_id: UUID, mfa_secret: MFASecret) -> None:
        """Store temporary MFA secret (mock implementation)."""
        # In production, store in Redis or database
        logger.info("Temp MFA secret stored (mock)", user_id=str(user_id))
    
    async def _get_temp_mfa_secret(self, user_id: UUID) -> Optional[MFASecret]:
        """Get temporary MFA secret (mock implementation)."""
        # In production, retrieve from Redis or database
        return None
    
    async def _remove_temp_mfa_secret(self, user_id: UUID) -> None:
        """Remove temporary MFA secret (mock implementation)."""
        # In production, remove from Redis or database
        logger.info("Temp MFA secret removed (mock)", user_id=str(user_id))
    
    async def _enable_mfa(self, user_id: UUID, mfa_secret: MFASecret) -> None:
        """Enable MFA for user (mock implementation)."""
        # In production, update database
        logger.info("MFA enabled (mock)", user_id=str(user_id))
    
    async def _verify_mfa_token(self, user_credentials: UserCredentials, token: str) -> MFAVerification:
        """Verify MFA token (mock implementation)."""
        # In production, use actual MFA secret from database
        if user_credentials.mfa_secret:
            return self.mfa_manager.verify_totp(
                str(user_credentials.user_id),
                user_credentials.mfa_secret,
                token
            )
        return MFAVerification(is_valid=False, method_used="totp")
    
    def _is_account_locked(self, user_credentials: UserCredentials) -> bool:
        """Check if account is locked."""
        if user_credentials.locked_until:
            return datetime.utcnow() < user_credentials.locked_until
        return user_credentials.failed_login_attempts >= self.max_login_attempts