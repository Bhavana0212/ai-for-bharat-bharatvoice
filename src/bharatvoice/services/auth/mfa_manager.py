"""
Multi-Factor Authentication (MFA) manager for BharatVoice Assistant.

This module provides TOTP-based MFA, backup codes, and SMS verification
for enhanced security with Indian mobile number support.
"""

import secrets
import string
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pyotp
import qrcode
from io import BytesIO
import base64
import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings


logger = structlog.get_logger(__name__)


class MFASecret(BaseModel):
    """MFA secret configuration."""
    
    user_id: str
    secret: str
    backup_codes: List[str]
    is_enabled: bool = False
    created_at: datetime
    last_used: Optional[datetime] = None


class MFAVerification(BaseModel):
    """MFA verification result."""
    
    is_valid: bool
    method_used: str  # "totp", "backup_code", "sms"
    remaining_attempts: Optional[int] = None
    next_attempt_at: Optional[datetime] = None


class SMSVerification(BaseModel):
    """SMS verification data."""
    
    phone_number: str
    code: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3


class MFAManager:
    """Multi-Factor Authentication manager."""
    
    def __init__(self, settings: Settings, redis_client=None, sms_service=None):
        """
        Initialize MFA manager.
        
        Args:
            settings: Application settings
            redis_client: Redis client for storing temporary data
            sms_service: SMS service for sending verification codes
        """
        self.settings = settings
        self.redis_client = redis_client
        self.sms_service = sms_service
        self.app_name = settings.app_name
        
        # MFA configuration
        self.totp_window = 1  # Allow 1 time step tolerance
        self.backup_code_length = 8
        self.backup_code_count = 10
        self.sms_code_length = 6
        self.sms_code_expiry = timedelta(minutes=5)
        self.max_sms_attempts = 3
        
        # Rate limiting
        self.max_verification_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        
        logger.info("MFA manager initialized")
    
    def generate_secret(self, user_id: str, username: str) -> MFASecret:
        """
        Generate MFA secret and backup codes for user.
        
        Args:
            user_id: User identifier
            username: Username for QR code
            
        Returns:
            MFA secret configuration
        """
        try:
            # Generate TOTP secret
            secret = pyotp.random_base32()
            
            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            
            mfa_secret = MFASecret(
                user_id=user_id,
                secret=secret,
                backup_codes=backup_codes,
                is_enabled=False,
                created_at=datetime.utcnow()
            )
            
            logger.info("MFA secret generated", user_id=user_id)
            return mfa_secret
            
        except Exception as e:
            logger.error("Failed to generate MFA secret", user_id=user_id, exc_info=e)
            raise
    
    def get_qr_code(self, mfa_secret: MFASecret, username: str) -> str:
        """
        Generate QR code for TOTP setup.
        
        Args:
            mfa_secret: MFA secret configuration
            username: Username for QR code
            
        Returns:
            Base64 encoded QR code image
        """
        try:
            # Create TOTP URI
            totp = pyotp.TOTP(mfa_secret.secret)
            provisioning_uri = totp.provisioning_uri(
                name=username,
                issuer_name=self.app_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info("QR code generated", user_id=mfa_secret.user_id)
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error("Failed to generate QR code", exc_info=e)
            raise
    
    def verify_totp_setup(self, mfa_secret: MFASecret, token: str) -> bool:
        """
        Verify TOTP token during setup.
        
        Args:
            mfa_secret: MFA secret configuration
            token: TOTP token to verify
            
        Returns:
            True if token is valid
        """
        try:
            totp = pyotp.TOTP(mfa_secret.secret)
            is_valid = totp.verify(token, valid_window=self.totp_window)
            
            logger.info(
                "TOTP setup verification",
                user_id=mfa_secret.user_id,
                is_valid=is_valid
            )
            
            return is_valid
            
        except Exception as e:
            logger.error("TOTP setup verification failed", exc_info=e)
            return False
    
    def verify_totp(self, user_id: str, secret: str, token: str) -> MFAVerification:
        """
        Verify TOTP token for authentication.
        
        Args:
            user_id: User identifier
            secret: TOTP secret
            token: TOTP token to verify
            
        Returns:
            MFA verification result
        """
        try:
            # Check rate limiting
            if self._is_rate_limited(user_id, "totp"):
                return MFAVerification(
                    is_valid=False,
                    method_used="totp",
                    remaining_attempts=0,
                    next_attempt_at=self._get_next_attempt_time(user_id, "totp")
                )
            
            # Verify TOTP
            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(token, valid_window=self.totp_window)
            
            if is_valid:
                self._reset_rate_limit(user_id, "totp")
                logger.info("TOTP verification successful", user_id=user_id)
            else:
                self._increment_failed_attempts(user_id, "totp")
                logger.warning("TOTP verification failed", user_id=user_id)
            
            return MFAVerification(
                is_valid=is_valid,
                method_used="totp",
                remaining_attempts=self._get_remaining_attempts(user_id, "totp")
            )
            
        except Exception as e:
            logger.error("TOTP verification error", user_id=user_id, exc_info=e)
            return MFAVerification(is_valid=False, method_used="totp")
    
    def verify_backup_code(self, user_id: str, backup_codes: List[str], code: str) -> MFAVerification:
        """
        Verify backup code for authentication.
        
        Args:
            user_id: User identifier
            backup_codes: List of valid backup codes
            code: Backup code to verify
            
        Returns:
            MFA verification result
        """
        try:
            # Check rate limiting
            if self._is_rate_limited(user_id, "backup"):
                return MFAVerification(
                    is_valid=False,
                    method_used="backup_code",
                    remaining_attempts=0,
                    next_attempt_at=self._get_next_attempt_time(user_id, "backup")
                )
            
            # Verify backup code
            code_upper = code.upper().replace('-', '').replace(' ', '')
            is_valid = any(bc.upper().replace('-', '') == code_upper for bc in backup_codes)
            
            if is_valid:
                self._reset_rate_limit(user_id, "backup")
                logger.info("Backup code verification successful", user_id=user_id)
                # Note: In production, you should remove the used backup code
            else:
                self._increment_failed_attempts(user_id, "backup")
                logger.warning("Backup code verification failed", user_id=user_id)
            
            return MFAVerification(
                is_valid=is_valid,
                method_used="backup_code",
                remaining_attempts=self._get_remaining_attempts(user_id, "backup")
            )
            
        except Exception as e:
            logger.error("Backup code verification error", user_id=user_id, exc_info=e)
            return MFAVerification(is_valid=False, method_used="backup_code")
    
    def send_sms_code(self, user_id: str, phone_number: str) -> bool:
        """
        Send SMS verification code.
        
        Args:
            user_id: User identifier
            phone_number: Indian mobile number (+91XXXXXXXXXX)
            
        Returns:
            True if SMS was sent successfully
        """
        try:
            # Validate Indian mobile number format
            if not self._is_valid_indian_mobile(phone_number):
                logger.warning("Invalid Indian mobile number", phone_number=phone_number[:5] + "...")
                return False
            
            # Check rate limiting
            if self._is_rate_limited(user_id, "sms"):
                logger.warning("SMS rate limited", user_id=user_id)
                return False
            
            # Generate verification code
            code = self._generate_sms_code()
            
            # Store verification data
            sms_verification = SMSVerification(
                phone_number=phone_number,
                code=code,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + self.sms_code_expiry
            )
            
            if self.redis_client:
                self._store_sms_verification(user_id, sms_verification)
            
            # Send SMS (mock implementation)
            if self.sms_service:
                message = f"Your {self.app_name} verification code is: {code}. Valid for 5 minutes."
                success = self.sms_service.send_sms(phone_number, message)
            else:
                # Mock SMS sending
                logger.info(f"SMS Code (MOCK): {code}", user_id=user_id, phone=phone_number[:5] + "...")
                success = True
            
            if success:
                logger.info("SMS verification code sent", user_id=user_id)
            else:
                logger.error("Failed to send SMS verification code", user_id=user_id)
            
            return success
            
        except Exception as e:
            logger.error("SMS code sending failed", user_id=user_id, exc_info=e)
            return False
    
    def verify_sms_code(self, user_id: str, code: str) -> MFAVerification:
        """
        Verify SMS verification code.
        
        Args:
            user_id: User identifier
            code: SMS verification code
            
        Returns:
            MFA verification result
        """
        try:
            if not self.redis_client:
                logger.error("Redis client not available for SMS verification")
                return MFAVerification(is_valid=False, method_used="sms")
            
            # Get stored verification data
            sms_verification = self._get_sms_verification(user_id)
            if not sms_verification:
                logger.warning("No SMS verification found", user_id=user_id)
                return MFAVerification(is_valid=False, method_used="sms")
            
            # Check if expired
            if datetime.utcnow() > sms_verification.expires_at:
                logger.warning("SMS verification code expired", user_id=user_id)
                self._remove_sms_verification(user_id)
                return MFAVerification(is_valid=False, method_used="sms")
            
            # Check attempts
            if sms_verification.attempts >= sms_verification.max_attempts:
                logger.warning("SMS verification max attempts reached", user_id=user_id)
                self._remove_sms_verification(user_id)
                return MFAVerification(is_valid=False, method_used="sms", remaining_attempts=0)
            
            # Verify code
            is_valid = sms_verification.code == code
            
            if is_valid:
                self._remove_sms_verification(user_id)
                logger.info("SMS verification successful", user_id=user_id)
            else:
                sms_verification.attempts += 1
                self._store_sms_verification(user_id, sms_verification)
                logger.warning("SMS verification failed", user_id=user_id)
            
            return MFAVerification(
                is_valid=is_valid,
                method_used="sms",
                remaining_attempts=sms_verification.max_attempts - sms_verification.attempts
            )
            
        except Exception as e:
            logger.error("SMS verification error", user_id=user_id, exc_info=e)
            return MFAVerification(is_valid=False, method_used="sms")
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes."""
        codes = []
        for _ in range(self.backup_code_count):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) 
                          for _ in range(self.backup_code_length))
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)
        return codes
    
    def _generate_sms_code(self) -> str:
        """Generate SMS verification code."""
        return ''.join(secrets.choice(string.digits) for _ in range(self.sms_code_length))
    
    def _is_valid_indian_mobile(self, phone_number: str) -> bool:
        """Validate Indian mobile number format."""
        # Remove spaces and dashes
        clean_number = phone_number.replace(' ', '').replace('-', '')
        
        # Check format: +91XXXXXXXXXX or 91XXXXXXXXXX or XXXXXXXXXX
        if clean_number.startswith('+91'):
            clean_number = clean_number[3:]
        elif clean_number.startswith('91'):
            clean_number = clean_number[2:]
        
        # Should be 10 digits starting with 6, 7, 8, or 9
        return (len(clean_number) == 10 and 
                clean_number.isdigit() and 
                clean_number[0] in '6789')
    
    def _is_rate_limited(self, user_id: str, method: str) -> bool:
        """Check if user is rate limited for MFA method."""
        if not self.redis_client:
            return False
        
        key = f"mfa_attempts:{user_id}:{method}"
        attempts = self.redis_client.get(key)
        return attempts and int(attempts) >= self.max_verification_attempts
    
    def _increment_failed_attempts(self, user_id: str, method: str) -> None:
        """Increment failed MFA attempts."""
        if not self.redis_client:
            return
        
        key = f"mfa_attempts:{user_id}:{method}"
        current = self.redis_client.get(key)
        attempts = int(current) + 1 if current else 1
        
        self.redis_client.setex(key, int(self.lockout_duration.total_seconds()), attempts)
    
    def _reset_rate_limit(self, user_id: str, method: str) -> None:
        """Reset rate limit for user and method."""
        if not self.redis_client:
            return
        
        key = f"mfa_attempts:{user_id}:{method}"
        self.redis_client.delete(key)
    
    def _get_remaining_attempts(self, user_id: str, method: str) -> int:
        """Get remaining MFA attempts."""
        if not self.redis_client:
            return self.max_verification_attempts
        
        key = f"mfa_attempts:{user_id}:{method}"
        attempts = self.redis_client.get(key)
        used_attempts = int(attempts) if attempts else 0
        return max(0, self.max_verification_attempts - used_attempts)
    
    def _get_next_attempt_time(self, user_id: str, method: str) -> Optional[datetime]:
        """Get next allowed attempt time."""
        if not self.redis_client:
            return None
        
        key = f"mfa_attempts:{user_id}:{method}"
        ttl = self.redis_client.ttl(key)
        if ttl > 0:
            return datetime.utcnow() + timedelta(seconds=ttl)
        return None
    
    def _store_sms_verification(self, user_id: str, sms_verification: SMSVerification) -> None:
        """Store SMS verification data in Redis."""
        if not self.redis_client:
            return
        
        key = f"sms_verification:{user_id}"
        data = sms_verification.dict()
        data["created_at"] = sms_verification.created_at.isoformat()
        data["expires_at"] = sms_verification.expires_at.isoformat()
        
        ttl = int(self.sms_code_expiry.total_seconds())
        self.redis_client.setex(key, ttl, str(data))
    
    def _get_sms_verification(self, user_id: str) -> Optional[SMSVerification]:
        """Get SMS verification data from Redis."""
        if not self.redis_client:
            return None
        
        key = f"sms_verification:{user_id}"
        data = self.redis_client.get(key)
        if not data:
            return None
        
        try:
            import ast
            data_dict = ast.literal_eval(data.decode() if isinstance(data, bytes) else data)
            data_dict["created_at"] = datetime.fromisoformat(data_dict["created_at"])
            data_dict["expires_at"] = datetime.fromisoformat(data_dict["expires_at"])
            return SMSVerification(**data_dict)
        except Exception:
            return None
    
    def _remove_sms_verification(self, user_id: str) -> None:
        """Remove SMS verification data from Redis."""
        if not self.redis_client:
            return
        
        key = f"sms_verification:{user_id}"
        self.redis_client.delete(key)