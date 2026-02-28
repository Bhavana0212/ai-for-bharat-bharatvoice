"""
Authentication service for BharatVoice Assistant.

This module provides comprehensive user authentication, session management,
security features, data encryption, privacy compliance, and Indian law compliance.
"""

from .auth_service import AuthService
from .jwt_manager import JWTManager
from .password_manager import PasswordManager
from .session_manager import SessionManager
from .mfa_manager import MFAManager
from .encryption_manager import EncryptionManager
from .privacy_manager import PrivacyManager
from .indian_compliance_manager import IndianComplianceManager

__all__ = [
    "AuthService",
    "JWTManager", 
    "PasswordManager",
    "SessionManager",
    "MFAManager",
    "EncryptionManager",
    "PrivacyManager",
    "IndianComplianceManager"
]