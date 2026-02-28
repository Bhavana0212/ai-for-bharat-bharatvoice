"""
Password management for BharatVoice Assistant.

This module provides secure password hashing, verification, and strength validation
using bcrypt with configurable work factors and security policies.
"""

import re
from typing import Optional, List, Dict, Any
import secrets
import string

import bcrypt
import structlog
from pydantic import BaseModel


logger = structlog.get_logger(__name__)


class PasswordPolicy(BaseModel):
    """Password policy configuration."""
    
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    min_special_chars: int = 1
    forbidden_patterns: List[str] = ["password", "123456", "qwerty"]
    max_repeated_chars: int = 3


class PasswordStrength(BaseModel):
    """Password strength assessment."""
    
    score: int  # 0-100
    is_valid: bool
    feedback: List[str]
    estimated_crack_time: str


class PasswordManager:
    """Secure password management with bcrypt."""
    
    def __init__(self, rounds: int = 12):
        """
        Initialize password manager.
        
        Args:
            rounds: bcrypt work factor (4-31, higher is more secure but slower)
        """
        self.rounds = max(4, min(31, rounds))  # Clamp to valid range
        self.policy = PasswordPolicy()
        
        logger.info("Password manager initialized", rounds=self.rounds)
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
            
        Raises:
            ValueError: If password is invalid
        """
        try:
            # Validate password first
            strength = self.assess_password_strength(password)
            if not strength.is_valid:
                raise ValueError(f"Password does not meet policy: {'; '.join(strength.feedback)}")
            
            # Hash password
            password_bytes = password.encode('utf-8')
            salt = bcrypt.gensalt(rounds=self.rounds)
            hashed = bcrypt.hashpw(password_bytes, salt)
            
            logger.info("Password hashed successfully")
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error("Password hashing failed", exc_info=e)
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to verify against
            
        Returns:
            True if password matches hash
        """
        try:
            password_bytes = password.encode('utf-8')
            hashed_bytes = hashed_password.encode('utf-8')
            
            result = bcrypt.checkpw(password_bytes, hashed_bytes)
            
            logger.debug("Password verification completed", result=result)
            return result
            
        except Exception as e:
            logger.error("Password verification failed", exc_info=e)
            return False
    
    def assess_password_strength(self, password: str) -> PasswordStrength:
        """
        Assess password strength against policy.
        
        Args:
            password: Password to assess
            
        Returns:
            Password strength assessment
        """
        feedback = []
        score = 0
        
        # Length check
        if len(password) < self.policy.min_length:
            feedback.append(f"Password must be at least {self.policy.min_length} characters long")
        elif len(password) >= self.policy.min_length:
            score += 20
            
        if len(password) > self.policy.max_length:
            feedback.append(f"Password must not exceed {self.policy.max_length} characters")
        
        # Character type checks
        if self.policy.require_uppercase and not re.search(r'[A-Z]', password):
            feedback.append("Password must contain at least one uppercase letter")
        elif re.search(r'[A-Z]', password):
            score += 15
            
        if self.policy.require_lowercase and not re.search(r'[a-z]', password):
            feedback.append("Password must contain at least one lowercase letter")
        elif re.search(r'[a-z]', password):
            score += 15
            
        if self.policy.require_digits and not re.search(r'\d', password):
            feedback.append("Password must contain at least one digit")
        elif re.search(r'\d', password):
            score += 15
            
        if self.policy.require_special_chars:
            special_chars = re.findall(r'[!@#$%^&*(),.?":{}|<>]', password)
            if len(special_chars) < self.policy.min_special_chars:
                feedback.append(f"Password must contain at least {self.policy.min_special_chars} special character(s)")
            else:
                score += 15
        
        # Pattern checks
        password_lower = password.lower()
        for pattern in self.policy.forbidden_patterns:
            if pattern.lower() in password_lower:
                feedback.append(f"Password contains forbidden pattern: {pattern}")
                score -= 10
        
        # Repeated character check
        repeated_count = 0
        for i in range(len(password) - 1):
            if password[i] == password[i + 1]:
                repeated_count += 1
                if repeated_count >= self.policy.max_repeated_chars:
                    feedback.append(f"Password has too many repeated characters")
                    score -= 5
                    break
            else:
                repeated_count = 0
        
        # Bonus points for length and complexity
        if len(password) >= 12:
            score += 10
        if len(password) >= 16:
            score += 10
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        # Determine crack time estimate
        if score >= 80:
            crack_time = "centuries"
        elif score >= 60:
            crack_time = "years"
        elif score >= 40:
            crack_time = "months"
        elif score >= 20:
            crack_time = "days"
        else:
            crack_time = "minutes"
        
        is_valid = len(feedback) == 0 and score >= 60
        
        return PasswordStrength(
            score=score,
            is_valid=is_valid,
            feedback=feedback,
            estimated_crack_time=crack_time
        )
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Password length (minimum 8)
            
        Returns:
            Secure random password
        """
        length = max(8, length)
        
        # Ensure we have at least one of each required character type
        chars = []
        
        if self.policy.require_uppercase:
            chars.append(secrets.choice(string.ascii_uppercase))
        if self.policy.require_lowercase:
            chars.append(secrets.choice(string.ascii_lowercase))
        if self.policy.require_digits:
            chars.append(secrets.choice(string.digits))
        if self.policy.require_special_chars:
            chars.append(secrets.choice("!@#$%^&*(),.?\":{}|<>"))
        
        # Fill remaining length with random characters
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*(),.?\":{}|<>"
        for _ in range(length - len(chars)):
            chars.append(secrets.choice(all_chars))
        
        # Shuffle the characters
        password_list = chars[:]
        for i in range(len(password_list)):
            j = secrets.randbelow(len(password_list))
            password_list[i], password_list[j] = password_list[j], password_list[i]
        
        password = ''.join(password_list)
        
        logger.info("Secure password generated", length=len(password))
        return password
    
    def update_policy(self, policy_updates: Dict[str, Any]) -> None:
        """
        Update password policy.
        
        Args:
            policy_updates: Dictionary of policy updates
        """
        try:
            for key, value in policy_updates.items():
                if hasattr(self.policy, key):
                    setattr(self.policy, key, value)
                    logger.info("Password policy updated", key=key, value=value)
                else:
                    logger.warning("Unknown policy key", key=key)
        except Exception as e:
            logger.error("Failed to update password policy", exc_info=e)
            raise