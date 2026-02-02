"""
Encryption manager for BharatVoice Assistant.

This module provides end-to-end encryption for voice data transmission,
local storage encryption, and secure key management with rotation capabilities.
"""

import os
import base64
import secrets
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings


logger = structlog.get_logger(__name__)


class EncryptionKey(BaseModel):
    """Encryption key metadata."""
    
    key_id: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    key_type: str  # "symmetric", "asymmetric_public", "asymmetric_private"


class EncryptedData(BaseModel):
    """Encrypted data container."""
    
    data: str  # Base64 encoded encrypted data
    key_id: str
    algorithm: str
    iv: Optional[str] = None  # Initialization vector for symmetric encryption
    timestamp: datetime


class EncryptionManager:
    """Encryption manager for data protection."""
    
    def __init__(self, settings: Settings, redis_client=None):
        """
        Initialize encryption manager.
        
        Args:
            settings: Application settings
            redis_client: Redis client for key storage
        """
        self.settings = settings
        self.redis_client = redis_client
        
        # Encryption configuration
        self.key_rotation_interval = timedelta(days=90)  # 3 months
        self.symmetric_key_size = 32  # 256 bits
        self.rsa_key_size = 2048
        
        # Initialize master key
        self.master_key = self._get_or_create_master_key()
        self.fernet = Fernet(self.master_key)
        
        # Key storage
        self.encryption_keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, EncryptionKey] = {}
        
        logger.info("Encryption manager initialized")
    
    def encrypt_voice_data(self, audio_data: bytes, user_id: str) -> EncryptedData:
        """
        Encrypt voice data for transmission or storage.
        
        Args:
            audio_data: Raw audio data
            user_id: User identifier for key derivation
            
        Returns:
            Encrypted data container
        """
        try:
            # Generate or get user-specific key
            key_id = f"voice_{user_id}"
            encryption_key = self._get_or_create_symmetric_key(key_id)
            
            # Generate random IV
            iv = os.urandom(16)
            
            # Encrypt data using AES-256-CBC
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(audio_data)
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Encode to base64
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            iv_b64 = base64.b64encode(iv).decode('utf-8')
            
            result = EncryptedData(
                data=encrypted_b64,
                key_id=key_id,
                algorithm="AES-256-CBC",
                iv=iv_b64,
                timestamp=datetime.utcnow()
            )
            
            logger.info("Voice data encrypted", user_id=user_id, size=len(audio_data))
            return result
            
        except Exception as e:
            logger.error("Voice data encryption failed", user_id=user_id, exc_info=e)
            raise
    
    def decrypt_voice_data(self, encrypted_data: EncryptedData) -> bytes:
        """
        Decrypt voice data.
        
        Args:
            encrypted_data: Encrypted data container
            
        Returns:
            Decrypted audio data
        """
        try:
            # Get encryption key
            encryption_key = self._get_symmetric_key(encrypted_data.key_id)
            if not encryption_key:
                raise ValueError(f"Encryption key not found: {encrypted_data.key_id}")
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.data)
            iv = base64.b64decode(encrypted_data.iv) if encrypted_data.iv else None
            
            if not iv:
                raise ValueError("IV required for AES-CBC decryption")
            
            # Decrypt data
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()
            
            # Remove padding
            decrypted_data = self._unpad_data(padded_data)
            
            logger.debug("Voice data decrypted", key_id=encrypted_data.key_id)
            return decrypted_data
            
        except Exception as e:
            logger.error("Voice data decryption failed", key_id=encrypted_data.key_id, exc_info=e)
            raise
    
    def encrypt_user_profile(self, profile_data: Dict[str, Any], user_id: str) -> EncryptedData:
        """
        Encrypt user profile data for local storage.
        
        Args:
            profile_data: User profile dictionary
            user_id: User identifier
            
        Returns:
            Encrypted profile data
        """
        try:
            # Serialize profile data
            import json
            profile_json = json.dumps(profile_data, default=str)
            profile_bytes = profile_json.encode('utf-8')
            
            # Use Fernet for profile encryption (simpler for structured data)
            encrypted_data = self.fernet.encrypt(profile_bytes)
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            result = EncryptedData(
                data=encrypted_b64,
                key_id="profile_master",
                algorithm="Fernet",
                timestamp=datetime.utcnow()
            )
            
            logger.info("User profile encrypted", user_id=user_id)
            return result
            
        except Exception as e:
            logger.error("User profile encryption failed", user_id=user_id, exc_info=e)
            raise
    
    def decrypt_user_profile(self, encrypted_data: EncryptedData) -> Dict[str, Any]:
        """
        Decrypt user profile data.
        
        Args:
            encrypted_data: Encrypted profile data
            
        Returns:
            Decrypted profile dictionary
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.data)
            
            # Decrypt using Fernet
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            profile_json = decrypted_bytes.decode('utf-8')
            
            # Deserialize profile data
            import json
            profile_data = json.loads(profile_json)
            
            logger.debug("User profile decrypted")
            return profile_data
            
        except Exception as e:
            logger.error("User profile decryption failed", exc_info=e)
            raise
    
    def generate_rsa_keypair(self, key_id: str) -> Tuple[str, str]:
        """
        Generate RSA key pair for asymmetric encryption.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Tuple of (public_key_pem, private_key_pem)
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.rsa_key_size,
                backend=default_backend()
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys to PEM format
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            # Store key metadata
            self.key_metadata[f"{key_id}_private"] = EncryptionKey(
                key_id=f"{key_id}_private",
                algorithm="RSA-2048",
                created_at=datetime.utcnow(),
                key_type="asymmetric_private"
            )
            
            self.key_metadata[f"{key_id}_public"] = EncryptionKey(
                key_id=f"{key_id}_public",
                algorithm="RSA-2048",
                created_at=datetime.utcnow(),
                key_type="asymmetric_public"
            )
            
            logger.info("RSA key pair generated", key_id=key_id)
            return public_pem, private_pem
            
        except Exception as e:
            logger.error("RSA key pair generation failed", key_id=key_id, exc_info=e)
            raise
    
    def encrypt_with_rsa(self, data: bytes, public_key_pem: str) -> str:
        """
        Encrypt data with RSA public key.
        
        Args:
            data: Data to encrypt
            public_key_pem: RSA public key in PEM format
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=default_backend()
            )
            
            # Encrypt data
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Encode to base64
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            logger.debug("Data encrypted with RSA")
            return encrypted_b64
            
        except Exception as e:
            logger.error("RSA encryption failed", exc_info=e)
            raise
    
    def decrypt_with_rsa(self, encrypted_data: str, private_key_pem: str) -> bytes:
        """
        Decrypt data with RSA private key.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            private_key_pem: RSA private key in PEM format
            
        Returns:
            Decrypted data
        """
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data)
            
            # Decrypt data
            decrypted_data = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            logger.debug("Data decrypted with RSA")
            return decrypted_data
            
        except Exception as e:
            logger.error("RSA decryption failed", exc_info=e)
            raise
    
    def rotate_keys(self) -> Dict[str, str]:
        """
        Rotate encryption keys.
        
        Returns:
            Dictionary of rotated key IDs and their status
        """
        try:
            rotated_keys = {}
            current_time = datetime.utcnow()
            
            # Check which keys need rotation
            for key_id, metadata in self.key_metadata.items():
                if (metadata.expires_at and current_time > metadata.expires_at) or \
                   (current_time - metadata.created_at > self.key_rotation_interval):
                    
                    # Generate new key
                    if metadata.key_type == "symmetric":
                        self._rotate_symmetric_key(key_id)
                        rotated_keys[key_id] = "rotated"
                    elif metadata.key_type.startswith("asymmetric"):
                        # For asymmetric keys, generate new pair
                        base_key_id = key_id.replace("_private", "").replace("_public", "")
                        self.generate_rsa_keypair(f"{base_key_id}_new")
                        rotated_keys[base_key_id] = "new_pair_generated"
            
            # Rotate master key if needed
            if self._should_rotate_master_key():
                self._rotate_master_key()
                rotated_keys["master_key"] = "rotated"
            
            logger.info("Key rotation completed", rotated_count=len(rotated_keys))
            return rotated_keys
            
        except Exception as e:
            logger.error("Key rotation failed", exc_info=e)
            raise
    
    def anonymize_data(self, data: Dict[str, Any], fields_to_anonymize: list) -> Dict[str, Any]:
        """
        Anonymize sensitive data fields for analytics.
        
        Args:
            data: Data dictionary
            fields_to_anonymize: List of field names to anonymize
            
        Returns:
            Anonymized data dictionary
        """
        try:
            anonymized_data = data.copy()
            
            for field in fields_to_anonymize:
                if field in anonymized_data:
                    if isinstance(anonymized_data[field], str):
                        # Hash string values
                        anonymized_data[field] = self._hash_string(anonymized_data[field])
                    elif isinstance(anonymized_data[field], (int, float)):
                        # Add noise to numeric values
                        anonymized_data[field] = self._add_noise(anonymized_data[field])
                    else:
                        # Remove complex data types
                        anonymized_data[field] = "[ANONYMIZED]"
            
            logger.debug("Data anonymized", fields=fields_to_anonymize)
            return anonymized_data
            
        except Exception as e:
            logger.error("Data anonymization failed", exc_info=e)
            raise
    
    def schedule_data_deletion(self, user_id: str, deletion_date: datetime) -> bool:
        """
        Schedule automatic data deletion for compliance.
        
        Args:
            user_id: User identifier
            deletion_date: When to delete the data
            
        Returns:
            True if scheduling successful
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available for deletion scheduling")
                return False
            
            # Store deletion schedule in Redis
            deletion_key = f"deletion_schedule:{user_id}"
            deletion_data = {
                "user_id": user_id,
                "deletion_date": deletion_date.isoformat(),
                "scheduled_at": datetime.utcnow().isoformat()
            }
            
            # Calculate TTL until deletion date
            ttl_seconds = int((deletion_date - datetime.utcnow()).total_seconds())
            if ttl_seconds > 0:
                self.redis_client.setex(deletion_key, ttl_seconds, str(deletion_data))
                
                logger.info(
                    "Data deletion scheduled",
                    user_id=user_id,
                    deletion_date=deletion_date.isoformat()
                )
                return True
            else:
                logger.warning("Deletion date is in the past", user_id=user_id)
                return False
            
        except Exception as e:
            logger.error("Data deletion scheduling failed", user_id=user_id, exc_info=e)
            return False
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key."""
        master_key_env = self.settings.security.encryption_key
        
        if master_key_env:
            # Use key from environment
            try:
                return base64.b64decode(master_key_env)
            except Exception:
                logger.warning("Invalid master key in environment, generating new one")
        
        # Generate new master key
        master_key = Fernet.generate_key()
        logger.warning(
            "Generated new master key - store this securely: %s",
            base64.b64encode(master_key).decode()
        )
        return master_key
    
    def _get_or_create_symmetric_key(self, key_id: str) -> bytes:
        """Get or create symmetric encryption key."""
        if key_id in self.encryption_keys:
            return self.encryption_keys[key_id]
        
        # Generate new key
        key = os.urandom(self.symmetric_key_size)
        self.encryption_keys[key_id] = key
        
        # Store metadata
        self.key_metadata[key_id] = EncryptionKey(
            key_id=key_id,
            algorithm="AES-256",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.key_rotation_interval,
            key_type="symmetric"
        )
        
        logger.info("New symmetric key generated", key_id=key_id)
        return key
    
    def _get_symmetric_key(self, key_id: str) -> Optional[bytes]:
        """Get symmetric encryption key."""
        return self.encryption_keys.get(key_id)
    
    def _rotate_symmetric_key(self, key_id: str) -> None:
        """Rotate symmetric encryption key."""
        # Generate new key
        new_key = os.urandom(self.symmetric_key_size)
        self.encryption_keys[key_id] = new_key
        
        # Update metadata
        self.key_metadata[key_id] = EncryptionKey(
            key_id=key_id,
            algorithm="AES-256",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.key_rotation_interval,
            key_type="symmetric"
        )
        
        logger.info("Symmetric key rotated", key_id=key_id)
    
    def _should_rotate_master_key(self) -> bool:
        """Check if master key should be rotated."""
        # In production, implement proper master key rotation logic
        return False
    
    def _rotate_master_key(self) -> None:
        """Rotate master encryption key."""
        # In production, implement secure master key rotation
        logger.warning("Master key rotation not implemented in this version")
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to AES block size using PKCS7."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding from data."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _hash_string(self, value: str) -> str:
        """Hash string value for anonymization."""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:16]
    
    def _add_noise(self, value: float, noise_factor: float = 0.1) -> float:
        """Add noise to numeric value for anonymization."""
        noise = secrets.randbelow(int(abs(value) * noise_factor * 2)) - (abs(value) * noise_factor)
        return value + noise