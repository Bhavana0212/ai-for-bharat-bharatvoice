<<<<<<< HEAD
"""
File encryption utilities for secure storage.
"""

import hashlib
import logging
import os
from typing import Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class FileEncryption:
    """File encryption and decryption utilities."""
    
    def __init__(self):
        self.settings = get_settings()
        self._fernet: Optional[Fernet] = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with key from settings."""
        try:
            encryption_key = self.settings.security.encryption_key
            
            if not encryption_key:
                # Generate a key from the secret key
                secret_key = self.settings.security.secret_key.encode()
                salt = b'bharatvoice_salt'  # In production, use a random salt per file
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(secret_key))
            else:
                key = encryption_key.encode()
                if len(key) != 44:  # Fernet key should be 44 bytes when base64 encoded
                    # Derive proper key from provided key
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=b'bharatvoice_salt',
                        iterations=100000,
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(key))
            
            self._fernet = Fernet(key)
            logger.info("File encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize file encryption: {e}")
            self._fernet = None
    
    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self._fernet is not None
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
            
        Raises:
            RuntimeError: If encryption is not available
        """
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data
            
        Raises:
            RuntimeError: If encryption is not available
        """
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        
        try:
            return self._fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """
        Encrypt a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file
            
        Returns:
            True if successful
        """
        if not self._fernet:
            logger.error("Encryption not available")
            return False
        
        try:
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self._fernet.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            logger.debug(f"File encrypted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt file {input_path}: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """
        Decrypt a file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to output decrypted file
            
        Returns:
            True if successful
        """
        if not self._fernet:
            logger.error("Encryption not available")
            return False
        
        try:
            with open(input_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            data = self._fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(data)
            
            logger.debug(f"File decrypted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decrypt file {input_path}: {e}")
            return False
    
    def calculate_checksum(self, data: bytes) -> str:
        """
        Calculate SHA-256 checksum of data.
        
        Args:
            data: Data to checksum
            
        Returns:
            Hexadecimal checksum string
        """
        return hashlib.sha256(data).hexdigest()
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal checksum string
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def verify_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """
        Verify data checksum.
        
        Args:
            data: Data to verify
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(data)
        return actual_checksum == expected_checksum
    
    def verify_file_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file checksum.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_file_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def generate_key_id(self, data: bytes) -> str:
        """
        Generate a key ID for data.
        
        Args:
            data: Data to generate key ID for
            
        Returns:
            Key ID string
        """
        return hashlib.md5(data).hexdigest()[:16]


# Global encryption instance
_file_encryption: Optional[FileEncryption] = None


def get_file_encryption() -> FileEncryption:
    """Get global file encryption instance."""
    global _file_encryption
    
    if _file_encryption is None:
        _file_encryption = FileEncryption()
    
=======
"""
File encryption utilities for secure storage.
"""

import hashlib
import logging
import os
from typing import Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class FileEncryption:
    """File encryption and decryption utilities."""
    
    def __init__(self):
        self.settings = get_settings()
        self._fernet: Optional[Fernet] = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with key from settings."""
        try:
            encryption_key = self.settings.security.encryption_key
            
            if not encryption_key:
                # Generate a key from the secret key
                secret_key = self.settings.security.secret_key.encode()
                salt = b'bharatvoice_salt'  # In production, use a random salt per file
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(secret_key))
            else:
                key = encryption_key.encode()
                if len(key) != 44:  # Fernet key should be 44 bytes when base64 encoded
                    # Derive proper key from provided key
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=b'bharatvoice_salt',
                        iterations=100000,
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(key))
            
            self._fernet = Fernet(key)
            logger.info("File encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize file encryption: {e}")
            self._fernet = None
    
    def is_available(self) -> bool:
        """Check if encryption is available."""
        return self._fernet is not None
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
            
        Raises:
            RuntimeError: If encryption is not available
        """
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data
            
        Raises:
            RuntimeError: If encryption is not available
        """
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        
        try:
            return self._fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """
        Encrypt a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file
            
        Returns:
            True if successful
        """
        if not self._fernet:
            logger.error("Encryption not available")
            return False
        
        try:
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self._fernet.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            logger.debug(f"File encrypted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt file {input_path}: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """
        Decrypt a file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to output decrypted file
            
        Returns:
            True if successful
        """
        if not self._fernet:
            logger.error("Encryption not available")
            return False
        
        try:
            with open(input_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            data = self._fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(data)
            
            logger.debug(f"File decrypted: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decrypt file {input_path}: {e}")
            return False
    
    def calculate_checksum(self, data: bytes) -> str:
        """
        Calculate SHA-256 checksum of data.
        
        Args:
            data: Data to checksum
            
        Returns:
            Hexadecimal checksum string
        """
        return hashlib.sha256(data).hexdigest()
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal checksum string
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def verify_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """
        Verify data checksum.
        
        Args:
            data: Data to verify
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_checksum(data)
        return actual_checksum == expected_checksum
    
    def verify_file_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file checksum.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.calculate_file_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def generate_key_id(self, data: bytes) -> str:
        """
        Generate a key ID for data.
        
        Args:
            data: Data to generate key ID for
            
        Returns:
            Key ID string
        """
        return hashlib.md5(data).hexdigest()[:16]


# Global encryption instance
_file_encryption: Optional[FileEncryption] = None


def get_file_encryption() -> FileEncryption:
    """Get global file encryption instance."""
    global _file_encryption
    
    if _file_encryption is None:
        _file_encryption = FileEncryption()
    
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    return _file_encryption