<<<<<<< HEAD
"""
Secure file storage system with encryption, compression, and lifecycle management.
"""

import asyncio
import logging
import mimetypes
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import aiofiles
from fastapi import UploadFile

from .compression import CompressionType, get_file_compression
from .encryption import get_file_encryption
from .lifecycle import get_lifecycle_manager
from bharatvoice.database.base import get_db_session
from bharatvoice.database.models import AudioFile
from bharatvoice.config.settings import get_settings
from sqlalchemy import select

logger = logging.getLogger(__name__)


class FileStorage:
    """Secure file storage system with comprehensive features."""
    
    def __init__(self):
        self.settings = get_settings()
        self.encryption = get_file_encryption()
        self.compression = get_file_compression()
        self._lifecycle_manager = None
        self._base_storage_path = Path("storage")
        self._ensure_storage_directories()
    
    def _ensure_storage_directories(self):
        """Ensure storage directories exist."""
        directories = [
            self._base_storage_path,
            self._base_storage_path / "audio",
            self._base_storage_path / "uploads",
            self._base_storage_path / "temp",
            self._base_storage_path / "encrypted",
            self._base_storage_path / "compressed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize file storage system."""
        try:
            self._lifecycle_manager = await get_lifecycle_manager()
            logger.info("File storage system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize file storage: {e}")
            raise
    
    async def store_file(
        self,
        file_data: Union[bytes, BinaryIO, UploadFile],
        user_id: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        encrypt: bool = True,
        compress: bool = True,
        is_temporary: bool = False,
        expires_hours: Optional[int] = None
    ) -> Optional[str]:
        """
        Store a file securely.
        
        Args:
            file_data: File data to store
            user_id: User ID
            session_id: Session ID (optional)
            filename: Original filename
            mime_type: MIME type
            encrypt: Whether to encrypt the file
            compress: Whether to compress the file
            is_temporary: Whether file is temporary
            expires_hours: Hours until file expires
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Read file data
            if isinstance(file_data, UploadFile):
                data = await file_data.read()
                if not filename:
                    filename = file_data.filename
                if not mime_type:
                    mime_type = file_data.content_type
            elif isinstance(file_data, (BinaryIO, bytes)):
                if isinstance(file_data, bytes):
                    data = file_data
                else:
                    data = file_data.read()
            else:
                raise ValueError("Unsupported file data type")
            
            # Determine MIME type if not provided
            if not mime_type and filename:
                mime_type, _ = mimetypes.guess_type(filename)
            
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Calculate original checksum
            original_checksum = self.encryption.calculate_checksum(data)
            
            # Compress if requested
            compression_type = CompressionType.NONE
            if compress and len(data) > 1024:  # Only compress files > 1KB
                compression_type = self.compression.choose_best_compression(data)
                if compression_type != CompressionType.NONE:
                    data = self.compression.compress_data(data, compression_type)
            
            # Encrypt if requested
            encryption_key_id = None
            if encrypt and self.encryption.is_available():
                encryption_key_id = self.encryption.generate_key_id(data)
                data = self.encryption.encrypt_data(data)
            
            # Determine storage path
            storage_subdir = "temp" if is_temporary else "audio"
            if encrypt:
                storage_subdir = "encrypted"
            elif compression_type != CompressionType.NONE:
                storage_subdir = "compressed"
            
            file_path = self._base_storage_path / storage_subdir / f"{file_id}.dat"
            
            # Write file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
            
            # Get file info
            file_size = len(data)
            
            # Calculate expiration
            expires_at = None
            if expires_hours:
                expires_at = datetime.now() + timedelta(hours=expires_hours)
            elif is_temporary:
                expires_at = datetime.now() + timedelta(hours=24)  # Default 24h for temp files
            
            # Store metadata in database
            async with get_db_session() as session:
                audio_file = AudioFile(
                    id=uuid.UUID(file_id),
                    user_id=uuid.UUID(user_id),
                    session_id=uuid.UUID(session_id) if session_id else None,
                    filename=f"{file_id}.dat",
                    original_filename=filename,
                    file_path=str(file_path),
                    file_size=file_size,
                    mime_type=mime_type,
                    is_encrypted=encrypt and self.encryption.is_available(),
                    encryption_key_id=encryption_key_id,
                    checksum=original_checksum,
                    is_temporary=is_temporary,
                    expires_at=expires_at
                )
                
                session.add(audio_file)
                await session.commit()
            
            logger.info(f"File stored successfully: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            return None
    
    async def retrieve_file(
        self, 
        file_id: str, 
        user_id: Optional[str] = None
    ) -> Optional[Tuple[bytes, Dict[str, any]]]:
        """
        Retrieve a file.
        
        Args:
            file_id: File ID
            user_id: User ID for access control
            
        Returns:
            Tuple of (file_data, metadata) if successful, None otherwise
        """
        try:
            # Get file metadata from database
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.id == uuid.UUID(file_id))
                if user_id:
                    stmt = stmt.where(AudioFile.user_id == uuid.UUID(user_id))
                
                result = await session.execute(stmt)
                file_record = result.scalar_one_or_none()
                
                if not file_record:
                    logger.warning(f"File not found: {file_id}")
                    return None
                
                if file_record.is_deleted:
                    logger.warning(f"File is deleted: {file_id}")
                    return None
                
                # Check if file has expired
                if file_record.expires_at and file_record.expires_at < datetime.now():
                    logger.warning(f"File has expired: {file_id}")
                    return None
                
                # Read file data
                if not os.path.exists(file_record.file_path):
                    logger.error(f"Physical file not found: {file_record.file_path}")
                    return None
                
                async with aiofiles.open(file_record.file_path, 'rb') as f:
                    data = await f.read()
                
                # Decrypt if encrypted
                if file_record.is_encrypted and self.encryption.is_available():
                    try:
                        data = self.encryption.decrypt_data(data)
                    except Exception as e:
                        logger.error(f"Failed to decrypt file {file_id}: {e}")
                        return None
                
                # Decompress if compressed (detect compression type from metadata or file)
                # For now, we'll assume GZIP if the file was compressed
                # In a real implementation, you'd store compression type in metadata
                
                # Update access count
                file_record.access_count += 1
                file_record.last_accessed = datetime.now()
                await session.commit()
                
                # Prepare metadata
                metadata = {
                    "filename": file_record.original_filename,
                    "mime_type": file_record.mime_type,
                    "file_size": len(data),
                    "original_size": file_record.file_size,
                    "created_at": file_record.created_at,
                    "is_encrypted": file_record.is_encrypted,
                    "checksum": file_record.checksum
                }
                
                # Verify checksum if available
                if file_record.checksum:
                    if not self.encryption.verify_checksum(data, file_record.checksum):
                        logger.error(f"Checksum verification failed for file {file_id}")
                        return None
                
                logger.debug(f"File retrieved successfully: {file_id}")
                return data, metadata
                
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a file.
        
        Args:
            file_id: File ID
            user_id: User ID for access control
            
        Returns:
            True if successful
        """
        try:
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.id == uuid.UUID(file_id))
                if user_id:
                    stmt = stmt.where(AudioFile.user_id == uuid.UUID(user_id))
                
                result = await session.execute(stmt)
                file_record = result.scalar_one_or_none()
                
                if not file_record:
                    logger.warning(f"File not found for deletion: {file_id}")
                    return False
                
                # Delete physical file
                try:
                    if os.path.exists(file_record.file_path):
                        os.remove(file_record.file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete physical file {file_record.file_path}: {e}")
                
                # Mark as deleted in database
                file_record.is_deleted = True
                file_record.deleted_at = datetime.now()
                await session.commit()
                
                logger.info(f"File deleted successfully: {file_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def list_user_files(
        self, 
        user_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, any]]:
        """
        List files for a user.
        
        Args:
            user_id: User ID
            include_deleted: Whether to include deleted files
            limit: Maximum number of files to return
            offset: Offset for pagination
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.user_id == uuid.UUID(user_id))
                
                if not include_deleted:
                    stmt = stmt.where(AudioFile.is_deleted == False)
                
                stmt = stmt.order_by(AudioFile.created_at.desc())
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                files = result.scalars().all()
                
                file_list = []
                for file_record in files:
                    file_info = {
                        "id": str(file_record.id),
                        "filename": file_record.original_filename,
                        "mime_type": file_record.mime_type,
                        "file_size": file_record.file_size,
                        "created_at": file_record.created_at,
                        "is_temporary": file_record.is_temporary,
                        "is_encrypted": file_record.is_encrypted,
                        "is_deleted": file_record.is_deleted,
                        "expires_at": file_record.expires_at,
                        "access_count": file_record.access_count,
                        "last_accessed": file_record.last_accessed
                    }
                    file_list.append(file_info)
                
                return file_list
                
        except Exception as e:
            logger.error(f"Failed to list files for user {user_id}: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage system statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            if self._lifecycle_manager:
                return await self._lifecycle_manager.get_storage_stats()
            else:
                return {"error": "Lifecycle manager not initialized"}
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_files(self) -> int:
        """
        Clean up expired files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            if self._lifecycle_manager:
                return await self._lifecycle_manager.cleanup_expired_files()
            else:
                logger.warning("Lifecycle manager not initialized")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform storage system health check.
        
        Returns:
            Health check results
        """
        try:
            health = {
                "status": "healthy",
                "encryption_available": self.encryption.is_available(),
                "storage_directories": {},
                "lifecycle_manager": self._lifecycle_manager is not None
            }
            
            # Check storage directories
            for directory in ["audio", "uploads", "temp", "encrypted", "compressed"]:
                dir_path = self._base_storage_path / directory
                health["storage_directories"][directory] = {
                    "exists": dir_path.exists(),
                    "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False
                }
            
            return health
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Global file storage instance
_file_storage: Optional[FileStorage] = None


async def get_file_storage() -> FileStorage:
    """Get global file storage instance."""
    global _file_storage
    
    if _file_storage is None:
        _file_storage = FileStorage()
        await _file_storage.initialize()
    
    return _file_storage


async def close_file_storage():
    """Close global file storage."""
    global _file_storage
    
    if _file_storage:
        # File storage doesn't need explicit closing, but we can clean up
=======
"""
Secure file storage system with encryption, compression, and lifecycle management.
"""

import asyncio
import logging
import mimetypes
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import aiofiles
from fastapi import UploadFile

from .compression import CompressionType, get_file_compression
from .encryption import get_file_encryption
from .lifecycle import get_lifecycle_manager
from bharatvoice.database.base import get_db_session
from bharatvoice.database.models import AudioFile
from bharatvoice.config.settings import get_settings
from sqlalchemy import select

logger = logging.getLogger(__name__)


class FileStorage:
    """Secure file storage system with comprehensive features."""
    
    def __init__(self):
        self.settings = get_settings()
        self.encryption = get_file_encryption()
        self.compression = get_file_compression()
        self._lifecycle_manager = None
        self._base_storage_path = Path("storage")
        self._ensure_storage_directories()
    
    def _ensure_storage_directories(self):
        """Ensure storage directories exist."""
        directories = [
            self._base_storage_path,
            self._base_storage_path / "audio",
            self._base_storage_path / "uploads",
            self._base_storage_path / "temp",
            self._base_storage_path / "encrypted",
            self._base_storage_path / "compressed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize file storage system."""
        try:
            self._lifecycle_manager = await get_lifecycle_manager()
            logger.info("File storage system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize file storage: {e}")
            raise
    
    async def store_file(
        self,
        file_data: Union[bytes, BinaryIO, UploadFile],
        user_id: str,
        session_id: Optional[str] = None,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        encrypt: bool = True,
        compress: bool = True,
        is_temporary: bool = False,
        expires_hours: Optional[int] = None
    ) -> Optional[str]:
        """
        Store a file securely.
        
        Args:
            file_data: File data to store
            user_id: User ID
            session_id: Session ID (optional)
            filename: Original filename
            mime_type: MIME type
            encrypt: Whether to encrypt the file
            compress: Whether to compress the file
            is_temporary: Whether file is temporary
            expires_hours: Hours until file expires
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Read file data
            if isinstance(file_data, UploadFile):
                data = await file_data.read()
                if not filename:
                    filename = file_data.filename
                if not mime_type:
                    mime_type = file_data.content_type
            elif isinstance(file_data, (BinaryIO, bytes)):
                if isinstance(file_data, bytes):
                    data = file_data
                else:
                    data = file_data.read()
            else:
                raise ValueError("Unsupported file data type")
            
            # Determine MIME type if not provided
            if not mime_type and filename:
                mime_type, _ = mimetypes.guess_type(filename)
            
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # Calculate original checksum
            original_checksum = self.encryption.calculate_checksum(data)
            
            # Compress if requested
            compression_type = CompressionType.NONE
            if compress and len(data) > 1024:  # Only compress files > 1KB
                compression_type = self.compression.choose_best_compression(data)
                if compression_type != CompressionType.NONE:
                    data = self.compression.compress_data(data, compression_type)
            
            # Encrypt if requested
            encryption_key_id = None
            if encrypt and self.encryption.is_available():
                encryption_key_id = self.encryption.generate_key_id(data)
                data = self.encryption.encrypt_data(data)
            
            # Determine storage path
            storage_subdir = "temp" if is_temporary else "audio"
            if encrypt:
                storage_subdir = "encrypted"
            elif compression_type != CompressionType.NONE:
                storage_subdir = "compressed"
            
            file_path = self._base_storage_path / storage_subdir / f"{file_id}.dat"
            
            # Write file to disk
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
            
            # Get file info
            file_size = len(data)
            
            # Calculate expiration
            expires_at = None
            if expires_hours:
                expires_at = datetime.now() + timedelta(hours=expires_hours)
            elif is_temporary:
                expires_at = datetime.now() + timedelta(hours=24)  # Default 24h for temp files
            
            # Store metadata in database
            async with get_db_session() as session:
                audio_file = AudioFile(
                    id=uuid.UUID(file_id),
                    user_id=uuid.UUID(user_id),
                    session_id=uuid.UUID(session_id) if session_id else None,
                    filename=f"{file_id}.dat",
                    original_filename=filename,
                    file_path=str(file_path),
                    file_size=file_size,
                    mime_type=mime_type,
                    is_encrypted=encrypt and self.encryption.is_available(),
                    encryption_key_id=encryption_key_id,
                    checksum=original_checksum,
                    is_temporary=is_temporary,
                    expires_at=expires_at
                )
                
                session.add(audio_file)
                await session.commit()
            
            logger.info(f"File stored successfully: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            return None
    
    async def retrieve_file(
        self, 
        file_id: str, 
        user_id: Optional[str] = None
    ) -> Optional[Tuple[bytes, Dict[str, any]]]:
        """
        Retrieve a file.
        
        Args:
            file_id: File ID
            user_id: User ID for access control
            
        Returns:
            Tuple of (file_data, metadata) if successful, None otherwise
        """
        try:
            # Get file metadata from database
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.id == uuid.UUID(file_id))
                if user_id:
                    stmt = stmt.where(AudioFile.user_id == uuid.UUID(user_id))
                
                result = await session.execute(stmt)
                file_record = result.scalar_one_or_none()
                
                if not file_record:
                    logger.warning(f"File not found: {file_id}")
                    return None
                
                if file_record.is_deleted:
                    logger.warning(f"File is deleted: {file_id}")
                    return None
                
                # Check if file has expired
                if file_record.expires_at and file_record.expires_at < datetime.now():
                    logger.warning(f"File has expired: {file_id}")
                    return None
                
                # Read file data
                if not os.path.exists(file_record.file_path):
                    logger.error(f"Physical file not found: {file_record.file_path}")
                    return None
                
                async with aiofiles.open(file_record.file_path, 'rb') as f:
                    data = await f.read()
                
                # Decrypt if encrypted
                if file_record.is_encrypted and self.encryption.is_available():
                    try:
                        data = self.encryption.decrypt_data(data)
                    except Exception as e:
                        logger.error(f"Failed to decrypt file {file_id}: {e}")
                        return None
                
                # Decompress if compressed (detect compression type from metadata or file)
                # For now, we'll assume GZIP if the file was compressed
                # In a real implementation, you'd store compression type in metadata
                
                # Update access count
                file_record.access_count += 1
                file_record.last_accessed = datetime.now()
                await session.commit()
                
                # Prepare metadata
                metadata = {
                    "filename": file_record.original_filename,
                    "mime_type": file_record.mime_type,
                    "file_size": len(data),
                    "original_size": file_record.file_size,
                    "created_at": file_record.created_at,
                    "is_encrypted": file_record.is_encrypted,
                    "checksum": file_record.checksum
                }
                
                # Verify checksum if available
                if file_record.checksum:
                    if not self.encryption.verify_checksum(data, file_record.checksum):
                        logger.error(f"Checksum verification failed for file {file_id}")
                        return None
                
                logger.debug(f"File retrieved successfully: {file_id}")
                return data, metadata
                
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a file.
        
        Args:
            file_id: File ID
            user_id: User ID for access control
            
        Returns:
            True if successful
        """
        try:
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.id == uuid.UUID(file_id))
                if user_id:
                    stmt = stmt.where(AudioFile.user_id == uuid.UUID(user_id))
                
                result = await session.execute(stmt)
                file_record = result.scalar_one_or_none()
                
                if not file_record:
                    logger.warning(f"File not found for deletion: {file_id}")
                    return False
                
                # Delete physical file
                try:
                    if os.path.exists(file_record.file_path):
                        os.remove(file_record.file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete physical file {file_record.file_path}: {e}")
                
                # Mark as deleted in database
                file_record.is_deleted = True
                file_record.deleted_at = datetime.now()
                await session.commit()
                
                logger.info(f"File deleted successfully: {file_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def list_user_files(
        self, 
        user_id: str,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, any]]:
        """
        List files for a user.
        
        Args:
            user_id: User ID
            include_deleted: Whether to include deleted files
            limit: Maximum number of files to return
            offset: Offset for pagination
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.user_id == uuid.UUID(user_id))
                
                if not include_deleted:
                    stmt = stmt.where(AudioFile.is_deleted == False)
                
                stmt = stmt.order_by(AudioFile.created_at.desc())
                stmt = stmt.limit(limit).offset(offset)
                
                result = await session.execute(stmt)
                files = result.scalars().all()
                
                file_list = []
                for file_record in files:
                    file_info = {
                        "id": str(file_record.id),
                        "filename": file_record.original_filename,
                        "mime_type": file_record.mime_type,
                        "file_size": file_record.file_size,
                        "created_at": file_record.created_at,
                        "is_temporary": file_record.is_temporary,
                        "is_encrypted": file_record.is_encrypted,
                        "is_deleted": file_record.is_deleted,
                        "expires_at": file_record.expires_at,
                        "access_count": file_record.access_count,
                        "last_accessed": file_record.last_accessed
                    }
                    file_list.append(file_info)
                
                return file_list
                
        except Exception as e:
            logger.error(f"Failed to list files for user {user_id}: {e}")
            return []
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage system statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            if self._lifecycle_manager:
                return await self._lifecycle_manager.get_storage_stats()
            else:
                return {"error": "Lifecycle manager not initialized"}
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_files(self) -> int:
        """
        Clean up expired files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            if self._lifecycle_manager:
                return await self._lifecycle_manager.cleanup_expired_files()
            else:
                logger.warning("Lifecycle manager not initialized")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform storage system health check.
        
        Returns:
            Health check results
        """
        try:
            health = {
                "status": "healthy",
                "encryption_available": self.encryption.is_available(),
                "storage_directories": {},
                "lifecycle_manager": self._lifecycle_manager is not None
            }
            
            # Check storage directories
            for directory in ["audio", "uploads", "temp", "encrypted", "compressed"]:
                dir_path = self._base_storage_path / directory
                health["storage_directories"][directory] = {
                    "exists": dir_path.exists(),
                    "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False
                }
            
            return health
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Global file storage instance
_file_storage: Optional[FileStorage] = None


async def get_file_storage() -> FileStorage:
    """Get global file storage instance."""
    global _file_storage
    
    if _file_storage is None:
        _file_storage = FileStorage()
        await _file_storage.initialize()
    
    return _file_storage


async def close_file_storage():
    """Close global file storage."""
    global _file_storage
    
    if _file_storage:
        # File storage doesn't need explicit closing, but we can clean up
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        _file_storage = None