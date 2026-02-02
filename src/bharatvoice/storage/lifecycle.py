"""
File lifecycle management for automated cleanup and retention.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from bharatvoice.database.base import get_db_session
from bharatvoice.database.models import AudioFile
from sqlalchemy import select, delete, and_

logger = logging.getLogger(__name__)


class FileLifecycleManager:
    """Manages file lifecycle including cleanup and retention policies."""
    
    def __init__(self):
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self):
        """Start the lifecycle management task."""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("File lifecycle manager started")
    
    async def stop(self):
        """Stop the lifecycle management task."""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("File lifecycle manager stopped")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_expired_files()
                await self.cleanup_temporary_files()
                await self.cleanup_orphaned_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lifecycle cleanup error: {e}")
    
    async def cleanup_expired_files(self) -> int:
        """
        Clean up expired files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            count = 0
            current_time = datetime.now()
            
            async with get_db_session() as session:
                # Find expired files
                stmt = select(AudioFile).where(
                    and_(
                        AudioFile.expires_at.isnot(None),
                        AudioFile.expires_at < current_time,
                        AudioFile.is_deleted == False
                    )
                )
                
                result = await session.execute(stmt)
                expired_files = result.scalars().all()
                
                for file_record in expired_files:
                    # Delete physical file
                    if await self._delete_physical_file(file_record.file_path):
                        # Mark as deleted in database
                        file_record.is_deleted = True
                        file_record.deleted_at = current_time
                        count += 1
                
                await session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired files")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {e}")
            return 0
    
    async def cleanup_temporary_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for temporary files
            
        Returns:
            Number of files cleaned up
        """
        try:
            count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            async with get_db_session() as session:
                # Find old temporary files
                stmt = select(AudioFile).where(
                    and_(
                        AudioFile.is_temporary == True,
                        AudioFile.created_at < cutoff_time,
                        AudioFile.is_deleted == False
                    )
                )
                
                result = await session.execute(stmt)
                temp_files = result.scalars().all()
                
                for file_record in temp_files:
                    # Delete physical file
                    if await self._delete_physical_file(file_record.file_path):
                        # Mark as deleted in database
                        file_record.is_deleted = True
                        file_record.deleted_at = datetime.now()
                        count += 1
                
                await session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} temporary files")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {e}")
            return 0
    
    async def cleanup_orphaned_files(self) -> int:
        """
        Clean up orphaned files (files on disk without database records).
        
        Returns:
            Number of files cleaned up
        """
        try:
            count = 0
            storage_dirs = ["uploads", "audio", "temp"]
            
            for storage_dir in storage_dirs:
                if not os.path.exists(storage_dir):
                    continue
                
                # Get all files in storage directory
                for root, dirs, files in os.walk(storage_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Check if file exists in database
                        async with get_db_session() as session:
                            stmt = select(AudioFile).where(AudioFile.file_path == file_path)
                            result = await session.execute(stmt)
                            file_record = result.scalar_one_or_none()
                            
                            if not file_record:
                                # Orphaned file - delete it
                                try:
                                    os.remove(file_path)
                                    count += 1
                                    logger.debug(f"Deleted orphaned file: {file_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to delete orphaned file {file_path}: {e}")
            
            if count > 0:
                logger.info(f"Cleaned up {count} orphaned files")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return 0
    
    async def cleanup_user_data(self, user_id: str, retention_days: int) -> int:
        """
        Clean up user data based on retention policy.
        
        Args:
            user_id: User ID
            retention_days: Number of days to retain data
            
        Returns:
            Number of files cleaned up
        """
        try:
            count = 0
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            async with get_db_session() as session:
                # Find old user files
                stmt = select(AudioFile).where(
                    and_(
                        AudioFile.user_id == user_id,
                        AudioFile.created_at < cutoff_time,
                        AudioFile.is_deleted == False
                    )
                )
                
                result = await session.execute(stmt)
                old_files = result.scalars().all()
                
                for file_record in old_files:
                    # Delete physical file
                    if await self._delete_physical_file(file_record.file_path):
                        # Mark as deleted in database
                        file_record.is_deleted = True
                        file_record.deleted_at = datetime.now()
                        count += 1
                
                await session.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} old files for user {user_id}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup user data for {user_id}: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics dictionary
        """
        try:
            stats = {
                "total_files": 0,
                "total_size": 0,
                "active_files": 0,
                "active_size": 0,
                "temporary_files": 0,
                "temporary_size": 0,
                "deleted_files": 0,
                "encrypted_files": 0,
                "by_type": {}
            }
            
            async with get_db_session() as session:
                stmt = select(AudioFile)
                result = await session.execute(stmt)
                files = result.scalars().all()
                
                for file_record in files:
                    stats["total_files"] += 1
                    stats["total_size"] += file_record.file_size or 0
                    
                    if not file_record.is_deleted:
                        stats["active_files"] += 1
                        stats["active_size"] += file_record.file_size or 0
                    else:
                        stats["deleted_files"] += 1
                    
                    if file_record.is_temporary:
                        stats["temporary_files"] += 1
                        stats["temporary_size"] += file_record.file_size or 0
                    
                    if file_record.is_encrypted:
                        stats["encrypted_files"] += 1
                    
                    # Count by MIME type
                    mime_type = file_record.mime_type or "unknown"
                    if mime_type not in stats["by_type"]:
                        stats["by_type"][mime_type] = {"count": 0, "size": 0}
                    
                    stats["by_type"][mime_type]["count"] += 1
                    stats["by_type"][mime_type]["size"] += file_record.file_size or 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def _delete_physical_file(self, file_path: str) -> bool:
        """
        Delete physical file from disk.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted physical file: {file_path}")
                return True
            else:
                logger.warning(f"Physical file not found: {file_path}")
                return True  # Consider it successful if file doesn't exist
                
        except Exception as e:
            logger.error(f"Failed to delete physical file {file_path}: {e}")
            return False
    
    async def schedule_file_deletion(
        self, 
        file_path: str, 
        delay_hours: int = 24
    ) -> bool:
        """
        Schedule a file for deletion after a delay.
        
        Args:
            file_path: Path to file
            delay_hours: Hours to wait before deletion
            
        Returns:
            True if scheduled successfully
        """
        try:
            async with get_db_session() as session:
                stmt = select(AudioFile).where(AudioFile.file_path == file_path)
                result = await session.execute(stmt)
                file_record = result.scalar_one_or_none()
                
                if file_record:
                    file_record.expires_at = datetime.now() + timedelta(hours=delay_hours)
                    await session.commit()
                    logger.info(f"Scheduled file for deletion: {file_path}")
                    return True
                else:
                    logger.warning(f"File record not found for scheduling deletion: {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to schedule file deletion {file_path}: {e}")
            return False


# Global lifecycle manager instance
_lifecycle_manager: Optional[FileLifecycleManager] = None


async def get_lifecycle_manager() -> FileLifecycleManager:
    """Get global file lifecycle manager instance."""
    global _lifecycle_manager
    
    if _lifecycle_manager is None:
        _lifecycle_manager = FileLifecycleManager()
        await _lifecycle_manager.start()
    
    return _lifecycle_manager


async def close_lifecycle_manager():
    """Close global file lifecycle manager."""
    global _lifecycle_manager
    
    if _lifecycle_manager:
        await _lifecycle_manager.stop()
        _lifecycle_manager = None