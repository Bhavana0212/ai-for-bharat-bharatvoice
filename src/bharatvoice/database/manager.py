"""
Database management utilities for BharatVoice Assistant.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from .base import get_async_engine, get_sync_engine, create_tables, close_database
from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database management and operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.alembic_cfg = self._get_alembic_config()
    
    def _get_alembic_config(self) -> Config:
        """Get Alembic configuration."""
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", self.settings.database.url)
        return alembic_cfg
    
    async def initialize_database(self) -> bool:
        """
        Initialize database with tables and initial data.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Initializing database...")
            
            # Create tables using SQLAlchemy
            await create_tables()
            
            # Run any pending migrations
            await self.run_migrations()
            
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def run_migrations(self) -> bool:
        """
        Run database migrations.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Running database migrations...")
            
            # Run migrations in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                command.upgrade, 
                self.alembic_cfg, 
                "head"
            )
            
            logger.info("Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to run migrations: {e}")
            return False
    
    async def create_migration(self, message: str) -> bool:
        """
        Create a new migration.
        
        Args:
            message: Migration message
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Creating migration: {message}")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                command.revision,
                self.alembic_cfg,
                message,
                True  # autogenerate
            )
            
            logger.info("Migration created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            return False
    
    async def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            bool: True if successful
        """
        try:
            if not backup_path:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_bharatvoice_{timestamp}.sql"
            
            logger.info(f"Creating database backup: {backup_path}")
            
            # For SQLite, we can copy the file
            if "sqlite" in self.settings.database.url:
                import shutil
                db_path = self.settings.database.url.replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
            else:
                # For PostgreSQL/MySQL, use pg_dump/mysqldump
                # This would require additional implementation
                logger.warning("Backup for non-SQLite databases not implemented")
                return False
            
            logger.info("Database backup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    async def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Restoring database from: {backup_path}")
            
            if not Path(backup_path).exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # For SQLite, we can copy the file
            if "sqlite" in self.settings.database.url:
                import shutil
                db_path = self.settings.database.url.replace("sqlite:///", "")
                shutil.copy2(backup_path, db_path)
            else:
                # For PostgreSQL/MySQL, use psql/mysql
                logger.warning("Restore for non-SQLite databases not implemented")
                return False
            
            logger.info("Database restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
    
    async def health_check(self) -> dict:
        """
        Perform database health check.
        
        Returns:
            dict: Health check results
        """
        try:
            engine = get_async_engine()
            
            async with engine.connect() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database info
                if "sqlite" in self.settings.database.url:
                    version_result = await conn.execute(text("SELECT sqlite_version()"))
                    version = version_result.fetchone()[0]
                    db_type = "SQLite"
                elif "postgresql" in self.settings.database.url:
                    version_result = await conn.execute(text("SELECT version()"))
                    version = version_result.fetchone()[0]
                    db_type = "PostgreSQL"
                else:
                    version = "Unknown"
                    db_type = "Unknown"
                
                return {
                    "status": "healthy",
                    "database_type": db_type,
                    "version": version,
                    "url": self.settings.database.url.split("@")[-1] if "@" in self.settings.database.url else "local",
                    "pool_size": self.settings.database.pool_size,
                    "max_overflow": self.settings.database.max_overflow
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def optimize_database(self) -> bool:
        """
        Optimize database performance.
        
        Returns:
            bool: True if successful
        """
        try:
            logger.info("Optimizing database...")
            
            engine = get_async_engine()
            
            async with engine.connect() as conn:
                if "sqlite" in self.settings.database.url:
                    # SQLite optimization
                    await conn.execute(text("VACUUM"))
                    await conn.execute(text("ANALYZE"))
                    await conn.commit()
                elif "postgresql" in self.settings.database.url:
                    # PostgreSQL optimization
                    await conn.execute(text("VACUUM ANALYZE"))
                    await conn.commit()
                
            logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            return False
    
    async def cleanup_old_data(self, days: int = 30) -> bool:
        """
        Clean up old data based on retention policies.
        
        Args:
            days: Number of days to retain data
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"Cleaning up data older than {days} days...")
            
            engine = get_async_engine()
            
            async with engine.connect() as conn:
                # Clean up old audio files
                await conn.execute(text("""
                    DELETE FROM audio_files 
                    WHERE is_temporary = true 
                    AND created_at < datetime('now', '-{} days')
                """.format(days)))
                
                # Clean up old cache entries
                await conn.execute(text("""
                    DELETE FROM cache_entries 
                    WHERE expires_at < datetime('now')
                """))
                
                # Clean up old system metrics (keep aggregated data)
                await conn.execute(text("""
                    DELETE FROM system_metrics 
                    WHERE aggregation_window IS NULL 
                    AND created_at < datetime('now', '-{} days')
                """.format(days)))
                
                await conn.commit()
            
            logger.info("Data cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()