"""
Database connection management and pooling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from .base import get_db_session, get_async_engine, health_check
from bharatvoice.config.settings import get_settings

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Database connection pool manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self._engine = None
        self._session_factory = None
        self._health_check_task = None
        self._is_healthy = True
    
    async def initialize(self):
        """Initialize connection pool."""
        try:
            self._engine = get_async_engine()
            logger.info(f"Database connection pool initialized with {self.settings.database.pool_size} connections")
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def close(self):
        """Close connection pool."""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._engine:
                await self._engine.dispose()
                logger.info("Database connection pool closed")
                
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session from pool.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._is_healthy:
            raise RuntimeError("Database connection pool is unhealthy")
        
        async with get_db_session() as session:
            yield session
    
    async def _periodic_health_check(self):
        """Periodic health check for connection pool."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                self._is_healthy = await health_check()
                
                if not self._is_healthy:
                    logger.warning("Database health check failed")
                else:
                    logger.debug("Database health check passed")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self._is_healthy = False
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection pool is healthy."""
        return self._is_healthy
    
    async def get_pool_status(self) -> dict:
        """
        Get connection pool status.
        
        Returns:
            dict: Pool status information
        """
        try:
            if not self._engine:
                return {"status": "not_initialized"}
            
            pool = self._engine.pool
            
            return {
                "status": "healthy" if self._is_healthy else "unhealthy",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
            }
            
        except Exception as e:
            logger.error(f"Failed to get pool status: {e}")
            return {"status": "error", "error": str(e)}


# Global connection pool instance
connection_pool = ConnectionPool()


async def init_connection_pool():
    """Initialize global connection pool."""
    await connection_pool.initialize()


async def close_connection_pool():
    """Close global connection pool."""
    await connection_pool.close()


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session from global connection pool.
    
    Yields:
        AsyncSession: Database session
    """
    async with connection_pool.get_session() as session:
        yield session