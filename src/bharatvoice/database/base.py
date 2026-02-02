"""
Database base configuration and session management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from bharatvoice.config.settings import get_settings

# Create declarative base
Base = declarative_base()

# Global variables for database engines and sessions
_async_engine = None
_sync_engine = None
_async_session_factory = None
_sync_session_factory = None


def _get_database_url(async_mode: bool = True) -> str:
    """Get database URL with appropriate driver for async/sync mode."""
    settings = get_settings()
    url = settings.database.url
    
    if async_mode:
        # Convert sync URLs to async
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///")
        elif url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        elif url.startswith("mysql://"):
            return url.replace("mysql://", "mysql+aiomysql://")
    
    return url


def init_database():
    """Initialize database engines and session factories."""
    global _async_engine, _sync_engine, _async_session_factory, _sync_session_factory
    
    settings = get_settings()
    
    # Create async engine
    async_url = _get_database_url(async_mode=True)
    _async_engine = create_async_engine(
        async_url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        # SQLite specific settings
        poolclass=StaticPool if "sqlite" in async_url else None,
        connect_args={"check_same_thread": False} if "sqlite" in async_url else {},
    )
    
    # Create sync engine for migrations
    sync_url = _get_database_url(async_mode=False)
    _sync_engine = create_engine(
        sync_url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        # SQLite specific settings
        poolclass=StaticPool if "sqlite" in sync_url else None,
        connect_args={"check_same_thread": False} if "sqlite" in sync_url else {},
    )
    
    # Create session factories
    _async_session_factory = async_sessionmaker(
        _async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    _sync_session_factory = sessionmaker(
        _sync_engine,
        expire_on_commit=False
    )
    
    # Enable WAL mode for SQLite
    if "sqlite" in sync_url:
        @event.listens_for(_sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()


async def create_tables():
    """Create all database tables."""
    if _async_engine is None:
        init_database()
    
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_sync_engine():
    """Get synchronous database engine for migrations."""
    if _sync_engine is None:
        init_database()
    return _sync_engine


def get_async_engine():
    """Get asynchronous database engine."""
    if _async_engine is None:
        init_database()
    return _async_engine


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.
    
    Yields:
        AsyncSession: Database session
    """
    if _async_session_factory is None:
        init_database()
    
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get synchronous database session for migrations."""
    if _sync_session_factory is None:
        init_database()
    return _sync_session_factory()


async def health_check() -> bool:
    """
    Check database connectivity.
    
    Returns:
        bool: True if database is accessible
    """
    try:
        async with get_db_session() as session:
            await session.execute("SELECT 1")
        return True
    except Exception:
        return False


async def close_database():
    """Close database connections."""
    global _async_engine, _sync_engine
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None