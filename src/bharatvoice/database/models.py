<<<<<<< HEAD
"""
Database models for BharatVoice Assistant.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    JSON, LargeBinary, String, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class User(Base):
    """User model for authentication and profile management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), unique=True, nullable=True, index=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # MFA
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32), nullable=True)
    backup_codes = Column(JSON, nullable=True)
    
    # Privacy and compliance
    data_retention_days = Column(Integer, default=365, nullable=False)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_date = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    audio_files = relationship("AudioFile", back_populates="user", cascade="all, delete-orphan")


class UserProfile(Base):
    """User profile model for personalization and preferences."""
    
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Personal information (encrypted)
    full_name = Column(String(255), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(20), nullable=True)
    
    # Location and regional context
    country = Column(String(100), default="India", nullable=False)
    state = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    timezone = Column(String(50), default="Asia/Kolkata", nullable=False)
    
    # Language preferences
    preferred_languages = Column(JSON, nullable=False, default=list)  # List of language codes
    primary_language = Column(String(10), default="hi", nullable=False)
    code_switching_preference = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0
    
    # Voice preferences
    preferred_voice_gender = Column(String(20), default="female", nullable=False)
    speech_rate = Column(Float, default=1.0, nullable=False)  # 0.5 to 2.0
    voice_volume = Column(Float, default=0.8, nullable=False)  # 0.0 to 1.0
    
    # Cultural preferences
    cultural_context = Column(JSON, nullable=True)  # Regional customs, festivals, etc.
    formality_preference = Column(String(20), default="medium", nullable=False)  # low, medium, high
    
    # Learning and adaptation
    interaction_count = Column(Integer, default=0, nullable=False)
    successful_interactions = Column(Integer, default=0, nullable=False)
    vocabulary_adaptations = Column(JSON, nullable=True)  # Learned terms and preferences
    accent_adaptations = Column(JSON, nullable=True)  # Speech recognition adaptations
    
    # Privacy settings
    data_sharing_consent = Column(Boolean, default=False, nullable=False)
    analytics_consent = Column(Boolean, default=False, nullable=False)
    voice_data_retention = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="profile")


class ConversationSession(Base):
    """Conversation session model for context management."""
    
    __tablename__ = "conversation_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_type = Column(String(50), default="voice", nullable=False)  # voice, text, mixed
    device_info = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Conversation context
    conversation_history = Column(JSON, nullable=False, default=list)
    context_variables = Column(JSON, nullable=False, default=dict)
    current_intent = Column(String(100), nullable=True)
    language_detected = Column(String(10), nullable=True)
    
    # Session state
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_timeout = Column(Integer, default=1800, nullable=False)  # 30 minutes
    
    # Performance metrics
    total_interactions = Column(Integer, default=0, nullable=False)
    successful_interactions = Column(Integer, default=0, nullable=False)
    average_response_time = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    audio_files = relationship("AudioFile", back_populates="session", cascade="all, delete-orphan")


class AudioFile(Base):
    """Audio file model for voice data management."""
    
    __tablename__ = "audio_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversation_sessions.id"), nullable=True)
    
    # File metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    mime_type = Column(String(100), nullable=False)
    
    # Audio properties
    duration = Column(Float, nullable=True)  # seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    bit_depth = Column(Integer, nullable=True)
    
    # Processing metadata
    is_processed = Column(Boolean, default=False, nullable=False)
    transcription = Column(Text, nullable=True)
    language_detected = Column(String(10), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Storage and security
    is_encrypted = Column(Boolean, default=True, nullable=False)
    encryption_key_id = Column(String(100), nullable=True)
    checksum = Column(String(64), nullable=True)  # SHA-256
    
    # Lifecycle management
    is_temporary = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audio_files")
    session = relationship("ConversationSession", back_populates="audio_files")


class CacheEntry(Base):
    """Cache entry model for Redis backup and persistence."""
    
    __tablename__ = "cache_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cache metadata
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    cache_type = Column(String(50), nullable=False, index=True)  # session, translation, recognition, etc.
    
    # Data storage
    data = Column(JSON, nullable=True)
    binary_data = Column(LargeBinary, nullable=True)
    data_size = Column(Integer, nullable=False, default=0)
    
    # Cache management
    ttl = Column(Integer, nullable=True)  # Time to live in seconds
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Metadata
    tags = Column(JSON, nullable=True)  # For cache invalidation strategies
    version = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('cache_key', name='uq_cache_key'),
    )


class SystemMetrics(Base):
    """System metrics model for monitoring and performance tracking."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # counter, gauge, histogram
    service_name = Column(String(100), nullable=False, index=True)
    
    # Metric data
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    labels = Column(JSON, nullable=True)  # Additional metric labels
    
    # Context
    user_id = Column(UUID(as_uuid=True), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Aggregation support
    aggregation_window = Column(String(20), nullable=True)  # minute, hour, day
    aggregation_value = Column(Float, nullable=True)
    sample_count = Column(Integer, default=1, nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes for time-series queries
    __table_args__ = (
        UniqueConstraint('metric_name', 'service_name', 'timestamp', 'labels', name='uq_metric_timestamp'),
=======
"""
Database models for BharatVoice Assistant.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    JSON, LargeBinary, String, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class User(Base):
    """User model for authentication and profile management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), unique=True, nullable=True, index=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # MFA
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32), nullable=True)
    backup_codes = Column(JSON, nullable=True)
    
    # Privacy and compliance
    data_retention_days = Column(Integer, default=365, nullable=False)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_date = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    audio_files = relationship("AudioFile", back_populates="user", cascade="all, delete-orphan")


class UserProfile(Base):
    """User profile model for personalization and preferences."""
    
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Personal information (encrypted)
    full_name = Column(String(255), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(20), nullable=True)
    
    # Location and regional context
    country = Column(String(100), default="India", nullable=False)
    state = Column(String(100), nullable=True)
    city = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    timezone = Column(String(50), default="Asia/Kolkata", nullable=False)
    
    # Language preferences
    preferred_languages = Column(JSON, nullable=False, default=list)  # List of language codes
    primary_language = Column(String(10), default="hi", nullable=False)
    code_switching_preference = Column(Float, default=0.5, nullable=False)  # 0.0 to 1.0
    
    # Voice preferences
    preferred_voice_gender = Column(String(20), default="female", nullable=False)
    speech_rate = Column(Float, default=1.0, nullable=False)  # 0.5 to 2.0
    voice_volume = Column(Float, default=0.8, nullable=False)  # 0.0 to 1.0
    
    # Cultural preferences
    cultural_context = Column(JSON, nullable=True)  # Regional customs, festivals, etc.
    formality_preference = Column(String(20), default="medium", nullable=False)  # low, medium, high
    
    # Learning and adaptation
    interaction_count = Column(Integer, default=0, nullable=False)
    successful_interactions = Column(Integer, default=0, nullable=False)
    vocabulary_adaptations = Column(JSON, nullable=True)  # Learned terms and preferences
    accent_adaptations = Column(JSON, nullable=True)  # Speech recognition adaptations
    
    # Privacy settings
    data_sharing_consent = Column(Boolean, default=False, nullable=False)
    analytics_consent = Column(Boolean, default=False, nullable=False)
    voice_data_retention = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="profile")


class ConversationSession(Base):
    """Conversation session model for context management."""
    
    __tablename__ = "conversation_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    session_type = Column(String(50), default="voice", nullable=False)  # voice, text, mixed
    device_info = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Conversation context
    conversation_history = Column(JSON, nullable=False, default=list)
    context_variables = Column(JSON, nullable=False, default=dict)
    current_intent = Column(String(100), nullable=True)
    language_detected = Column(String(10), nullable=True)
    
    # Session state
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    session_timeout = Column(Integer, default=1800, nullable=False)  # 30 minutes
    
    # Performance metrics
    total_interactions = Column(Integer, default=0, nullable=False)
    successful_interactions = Column(Integer, default=0, nullable=False)
    average_response_time = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    audio_files = relationship("AudioFile", back_populates="session", cascade="all, delete-orphan")


class AudioFile(Base):
    """Audio file model for voice data management."""
    
    __tablename__ = "audio_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("conversation_sessions.id"), nullable=True)
    
    # File metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    mime_type = Column(String(100), nullable=False)
    
    # Audio properties
    duration = Column(Float, nullable=True)  # seconds
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    bit_depth = Column(Integer, nullable=True)
    
    # Processing metadata
    is_processed = Column(Boolean, default=False, nullable=False)
    transcription = Column(Text, nullable=True)
    language_detected = Column(String(10), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Storage and security
    is_encrypted = Column(Boolean, default=True, nullable=False)
    encryption_key_id = Column(String(100), nullable=True)
    checksum = Column(String(64), nullable=True)  # SHA-256
    
    # Lifecycle management
    is_temporary = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="audio_files")
    session = relationship("ConversationSession", back_populates="audio_files")


class CacheEntry(Base):
    """Cache entry model for Redis backup and persistence."""
    
    __tablename__ = "cache_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cache metadata
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    cache_type = Column(String(50), nullable=False, index=True)  # session, translation, recognition, etc.
    
    # Data storage
    data = Column(JSON, nullable=True)
    binary_data = Column(LargeBinary, nullable=True)
    data_size = Column(Integer, nullable=False, default=0)
    
    # Cache management
    ttl = Column(Integer, nullable=True)  # Time to live in seconds
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Metadata
    tags = Column(JSON, nullable=True)  # For cache invalidation strategies
    version = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Indexes
    __table_args__ = (
        UniqueConstraint('cache_key', name='uq_cache_key'),
    )


class SystemMetrics(Base):
    """System metrics model for monitoring and performance tracking."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)  # counter, gauge, histogram
    service_name = Column(String(100), nullable=False, index=True)
    
    # Metric data
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    labels = Column(JSON, nullable=True)  # Additional metric labels
    
    # Context
    user_id = Column(UUID(as_uuid=True), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Aggregation support
    aggregation_window = Column(String(20), nullable=True)  # minute, hour, day
    aggregation_value = Column(Float, nullable=True)
    sample_count = Column(Integer, default=1, nullable=False)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Indexes for time-series queries
    __table_args__ = (
        UniqueConstraint('metric_name', 'service_name', 'timestamp', 'labels', name='uq_metric_timestamp'),
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )