# BharatVoice Assistant - Developer Documentation

## Overview

This documentation provides comprehensive guidance for developers working on or extending the BharatVoice Assistant system. It covers architecture, development setup, coding standards, testing practices, and contribution guidelines.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment Setup](#development-environment-setup)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [API Development](#api-development)
6. [Testing Framework](#testing-framework)
7. [Database Schema](#database-schema)
8. [Coding Standards](#coding-standards)
9. [Performance Guidelines](#performance-guidelines)
10. [Security Considerations](#security-considerations)
11. [Deployment Pipeline](#deployment-pipeline)
12. [Contributing Guidelines](#contributing-guidelines)

---

## Architecture Overview

### System Architecture

BharatVoice Assistant follows a microservices architecture with the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Load Balancer  │    │   Web Client    │
│   (FastAPI)     │    │    (Nginx)      │    │   (React/Vue)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Voice Processing│    │ Language Engine │    │Context Management│
│    Service      │    │    Service      │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Response Generation│  │External Services│    │   Auth Service  │
│    Service      │    │  Integration    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │   File Storage  │
│   Database      │    │     Cache       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Design Principles

1. **Microservices**: Loosely coupled, independently deployable services
2. **Event-Driven**: Asynchronous communication between services
3. **Scalability**: Horizontal scaling capabilities
4. **Resilience**: Fault tolerance and graceful degradation
5. **Security**: End-to-end encryption and secure authentication
6. **Cultural Awareness**: Deep understanding of Indian languages and culture

---

## Development Environment Setup

### Prerequisites

```bash
# System requirements
- Python 3.9+
- Node.js 16+ (for frontend development)
- PostgreSQL 13+
- Redis 6+
- Docker 20.10+
- Git 2.30+

# Python dependencies
pip install -e '.[dev]'

# Additional ML dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers openai-whisper
```

### Local Development Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/bharatvoice/assistant.git
   cd assistant
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env.development
   # Edit .env.development with local settings
   ```

3. **Database Setup**:
   ```bash
   # Start PostgreSQL and Redis
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   
   # Run migrations
   alembic upgrade head
   
   # Load sample data
   python scripts/load_sample_data.py
   ```

4. **Start Development Server**:
   ```bash
   # Start API server
   uvicorn bharatvoice.main:app --reload --host 0.0.0.0 --port 8000
   
   # Start frontend (if applicable)
   cd frontend && npm run dev
   ```

### Development Tools

```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
mypy src/

# Testing
pytest tests/ -v --cov=src/bharatvoice

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

## Project Structure

```
bharatvoice-assistant/
├── src/bharatvoice/           # Main application code
│   ├── api/                   # FastAPI routes and middleware
│   ├── core/                  # Core models and interfaces
│   ├── services/              # Business logic services
│   │   ├── voice_processing/  # Voice/audio processing
│   │   ├── language_engine/   # Language processing
│   │   ├── context_management/# User context and profiles
│   │   ├── response_generation/# NLU and response generation
│   │   ├── auth/              # Authentication services
│   │   ├── external_integrations/# External API integrations
│   │   ├── platform_integrations/# Platform integrations
│   │   ├── learning/          # Adaptive learning
│   │   └── offline_sync/      # Offline capabilities
│   ├── database/              # Database models and connections
│   ├── cache/                 # Caching layer
│   ├── storage/               # File storage
│   ├── utils/                 # Utility functions
│   ├── config/                # Configuration management
│   └── main.py                # Application entry point
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   ├── property/              # Property-based tests
│   └── conftest.py            # Test configuration
├── alembic/                   # Database migrations
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── k8s/                       # Kubernetes configurations
├── docker/                    # Docker configurations
├── .github/                   # GitHub workflows
├── pyproject.toml             # Project configuration
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Local development
└── README.md                  # Project overview
```

---

## Core Components

### 1. Voice Processing Service

**Location**: `src/bharatvoice/services/voice_processing/`

**Key Classes**:
- `AudioProcessor`: Audio preprocessing and noise filtering
- `TTSEngine`: Text-to-speech synthesis
- `AdaptiveTTSEngine`: User-adaptive TTS
- `VoiceProcessingService`: Main service orchestrator

**Example Usage**:
```python
from bharatvoice.services.voice_processing.service import VoiceProcessingService
from bharatvoice.core.models import AudioBuffer, LanguageCode

service = VoiceProcessingService()

# Process audio input
result = await service.process_voice_input(
    audio_buffer=audio_buffer,
    user_id="user_123",
    language_hint=LanguageCode.HINDI
)

# Generate speech
audio_response = await service.synthesize_response(
    text="नमस्ते, आप कैसे हैं?",
    language=LanguageCode.HINDI,
    user_id="user_123"
)
```

### 2. Language Engine Service

**Location**: `src/bharatvoice/services/language_engine/`

**Key Classes**:
- `MultilingualASREngine`: Speech recognition
- `TranslationEngine`: Language translation
- `CodeSwitchingDetector`: Code-switching detection
- `LanguageEngineService`: Main service orchestrator

**Example Usage**:
```python
from bharatvoice.services.language_engine.service import LanguageEngineService

service = LanguageEngineService()

# Recognize speech
recognition_result = await service.recognize_speech(
    audio_buffer=audio_buffer,
    language_hint=LanguageCode.HINDI
)

# Translate text
translation_result = await service.translate_text(
    text="Hello, how are you?",
    source_language=LanguageCode.ENGLISH_IN,
    target_language=LanguageCode.HINDI
)
```

### 3. Context Management Service

**Location**: `src/bharatvoice/services/context_management/`

**Key Classes**:
- `UserProfileManager`: User profile management
- `RegionalContextManager`: Regional context handling
- `ConversationManager`: Conversation state management
- `ContextManagementService`: Main service orchestrator

**Example Usage**:
```python
from bharatvoice.services.context_management.service import ContextManagementService

service = ContextManagementService()

# Get user context
user_context = await service.get_user_context(user_id="user_123")

# Update regional context
await service.update_regional_context(
    user_id="user_123",
    location_data=location_data
)
```

### 4. Response Generation Service

**Location**: `src/bharatvoice/services/response_generation/`

**Key Classes**:
- `NLUService`: Natural language understanding
- `ResponseGenerator`: Response generation
- `CulturalContextInterpreter`: Cultural context analysis
- `ResponseGenerationService`: Main service orchestrator

**Example Usage**:
```python
from bharatvoice.services.response_generation.response_generator import ResponseGenerator

generator = ResponseGenerator()

# Generate response
response = await generator.generate_response(
    user_input="मौसम कैसा है?",
    user_context=user_context,
    conversation_state=conversation_state
)
```

---

## API Development

### FastAPI Application Structure

**Main Application** (`src/bharatvoice/main.py`):
```python
from fastapi import FastAPI
from bharatvoice.api import voice, language, context, auth
from bharatvoice.api.middleware import setup_middleware

app = FastAPI(
    title="BharatVoice Assistant API",
    description="AI-powered multilingual voice assistant for India",
    version="1.0.0"
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(voice.router, prefix="/v1/voice", tags=["voice"])
app.include_router(language.router, prefix="/v1/language", tags=["language"])
app.include_router(context.router, prefix="/v1/context", tags=["context"])
app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
```

### Creating New API Endpoints

1. **Define Pydantic Models**:
```python
# src/bharatvoice/api/models/voice.py
from pydantic import BaseModel
from bharatvoice.core.models import LanguageCode

class VoiceProcessRequest(BaseModel):
    audio_data: bytes
    language: LanguageCode = None
    user_id: str
    context: dict = {}

class VoiceProcessResponse(BaseModel):
    transcription: str
    confidence: float
    intent: dict
    response: dict
    processing_time: float
```

2. **Create Router**:
```python
# src/bharatvoice/api/voice.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from bharatvoice.api.models.voice import VoiceProcessRequest, VoiceProcessResponse
from bharatvoice.services.voice_processing.service import VoiceProcessingService
from bharatvoice.api.dependencies import get_current_user, get_voice_service

router = APIRouter()

@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    audio_file: UploadFile = File(...),
    language: str = None,
    current_user = Depends(get_current_user),
    voice_service: VoiceProcessingService = Depends(get_voice_service)
):
    """Process voice input and return transcription with intent analysis."""
    try:
        # Convert uploaded file to AudioBuffer
        audio_buffer = await convert_upload_to_audio_buffer(audio_file)
        
        # Process voice input
        result = await voice_service.process_voice_input(
            audio_buffer=audio_buffer,
            user_id=current_user.id,
            language_hint=LanguageCode(language) if language else None
        )
        
        return VoiceProcessResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Error Handling

**Custom Exception Classes**:
```python
# src/bharatvoice/core/exceptions.py
class BharatVoiceException(Exception):
    """Base exception for BharatVoice application."""
    pass

class VoiceProcessingError(BharatVoiceException):
    """Voice processing related errors."""
    pass

class LanguageNotSupportedError(BharatVoiceException):
    """Language not supported error."""
    pass

class AuthenticationError(BharatVoiceException):
    """Authentication related errors."""
    pass
```

**Error Handler**:
```python
# src/bharatvoice/api/middleware.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from bharatvoice.core.exceptions import BharatVoiceException

@app.exception_handler(BharatVoiceException)
async def bharatvoice_exception_handler(request: Request, exc: BharatVoiceException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.__class__.__name__,
                "message": str(exc),
                "request_id": request.state.request_id
            }
        }
    )
```

---

## Testing Framework

### Property-Based Testing

BharatVoice uses property-based testing with Hypothesis to ensure system reliability:

**Example Property Test**:
```python
# tests/test_voice_processing_properties.py
import pytest
from hypothesis import given, strategies as st
from bharatvoice.services.voice_processing.tts_engine import TTSEngine
from bharatvoice.core.models import LanguageCode, AccentType

class TestTTSProperties:
    @pytest.fixture
    def tts_engine(self):
        return TTSEngine(sample_rate=22050, quality='high')
    
    @given(
        text=st.text(min_size=1, max_size=200),
        language=st.sampled_from(list(LanguageCode)),
        accent=st.sampled_from(list(AccentType))
    )
    @pytest.mark.asyncio
    async def test_tts_synthesis_completeness(self, tts_engine, text, language, accent):
        """Property: TTS should always produce valid audio output."""
        result = await tts_engine.synthesize_speech(text, language, accent)
        
        # Properties that should always hold
        assert len(result.data) > 0, "Audio data should not be empty"
        assert result.duration > 0, "Duration should be positive"
        assert result.sample_rate > 0, "Sample rate should be positive"
        assert 0.0 <= max(result.data) <= 1.0, "Audio should be normalized"
```

### Unit Testing

**Service Testing Example**:
```python
# tests/unit/test_language_engine.py
import pytest
from unittest.mock import Mock, AsyncMock
from bharatvoice.services.language_engine.service import LanguageEngineService

class TestLanguageEngineService:
    @pytest.fixture
    def mock_asr_engine(self):
        mock = Mock()
        mock.recognize_speech = AsyncMock(return_value={
            "text": "नमस्ते",
            "confidence": 0.95,
            "language": "hi"
        })
        return mock
    
    @pytest.fixture
    def service(self, mock_asr_engine):
        service = LanguageEngineService()
        service.asr_engine = mock_asr_engine
        return service
    
    @pytest.mark.asyncio
    async def test_recognize_speech_success(self, service, mock_asr_engine):
        """Test successful speech recognition."""
        audio_buffer = Mock()
        
        result = await service.recognize_speech(audio_buffer)
        
        assert result["text"] == "नमस्ते"
        assert result["confidence"] == 0.95
        mock_asr_engine.recognize_speech.assert_called_once_with(audio_buffer)
```

### Integration Testing

**API Integration Test**:
```python
# tests/integration/test_voice_api.py
import pytest
from fastapi.testclient import TestClient
from bharatvoice.main import app

client = TestClient(app)

class TestVoiceAPI:
    def test_process_voice_endpoint(self):
        """Test voice processing endpoint."""
        # Prepare test audio file
        with open("tests/fixtures/test_audio.wav", "rb") as audio_file:
            response = client.post(
                "/v1/voice/process",
                files={"audio_file": audio_file},
                data={"language": "hi"},
                headers={"Authorization": "Bearer test_token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "transcription" in data
        assert "confidence" in data
        assert "intent" in data
```

### Test Data Management

**Fixtures and Test Data**:
```python
# tests/conftest.py
import pytest
from bharatvoice.core.models import UserProfile, LanguageCode

@pytest.fixture
def sample_user_profile():
    return UserProfile(
        user_id="test_user_123",
        name="Test User",
        preferred_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
        location={"city": "Delhi", "state": "Delhi"},
        accessibility_settings={
            "volume_level": "medium",
            "speech_rate": "normal"
        }
    )

@pytest.fixture
def sample_audio_buffer():
    # Generate or load sample audio data
    return AudioBuffer(
        data=[0.1, 0.2, 0.3] * 1000,  # Sample audio data
        sample_rate=16000,
        channels=1,
        format=AudioFormat.WAV,
        duration=3.0
    )
```

---

## Database Schema

### Core Tables

**Users Table**:
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20) UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE
);
```

**User Profiles Table**:
```sql
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preferred_languages TEXT[] DEFAULT '{}',
    location JSONB,
    accessibility_settings JSONB DEFAULT '{}',
    cultural_preferences JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Conversation History Table**:
```sql
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    input_text TEXT,
    input_language VARCHAR(10),
    response_text TEXT,
    response_language VARCHAR(10),
    intent VARCHAR(100),
    confidence DECIMAL(3,2),
    processing_time DECIMAL(6,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_conversation_user_session (user_id, session_id),
    INDEX idx_conversation_created_at (created_at)
);
```

### Database Models

**SQLAlchemy Models**:
```python
# src/bharatvoice/database/models.py
from sqlalchemy import Column, String, DateTime, Boolean, Text, DECIMAL, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    phone = Column(String(20), unique=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    phone_verified = Column(Boolean, default=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    preferred_languages = Column(ARRAY(String), default=[])
    location = Column(JSON)
    accessibility_settings = Column(JSON, default={})
    cultural_preferences = Column(JSON, default={})
    privacy_settings = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

### Migration Management

**Creating Migrations**:
```bash
# Create new migration
alembic revision --autogenerate -m "Add user profiles table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

**Migration Example**:
```python
# alembic/versions/001_add_user_profiles.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    op.create_table(
        'user_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('preferred_languages', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('location', postgresql.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )

def downgrade():
    op.drop_table('user_profiles')
```

---

## Coding Standards

### Python Code Style

**Follow PEP 8 with these additions**:

1. **Line Length**: 88 characters (Black default)
2. **Import Organization**:
   ```python
   # Standard library imports
   import os
   import sys
   from typing import List, Dict, Optional
   
   # Third-party imports
   import fastapi
   import sqlalchemy
   from pydantic import BaseModel
   
   # Local imports
   from bharatvoice.core.models import LanguageCode
   from bharatvoice.services.base import BaseService
   ```

3. **Type Hints**: Always use type hints
   ```python
   async def process_audio(
       audio_buffer: AudioBuffer,
       language: Optional[LanguageCode] = None
   ) -> ProcessingResult:
       """Process audio buffer and return results."""
       pass
   ```

4. **Docstrings**: Use Google-style docstrings
   ```python
   def calculate_confidence(
       predictions: List[float],
       threshold: float = 0.5
   ) -> float:
       """Calculate confidence score from predictions.
       
       Args:
           predictions: List of prediction scores
           threshold: Minimum threshold for confidence
           
       Returns:
           Calculated confidence score between 0 and 1
           
       Raises:
           ValueError: If predictions list is empty
       """
       pass
   ```

### Code Organization

**Service Layer Pattern**:
```python
# src/bharatvoice/services/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import structlog

logger = structlog.get_logger(__name__)

class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(self):
        self.logger = logger.bind(service=self.__class__.__name__)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service resources."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass
```

**Dependency Injection**:
```python
# src/bharatvoice/api/dependencies.py
from fastapi import Depends
from bharatvoice.services.voice_processing.service import VoiceProcessingService
from bharatvoice.database.connection import get_database_session

async def get_voice_service(
    db_session = Depends(get_database_session)
) -> VoiceProcessingService:
    """Get voice processing service instance."""
    service = VoiceProcessingService(db_session=db_session)
    await service.initialize()
    return service
```

### Error Handling Patterns

**Structured Error Handling**:
```python
from bharatvoice.core.exceptions import VoiceProcessingError
import structlog

logger = structlog.get_logger(__name__)

async def process_voice_input(audio_buffer: AudioBuffer) -> ProcessingResult:
    """Process voice input with proper error handling."""
    try:
        # Validate input
        if not audio_buffer or len(audio_buffer.data) == 0:
            raise VoiceProcessingError("Empty audio buffer provided")
        
        # Process audio
        result = await _internal_processing(audio_buffer)
        
        logger.info(
            "Voice processing completed",
            duration=result.processing_time,
            confidence=result.confidence
        )
        
        return result
        
    except VoiceProcessingError:
        # Re-raise known errors
        raise
    except Exception as e:
        # Log and wrap unexpected errors
        logger.error(
            "Unexpected error in voice processing",
            error=str(e),
            audio_duration=audio_buffer.duration
        )
        raise VoiceProcessingError(f"Processing failed: {str(e)}") from e
```

---

## Performance Guidelines

### Async Programming

**Use async/await consistently**:
```python
# Good: Async all the way
async def process_request(request: ProcessingRequest) -> ProcessingResult:
    # Concurrent processing
    audio_task = asyncio.create_task(process_audio(request.audio))
    context_task = asyncio.create_task(get_user_context(request.user_id))
    
    # Wait for both to complete
    audio_result, user_context = await asyncio.gather(audio_task, context_task)
    
    return ProcessingResult(audio=audio_result, context=user_context)

# Bad: Blocking operations
def process_request_sync(request: ProcessingRequest) -> ProcessingResult:
    audio_result = process_audio_sync(request.audio)  # Blocks
    user_context = get_user_context_sync(request.user_id)  # Blocks
    return ProcessingResult(audio=audio_result, context=user_context)
```

### Database Optimization

**Connection Pooling**:
```python
# src/bharatvoice/database/connection.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
```

**Query Optimization**:
```python
# Use select with specific columns
async def get_user_profile(user_id: str) -> UserProfile:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserProfile.preferred_languages, UserProfile.location)
            .where(UserProfile.user_id == user_id)
        )
        return result.first()

# Use bulk operations for multiple records
async def update_user_preferences(updates: List[Dict]) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(UserProfile),
            updates
        )
        await session.commit()
```

### Caching Strategies

**Redis Caching**:
```python
# src/bharatvoice/cache/decorators.py
from functools import wraps
from bharatvoice.cache.redis_cache import get_redis_client
import json

def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Cache function result in Redis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            redis_client = await get_redis_client()
            cached_result = await redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl=1800, key_prefix="weather")
async def get_weather_data(city: str) -> Dict:
    """Get weather data with caching."""
    return await external_weather_api.get_weather(city)
```

---

## Security Considerations

### Authentication and Authorization

**JWT Token Management**:
```python
# src/bharatvoice/services/auth/jwt_manager.py
from jose import JWTError, jwt
from datetime import datetime, timedelta
from bharatvoice.core.exceptions import AuthenticationError

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
```

### Data Encryption

**Voice Data Encryption**:
```python
# src/bharatvoice/services/auth/encryption_manager.py
from cryptography.fernet import Fernet
from bharatvoice.core.models import AudioBuffer

class EncryptionManager:
    def __init__(self, encryption_key: bytes):
        self.cipher_suite = Fernet(encryption_key)
    
    def encrypt_audio_data(self, audio_buffer: AudioBuffer) -> bytes:
        """Encrypt audio data."""
        audio_bytes = bytes(audio_buffer.data)
        encrypted_data = self.cipher_suite.encrypt(audio_bytes)
        return encrypted_data
    
    def decrypt_audio_data(self, encrypted_data: bytes) -> AudioBuffer:
        """Decrypt audio data."""
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_data)
        # Convert back to AudioBuffer
        return AudioBuffer.from_bytes(decrypted_bytes)
```

### Input Validation

**Pydantic Models with Validation**:
```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
from bharatvoice.core.models import LanguageCode

class VoiceProcessRequest(BaseModel):
    audio_data: bytes = Field(..., min_length=1, max_length=50*1024*1024)  # Max 50MB
    language: Optional[LanguageCode] = None
    user_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        """Validate audio data format."""
        if not v or len(v) < 100:  # Minimum audio size
            raise ValueError("Audio data too small")
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user ID format."""
        if len(v) < 3 or len(v) > 50:
            raise ValueError("User ID must be between 3 and 50 characters")
        return v
```

---

## Deployment Pipeline

### CI/CD Configuration

**GitHub Actions Workflow**:
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: bharatvoice_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e '.[dev]'
    
    - name: Run linting
      run: |
        black --check src/ tests/
        isort --check-only src/ tests/
        flake8 src/ tests/
        mypy src/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src/bharatvoice --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t bharatvoice-assistant:${{ github.sha }} .
        docker tag bharatvoice-assistant:${{ github.sha }} bharatvoice-assistant:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push bharatvoice-assistant:${{ github.sha }}
        docker push bharatvoice-assistant:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # Deployment script here
        echo "Deploying to production..."
```

### Docker Configuration

**Multi-stage Dockerfile**:
```dockerfile
# Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.9-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create non-root user
RUN useradd --create-home --shell /bin/bash bharatvoice
USER bharatvoice

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "bharatvoice.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Contributing Guidelines

### Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/bharatvoice-assistant.git
   cd bharatvoice-assistant
   git remote add upstream https://github.com/bharatvoice/assistant.git
   ```

2. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**:
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

4. **Test Changes**:
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run specific test categories
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   pytest tests/property/ -v
   
   # Check code quality
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit and Push**:
   ```bash
   git add .
   git commit -m "feat: add new voice processing feature"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**:
   - Use descriptive title and description
   - Link related issues
   - Add screenshots/demos if applicable

### Code Review Process

**Pull Request Checklist**:
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

**Review Criteria**:
1. **Functionality**: Does the code work as intended?
2. **Code Quality**: Is the code clean, readable, and maintainable?
3. **Testing**: Are there adequate tests with good coverage?
4. **Performance**: Are there any performance implications?
5. **Security**: Are there any security concerns?
6. **Documentation**: Is the code properly documented?

### Issue Reporting

**Bug Report Template**:
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- BharatVoice version: [e.g., 1.0.0]

## Additional Context
Any other relevant information
```

**Feature Request Template**:
```markdown
## Feature Description
Brief description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches considered

## Additional Context
Any other relevant information
```

---

## Resources and References

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)

### Indian Language Resources
- [Universal Dependencies for Indian Languages](https://universaldependencies.org/)
- [Indian Language Technology Proliferation](https://www.cdac.in/index.aspx?id=st_iltp)
- [AI4Bharat Resources](https://ai4bharat.org/)

### Machine Learning
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Text-to-Speech](https://cloud.google.com/text-to-speech)

### Community
- **GitHub**: https://github.com/bharatvoice/assistant
- **Discord**: https://discord.gg/bharatvoice
- **Forum**: https://community.bharatvoice.ai
- **Email**: developers@bharatvoice.ai

---

**Happy Coding! 🚀**

*Building the future of voice interaction for India, one commit at a time.*