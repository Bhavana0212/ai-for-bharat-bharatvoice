<<<<<<< HEAD
"""
Pytest configuration and fixtures for BharatVoice Assistant tests.

This module provides common test fixtures, configuration, and utilities
for unit tests, integration tests, and property-based tests.
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import settings, Verbosity

from bharatvoice.main import create_app
from bharatvoice.config import get_settings
from bharatvoice.core.models import (
    AudioBuffer,
    ConversationState,
    LanguageCode,
    UserProfile,
    LocationData,
)


# Hypothesis settings for property-based tests
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=10, verbosity=Verbosity.normal)

# Load appropriate profile based on environment
settings.load_profile("default")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    """Create FastAPI application for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def settings():
    """Get application settings for testing."""
    return get_settings()


@pytest.fixture
def sample_audio_buffer() -> AudioBuffer:
    """Create sample audio buffer for testing."""
    return AudioBuffer(
        data=[0.1, 0.2, -0.1, -0.2] * 1000,  # 4000 samples
        sample_rate=16000,
        channels=1,
        duration=0.25  # 250ms
    )


@pytest.fixture
def sample_user_profile() -> UserProfile:
    """Create sample user profile for testing."""
    return UserProfile(
        preferred_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
        primary_language=LanguageCode.HINDI,
        location=LocationData(
            latitude=28.6139,
            longitude=77.2090,
            city="New Delhi",
            state="Delhi",
            country="India"
        )
    )


@pytest.fixture
def sample_conversation_state(sample_user_profile) -> ConversationState:
    """Create sample conversation state for testing."""
    return ConversationState(
        user_id=sample_user_profile.user_id,
        current_language=LanguageCode.HINDI,
        conversation_history=[],
        context_variables={},
        is_active=True
    )


@pytest.fixture
def mock_audio_processor():
    """Create mock audio processor for testing."""
    mock = Mock()
    mock.process_audio_stream.return_value = Mock()
    mock.detect_voice_activity.return_value = Mock()
    mock.synthesize_speech.return_value = Mock()
    mock.filter_background_noise.return_value = Mock()
    return mock


@pytest.fixture
def mock_language_engine():
    """Create mock language engine for testing."""
    mock = Mock()
    mock.recognize_speech.return_value = Mock()
    mock.detect_code_switching.return_value = []
    mock.translate_text.return_value = "translated text"
    mock.adapt_to_regional_accent.return_value = "adapted_model_id"
    mock.detect_language.return_value = LanguageCode.HINDI
    return mock


@pytest.fixture
def mock_context_manager():
    """Create mock context manager for testing."""
    mock = Mock()
    mock.maintain_conversation_state.return_value = Mock()
    mock.update_user_profile.return_value = Mock()
    mock.get_regional_context.return_value = Mock()
    mock.learn_from_interaction.return_value = {}
    mock.get_conversation_state.return_value = Mock()
    mock.get_user_profile.return_value = Mock()
    return mock


@pytest.fixture
def mock_response_generator():
    """Create mock response generator for testing."""
    mock = Mock()
    mock.process_query.return_value = {"intent": "greeting", "entities": []}
    mock.generate_response.return_value = Mock()
    mock.integrate_external_service.return_value = Mock()
    mock.format_cultural_response.return_value = Mock()
    return mock


# Property-based test strategies
from hypothesis import strategies as st


@st.composite
def audio_buffer_strategy(draw):
    """Hypothesis strategy for generating AudioBuffer instances."""
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    duration = draw(st.floats(min_value=0.1, max_value=10.0))
    num_samples = int(sample_rate * duration)
    
    data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=num_samples,
        max_size=num_samples
    ))
    
    return AudioBuffer(
        data=data,
        sample_rate=sample_rate,
        channels=draw(st.sampled_from([1, 2])),
        duration=duration
    )


@st.composite
def language_code_strategy(draw):
    """Hypothesis strategy for generating LanguageCode instances."""
    return draw(st.sampled_from(list(LanguageCode)))


@st.composite
def user_profile_strategy(draw):
    """Hypothesis strategy for generating UserProfile instances."""
    preferred_languages = draw(st.lists(
        language_code_strategy(),
        min_size=1,
        max_size=5,
        unique=True
    ))
    
    primary_language = draw(st.sampled_from(preferred_languages))
    
    return UserProfile(
        preferred_languages=preferred_languages,
        primary_language=primary_language,
        location=draw(location_data_strategy()) if draw(st.booleans()) else None
    )


@st.composite
def location_data_strategy(draw):
    """Hypothesis strategy for generating LocationData instances."""
    return LocationData(
        latitude=draw(st.floats(min_value=-90.0, max_value=90.0)),
        longitude=draw(st.floats(min_value=-180.0, max_value=180.0)),
        city=draw(st.text(min_size=1, max_size=50)),
        state=draw(st.text(min_size=1, max_size=50)),
        country=draw(st.just("India")),
        postal_code=draw(st.text(min_size=6, max_size=6)) if draw(st.booleans()) else None
    )


# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


# Async test utilities
@pytest.fixture
async def async_client(app):
    """Create async test client for FastAPI application."""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
=======
"""
Pytest configuration and fixtures for BharatVoice Assistant tests.

This module provides common test fixtures, configuration, and utilities
for unit tests, integration tests, and property-based tests.
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator
from unittest.mock import Mock

from fastapi.testclient import TestClient
from hypothesis import settings, Verbosity

from bharatvoice.main import create_app
from bharatvoice.config import get_settings
from bharatvoice.core.models import (
    AudioBuffer,
    ConversationState,
    LanguageCode,
    UserProfile,
    LocationData,
)


# Hypothesis settings for property-based tests
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=10, verbosity=Verbosity.normal)

# Load appropriate profile based on environment
settings.load_profile("default")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    """Create FastAPI application for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def settings():
    """Get application settings for testing."""
    return get_settings()


@pytest.fixture
def sample_audio_buffer() -> AudioBuffer:
    """Create sample audio buffer for testing."""
    return AudioBuffer(
        data=[0.1, 0.2, -0.1, -0.2] * 1000,  # 4000 samples
        sample_rate=16000,
        channels=1,
        duration=0.25  # 250ms
    )


@pytest.fixture
def sample_user_profile() -> UserProfile:
    """Create sample user profile for testing."""
    return UserProfile(
        preferred_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
        primary_language=LanguageCode.HINDI,
        location=LocationData(
            latitude=28.6139,
            longitude=77.2090,
            city="New Delhi",
            state="Delhi",
            country="India"
        )
    )


@pytest.fixture
def sample_conversation_state(sample_user_profile) -> ConversationState:
    """Create sample conversation state for testing."""
    return ConversationState(
        user_id=sample_user_profile.user_id,
        current_language=LanguageCode.HINDI,
        conversation_history=[],
        context_variables={},
        is_active=True
    )


@pytest.fixture
def mock_audio_processor():
    """Create mock audio processor for testing."""
    mock = Mock()
    mock.process_audio_stream.return_value = Mock()
    mock.detect_voice_activity.return_value = Mock()
    mock.synthesize_speech.return_value = Mock()
    mock.filter_background_noise.return_value = Mock()
    return mock


@pytest.fixture
def mock_language_engine():
    """Create mock language engine for testing."""
    mock = Mock()
    mock.recognize_speech.return_value = Mock()
    mock.detect_code_switching.return_value = []
    mock.translate_text.return_value = "translated text"
    mock.adapt_to_regional_accent.return_value = "adapted_model_id"
    mock.detect_language.return_value = LanguageCode.HINDI
    return mock


@pytest.fixture
def mock_context_manager():
    """Create mock context manager for testing."""
    mock = Mock()
    mock.maintain_conversation_state.return_value = Mock()
    mock.update_user_profile.return_value = Mock()
    mock.get_regional_context.return_value = Mock()
    mock.learn_from_interaction.return_value = {}
    mock.get_conversation_state.return_value = Mock()
    mock.get_user_profile.return_value = Mock()
    return mock


@pytest.fixture
def mock_response_generator():
    """Create mock response generator for testing."""
    mock = Mock()
    mock.process_query.return_value = {"intent": "greeting", "entities": []}
    mock.generate_response.return_value = Mock()
    mock.integrate_external_service.return_value = Mock()
    mock.format_cultural_response.return_value = Mock()
    return mock


# Property-based test strategies
from hypothesis import strategies as st


@st.composite
def audio_buffer_strategy(draw):
    """Hypothesis strategy for generating AudioBuffer instances."""
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    duration = draw(st.floats(min_value=0.1, max_value=10.0))
    num_samples = int(sample_rate * duration)
    
    data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=num_samples,
        max_size=num_samples
    ))
    
    return AudioBuffer(
        data=data,
        sample_rate=sample_rate,
        channels=draw(st.sampled_from([1, 2])),
        duration=duration
    )


@st.composite
def language_code_strategy(draw):
    """Hypothesis strategy for generating LanguageCode instances."""
    return draw(st.sampled_from(list(LanguageCode)))


@st.composite
def user_profile_strategy(draw):
    """Hypothesis strategy for generating UserProfile instances."""
    preferred_languages = draw(st.lists(
        language_code_strategy(),
        min_size=1,
        max_size=5,
        unique=True
    ))
    
    primary_language = draw(st.sampled_from(preferred_languages))
    
    return UserProfile(
        preferred_languages=preferred_languages,
        primary_language=primary_language,
        location=draw(location_data_strategy()) if draw(st.booleans()) else None
    )


@st.composite
def location_data_strategy(draw):
    """Hypothesis strategy for generating LocationData instances."""
    return LocationData(
        latitude=draw(st.floats(min_value=-90.0, max_value=90.0)),
        longitude=draw(st.floats(min_value=-180.0, max_value=180.0)),
        city=draw(st.text(min_size=1, max_size=50)),
        state=draw(st.text(min_size=1, max_size=50)),
        country=draw(st.just("India")),
        postal_code=draw(st.text(min_size=6, max_size=6)) if draw(st.booleans()) else None
    )


# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "property: Property-based tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


# Async test utilities
@pytest.fixture
async def async_client(app):
    """Create async test client for FastAPI application."""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as client:
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        yield client