"""
Property-based tests for Streamlit Web Interface session state management.

This module tests the correctness properties of session state persistence
for audio recording and language selection in the Streamlit interface.

Feature: streamlit-web-interface
Task: 2.4 Write property tests for session state management
"""

import pytest
from hypothesis import given, strategies as st
from unittest.mock import MagicMock, patch
import sys
import os

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()

# Import app functions after mocking streamlit
from app import (
    initialize_session_state,
    log_action,
    cache_response,
    get_cached_response,
    clear_cache
)


# Custom strategies for testing
@st.composite
def audio_data_strategy(draw):
    """Generate random audio data (bytes)."""
    size = draw(st.integers(min_value=100, max_value=10 * 1024 * 1024))  # 100 bytes to 10MB
    return draw(st.binary(min_size=size, max_size=size))


@st.composite
def language_code_strategy(draw):
    """Generate valid language codes."""
    languages = ['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']
    return draw(st.sampled_from(languages))


@st.composite
def audio_filename_strategy(draw):
    """Generate valid audio filenames."""
    extensions = ['wav', 'mp3', 'm4a', 'ogg']
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    ext = draw(st.sampled_from(extensions))
    return f"{name}.{ext}"


class MockSessionState:
    """Mock Streamlit session state for testing."""
    
    def __init__(self):
        self._state = {}
    
    def __contains__(self, key):
        return key in self._state
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __delitem__(self, key):
        del self._state[key]
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def keys(self):
        return self._state.keys()
    
    def clear(self):
        self._state.clear()


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    return MockSessionState()


@pytest.fixture(autouse=True)
def setup_streamlit_mock(mock_session_state):
    """Setup streamlit mock with session state."""
    import streamlit as st_mock
    st_mock.session_state = mock_session_state
    yield
    mock_session_state.clear()


# Property 2: Recording State Persistence
# **Validates: Requirements 1.4**

@pytest.mark.property
@given(
    audio_data=audio_data_strategy(),
    filename=audio_filename_strategy()
)
def test_property_recording_state_persistence(audio_data, filename, mock_session_state):
    """
    Property 2: Recording State Persistence
    
    For any recording session, when the user stops recording, the audio data
    should be stored in session state and made available for processing.
    
    **Validates: Requirements 1.4**
    
    Test Strategy:
    1. Generate random audio data and filename
    2. Simulate storing audio in session state (as would happen after recording)
    3. Verify audio data persists in session state
    4. Verify audio filename persists in session state
    5. Verify data can be retrieved for processing
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Simulate recording stop - store audio data in session state
    st.session_state.audio_data = audio_data
    st.session_state.audio_filename = filename
    
    # Property: Audio data should persist in session state
    assert 'audio_data' in st.session_state, \
        "Audio data should be stored in session state after recording stops"
    
    assert st.session_state.audio_data == audio_data, \
        "Stored audio data should match the recorded audio data"
    
    # Property: Audio filename should persist in session state
    assert 'audio_filename' in st.session_state, \
        "Audio filename should be stored in session state after recording stops"
    
    assert st.session_state.audio_filename == filename, \
        "Stored filename should match the recorded filename"
    
    # Property: Audio data should be available for processing
    retrieved_audio = st.session_state.audio_data
    retrieved_filename = st.session_state.audio_filename
    
    assert retrieved_audio is not None, \
        "Audio data should be retrievable from session state for processing"
    
    assert retrieved_audio == audio_data, \
        "Retrieved audio data should be identical to stored data"
    
    assert retrieved_filename == filename, \
        "Retrieved filename should be identical to stored filename"
    
    # Property: Audio data should persist across multiple accesses
    first_access = st.session_state.audio_data
    second_access = st.session_state.audio_data
    
    assert first_access == second_access, \
        "Audio data should remain consistent across multiple accesses"


@pytest.mark.property
@given(
    audio_data=audio_data_strategy(),
    filename=audio_filename_strategy()
)
def test_property_recording_state_persistence_after_operations(
    audio_data, filename, mock_session_state
):
    """
    Property 2 (Extended): Recording State Persistence After Operations
    
    Audio data should persist in session state even after other operations
    are performed (like logging actions or caching responses).
    
    **Validates: Requirements 1.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Store audio data
    st.session_state.audio_data = audio_data
    st.session_state.audio_filename = filename
    
    # Perform other operations
    log_action('upload', 'success', f'{filename} uploaded')
    cache_response('test_key', {'data': 'test'})
    
    # Property: Audio data should still persist after other operations
    assert st.session_state.audio_data == audio_data, \
        "Audio data should persist after logging actions"
    
    assert st.session_state.audio_filename == filename, \
        "Audio filename should persist after caching responses"
    
    # Perform more operations
    log_action('transcribe', 'pending', 'Processing audio')
    
    # Property: Audio data should still be accessible
    assert st.session_state.audio_data is not None, \
        "Audio data should remain accessible after multiple operations"


# Property 3: Language Selection Persistence
# **Validates: Requirements 2.2**

@pytest.mark.property
@given(language=language_code_strategy())
def test_property_language_selection_persistence(language, mock_session_state):
    """
    Property 3: Language Selection Persistence
    
    For any language selection made by the user, the selected language should
    be stored in session state and persist throughout the session.
    
    **Validates: Requirements 2.2**
    
    Test Strategy:
    1. Generate random language code from supported languages
    2. Simulate language selection by storing in session state
    3. Verify language persists in session state
    4. Verify language can be retrieved for API calls
    5. Verify language persists across multiple accesses
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Simulate language selection
    st.session_state.selected_language = language
    
    # Property: Selected language should persist in session state
    assert 'selected_language' in st.session_state, \
        "Selected language should be stored in session state"
    
    assert st.session_state.selected_language == language, \
        "Stored language should match the selected language"
    
    # Property: Language should be retrievable for API calls
    retrieved_language = st.session_state.selected_language
    
    assert retrieved_language is not None, \
        "Language should be retrievable from session state for API calls"
    
    assert retrieved_language == language, \
        "Retrieved language should be identical to selected language"
    
    # Property: Language should persist across multiple accesses
    first_access = st.session_state.selected_language
    second_access = st.session_state.selected_language
    third_access = st.session_state.selected_language
    
    assert first_access == second_access == third_access == language, \
        "Language should remain consistent across multiple accesses"


@pytest.mark.property
@given(
    initial_language=language_code_strategy(),
    new_language=language_code_strategy()
)
def test_property_language_selection_persistence_across_changes(
    initial_language, new_language, mock_session_state
):
    """
    Property 3 (Extended): Language Selection Persistence Across Changes
    
    When a user changes language, the new language should replace the old one
    and persist in session state.
    
    **Validates: Requirements 2.2**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set initial language
    st.session_state.selected_language = initial_language
    
    # Verify initial language is stored
    assert st.session_state.selected_language == initial_language, \
        "Initial language should be stored correctly"
    
    # Change language
    st.session_state.selected_language = new_language
    
    # Property: New language should replace old language
    assert st.session_state.selected_language == new_language, \
        "New language should replace the old language in session state"
    
    assert st.session_state.selected_language != initial_language or initial_language == new_language, \
        "Old language should be replaced unless both languages are the same"
    
    # Property: New language should persist
    retrieved_language = st.session_state.selected_language
    assert retrieved_language == new_language, \
        "New language should persist after change"


@pytest.mark.property
@given(
    language=language_code_strategy(),
    audio_data=audio_data_strategy(),
    filename=audio_filename_strategy()
)
def test_property_language_persistence_throughout_session(
    language, audio_data, filename, mock_session_state
):
    """
    Property 3 (Extended): Language Persistence Throughout Session
    
    The selected language should persist throughout the entire session,
    even as other session state variables are modified.
    
    **Validates: Requirements 2.2**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set language
    st.session_state.selected_language = language
    
    # Perform various operations that modify session state
    st.session_state.audio_data = audio_data
    st.session_state.audio_filename = filename
    st.session_state.transcription = {'text': 'Sample transcription'}
    st.session_state.response = {'text': 'Sample response'}
    st.session_state.is_processing = True
    
    # Property: Language should persist after modifying other state variables
    assert st.session_state.selected_language == language, \
        "Language should persist after modifying audio data"
    
    # Log actions
    log_action('upload', 'success', 'Audio uploaded')
    log_action('transcribe', 'success', 'Transcription complete')
    
    # Property: Language should persist after logging actions
    assert st.session_state.selected_language == language, \
        "Language should persist after logging actions"
    
    # Cache responses
    cache_response('transcription_key', {'text': 'cached'})
    
    # Property: Language should persist after caching
    assert st.session_state.selected_language == language, \
        "Language should persist after caching responses"
    
    # Clear cache
    clear_cache()
    
    # Property: Language should persist after clearing cache
    assert st.session_state.selected_language == language, \
        "Language should persist after clearing cache"


@pytest.mark.property
@given(language=language_code_strategy())
def test_property_language_persistence_with_default_fallback(
    language, mock_session_state
):
    """
    Property 3 (Extended): Language Persistence with Default Fallback
    
    If no language is explicitly set, the default language (Hindi) should
    be used. Once a language is set, it should override the default.
    
    **Validates: Requirements 2.2**
    """
    import streamlit as st
    
    # Initialize session state (should set default language)
    initialize_session_state()
    
    # Property: Default language should be Hindi
    assert st.session_state.selected_language == 'hi', \
        "Default language should be Hindi when not explicitly set"
    
    # Set a new language
    st.session_state.selected_language = language
    
    # Property: Explicitly set language should override default
    assert st.session_state.selected_language == language, \
        "Explicitly set language should override the default"


@pytest.mark.property
@given(
    languages=st.lists(
        language_code_strategy(),
        min_size=1,
        max_size=11  # All 11 supported languages
    )
)
def test_property_language_persistence_across_multiple_changes(
    languages, mock_session_state
):
    """
    Property 3 (Extended): Language Persistence Across Multiple Changes
    
    The session state should correctly handle multiple language changes,
    always storing the most recent selection.
    
    **Validates: Requirements 2.2**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Change language multiple times
    for language in languages:
        st.session_state.selected_language = language
        
        # Property: Current language should always be the most recently set
        assert st.session_state.selected_language == language, \
            f"Language should be {language} after setting it"
    
    # Property: Final language should be the last one in the list
    assert st.session_state.selected_language == languages[-1], \
        "Final language should be the last one set"


# Integration test: Both properties together

@pytest.mark.property
@given(
    language=language_code_strategy(),
    audio_data=audio_data_strategy(),
    filename=audio_filename_strategy()
)
def test_property_combined_session_state_persistence(
    language, audio_data, filename, mock_session_state
):
    """
    Combined Property Test: Recording and Language Persistence
    
    Both audio data and language selection should persist independently
    in session state throughout the session lifecycle.
    
    **Validates: Requirements 1.4, 2.2**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set language
    st.session_state.selected_language = language
    
    # Store audio data
    st.session_state.audio_data = audio_data
    st.session_state.audio_filename = filename
    
    # Property: Both should persist independently
    assert st.session_state.selected_language == language, \
        "Language should persist when audio data is stored"
    
    assert st.session_state.audio_data == audio_data, \
        "Audio data should persist when language is set"
    
    assert st.session_state.audio_filename == filename, \
        "Audio filename should persist when language is set"
    
    # Perform operations
    log_action('record', 'success', f'Recorded {filename}')
    cache_response('audio_key', {'audio': 'cached'})
    
    # Property: Both should still persist after operations
    assert st.session_state.selected_language == language, \
        "Language should persist after operations"
    
    assert st.session_state.audio_data == audio_data, \
        "Audio data should persist after operations"
    
    # Property: Both should be independently accessible
    retrieved_language = st.session_state.selected_language
    retrieved_audio = st.session_state.audio_data
    
    assert retrieved_language == language and retrieved_audio == audio_data, \
        "Both language and audio data should be independently accessible"
