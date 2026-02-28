"""
Property-based tests for Streamlit Web Interface audio input components.

This module tests the correctness properties of audio file size validation
and recording state persistence for the audio input UI components.

Feature: streamlit-web-interface
Task: 7.3 Write property tests for audio input
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import MagicMock, patch, Mock
import sys

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()
sys.modules['audio_recorder_streamlit'] = MagicMock()

# Import app functions after mocking streamlit
from app import (
    initialize_session_state,
    log_action,
    render_audio_uploader,
    render_voice_recorder
)


# Custom strategies for testing
@st.composite
def audio_data_strategy(draw):
    """Generate random audio data (bytes) with various sizes."""
    # Generate sizes from 100 bytes to 15MB to test both valid and invalid sizes
    size = draw(st.integers(min_value=100, max_value=15 * 1024 * 1024))
    return draw(st.binary(min_size=size, max_size=size))


@st.composite
def audio_size_strategy(draw):
    """Generate audio sizes in bytes for testing file size validation."""
    # Generate sizes from 0 to 15MB to test boundary conditions
    return draw(st.integers(min_value=0, max_value=15 * 1024 * 1024))


@st.composite
def audio_filename_strategy(draw):
    """Generate valid audio filenames with supported extensions."""
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


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""
    
    def __init__(self, name, data):
        self.name = name
        self.size = len(data)
        self._data = data
    
    def read(self):
        return self._data


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    return MockSessionState()


@pytest.fixture(autouse=True)
def setup_streamlit_mock(mock_session_state):
    """Setup streamlit mock with session state."""
    import streamlit as st_mock
    st_mock.session_state = mock_session_state
    
    # Mock streamlit UI functions
    st_mock.file_uploader = MagicMock(return_value=None)
    st_mock.error = MagicMock()
    st_mock.success = MagicMock()
    st_mock.write = MagicMock()
    st_mock.caption = MagicMock()
    st_mock.audio = MagicMock()
    
    yield
    mock_session_state.clear()


# Property 1: Audio File Size Validation
# **Validates: Requirements 1.5**

@pytest.mark.property
@given(file_size=audio_size_strategy())
def test_property_audio_file_size_validation(file_size, mock_session_state):
    """
    Property 1: Audio File Size Validation
    
    For any audio file (uploaded or recorded) that exceeds 10MB, the interface
    should reject the file and display a warning message.
    
    **Validates: Requirements 1.5**
    
    Test Strategy:
    1. Generate random file sizes from 0 to 15MB
    2. Create mock uploaded file with that size
    3. Simulate file upload
    4. Verify files over 10MB are rejected with error message
    5. Verify files under 10MB are accepted
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Create mock audio data of specified size
    audio_data = b'0' * file_size
    filename = "test_audio.wav"
    
    # Create mock uploaded file
    mock_file = MockUploadedFile(filename, audio_data)
    
    # Mock file_uploader to return our mock file
    st.file_uploader = MagicMock(return_value=mock_file)
    
    # Call render_audio_uploader
    result = render_audio_uploader()
    
    # Property: Files over 10MB should be rejected
    if file_size > 10 * 1024 * 1024:
        # Should return None for oversized files
        assert result is None, \
            f"Files over 10MB should be rejected (size: {file_size / (1024*1024):.2f}MB)"
        
        # Should display error message
        assert st.error.called, \
            "Error message should be displayed for oversized files"
        
        # Error message should mention 10MB limit
        error_call_args = str(st.error.call_args)
        assert "10MB" in error_call_args, \
            "Error message should mention the 10MB limit"
        
        # Should log error action
        assert 'action_history' in st.session_state, \
            "Action history should exist"
        
        if len(st.session_state.action_history) > 0:
            last_action = st.session_state.action_history[-1]
            assert last_action['type'] == 'upload', \
                "Last action should be upload"
            assert last_action['status'] == 'error', \
                "Upload status should be error for oversized files"
    
    # Property: Files under or equal to 10MB should be accepted
    else:
        # Should return audio data for valid files
        assert result == audio_data, \
            f"Files under 10MB should be accepted (size: {file_size / (1024*1024):.2f}MB)"
        
        # Should store in session state
        assert st.session_state.audio_data == audio_data, \
            "Audio data should be stored in session state for valid files"
        
        assert st.session_state.audio_filename == filename, \
            "Audio filename should be stored in session state for valid files"
        
        # Should display success message
        assert st.success.called, \
            "Success message should be displayed for valid files"
        
        # Should log success action
        assert 'action_history' in st.session_state, \
            "Action history should exist"
        
        if len(st.session_state.action_history) > 0:
            last_action = st.session_state.action_history[-1]
            assert last_action['type'] == 'upload', \
                "Last action should be upload"
            assert last_action['status'] == 'success', \
                "Upload status should be success for valid files"


@pytest.mark.property
@given(
    file_size=audio_size_strategy(),
    filename=audio_filename_strategy()
)
def test_property_audio_file_size_validation_with_various_formats(
    file_size, filename, mock_session_state
):
    """
    Property 1 (Extended): Audio File Size Validation Across Formats
    
    File size validation should work consistently across all supported
    audio formats (WAV, MP3, M4A, OGG).
    
    **Validates: Requirements 1.5**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Create mock audio data
    audio_data = b'0' * file_size
    
    # Create mock uploaded file
    mock_file = MockUploadedFile(filename, audio_data)
    
    # Mock file_uploader to return our mock file
    st.file_uploader = MagicMock(return_value=mock_file)
    
    # Call render_audio_uploader
    result = render_audio_uploader()
    
    # Property: Validation should be consistent across all formats
    max_size = 10 * 1024 * 1024
    
    if file_size > max_size:
        assert result is None, \
            f"File {filename} over 10MB should be rejected regardless of format"
        assert st.error.called, \
            f"Error should be displayed for oversized {filename}"
    else:
        assert result == audio_data, \
            f"File {filename} under 10MB should be accepted regardless of format"
        assert st.session_state.audio_data == audio_data, \
            f"Audio data should be stored for valid {filename}"


@pytest.mark.property
@given(file_size=st.integers(min_value=10*1024*1024 - 1000, max_value=10*1024*1024 + 1000))
def test_property_audio_file_size_validation_boundary(file_size, mock_session_state):
    """
    Property 1 (Extended): Audio File Size Validation at Boundary
    
    Test file size validation at the exact 10MB boundary to ensure
    correct handling of edge cases.
    
    **Validates: Requirements 1.5**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Create mock audio data at boundary
    audio_data = b'0' * file_size
    filename = "boundary_test.wav"
    
    # Create mock uploaded file
    mock_file = MockUploadedFile(filename, audio_data)
    
    # Mock file_uploader
    st.file_uploader = MagicMock(return_value=mock_file)
    
    # Call render_audio_uploader
    result = render_audio_uploader()
    
    # Property: Exact boundary behavior
    max_size = 10 * 1024 * 1024
    
    if file_size > max_size:
        # Even 1 byte over should be rejected
        assert result is None, \
            f"File of {file_size} bytes (>{max_size}) should be rejected"
        assert st.error.called, \
            "Error should be displayed for file just over limit"
    else:
        # Exactly at or below limit should be accepted
        assert result == audio_data, \
            f"File of {file_size} bytes (<={max_size}) should be accepted"
        assert st.session_state.audio_data == audio_data, \
            "Audio data should be stored for file at or below limit"


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
    1. Generate random audio data
    2. Mock audio_recorder to return recorded audio
    3. Call render_voice_recorder
    4. Verify audio data is stored in session state
    5. Verify audio is available for processing
    """
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
    
    # Initialize session state
    initialize_session_state()
    
    # Mock audio_recorder to return our audio data
    audio_recorder_mock = MagicMock(return_value=audio_data)
    
    with patch('app.audio_recorder', audio_recorder_mock):
        # Call render_voice_recorder
        result = render_voice_recorder()
        
        # Property: Recorded audio should be returned
        assert result == audio_data, \
            "Recorded audio data should be returned from render_voice_recorder"
        
        # Property: Audio data should be stored in session state
        assert 'audio_data' in st.session_state, \
            "Audio data should be stored in session state after recording"
        
        assert st.session_state.audio_data == audio_data, \
            "Stored audio data should match the recorded audio"
        
        # Property: Audio filename should be set
        assert 'audio_filename' in st.session_state, \
            "Audio filename should be stored in session state after recording"
        
        assert st.session_state.audio_filename == "recorded_audio.wav", \
            "Recorded audio should have default filename"
        
        # Property: Audio should be available for processing
        retrieved_audio = st.session_state.audio_data
        assert retrieved_audio is not None, \
            "Audio data should be retrievable for processing"
        
        assert retrieved_audio == audio_data, \
            "Retrieved audio should match recorded audio"
        
        # Property: Success message should be displayed
        assert st.success.called, \
            "Success message should be displayed after recording"
        
        # Property: Audio player should be displayed
        assert st.audio.called, \
            "Audio player should be displayed for recorded audio"
        
        # Property: Recording action should be logged
        assert 'action_history' in st.session_state, \
            "Action history should exist"
        
        if len(st.session_state.action_history) > 0:
            last_action = st.session_state.action_history[-1]
            assert last_action['type'] == 'record', \
                "Last action should be record"
            assert last_action['status'] == 'success', \
                "Recording status should be success"


@pytest.mark.property
@given(audio_data=audio_data_strategy())
def test_property_recording_state_persistence_across_operations(
    audio_data, mock_session_state
):
    """
    Property 2 (Extended): Recording State Persistence Across Operations
    
    Recorded audio data should persist in session state even after other
    operations are performed.
    
    **Validates: Requirements 1.4**
    """
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
    
    # Initialize session state
    initialize_session_state()
    
    # Mock audio_recorder
    audio_recorder_mock = MagicMock(return_value=audio_data)
    
    with patch('app.audio_recorder', audio_recorder_mock):
        # Record audio
        render_voice_recorder()
        
        # Perform other operations
        log_action('transcribe', 'pending', 'Processing audio')
        st.session_state.transcription = {'text': 'Sample transcription'}
        st.session_state.is_processing = True
        
        # Property: Audio data should still persist
        assert st.session_state.audio_data == audio_data, \
            "Audio data should persist after other operations"
        
        assert st.session_state.audio_filename == "recorded_audio.wav", \
            "Audio filename should persist after other operations"
        
        # Property: Audio should still be accessible
        retrieved_audio = st.session_state.audio_data
        assert retrieved_audio == audio_data, \
            "Audio should remain accessible after operations"


@pytest.mark.property
@given(
    first_audio=audio_data_strategy(),
    second_audio=audio_data_strategy()
)
def test_property_recording_state_persistence_with_multiple_recordings(
    first_audio, second_audio, mock_session_state
):
    """
    Property 2 (Extended): Recording State Persistence with Multiple Recordings
    
    When multiple recordings are made, the most recent recording should
    replace the previous one in session state.
    
    **Validates: Requirements 1.4**
    """
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
    
    # Assume recordings are different
    assume(first_audio != second_audio)
    
    # Initialize session state
    initialize_session_state()
    
    # First recording
    audio_recorder_mock = MagicMock(return_value=first_audio)
    with patch('app.audio_recorder', audio_recorder_mock):
        render_voice_recorder()
        
        # Verify first recording is stored
        assert st.session_state.audio_data == first_audio, \
            "First recording should be stored"
    
    # Second recording
    audio_recorder_mock = MagicMock(return_value=second_audio)
    with patch('app.audio_recorder', audio_recorder_mock):
        render_voice_recorder()
        
        # Property: Second recording should replace first
        assert st.session_state.audio_data == second_audio, \
            "Second recording should replace first recording"
        
        assert st.session_state.audio_data != first_audio, \
            "First recording should be replaced"
        
        # Property: Most recent recording should be accessible
        retrieved_audio = st.session_state.audio_data
        assert retrieved_audio == second_audio, \
            "Most recent recording should be accessible"


@pytest.mark.property
@given(audio_data=audio_data_strategy())
def test_property_recording_state_persistence_multiple_accesses(
    audio_data, mock_session_state
):
    """
    Property 2 (Extended): Recording State Persistence Across Multiple Accesses
    
    Recorded audio data should remain consistent across multiple accesses
    from session state.
    
    **Validates: Requirements 1.4**
    """
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
    
    # Initialize session state
    initialize_session_state()
    
    # Mock audio_recorder
    audio_recorder_mock = MagicMock(return_value=audio_data)
    
    with patch('app.audio_recorder', audio_recorder_mock):
        # Record audio
        render_voice_recorder()
        
        # Access audio data multiple times
        first_access = st.session_state.audio_data
        second_access = st.session_state.audio_data
        third_access = st.session_state.audio_data
        
        # Property: All accesses should return the same data
        assert first_access == second_access == third_access == audio_data, \
            "Audio data should remain consistent across multiple accesses"
        
        # Property: Data should not be corrupted by multiple accesses
        assert len(first_access) == len(audio_data), \
            "Audio data length should remain unchanged"


# Integration test: Both properties together

@pytest.mark.property
@given(
    upload_size=audio_size_strategy(),
    record_data=audio_data_strategy(),
    filename=audio_filename_strategy()
)
def test_property_combined_audio_input_validation_and_persistence(
    upload_size, record_data, filename, mock_session_state
):
    """
    Combined Property Test: File Size Validation and Recording Persistence
    
    Both file size validation and recording state persistence should work
    correctly when used together in the same session.
    
    **Validates: Requirements 1.4, 1.5**
    """
    import streamlit as st
    from audio_recorder_streamlit import audio_recorder
    
    # Initialize session state
    initialize_session_state()
    
    # Test file upload first
    upload_data = b'0' * upload_size
    mock_file = MockUploadedFile(filename, upload_data)
    st.file_uploader = MagicMock(return_value=mock_file)
    
    upload_result = render_audio_uploader()
    
    # Property: Upload validation should work correctly
    if upload_size > 10 * 1024 * 1024:
        assert upload_result is None, \
            "Oversized upload should be rejected"
    else:
        assert upload_result == upload_data, \
            "Valid upload should be accepted"
        assert st.session_state.audio_data == upload_data, \
            "Upload data should be stored"
    
    # Test recording
    audio_recorder_mock = MagicMock(return_value=record_data)
    with patch('app.audio_recorder', audio_recorder_mock):
        record_result = render_voice_recorder()
        
        # Property: Recording should work and replace upload
        assert record_result == record_data, \
            "Recording should work after upload"
        
        assert st.session_state.audio_data == record_data, \
            "Recording should replace uploaded audio in session state"
        
        # Property: Recording should override upload regardless of upload validity
        if upload_size <= 10 * 1024 * 1024:
            assert st.session_state.audio_data != upload_data, \
                "Recording should replace valid upload"
