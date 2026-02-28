"""
Property-based tests for Streamlit processing workflow

Tests the main processing workflow including automatic response generation,
automatic TTS request, graceful TTS degradation, and action logging completeness.

Feature: streamlit-web-interface
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import MagicMock, patch, Mock
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()
sys.modules['audio_recorder_streamlit'] = MagicMock()

import app


# Strategy generators
@st.composite
def generate_audio_data(draw):
    """Generate random audio data"""
    size = draw(st.integers(min_value=100, max_value=1024 * 1024))  # 100 bytes to 1MB
    return b'0' * size


@st.composite
def generate_language_code(draw):
    """Generate valid language codes"""
    languages = ['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']
    return draw(st.sampled_from(languages))


@st.composite
def generate_transcription_response(draw):
    """Generate mock transcription API response"""
    text = draw(st.text(min_size=1, max_size=500))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    language = draw(generate_language_code())
    processing_time = draw(st.floats(min_value=0.1, max_value=5.0))
    
    return {
        'request_id': 'test_request_id',
        'result': {
            'transcribed_text': text,
            'confidence': confidence,
            'detected_language': language,
            'code_switching_points': [],
            'alternative_transcriptions': [],
            'processing_time': processing_time
        },
        'processing_time': processing_time
    }


@st.composite
def generate_response_response(draw):
    """Generate mock response generation API response"""
    text = draw(st.text(min_size=1, max_size=500))
    language = draw(generate_language_code())
    processing_time = draw(st.floats(min_value=0.1, max_value=5.0))
    
    return {
        'request_id': 'test_request_id',
        'text': text,
        'language': language,
        'suggested_actions': [],
        'processing_time': processing_time
    }


@st.composite
def generate_tts_audio(draw):
    """Generate mock TTS audio bytes"""
    size = draw(st.integers(min_value=1000, max_value=100000))
    return b'AUDIO' * (size // 5)


# Feature: streamlit-web-interface, Property 11: Automatic Response Generation
@given(
    audio_data=generate_audio_data(),
    language=generate_language_code(),
    transcription_response=generate_transcription_response(),
    response_response=generate_response_response()
)
@settings(max_examples=50, deadline=None)
def test_automatic_response_generation(audio_data, language, transcription_response, response_response):
    """
    **Validates: Requirements 4.1**
    
    Property 11: Automatic Response Generation
    For any successful transcription, the interface should automatically trigger
    a response generation request to the backend.
    
    This test verifies that:
    1. After transcription succeeds, response generation is automatically called
    2. The transcribed text is passed to response generation
    3. The response is stored in session state
    4. Both operations are logged
    """
    # Setup mock session state
    mock_session_state = {
        'audio_data': audio_data,
        'selected_language': language,
        'is_online': True,
        'is_processing': False,
        'action_history': []
    }
    
    # Mock streamlit session state
    with patch.object(app.st, 'session_state', mock_session_state):
        # Mock API client
        with patch.object(app, 'BharatVoiceAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock API responses
            mock_client.recognize_speech.return_value = transcription_response
            mock_client.generate_response.return_value = response_response
            mock_client.synthesize_speech.return_value = b'AUDIO_DATA'
            
            # Mock streamlit UI functions
            with patch.object(app.st, 'spinner', MagicMock()):
                with patch.object(app.st, 'success', MagicMock()):
                    with patch.object(app.st, 'error', MagicMock()):
                        with patch.object(app.st, 'warning', MagicMock()):
                            # Call process_audio
                            app.process_audio()
            
            # Verify transcription was called
            assert mock_client.recognize_speech.called
            
            # Verify response generation was automatically called after transcription
            assert mock_client.generate_response.called
            
            # Verify the transcribed text was passed to response generation
            call_args = mock_client.generate_response.call_args
            assert call_args is not None
            assert call_args[1]['text'] == transcription_response['result']['transcribed_text']
            
            # Verify response is stored in session state
            assert 'response' in mock_session_state
            assert mock_session_state['response']['text'] == response_response['text']
            
            # Verify both operations are logged
            action_types = [action['type'] for action in mock_session_state['action_history']]
            assert 'transcribe' in action_types
            assert 'respond' in action_types


# Feature: streamlit-web-interface, Property 14: Automatic TTS Request
@given(
    audio_data=generate_audio_data(),
    language=generate_language_code(),
    transcription_response=generate_transcription_response(),
    response_response=generate_response_response(),
    tts_audio=generate_tts_audio()
)
@settings(max_examples=50, deadline=None)
def test_automatic_tts_request(audio_data, language, transcription_response, response_response, tts_audio):
    """
    **Validates: Requirements 5.1**
    
    Property 14: Automatic TTS Request
    For any text response received, the interface should automatically trigger
    a TTS synthesis request to the backend.
    
    This test verifies that:
    1. After response generation succeeds, TTS is automatically called
    2. The response text is passed to TTS synthesis
    3. The TTS audio is stored in session state
    4. All three operations are logged
    """
    # Setup mock session state
    mock_session_state = {
        'audio_data': audio_data,
        'selected_language': language,
        'is_online': True,
        'is_processing': False,
        'action_history': []
    }
    
    # Mock streamlit session state
    with patch.object(app.st, 'session_state', mock_session_state):
        # Mock API client
        with patch.object(app, 'BharatVoiceAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock API responses
            mock_client.recognize_speech.return_value = transcription_response
            mock_client.generate_response.return_value = response_response
            mock_client.synthesize_speech.return_value = tts_audio
            
            # Mock streamlit UI functions
            with patch.object(app.st, 'spinner', MagicMock()):
                with patch.object(app.st, 'success', MagicMock()):
                    with patch.object(app.st, 'error', MagicMock()):
                        with patch.object(app.st, 'warning', MagicMock()):
                            # Call process_audio
                            app.process_audio()
            
            # Verify TTS was automatically called after response generation
            assert mock_client.synthesize_speech.called
            
            # Verify the response text was passed to TTS synthesis
            call_args = mock_client.synthesize_speech.call_args
            assert call_args is not None
            assert call_args[1]['text'] == response_response['text']
            
            # Verify TTS audio is stored in session state
            assert 'tts_audio' in mock_session_state
            assert mock_session_state['tts_audio'] == tts_audio
            
            # Verify all three operations are logged
            action_types = [action['type'] for action in mock_session_state['action_history']]
            assert 'transcribe' in action_types
            assert 'respond' in action_types
            assert 'tts' in action_types


# Feature: streamlit-web-interface, Property 16: Graceful TTS Degradation
@given(
    audio_data=generate_audio_data(),
    language=generate_language_code(),
    transcription_response=generate_transcription_response(),
    response_response=generate_response_response(),
    error_type=st.sampled_from(['timeout', 'connection', 'http', 'generic'])
)
@settings(max_examples=50, deadline=None)
def test_graceful_tts_degradation(audio_data, language, transcription_response, response_response, error_type):
    """
    **Validates: Requirements 5.5**
    
    Property 16: Graceful TTS Degradation
    For any TTS generation failure, the text response should still be displayed
    and the error should be logged.
    
    This test verifies that:
    1. When TTS fails, the workflow continues (doesn't crash)
    2. The text response is still available in session state
    3. The TTS failure is logged as a warning (not error)
    4. A warning message is displayed to the user
    """
    # Setup mock session state
    mock_session_state = {
        'audio_data': audio_data,
        'selected_language': language,
        'is_online': True,
        'is_processing': False,
        'action_history': []
    }
    
    # Mock streamlit session state
    with patch.object(app.st, 'session_state', mock_session_state):
        # Mock API client
        with patch.object(app, 'BharatVoiceAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock API responses
            mock_client.recognize_speech.return_value = transcription_response
            mock_client.generate_response.return_value = response_response
            
            # Mock TTS failure based on error type
            import requests
            if error_type == 'timeout':
                mock_client.synthesize_speech.side_effect = requests.exceptions.Timeout("TTS timeout")
            elif error_type == 'connection':
                mock_client.synthesize_speech.side_effect = requests.exceptions.ConnectionError("TTS connection error")
            elif error_type == 'http':
                mock_response = Mock()
                mock_response.status_code = 500
                mock_client.synthesize_speech.side_effect = requests.exceptions.HTTPError(response=mock_response)
            else:
                mock_client.synthesize_speech.side_effect = Exception("Generic TTS error")
            
            # Mock streamlit UI functions
            warning_called = []
            def mock_warning(msg):
                warning_called.append(msg)
            
            with patch.object(app.st, 'spinner', MagicMock()):
                with patch.object(app.st, 'success', MagicMock()):
                    with patch.object(app.st, 'error', MagicMock()):
                        with patch.object(app.st, 'warning', mock_warning):
                            # Call process_audio
                            app.process_audio()
            
            # Verify TTS was attempted
            assert mock_client.synthesize_speech.called
            
            # Verify text response is still available (graceful degradation)
            assert 'response' in mock_session_state
            assert mock_session_state['response']['text'] == response_response['text']
            
            # Verify TTS failure is logged as warning (not error)
            tts_actions = [action for action in mock_session_state['action_history'] if action['type'] == 'tts']
            assert len(tts_actions) > 0
            assert tts_actions[0]['status'] == 'warning'
            
            # Verify warning message was displayed
            assert len(warning_called) > 0
            assert any('TTS' in msg or 'text response only' in msg for msg in warning_called)


# Feature: streamlit-web-interface, Property 10: Action Logging Completeness
@given(
    audio_data=generate_audio_data(),
    language=generate_language_code(),
    transcription_response=generate_transcription_response(),
    response_response=generate_response_response(),
    tts_audio=generate_tts_audio()
)
@settings(max_examples=50, deadline=None)
def test_action_logging_completeness(audio_data, language, transcription_response, response_response, tts_audio):
    """
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    
    Property 10: Action Logging Completeness
    For any user interaction (audio upload, recording, transcription, response
    generation, TTS playback), an entry with timestamp should be added to the
    action log.
    
    This test verifies that:
    1. All operations in the workflow are logged
    2. Each log entry has a timestamp
    3. Each log entry has a type, status, and details
    4. Log entries are in chronological order
    """
    # Setup mock session state
    mock_session_state = {
        'audio_data': audio_data,
        'selected_language': language,
        'is_online': True,
        'is_processing': False,
        'action_history': []
    }
    
    # Mock streamlit session state
    with patch.object(app.st, 'session_state', mock_session_state):
        # Mock API client
        with patch.object(app, 'BharatVoiceAPIClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock API responses
            mock_client.recognize_speech.return_value = transcription_response
            mock_client.generate_response.return_value = response_response
            mock_client.synthesize_speech.return_value = tts_audio
            
            # Mock streamlit UI functions
            with patch.object(app.st, 'spinner', MagicMock()):
                with patch.object(app.st, 'success', MagicMock()):
                    with patch.object(app.st, 'error', MagicMock()):
                        with patch.object(app.st, 'warning', MagicMock()):
                            # Call process_audio
                            app.process_audio()
            
            # Verify all operations are logged
            action_types = [action['type'] for action in mock_session_state['action_history']]
            assert 'transcribe' in action_types
            assert 'respond' in action_types
            assert 'tts' in action_types
            
            # Verify each log entry has required fields
            for action in mock_session_state['action_history']:
                assert 'timestamp' in action
                assert 'type' in action
                assert 'status' in action
                assert 'details' in action
                
                # Verify timestamp is in ISO format
                from datetime import datetime
                try:
                    datetime.fromisoformat(action['timestamp'])
                except ValueError:
                    pytest.fail(f"Invalid timestamp format: {action['timestamp']}")
            
            # Verify log entries are in chronological order
            timestamps = [action['timestamp'] for action in mock_session_state['action_history']]
            assert timestamps == sorted(timestamps)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
