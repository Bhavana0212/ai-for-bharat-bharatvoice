"""
Unit tests for BharatVoiceAPIClient synthesize_speech method

This test file validates Task 3.4 implementation:
- Create synthesize_speech() method with JSON payload
- Send text and language to /api/voice/synthesize endpoint
- Fetch and return audio file from audio_url
- Handle base64 decoding if needed

Requirements: 5.1, 12.3
"""

import pytest
import base64
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import BharatVoiceAPIClient


class TestBharatVoiceAPIClientTTS:
    """Test suite for synthesize_speech method"""
    
    @pytest.fixture
    def api_client(self):
        """Create API client instance for testing"""
        return BharatVoiceAPIClient(
            base_url='http://localhost:8000',
            timeout=30
        )
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data"""
        return b'\x00\x01\x02\x03\x04\x05' * 100  # Mock WAV data
    
    def test_synthesize_speech_with_relative_url(self, api_client, mock_audio_data):
        """Test TTS synthesis with relative audio URL"""
        # Mock the POST request to /api/voice/synthesize
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'request_id': 'test-123',
            'audio_url': '/audio/test-audio.wav',
            'duration': 2.5,
            'format': 'wav'
        }
        
        # Mock the GET request to fetch audio file
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = mock_audio_data
        mock_get_response.headers = {'Content-Type': 'audio/wav'}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="नमस्ते",
                    language='hi',
                    accent='standard'
                )
        
        # Verify the result
        assert result == mock_audio_data
        
        # Verify POST request was made with correct parameters
        api_client.session.post.assert_called_once()
        call_args = api_client.session.post.call_args
        assert call_args[1]['json']['text'] == "नमस्ते"
        assert call_args[1]['json']['language'] == 'hi'
        assert call_args[1]['json']['accent'] == 'standard'
        assert call_args[1]['json']['speed'] == 1.0
        assert call_args[1]['json']['pitch'] == 1.0
        
        # Verify GET request was made to fetch audio
        api_client.session.get.assert_called_once_with(
            'http://localhost:8000/audio/test-audio.wav',
            timeout=30
        )
    
    def test_synthesize_speech_with_absolute_url(self, api_client, mock_audio_data):
        """Test TTS synthesis with absolute audio URL"""
        # Mock the POST request
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'audio_url': 'http://cdn.example.com/audio/test-audio.wav'
        }
        
        # Mock the GET request
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = mock_audio_data
        mock_get_response.headers = {'Content-Type': 'audio/wav'}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="Hello",
                    language='en-IN'
                )
        
        # Verify GET request used absolute URL
        api_client.session.get.assert_called_once_with(
            'http://cdn.example.com/audio/test-audio.wav',
            timeout=30
        )
    
    def test_synthesize_speech_with_base64_json_response(self, api_client, mock_audio_data):
        """Test TTS synthesis with base64-encoded audio in JSON response"""
        # Encode audio data as base64
        encoded_audio = base64.b64encode(mock_audio_data).decode('utf-8')
        
        # Mock the POST request
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'audio_url': '/audio/test-audio.json'
        }
        
        # Mock the GET request returning JSON with base64 data
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = b'{"audio_data": "' + encoded_audio.encode() + b'"}'
        mock_get_response.headers = {'Content-Type': 'application/json'}
        mock_get_response.json.return_value = {'audio_data': encoded_audio}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="Test",
                    language='hi'
                )
        
        # Verify the audio was decoded correctly
        assert result == mock_audio_data
    
    def test_synthesize_speech_with_base64_text_response(self, api_client, mock_audio_data):
        """Test TTS synthesis with base64-encoded audio as text response"""
        # Encode audio data as base64
        encoded_audio = base64.b64encode(mock_audio_data)
        
        # Mock the POST request
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'audio_url': '/audio/test-audio.txt'
        }
        
        # Mock the GET request returning base64 text
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = encoded_audio
        mock_get_response.headers = {'Content-Type': 'text/plain'}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="Test",
                    language='hi'
                )
        
        # Verify the audio was decoded correctly
        assert result == mock_audio_data
    
    def test_synthesize_speech_with_custom_parameters(self, api_client, mock_audio_data):
        """Test TTS synthesis with custom speed and pitch parameters"""
        # Mock responses
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'audio_url': '/audio/test.wav'}
        
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = mock_audio_data
        mock_get_response.headers = {'Content-Type': 'audio/wav'}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="Test",
                    language='hi',
                    accent='mumbai',
                    speed=1.2,
                    pitch=0.9
                )
        
        # Verify custom parameters were sent
        call_args = api_client.session.post.call_args
        assert call_args[1]['json']['accent'] == 'mumbai'
        assert call_args[1]['json']['speed'] == 1.2
        assert call_args[1]['json']['pitch'] == 0.9
    
    def test_synthesize_speech_missing_audio_url(self, api_client):
        """Test TTS synthesis when response is missing audio_url"""
        # Mock POST response without audio_url
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            'request_id': 'test-123',
            'duration': 2.5
        }
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with pytest.raises(ValueError, match="Response missing 'audio_url' field"):
                api_client.synthesize_speech(
                    text="Test",
                    language='hi'
                )
    
    def test_synthesize_speech_http_error(self, api_client):
        """Test TTS synthesis with HTTP error response"""
        # Mock POST response with error
        mock_post_response = Mock()
        mock_post_response.status_code = 500
        mock_post_response.raise_for_status.side_effect = Exception("Server error")
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with pytest.raises(Exception, match="Server error"):
                api_client.synthesize_speech(
                    text="Test",
                    language='hi'
                )
    
    def test_synthesize_speech_audio_fetch_error(self, api_client):
        """Test TTS synthesis when audio fetch fails"""
        # Mock successful POST
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'audio_url': '/audio/test.wav'}
        
        # Mock failed GET
        mock_get_response = Mock()
        mock_get_response.status_code = 404
        mock_get_response.raise_for_status.side_effect = Exception("Audio not found")
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                with pytest.raises(Exception, match="Audio not found"):
                    api_client.synthesize_speech(
                        text="Test",
                        language='hi'
                    )
    
    def test_synthesize_speech_invalid_base64(self, api_client):
        """Test TTS synthesis with invalid base64 data (should fallback to raw content)"""
        invalid_base64 = b'This is not valid base64!!!'
        
        # Mock the POST request
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'audio_url': '/audio/test.txt'}
        
        # Mock the GET request with invalid base64
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.content = invalid_base64
        mock_get_response.headers = {'Content-Type': 'text/plain'}
        
        with patch.object(api_client.session, 'post', return_value=mock_post_response):
            with patch.object(api_client.session, 'get', return_value=mock_get_response):
                result = api_client.synthesize_speech(
                    text="Test",
                    language='hi'
                )
        
        # Should return raw content when base64 decoding fails
        assert result == invalid_base64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
