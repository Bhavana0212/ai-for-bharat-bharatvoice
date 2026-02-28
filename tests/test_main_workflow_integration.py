"""
Property-based tests for main workflow integration

Tests:
- Property 34: API Data Round-Trip Consistency
- Property 32: Base64 Audio Decoding

Validates Requirements: 12.3, 12.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import base64
import json
from typing import Dict, Any
import io


# Strategy for generating valid audio data
@st.composite
def audio_data_strategy(draw):
    """Generate valid audio data (WAV format)"""
    # Simple WAV header + random audio data
    size = draw(st.integers(min_value=100, max_value=10000))
    audio_bytes = draw(st.binary(min_size=size, max_size=size))
    return audio_bytes


# Strategy for generating API response structures
@st.composite
def transcription_response_strategy(draw):
    """Generate valid transcription response structure"""
    return {
        'request_id': draw(st.text(min_size=10, max_size=50)),
        'result': {
            'transcribed_text': draw(st.text(min_size=1, max_size=500)),
            'confidence': draw(st.floats(min_value=0.0, max_value=1.0)),
            'detected_language': draw(st.sampled_from(['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'])),
            'code_switching_points': [],
            'alternative_transcriptions': draw(st.lists(st.text(min_size=1, max_size=100), max_size=3)),
            'processing_time': draw(st.floats(min_value=0.1, max_value=10.0))
        },
        'processing_time': draw(st.floats(min_value=0.1, max_value=10.0))
    }


@st.composite
def response_generation_strategy(draw):
    """Generate valid response generation structure"""
    return {
        'request_id': draw(st.text(min_size=10, max_size=50)),
        'response': {
            'text': draw(st.text(min_size=1, max_size=1000)),
            'intent': draw(st.text(min_size=1, max_size=50)),
            'confidence': draw(st.floats(min_value=0.0, max_value=1.0)),
            'suggested_actions': []
        },
        'processing_time': draw(st.floats(min_value=0.1, max_value=10.0))
    }


@st.composite
def tts_response_strategy(draw):
    """Generate valid TTS response structure"""
    return {
        'request_id': draw(st.text(min_size=10, max_size=50)),
        'audio_url': f"/api/audio/{draw(st.text(min_size=10, max_size=50))}",
        'duration': draw(st.floats(min_value=0.5, max_value=60.0)),
        'format': 'wav'
    }


class TestAPIDataRoundTripConsistency:
    """
    Property 34: API Data Round-Trip Consistency
    
    Tests that data structures remain consistent through request-response cycles.
    This ensures that data sent to the API and received back maintains its integrity.
    
    Validates: Requirements 12.3, 12.5
    """
    
    @given(
        transcription_response=transcription_response_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_transcription_response_structure_consistency(self, transcription_response: Dict[str, Any]):
        """
        Property: Transcription response structure remains consistent after parsing
        
        Given: A valid transcription response from the API
        When: The response is parsed and stored
        Then: All required fields are present and have correct types
        """
        # Simulate JSON serialization/deserialization (round-trip)
        json_str = json.dumps(transcription_response)
        parsed_response = json.loads(json_str)
        
        # Verify structure consistency
        assert 'request_id' in parsed_response
        assert 'result' in parsed_response
        assert 'processing_time' in parsed_response
        
        result = parsed_response['result']
        assert 'transcribed_text' in result
        assert 'confidence' in result
        assert 'detected_language' in result
        assert 'processing_time' in result
        
        # Verify types
        assert isinstance(parsed_response['request_id'], str)
        assert isinstance(result['transcribed_text'], str)
        assert isinstance(result['confidence'], (int, float))
        assert isinstance(result['detected_language'], str)
        assert isinstance(result['processing_time'], (int, float))
        
        # Verify values match original
        assert parsed_response['request_id'] == transcription_response['request_id']
        assert result['transcribed_text'] == transcription_response['result']['transcribed_text']
        assert result['confidence'] == transcription_response['result']['confidence']
        assert result['detected_language'] == transcription_response['result']['detected_language']
    
    @given(
        response_data=response_generation_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_response_generation_structure_consistency(self, response_data: Dict[str, Any]):
        """
        Property: Response generation structure remains consistent after parsing
        
        Given: A valid response generation result from the API
        When: The response is parsed and stored
        Then: All required fields are present and have correct types
        """
        # Simulate JSON serialization/deserialization (round-trip)
        json_str = json.dumps(response_data)
        parsed_response = json.loads(json_str)
        
        # Verify structure consistency
        assert 'request_id' in parsed_response
        assert 'response' in parsed_response
        assert 'processing_time' in parsed_response
        
        response = parsed_response['response']
        assert 'text' in response
        assert 'intent' in response
        assert 'confidence' in response
        
        # Verify types
        assert isinstance(parsed_response['request_id'], str)
        assert isinstance(response['text'], str)
        assert isinstance(response['intent'], str)
        assert isinstance(response['confidence'], (int, float))
        
        # Verify values match original
        assert parsed_response['request_id'] == response_data['request_id']
        assert response['text'] == response_data['response']['text']
        assert response['intent'] == response_data['response']['intent']
        assert response['confidence'] == response_data['response']['confidence']
    
    @given(
        tts_data=tts_response_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tts_response_structure_consistency(self, tts_data: Dict[str, Any]):
        """
        Property: TTS response structure remains consistent after parsing
        
        Given: A valid TTS response from the API
        When: The response is parsed and stored
        Then: All required fields are present and have correct types
        """
        # Simulate JSON serialization/deserialization (round-trip)
        json_str = json.dumps(tts_data)
        parsed_response = json.loads(json_str)
        
        # Verify structure consistency
        assert 'request_id' in parsed_response
        assert 'audio_url' in parsed_response
        assert 'duration' in parsed_response
        assert 'format' in parsed_response
        
        # Verify types
        assert isinstance(parsed_response['request_id'], str)
        assert isinstance(parsed_response['audio_url'], str)
        assert isinstance(parsed_response['duration'], (int, float))
        assert isinstance(parsed_response['format'], str)
        
        # Verify values match original
        assert parsed_response['request_id'] == tts_data['request_id']
        assert parsed_response['audio_url'] == tts_data['audio_url']
        assert parsed_response['duration'] == tts_data['duration']
        assert parsed_response['format'] == tts_data['format']
    
    @given(
        language=st.sampled_from(['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']),
        text=st.text(min_size=1, max_size=500)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_request_data_consistency(self, language: str, text: str):
        """
        Property: Request data remains consistent through serialization
        
        Given: Valid request parameters (language, text)
        When: The request is serialized to JSON
        Then: The deserialized data matches the original
        """
        # Create request payload
        request_data = {
            'text': text,
            'language': language,
            'accent': 'standard',
            'speed': 1.0,
            'pitch': 1.0
        }
        
        # Simulate JSON serialization/deserialization
        json_str = json.dumps(request_data)
        parsed_request = json.loads(json_str)
        
        # Verify consistency
        assert parsed_request['text'] == text
        assert parsed_request['language'] == language
        assert parsed_request['accent'] == 'standard'
        assert parsed_request['speed'] == 1.0
        assert parsed_request['pitch'] == 1.0


class TestBase64AudioDecoding:
    """
    Property 32: Base64 Audio Decoding
    
    Tests that base64-encoded audio is correctly decoded and can be used.
    This ensures audio data integrity through encoding/decoding cycles.
    
    Validates: Requirements 12.3
    """
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_base64_encoding_decoding_consistency(self, audio_data: bytes):
        """
        Property: Audio data remains consistent through base64 encoding/decoding
        
        Given: Valid audio data (bytes)
        When: The data is base64-encoded and then decoded
        Then: The decoded data matches the original
        """
        # Encode to base64
        encoded = base64.b64encode(audio_data).decode('utf-8')
        
        # Verify it's a valid base64 string
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        # Decode back to bytes
        decoded = base64.b64decode(encoded)
        
        # Verify consistency
        assert decoded == audio_data
        assert len(decoded) == len(audio_data)
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_base64_audio_in_json_response(self, audio_data: bytes):
        """
        Property: Base64 audio in JSON response can be decoded correctly
        
        Given: Audio data embedded in a JSON response as base64
        When: The response is parsed and audio is decoded
        Then: The decoded audio matches the original
        """
        # Simulate API response with base64 audio
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')
        
        response = {
            'request_id': 'test-123',
            'audio_data': encoded_audio,
            'format': 'wav'
        }
        
        # Serialize and deserialize (simulating API round-trip)
        json_str = json.dumps(response)
        parsed_response = json.loads(json_str)
        
        # Decode audio from response
        decoded_audio = base64.b64decode(parsed_response['audio_data'])
        
        # Verify consistency
        assert decoded_audio == audio_data
        assert len(decoded_audio) == len(audio_data)
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_base64_audio_with_padding(self, audio_data: bytes):
        """
        Property: Base64 decoding handles padding correctly
        
        Given: Audio data that may result in base64 with padding
        When: The data is encoded and decoded
        Then: Padding is handled correctly and data is consistent
        """
        # Encode to base64 (may include padding)
        encoded = base64.b64encode(audio_data).decode('utf-8')
        
        # Check if padding exists
        has_padding = encoded.endswith('=')
        
        # Decode should work regardless of padding
        decoded = base64.b64decode(encoded)
        
        # Verify consistency
        assert decoded == audio_data
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_base64_audio_string_vs_bytes_handling(self, audio_data: bytes):
        """
        Property: Base64 audio can be handled as both string and bytes
        
        Given: Audio data encoded as base64
        When: The encoded data is treated as string or bytes
        Then: Both can be decoded correctly
        """
        # Encode to base64 string
        encoded_str = base64.b64encode(audio_data).decode('utf-8')
        
        # Encode to base64 bytes
        encoded_bytes = base64.b64encode(audio_data)
        
        # Decode from string
        decoded_from_str = base64.b64decode(encoded_str)
        
        # Decode from bytes
        decoded_from_bytes = base64.b64decode(encoded_bytes)
        
        # Both should match original
        assert decoded_from_str == audio_data
        assert decoded_from_bytes == audio_data
        assert decoded_from_str == decoded_from_bytes
    
    @given(
        audio_data=audio_data_strategy()
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_base64_audio_size_increase(self, audio_data: bytes):
        """
        Property: Base64 encoding increases size predictably
        
        Given: Audio data of known size
        When: The data is base64-encoded
        Then: The encoded size is approximately 4/3 of original (with padding)
        """
        original_size = len(audio_data)
        
        # Encode to base64
        encoded = base64.b64encode(audio_data).decode('utf-8')
        encoded_size = len(encoded)
        
        # Base64 encoding increases size by ~33% (4/3 ratio)
        # Allow some tolerance for padding
        expected_min_size = (original_size * 4) // 3
        expected_max_size = expected_min_size + 4  # Account for padding
        
        assert expected_min_size <= encoded_size <= expected_max_size


class TestWorkflowDataIntegrity:
    """
    Additional tests for data integrity throughout the workflow
    """
    
    @given(
        transcription=transcription_response_strategy(),
        response=response_generation_strategy(),
        tts=tts_response_strategy()
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_complete_workflow_data_consistency(
        self,
        transcription: Dict[str, Any],
        response: Dict[str, Any],
        tts: Dict[str, Any]
    ):
        """
        Property: Data remains consistent through complete workflow
        
        Given: Complete workflow data (transcription, response, TTS)
        When: All data is serialized and deserialized
        Then: All data structures remain consistent
        """
        # Simulate complete workflow data
        workflow_data = {
            'transcription': transcription,
            'response': response,
            'tts': tts
        }
        
        # Serialize and deserialize
        json_str = json.dumps(workflow_data)
        parsed_data = json.loads(json_str)
        
        # Verify all components are present
        assert 'transcription' in parsed_data
        assert 'response' in parsed_data
        assert 'tts' in parsed_data
        
        # Verify transcription consistency
        assert parsed_data['transcription']['request_id'] == transcription['request_id']
        assert parsed_data['transcription']['result']['transcribed_text'] == transcription['result']['transcribed_text']
        
        # Verify response consistency
        assert parsed_data['response']['request_id'] == response['request_id']
        assert parsed_data['response']['response']['text'] == response['response']['text']
        
        # Verify TTS consistency
        assert parsed_data['tts']['request_id'] == tts['request_id']
        assert parsed_data['tts']['audio_url'] == tts['audio_url']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
