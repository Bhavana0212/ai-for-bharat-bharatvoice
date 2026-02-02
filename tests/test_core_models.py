"""
Unit tests for core data models.

This module tests the Pydantic models used throughout the BharatVoice system,
ensuring proper validation, serialization, and business logic.
"""

import pytest
from datetime import datetime
from uuid import UUID
from hypothesis import given, strategies as st

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    UserProfile,
    ConversationState,
    RecognitionResult,
    LocationData,
    VoiceActivityResult,
    LanguageSwitchPoint,
)
from tests.conftest import (
    audio_buffer_strategy,
    language_code_strategy,
    user_profile_strategy,
    location_data_strategy,
)


class TestAudioBuffer:
    """Test AudioBuffer model."""
    
    def test_valid_audio_buffer_creation(self):
        """Test creating a valid AudioBuffer."""
        buffer = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2],
            sample_rate=16000,
            channels=1,
            duration=0.25
        )
        
        assert len(buffer.data) == 4
        assert buffer.sample_rate == 16000
        assert buffer.channels == 1
        assert buffer.duration == 0.25
        assert buffer.format == AudioFormat.WAV
    
    def test_invalid_sample_rate(self):
        """Test AudioBuffer with invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate must be one of"):
            AudioBuffer(
                data=[0.1, 0.2],
                sample_rate=12000,  # Invalid sample rate
                duration=0.1
            )
    
    def test_invalid_channels(self):
        """Test AudioBuffer with invalid channel count."""
        with pytest.raises(ValueError, match="Channels must be 1 \\(mono\\) or 2 \\(stereo\\)"):
            AudioBuffer(
                data=[0.1, 0.2],
                channels=3,  # Invalid channel count
                duration=0.1
            )
    
    def test_numpy_array_conversion(self):
        """Test conversion to numpy array."""
        buffer = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2],
            duration=0.25
        )
        
        np_array = buffer.numpy_array
        assert np_array.dtype.name == 'float32'
        assert len(np_array) == 4
        assert np_array[0] == pytest.approx(0.1)
    
    @given(audio_buffer_strategy())
    def test_audio_buffer_property_invariants(self, buffer):
        """Property test: AudioBuffer invariants."""
        # Duration should be positive
        assert buffer.duration > 0
        
        # Sample rate should be valid
        assert buffer.sample_rate in [8000, 16000, 22050, 44100, 48000]
        
        # Channels should be 1 or 2
        assert buffer.channels in [1, 2]
        
        # Data should not be empty
        assert len(buffer.data) > 0
        
        # All audio samples should be in valid range
        assert all(-1.0 <= sample <= 1.0 for sample in buffer.data)


class TestUserProfile:
    """Test UserProfile model."""
    
    def test_valid_user_profile_creation(self):
        """Test creating a valid UserProfile."""
        profile = UserProfile(
            preferred_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
            primary_language=LanguageCode.HINDI
        )
        
        assert len(profile.preferred_languages) == 2
        assert profile.primary_language == LanguageCode.HINDI
        assert isinstance(profile.user_id, UUID)
        assert isinstance(profile.created_at, datetime)
    
    def test_empty_preferred_languages(self):
        """Test UserProfile with empty preferred languages."""
        with pytest.raises(ValueError, match="At least one preferred language must be specified"):
            UserProfile(
                preferred_languages=[],
                primary_language=LanguageCode.HINDI
            )
    
    def test_default_values(self):
        """Test UserProfile default values."""
        profile = UserProfile()
        
        assert LanguageCode.HINDI in profile.preferred_languages
        assert LanguageCode.ENGLISH_IN in profile.preferred_languages
        assert profile.primary_language == LanguageCode.HINDI
        assert profile.privacy_settings.data_retention_days == 90
    
    @given(user_profile_strategy())
    def test_user_profile_property_invariants(self, profile):
        """Property test: UserProfile invariants."""
        # Must have at least one preferred language
        assert len(profile.preferred_languages) >= 1
        
        # Primary language must be in preferred languages
        assert profile.primary_language in profile.preferred_languages
        
        # User ID should be valid UUID
        assert isinstance(profile.user_id, UUID)
        
        # Timestamps should be valid
        assert isinstance(profile.created_at, datetime)
        assert isinstance(profile.last_updated, datetime)


class TestConversationState:
    """Test ConversationState model."""
    
    def test_valid_conversation_state_creation(self, sample_user_profile):
        """Test creating a valid ConversationState."""
        state = ConversationState(
            user_id=sample_user_profile.user_id,
            current_language=LanguageCode.HINDI
        )
        
        assert state.user_id == sample_user_profile.user_id
        assert state.current_language == LanguageCode.HINDI
        assert state.is_active is True
        assert len(state.conversation_history) == 0
        assert state.interaction_count == 0
    
    def test_session_duration_calculation(self, sample_user_profile):
        """Test session duration calculation."""
        state = ConversationState(
            user_id=sample_user_profile.user_id,
            current_language=LanguageCode.HINDI
        )
        
        # Empty conversation should have 0 duration
        assert state.session_duration == 0.0
    
    @given(st.uuids(), language_code_strategy())
    def test_conversation_state_property_invariants(self, user_id, language):
        """Property test: ConversationState invariants."""
        state = ConversationState(
            user_id=user_id,
            current_language=language
        )
        
        # Session ID should be valid UUID
        assert isinstance(state.session_id, UUID)
        
        # User ID should match input
        assert state.user_id == user_id
        
        # Language should match input
        assert state.current_language == language
        
        # Should start as active
        assert state.is_active is True
        
        # Should start with empty history
        assert len(state.conversation_history) == 0


class TestRecognitionResult:
    """Test RecognitionResult model."""
    
    def test_valid_recognition_result_creation(self):
        """Test creating a valid RecognitionResult."""
        result = RecognitionResult(
            transcribed_text="Hello world",
            confidence=0.95,
            detected_language=LanguageCode.ENGLISH_IN,
            processing_time=1.2
        )
        
        assert result.transcribed_text == "Hello world"
        assert result.confidence == 0.95
        assert result.detected_language == LanguageCode.ENGLISH_IN
        assert result.processing_time == 1.2
        assert result.has_code_switching is False
    
    def test_code_switching_detection(self):
        """Test code-switching detection in RecognitionResult."""
        switch_point = LanguageSwitchPoint(
            position=5,
            from_language=LanguageCode.HINDI,
            to_language=LanguageCode.ENGLISH_IN,
            confidence=0.8
        )
        
        result = RecognitionResult(
            transcribed_text="नमस्ते hello",
            confidence=0.9,
            detected_language=LanguageCode.HINDI,
            code_switching_points=[switch_point],
            processing_time=1.5
        )
        
        assert result.has_code_switching is True
        assert len(result.code_switching_points) == 1
        assert result.code_switching_points[0].position == 5


class TestLocationData:
    """Test LocationData model."""
    
    def test_valid_location_data_creation(self):
        """Test creating valid LocationData."""
        location = LocationData(
            latitude=28.6139,
            longitude=77.2090,
            city="New Delhi",
            state="Delhi",
            country="India"
        )
        
        assert location.latitude == 28.6139
        assert location.longitude == 77.2090
        assert location.city == "New Delhi"
        assert location.state == "Delhi"
        assert location.country == "India"
        assert location.timezone == "Asia/Kolkata"
    
    def test_invalid_coordinates(self):
        """Test LocationData with invalid coordinates."""
        with pytest.raises(ValueError):
            LocationData(
                latitude=100.0,  # Invalid latitude
                longitude=77.2090,
                city="Test",
                state="Test"
            )
        
        with pytest.raises(ValueError):
            LocationData(
                latitude=28.6139,
                longitude=200.0,  # Invalid longitude
                city="Test",
                state="Test"
            )
    
    @given(location_data_strategy())
    def test_location_data_property_invariants(self, location):
        """Property test: LocationData invariants."""
        # Latitude should be in valid range
        assert -90.0 <= location.latitude <= 90.0
        
        # Longitude should be in valid range
        assert -180.0 <= location.longitude <= 180.0
        
        # City and state should not be empty
        assert len(location.city) > 0
        assert len(location.state) > 0
        
        # Country should be India for this system
        assert location.country == "India"


class TestVoiceActivityResult:
    """Test VoiceActivityResult model."""
    
    def test_valid_voice_activity_result(self):
        """Test creating valid VoiceActivityResult."""
        result = VoiceActivityResult(
            is_speech=True,
            confidence=0.85,
            start_time=0.5,
            end_time=2.3,
            energy_level=0.7
        )
        
        assert result.is_speech is True
        assert result.confidence == 0.85
        assert result.start_time == 0.5
        assert result.end_time == 2.3
        assert result.energy_level == 0.7
    
    @given(
        st.booleans(),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    def test_voice_activity_result_property_invariants(
        self, is_speech, confidence, start_time, end_time, energy_level
    ):
        """Property test: VoiceActivityResult invariants."""
        # Ensure end_time >= start_time
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        
        result = VoiceActivityResult(
            is_speech=is_speech,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            energy_level=energy_level
        )
        
        # Confidence should be in valid range
        assert 0.0 <= result.confidence <= 1.0
        
        # Energy level should be in valid range
        assert 0.0 <= result.energy_level <= 1.0
        
        # End time should be >= start time
        assert result.end_time >= result.start_time
        
        # Times should be non-negative
        assert result.start_time >= 0.0
        assert result.end_time >= 0.0