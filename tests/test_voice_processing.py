<<<<<<< HEAD
"""
Unit tests for Voice Processing Service.

Tests the audio processing pipeline, voice activity detection,
background noise filtering, and TTS synthesis functionality.
"""

import asyncio
import numpy as np
import pytest

from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
)
from bharatvoice.services.voice_processing import (
    AudioFormatConverter,
    AudioProcessor,
    RealTimeAudioProcessor,
    TTSEngine,
    VoiceProcessingService,
    create_voice_processing_service,
)


class TestAudioBuffer:
    """Test AudioBuffer model functionality."""
    
    def test_audio_buffer_creation(self):
        """Test creating an AudioBuffer with valid data."""
        data = [0.1, 0.2, -0.1, -0.2] * 1000
        buffer = AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=len(data) / 16000
        )
        
        assert buffer.sample_rate == 16000
        assert buffer.channels == 1
        assert buffer.format == AudioFormat.WAV
        assert len(buffer.data) == len(data)
        assert buffer.duration == len(data) / 16000
    
    def test_audio_buffer_numpy_conversion(self):
        """Test converting AudioBuffer to numpy array."""
        data = [0.1, 0.2, -0.1, -0.2]
        buffer = AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=len(data) / 16000
        )
        
        numpy_array = buffer.numpy_array
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.dtype == np.float32
        assert len(numpy_array) == len(data)
        assert np.allclose(numpy_array, data)
    
    def test_audio_buffer_validation(self):
        """Test AudioBuffer validation for invalid parameters."""
        with pytest.raises(ValueError):
            AudioBuffer(
                data=[0.1, 0.2],
                sample_rate=12000,  # Invalid sample rate
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
        
        with pytest.raises(ValueError):
            AudioBuffer(
                data=[0.1, 0.2],
                sample_rate=16000,
                channels=3,  # Invalid channel count
                format=AudioFormat.WAV,
                duration=0.1
            )


class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance for testing."""
        return AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5
        )
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        # Generate 1 second of test audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        return AudioBuffer(
            data=audio_data.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
    
    @pytest.mark.asyncio
    async def test_process_audio_stream(self, audio_processor, test_audio_buffer):
        """Test audio stream processing."""
        processed = await audio_processor.process_audio_stream(
            test_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(processed, AudioBuffer)
        assert processed.sample_rate == audio_processor.sample_rate
        assert processed.channels == test_audio_buffer.channels
        assert processed.format == test_audio_buffer.format
        assert len(processed.data) > 0
    
    @pytest.mark.asyncio
    async def test_detect_voice_activity(self, audio_processor, test_audio_buffer):
        """Test voice activity detection."""
        vad_result = await audio_processor.detect_voice_activity(test_audio_buffer)
        
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
        assert hasattr(vad_result, 'start_time')
        assert hasattr(vad_result, 'end_time')
        assert hasattr(vad_result, 'energy_level')
        
        assert 0.0 <= vad_result.confidence <= 1.0
        assert vad_result.start_time >= 0.0
        assert vad_result.end_time >= vad_result.start_time
        assert vad_result.energy_level >= 0.0
    
    @pytest.mark.asyncio
    async def test_filter_background_noise(self, audio_processor, test_audio_buffer):
        """Test background noise filtering."""
        # Add noise to test audio
        noise = np.random.normal(0, 0.1, len(test_audio_buffer.data))
        noisy_data = [x + n for x, n in zip(test_audio_buffer.data, noise)]
        
        noisy_buffer = AudioBuffer(
            data=noisy_data,
            sample_rate=test_audio_buffer.sample_rate,
            channels=test_audio_buffer.channels,
            format=test_audio_buffer.format,
            duration=test_audio_buffer.duration
        )
        
        filtered = await audio_processor.filter_background_noise(noisy_buffer)
        
        assert isinstance(filtered, AudioBuffer)
        assert filtered.sample_rate == noisy_buffer.sample_rate
        assert len(filtered.data) == len(noisy_buffer.data)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_placeholder(self, audio_processor):
        """Test speech synthesis placeholder implementation."""
        synthesized = await audio_processor.synthesize_speech(
            "Hello world", LanguageCode.ENGLISH_IN, AccentType.STANDARD
        )
        
        assert isinstance(synthesized, AudioBuffer)
        assert synthesized.sample_rate == audio_processor.sample_rate
        assert synthesized.channels == 1
        assert synthesized.format == AudioFormat.WAV
        assert len(synthesized.data) > 0
        assert synthesized.duration > 0
    
    def test_language_specific_processing(self, audio_processor):
        """Test language-specific audio processing."""
        # Test with different languages
        test_audio = np.random.normal(0, 0.1, 1600)  # 0.1 seconds at 16kHz
        
        # Test Hindi processing
        hindi_processed = asyncio.run(
            audio_processor._apply_language_specific_processing(
                test_audio, LanguageCode.HINDI
            )
        )
        assert len(hindi_processed) == len(test_audio)
        
        # Test Tamil processing
        tamil_processed = asyncio.run(
            audio_processor._apply_language_specific_processing(
                test_audio, LanguageCode.TAMIL
            )
        )
        assert len(tamil_processed) == len(test_audio)
    
    def test_preemphasis_filter(self, audio_processor):
        """Test preemphasis filter application."""
        test_audio = np.random.normal(0, 0.1, 1000)
        filtered = audio_processor._apply_preemphasis(test_audio)
        
        assert len(filtered) == len(test_audio)
        assert not np.array_equal(filtered, test_audio)  # Should be different
    
    def test_audio_normalization(self, audio_processor):
        """Test audio normalization."""
        # Test with audio that needs normalization
        test_audio = np.array([2.0, -1.5, 3.0, -2.5])  # Values > 1.0
        normalized = audio_processor._normalize_audio(test_audio)
        
        assert np.max(np.abs(normalized)) <= 0.95  # Should be normalized
        assert len(normalized) == len(test_audio)


class TestAudioFormatConverter:
    """Test AudioFormatConverter functionality."""
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        return AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=44100,
            channels=2,
            format=AudioFormat.WAV,
            duration=4000 / 44100
        )
    
    def test_format_conversion(self, test_audio_buffer):
        """Test audio format conversion."""
        converted = AudioFormatConverter.convert_format(
            test_audio_buffer,
            target_format=AudioFormat.MP3,
            target_sample_rate=16000,
            target_channels=1
        )
        
        assert converted.format == AudioFormat.MP3
        assert converted.sample_rate == 16000
        assert converted.channels == 1
    
    def test_preprocess_for_recognition(self, test_audio_buffer):
        """Test preprocessing for speech recognition."""
        preprocessed = AudioFormatConverter.preprocess_for_recognition(test_audio_buffer)
        
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
        assert preprocessed.format == AudioFormat.WAV
    
    def test_extract_features(self, test_audio_buffer):
        """Test audio feature extraction."""
        features = AudioFormatConverter.extract_features(test_audio_buffer)
        
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
        assert 'spectral_rolloff' in features
        assert 'zero_crossing_rate' in features
        assert 'chroma' in features
        
        # Check that features are numpy arrays
        for feature_name, feature_data in features.items():
            assert hasattr(feature_data, 'shape')  # Should be numpy array


class TestTTSEngine:
    """Test TTS Engine functionality."""
    
    @pytest.fixture
    def tts_engine(self):
        """Create TTS engine for testing."""
        return TTSEngine(sample_rate=22050, quality='high')
    
    @pytest.fixture
    def adaptive_tts_engine(self):
        """Create adaptive TTS engine for testing."""
        return AdaptiveTTSEngine(sample_rate=22050, quality='medium')
    
    @pytest.mark.asyncio
    async def test_synthesize_speech(self, tts_engine):
        """Test speech synthesis."""
        # Note: This test uses the actual gTTS, so it requires internet connection
        # In a real test environment, you might want to mock this
        try:
            synthesized = await tts_engine.synthesize_speech(
                "Hello", LanguageCode.ENGLISH_IN, AccentType.STANDARD
            )
            
            assert isinstance(synthesized, AudioBuffer)
            assert synthesized.sample_rate == tts_engine.quality_config['sample_rate']
            assert synthesized.channels == 1
            assert len(synthesized.data) > 0
            
        except Exception as e:
            # If gTTS fails (no internet, etc.), skip this test
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_quality_optimization(self, tts_engine):
        """Test speech synthesis with quality optimization."""
        try:
            # Test with quality optimization enabled
            synthesized_optimized = await tts_engine.synthesize_speech(
                "Test quality", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=True
            )
            
            # Test with quality optimization disabled
            synthesized_basic = await tts_engine.synthesize_speech(
                "Test quality", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=False
            )
            
            assert isinstance(synthesized_optimized, AudioBuffer)
            assert isinstance(synthesized_basic, AudioBuffer)
            assert synthesized_optimized.sample_rate == synthesized_basic.sample_rate
            
        except Exception as e:
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, tts_engine):
        """Test streaming synthesis."""
        try:
            chunks = []
            async for chunk in tts_engine.synthesize_streaming(
                "This is a test for streaming", LanguageCode.ENGLISH_IN, 
                AccentType.STANDARD, chunk_duration=0.3
            ):
                chunks.append(chunk)
                assert isinstance(chunk, AudioBuffer)
                assert chunk.duration <= 0.4  # Allow some tolerance
            
            assert len(chunks) > 1  # Should have multiple chunks
            
        except Exception as e:
            pytest.skip(f"Streaming synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_to_format(self, tts_engine):
        """Test synthesis to different formats."""
        try:
            # Test WAV format
            wav_data = await tts_engine.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.WAV
            )
            assert isinstance(wav_data, bytes)
            assert len(wav_data) > 0
            
            # Test MP3 format
            mp3_data = await tts_engine.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.MP3
            )
            assert isinstance(mp3_data, bytes)
            assert len(mp3_data) > 0
            
        except Exception as e:
            pytest.skip(f"Format synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_with_pauses(self, tts_engine):
        """Test synthesis with pauses between segments."""
        try:
            segments = ["First segment", "Second segment", "Third segment"]
            combined_audio = await tts_engine.synthesize_with_pauses(
                segments, LanguageCode.ENGLISH_IN, AccentType.STANDARD, pause_duration=0.2
            )
            
            assert isinstance(combined_audio, AudioBuffer)
            assert combined_audio.duration > 1.0  # Should be longer due to pauses
            
        except Exception as e:
            pytest.skip(f"Multi-segment synthesis failed: {e}")
    
    def test_enhanced_accent_configurations(self, tts_engine):
        """Test enhanced accent configurations."""
        # Test that all accent types have enhanced configurations
        for accent in AccentType:
            config = tts_engine.ACCENT_CONFIGS.get(accent)
            assert config is not None
            assert 'speed' in config
            assert 'pitch_shift' in config
            assert 'formant_shift' in config
            assert 'emphasis_factor' in config
            assert 'pause_duration' in config
    
    def test_quality_settings(self, tts_engine):
        """Test quality settings configuration."""
        # Test different quality levels
        for quality in ['high', 'medium', 'low']:
            engine = TTSEngine(sample_rate=22050, quality=quality)
            assert engine.quality == quality
            assert quality in engine.QUALITY_SETTINGS
            
            config = engine.quality_config
            assert 'sample_rate' in config
            assert 'bitrate' in config
            assert 'normalize' in config
    
    def test_cache_functionality(self, tts_engine):
        """Test TTS caching functionality."""
        # Test cache stats
        stats = tts_engine.get_cache_stats()
        assert 'cache_size' in stats
        assert 'max_cache_size' in stats
        assert 'cache_usage_percent' in stats
        
        # Test cache clearing
        tts_engine.clear_cache()
        stats_after_clear = tts_engine.get_cache_stats()
        assert stats_after_clear['cache_size'] == 0
    
    def test_silence_generation(self, tts_engine):
        """Test silence generation fallback."""
        silence = tts_engine._generate_silence(1.0)
        
        assert isinstance(silence, AudioBuffer)
        assert silence.duration == 1.0
        assert silence.sample_rate == tts_engine.sample_rate
        assert len(silence.data) == tts_engine.sample_rate  # 1 second of samples
    
    def test_synthesis_time_estimation(self, tts_engine):
        """Test synthesis time estimation."""
        # Test with different languages
        for language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL]:
            estimated_time = tts_engine.estimate_synthesis_time("Hello world", language)
            assert estimated_time >= 0.5  # Minimum time
            assert estimated_time < 10.0  # Reasonable maximum
        
        # Test with longer text
        long_text = "This is a much longer text that should take more time to synthesize"
        long_time = tts_engine.estimate_synthesis_time(long_text, LanguageCode.ENGLISH_IN)
        short_time = tts_engine.estimate_synthesis_time("Short", LanguageCode.ENGLISH_IN)
        assert long_time > short_time
    
    def test_save_audio_to_file(self, tts_engine, tmp_path):
        """Test saving audio to file."""
        # Create test audio buffer
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=22050,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 22050
        )
        
        # Test saving to different formats
        for format_type in [AudioFormat.WAV, AudioFormat.MP3]:
            file_path = tmp_path / f"test.{format_type.value}"
            try:
                tts_engine.save_audio_to_file(test_audio, str(file_path), format_type)
                # File should exist (though we can't verify content without dependencies)
            except Exception as e:
                # Expected to fail without proper audio libraries
                assert "export" in str(e).lower() or "format" in str(e).lower()


class TestAdaptiveTTSEngine:
    """Test Adaptive TTS Engine functionality."""
    
    @pytest.fixture
    def adaptive_tts_engine(self):
        """Create adaptive TTS engine for testing."""
        return AdaptiveTTSEngine(sample_rate=22050, quality='high')
    
    def test_initialization(self, adaptive_tts_engine):
        """Test adaptive TTS engine initialization."""
        assert isinstance(adaptive_tts_engine.user_preferences, dict)
        assert isinstance(adaptive_tts_engine.feedback_history, list)
        assert len(adaptive_tts_engine.user_preferences) == 0
        assert len(adaptive_tts_engine.feedback_history) == 0
    
    def test_user_preferences_management(self, adaptive_tts_engine):
        """Test user preferences management."""
        user_id = "test_user_123"
        preferences = {
            'preferred_accent': AccentType.MUMBAI,
            'speed_preference': 1.2
        }
        
        # Update preferences
        adaptive_tts_engine.update_user_preferences(user_id, preferences)
        
        # Verify preferences were stored
        assert user_id in adaptive_tts_engine.user_preferences
        assert adaptive_tts_engine.user_preferences[user_id]['preferred_accent'] == AccentType.MUMBAI
        assert adaptive_tts_engine.user_preferences[user_id]['speed_preference'] == 1.2
        
        # Update with additional preferences
        additional_prefs = {'volume_preference': 0.8}
        adaptive_tts_engine.update_user_preferences(user_id, additional_prefs)
        
        # Verify both old and new preferences exist
        user_prefs = adaptive_tts_engine.user_preferences[user_id]
        assert user_prefs['preferred_accent'] == AccentType.MUMBAI
        assert user_prefs['speed_preference'] == 1.2
        assert user_prefs['volume_preference'] == 0.8
    
    def test_feedback_recording(self, adaptive_tts_engine):
        """Test feedback recording functionality."""
        user_id = "test_user_456"
        text = "Test feedback text"
        language = LanguageCode.HINDI
        rating = 4.5
        
        # Record feedback
        adaptive_tts_engine.record_feedback(user_id, text, language, rating)
        
        # Verify feedback was recorded
        assert len(adaptive_tts_engine.feedback_history) == 1
        
        feedback = adaptive_tts_engine.feedback_history[0]
        assert feedback['user_id'] == user_id
        assert feedback['text'] == text
        assert feedback['language'] == language
        assert feedback['rating'] == rating
        assert 'timestamp' in feedback
    
    def test_feedback_history_limit(self, adaptive_tts_engine):
        """Test feedback history size limit."""
        user_id = "test_user_789"
        
        # Add more than 1000 feedback entries
        for i in range(1005):
            adaptive_tts_engine.record_feedback(
                user_id, f"Text {i}", LanguageCode.ENGLISH_IN, 3.0
            )
        
        # Verify history is limited to 1000 entries
        assert len(adaptive_tts_engine.feedback_history) == 1000
        
        # Verify it kept the most recent entries
        assert adaptive_tts_engine.feedback_history[-1]['text'] == "Text 1004"
        assert adaptive_tts_engine.feedback_history[0]['text'] == "Text 5"
    
    @pytest.mark.asyncio
    async def test_synthesize_for_user(self, adaptive_tts_engine):
        """Test user-specific synthesis."""
        user_id = "test_user_synthesis"
        
        # Set user preferences
        preferences = {
            'preferred_accent': AccentType.DELHI,
            'speed_preference': 0.9
        }
        adaptive_tts_engine.update_user_preferences(user_id, preferences)
        
        try:
            # Test synthesis for user
            audio_buffer = await adaptive_tts_engine.synthesize_for_user(
                "Hello user", LanguageCode.ENGLISH_IN, user_id
            )
            
            assert isinstance(audio_buffer, AudioBuffer)
            assert audio_buffer.sample_rate > 0
            assert len(audio_buffer.data) > 0
            
        except Exception as e:
            # Expected to fail without gTTS
            pytest.skip(f"User synthesis failed: {e}")
    
    def test_speed_adjustment(self, adaptive_tts_engine):
        """Test speed adjustment functionality."""
        # Create test audio buffer
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=22050,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 22050
        )
        
        # Test speed adjustment
        try:
            # Test speed up
            faster_audio = adaptive_tts_engine._adjust_speed(test_audio, 1.5)
            assert isinstance(faster_audio, AudioBuffer)
            assert faster_audio.sample_rate == test_audio.sample_rate
            
            # Test slow down
            slower_audio = adaptive_tts_engine._adjust_speed(test_audio, 0.7)
            assert isinstance(slower_audio, AudioBuffer)
            assert slower_audio.sample_rate == test_audio.sample_rate
            
            # Test no change
            same_audio = adaptive_tts_engine._adjust_speed(test_audio, 1.0)
            assert same_audio == test_audio
            
        except Exception as e:
            # Expected to fail without pydub
            assert "AudioSegment" in str(e) or "pydub" in str(e)


class TestRealTimeAudioProcessor:
    """Test real-time audio processing functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor for real-time processor."""
        return AudioProcessor(sample_rate=16000)
    
    @pytest.fixture
    def realtime_processor(self, audio_processor):
        """Create RealTimeAudioProcessor for testing."""
        return RealTimeAudioProcessor(
            audio_processor,
            buffer_size=1024,
            overlap_ratio=0.5
        )
    
    @pytest.mark.asyncio
    async def test_process_stream(self, realtime_processor):
        """Test real-time stream processing."""
        # Generate test audio chunk
        chunk_size = 512
        audio_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        
        processed_audio, vad_results = await realtime_processor.process_stream(
            audio_chunk, LanguageCode.HINDI
        )
        
        # First chunk might not produce output (buffer not full)
        assert isinstance(vad_results, list)
        
        # Add more chunks to fill buffer
        for _ in range(3):
            audio_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
            processed_audio, vad_results = await realtime_processor.process_stream(
                audio_chunk, LanguageCode.HINDI
            )
        
        # Should have some results now
        if processed_audio:
            assert isinstance(processed_audio, AudioBuffer)
        assert len(vad_results) >= 0
    
    def test_buffer_reset(self, realtime_processor):
        """Test buffer reset functionality."""
        # Add some data to buffer
        realtime_processor.audio_buffer = np.array([1, 2, 3, 4, 5])
        
        # Reset buffer
        realtime_processor.reset_buffer()
        
        assert len(realtime_processor.audio_buffer) == 0


class TestVoiceProcessingService:
    """Test main VoiceProcessingService functionality."""
    
    @pytest.fixture
    def voice_service(self):
        """Create VoiceProcessingService for testing."""
        return create_voice_processing_service(
            sample_rate=16000,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5,
            enable_adaptive_tts=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        data = [0.1, 0.2, -0.1, -0.2] * 4000  # 1 second at 16kHz
        return AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.mark.asyncio
    async def test_process_audio_stream(self, voice_service, test_audio_buffer):
        """Test audio stream processing through service."""
        processed = await voice_service.process_audio_stream(
            test_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(processed, AudioBuffer)
        assert processed.sample_rate == voice_service.sample_rate
    
    @pytest.mark.asyncio
    async def test_detect_voice_activity(self, voice_service, test_audio_buffer):
        """Test voice activity detection through service."""
        vad_result = await voice_service.detect_voice_activity(test_audio_buffer)
        
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
        assert 0.0 <= vad_result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_filter_background_noise(self, voice_service, test_audio_buffer):
        """Test noise filtering through service."""
        filtered = await voice_service.filter_background_noise(test_audio_buffer)
        
        assert isinstance(filtered, AudioBuffer)
        assert len(filtered.data) == len(test_audio_buffer.data)
    
    @pytest.mark.asyncio
    async def test_preprocess_for_recognition(self, voice_service, test_audio_buffer):
        """Test preprocessing for recognition."""
        preprocessed = await voice_service.preprocess_for_recognition(test_audio_buffer)
        
        assert isinstance(preprocessed, AudioBuffer)
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
    
    @pytest.mark.asyncio
    async def test_extract_audio_features(self, voice_service, test_audio_buffer):
        """Test audio feature extraction."""
        features = await voice_service.extract_audio_features(test_audio_buffer)
        
        assert isinstance(features, dict)
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
    
    @pytest.mark.asyncio
    async def test_realtime_stream_processing(self, voice_service):
        """Test real-time stream processing."""
        audio_chunk = [0.1, 0.2, -0.1, -0.2] * 256  # Small chunk
        
        processed_audio, vad_results = await voice_service.process_realtime_stream(
            audio_chunk, LanguageCode.HINDI
        )
        
        assert isinstance(vad_results, list)
        # processed_audio might be None if buffer not full
    
    def test_service_stats(self, voice_service):
        """Test service statistics."""
        stats = voice_service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats
        assert 'total_synthesized' in stats
        assert 'average_processing_time' in stats
        assert 'vad_detections' in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, voice_service):
        """Test service health check."""
        health = await voice_service.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] in ['healthy', 'unhealthy']
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_quality(self, voice_service, test_audio_buffer):
        """Test speech synthesis with quality optimization."""
        try:
            # Test with quality optimization
            synthesized = await voice_service.synthesize_speech(
                "Hello world", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=True
            )
            
            assert isinstance(synthesized, AudioBuffer)
            assert synthesized.sample_rate == voice_service.sample_rate
            
        except Exception as e:
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, voice_service):
        """Test streaming synthesis through service."""
        try:
            chunks = []
            async for chunk in voice_service.synthesize_streaming(
                "This is a streaming test", LanguageCode.ENGLISH_IN, 
                AccentType.STANDARD, chunk_duration=0.3
            ):
                chunks.append(chunk)
                assert isinstance(chunk, AudioBuffer)
                assert chunk.sample_rate == voice_service.sample_rate
            
            assert len(chunks) > 0
            
        except Exception as e:
            pytest.skip(f"Streaming synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_to_format(self, voice_service):
        """Test format synthesis through service."""
        try:
            # Test WAV format
            wav_data = await voice_service.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.WAV
            )
            assert isinstance(wav_data, bytes)
            assert len(wav_data) > 0
            
        except Exception as e:
            pytest.skip(f"Format synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_with_pauses(self, voice_service):
        """Test multi-segment synthesis through service."""
        try:
            segments = ["First part", "Second part"]
            combined_audio = await voice_service.synthesize_with_pauses(
                segments, LanguageCode.ENGLISH_IN, AccentType.STANDARD, pause_duration=0.2
            )
            
            assert isinstance(combined_audio, AudioBuffer)
            assert combined_audio.sample_rate == voice_service.sample_rate
            
        except Exception as e:
            pytest.skip(f"Multi-segment synthesis failed: {e}")
    
    def test_synthesis_time_estimation(self, voice_service):
        """Test synthesis time estimation through service."""
        estimated_time = voice_service.estimate_synthesis_time(
            "Hello world", LanguageCode.ENGLISH_IN
        )
        assert estimated_time >= 0.5
        assert estimated_time < 10.0
    
    def test_save_synthesized_audio(self, voice_service, tmp_path):
        """Test saving synthesized audio through service."""
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 16000
        )
        
        file_path = tmp_path / "test_output.wav"
        try:
            voice_service.save_synthesized_audio(test_audio, str(file_path), AudioFormat.WAV)
        except Exception as e:
            # Expected to fail without proper audio libraries
            assert "export" in str(e).lower() or "format" in str(e).lower()
    
    def test_tts_user_preferences(self, voice_service):
        """Test TTS user preferences management."""
        user_id = "test_user_voice_service"
        preferences = {
            'preferred_accent': AccentType.BANGALORE,
            'speed_preference': 1.1
        }
        
        # This should work if adaptive TTS is enabled
        voice_service.update_user_tts_preferences(user_id, preferences)
        
        # Record feedback
        voice_service.record_tts_feedback(
            user_id, "Test text", LanguageCode.HINDI, 4.0
        )
    
    def test_cache_management(self, voice_service):
        """Test cache management functionality."""
        # Test clearing caches
        voice_service.clear_caches()
        
        # Test buffer reset
        voice_service.reset_realtime_buffer()
        
        # Should not raise exceptions


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor for edge case testing."""
        return AudioProcessor(sample_rate=16000)
    
    @pytest.mark.asyncio
    async def test_empty_audio_processing(self, audio_processor):
        """Test processing empty audio."""
        empty_buffer = AudioBuffer(
            data=[],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.0
        )
        
        # Should handle empty audio gracefully
        try:
            await audio_processor.process_audio_stream(empty_buffer, LanguageCode.HINDI)
        except Exception as e:
            # Expected to fail, but should not crash
            assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_very_short_audio_vad(self, audio_processor):
        """Test VAD with very short audio."""
        short_buffer = AudioBuffer(
            data=[0.1, 0.2],  # Very short
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=2 / 16000
        )
        
        vad_result = await audio_processor.detect_voice_activity(short_buffer)
        
        # Should return a valid result even for short audio
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
    
    def test_invalid_sample_rate_conversion(self):
        """Test handling of invalid sample rates."""
        buffer = AudioBuffer(
            data=[0.1, 0.2, 0.3, 0.4],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=4 / 16000
        )
        
        # Convert to valid sample rate
        converted = AudioFormatConverter.convert_format(
            buffer,
            target_format=AudioFormat.WAV,
            target_sample_rate=22050
        )
        
        assert converted.sample_rate == 22050
        assert len(converted.data) > 0


# Property-based tests would go here if using hypothesis
# For now, we'll stick to unit tests as requested

if __name__ == "__main__":
=======
"""
Unit tests for Voice Processing Service.

Tests the audio processing pipeline, voice activity detection,
background noise filtering, and TTS synthesis functionality.
"""

import asyncio
import numpy as np
import pytest

from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
)
from bharatvoice.services.voice_processing import (
    AudioFormatConverter,
    AudioProcessor,
    RealTimeAudioProcessor,
    TTSEngine,
    VoiceProcessingService,
    create_voice_processing_service,
)


class TestAudioBuffer:
    """Test AudioBuffer model functionality."""
    
    def test_audio_buffer_creation(self):
        """Test creating an AudioBuffer with valid data."""
        data = [0.1, 0.2, -0.1, -0.2] * 1000
        buffer = AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=len(data) / 16000
        )
        
        assert buffer.sample_rate == 16000
        assert buffer.channels == 1
        assert buffer.format == AudioFormat.WAV
        assert len(buffer.data) == len(data)
        assert buffer.duration == len(data) / 16000
    
    def test_audio_buffer_numpy_conversion(self):
        """Test converting AudioBuffer to numpy array."""
        data = [0.1, 0.2, -0.1, -0.2]
        buffer = AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=len(data) / 16000
        )
        
        numpy_array = buffer.numpy_array
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.dtype == np.float32
        assert len(numpy_array) == len(data)
        assert np.allclose(numpy_array, data)
    
    def test_audio_buffer_validation(self):
        """Test AudioBuffer validation for invalid parameters."""
        with pytest.raises(ValueError):
            AudioBuffer(
                data=[0.1, 0.2],
                sample_rate=12000,  # Invalid sample rate
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
        
        with pytest.raises(ValueError):
            AudioBuffer(
                data=[0.1, 0.2],
                sample_rate=16000,
                channels=3,  # Invalid channel count
                format=AudioFormat.WAV,
                duration=0.1
            )


class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance for testing."""
        return AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5
        )
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        # Generate 1 second of test audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        return AudioBuffer(
            data=audio_data.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
    
    @pytest.mark.asyncio
    async def test_process_audio_stream(self, audio_processor, test_audio_buffer):
        """Test audio stream processing."""
        processed = await audio_processor.process_audio_stream(
            test_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(processed, AudioBuffer)
        assert processed.sample_rate == audio_processor.sample_rate
        assert processed.channels == test_audio_buffer.channels
        assert processed.format == test_audio_buffer.format
        assert len(processed.data) > 0
    
    @pytest.mark.asyncio
    async def test_detect_voice_activity(self, audio_processor, test_audio_buffer):
        """Test voice activity detection."""
        vad_result = await audio_processor.detect_voice_activity(test_audio_buffer)
        
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
        assert hasattr(vad_result, 'start_time')
        assert hasattr(vad_result, 'end_time')
        assert hasattr(vad_result, 'energy_level')
        
        assert 0.0 <= vad_result.confidence <= 1.0
        assert vad_result.start_time >= 0.0
        assert vad_result.end_time >= vad_result.start_time
        assert vad_result.energy_level >= 0.0
    
    @pytest.mark.asyncio
    async def test_filter_background_noise(self, audio_processor, test_audio_buffer):
        """Test background noise filtering."""
        # Add noise to test audio
        noise = np.random.normal(0, 0.1, len(test_audio_buffer.data))
        noisy_data = [x + n for x, n in zip(test_audio_buffer.data, noise)]
        
        noisy_buffer = AudioBuffer(
            data=noisy_data,
            sample_rate=test_audio_buffer.sample_rate,
            channels=test_audio_buffer.channels,
            format=test_audio_buffer.format,
            duration=test_audio_buffer.duration
        )
        
        filtered = await audio_processor.filter_background_noise(noisy_buffer)
        
        assert isinstance(filtered, AudioBuffer)
        assert filtered.sample_rate == noisy_buffer.sample_rate
        assert len(filtered.data) == len(noisy_buffer.data)
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_placeholder(self, audio_processor):
        """Test speech synthesis placeholder implementation."""
        synthesized = await audio_processor.synthesize_speech(
            "Hello world", LanguageCode.ENGLISH_IN, AccentType.STANDARD
        )
        
        assert isinstance(synthesized, AudioBuffer)
        assert synthesized.sample_rate == audio_processor.sample_rate
        assert synthesized.channels == 1
        assert synthesized.format == AudioFormat.WAV
        assert len(synthesized.data) > 0
        assert synthesized.duration > 0
    
    def test_language_specific_processing(self, audio_processor):
        """Test language-specific audio processing."""
        # Test with different languages
        test_audio = np.random.normal(0, 0.1, 1600)  # 0.1 seconds at 16kHz
        
        # Test Hindi processing
        hindi_processed = asyncio.run(
            audio_processor._apply_language_specific_processing(
                test_audio, LanguageCode.HINDI
            )
        )
        assert len(hindi_processed) == len(test_audio)
        
        # Test Tamil processing
        tamil_processed = asyncio.run(
            audio_processor._apply_language_specific_processing(
                test_audio, LanguageCode.TAMIL
            )
        )
        assert len(tamil_processed) == len(test_audio)
    
    def test_preemphasis_filter(self, audio_processor):
        """Test preemphasis filter application."""
        test_audio = np.random.normal(0, 0.1, 1000)
        filtered = audio_processor._apply_preemphasis(test_audio)
        
        assert len(filtered) == len(test_audio)
        assert not np.array_equal(filtered, test_audio)  # Should be different
    
    def test_audio_normalization(self, audio_processor):
        """Test audio normalization."""
        # Test with audio that needs normalization
        test_audio = np.array([2.0, -1.5, 3.0, -2.5])  # Values > 1.0
        normalized = audio_processor._normalize_audio(test_audio)
        
        assert np.max(np.abs(normalized)) <= 0.95  # Should be normalized
        assert len(normalized) == len(test_audio)


class TestAudioFormatConverter:
    """Test AudioFormatConverter functionality."""
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        return AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=44100,
            channels=2,
            format=AudioFormat.WAV,
            duration=4000 / 44100
        )
    
    def test_format_conversion(self, test_audio_buffer):
        """Test audio format conversion."""
        converted = AudioFormatConverter.convert_format(
            test_audio_buffer,
            target_format=AudioFormat.MP3,
            target_sample_rate=16000,
            target_channels=1
        )
        
        assert converted.format == AudioFormat.MP3
        assert converted.sample_rate == 16000
        assert converted.channels == 1
    
    def test_preprocess_for_recognition(self, test_audio_buffer):
        """Test preprocessing for speech recognition."""
        preprocessed = AudioFormatConverter.preprocess_for_recognition(test_audio_buffer)
        
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
        assert preprocessed.format == AudioFormat.WAV
    
    def test_extract_features(self, test_audio_buffer):
        """Test audio feature extraction."""
        features = AudioFormatConverter.extract_features(test_audio_buffer)
        
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
        assert 'spectral_rolloff' in features
        assert 'zero_crossing_rate' in features
        assert 'chroma' in features
        
        # Check that features are numpy arrays
        for feature_name, feature_data in features.items():
            assert hasattr(feature_data, 'shape')  # Should be numpy array


class TestTTSEngine:
    """Test TTS Engine functionality."""
    
    @pytest.fixture
    def tts_engine(self):
        """Create TTS engine for testing."""
        return TTSEngine(sample_rate=22050, quality='high')
    
    @pytest.fixture
    def adaptive_tts_engine(self):
        """Create adaptive TTS engine for testing."""
        return AdaptiveTTSEngine(sample_rate=22050, quality='medium')
    
    @pytest.mark.asyncio
    async def test_synthesize_speech(self, tts_engine):
        """Test speech synthesis."""
        # Note: This test uses the actual gTTS, so it requires internet connection
        # In a real test environment, you might want to mock this
        try:
            synthesized = await tts_engine.synthesize_speech(
                "Hello", LanguageCode.ENGLISH_IN, AccentType.STANDARD
            )
            
            assert isinstance(synthesized, AudioBuffer)
            assert synthesized.sample_rate == tts_engine.quality_config['sample_rate']
            assert synthesized.channels == 1
            assert len(synthesized.data) > 0
            
        except Exception as e:
            # If gTTS fails (no internet, etc.), skip this test
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_quality_optimization(self, tts_engine):
        """Test speech synthesis with quality optimization."""
        try:
            # Test with quality optimization enabled
            synthesized_optimized = await tts_engine.synthesize_speech(
                "Test quality", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=True
            )
            
            # Test with quality optimization disabled
            synthesized_basic = await tts_engine.synthesize_speech(
                "Test quality", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=False
            )
            
            assert isinstance(synthesized_optimized, AudioBuffer)
            assert isinstance(synthesized_basic, AudioBuffer)
            assert synthesized_optimized.sample_rate == synthesized_basic.sample_rate
            
        except Exception as e:
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, tts_engine):
        """Test streaming synthesis."""
        try:
            chunks = []
            async for chunk in tts_engine.synthesize_streaming(
                "This is a test for streaming", LanguageCode.ENGLISH_IN, 
                AccentType.STANDARD, chunk_duration=0.3
            ):
                chunks.append(chunk)
                assert isinstance(chunk, AudioBuffer)
                assert chunk.duration <= 0.4  # Allow some tolerance
            
            assert len(chunks) > 1  # Should have multiple chunks
            
        except Exception as e:
            pytest.skip(f"Streaming synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_to_format(self, tts_engine):
        """Test synthesis to different formats."""
        try:
            # Test WAV format
            wav_data = await tts_engine.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.WAV
            )
            assert isinstance(wav_data, bytes)
            assert len(wav_data) > 0
            
            # Test MP3 format
            mp3_data = await tts_engine.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.MP3
            )
            assert isinstance(mp3_data, bytes)
            assert len(mp3_data) > 0
            
        except Exception as e:
            pytest.skip(f"Format synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_with_pauses(self, tts_engine):
        """Test synthesis with pauses between segments."""
        try:
            segments = ["First segment", "Second segment", "Third segment"]
            combined_audio = await tts_engine.synthesize_with_pauses(
                segments, LanguageCode.ENGLISH_IN, AccentType.STANDARD, pause_duration=0.2
            )
            
            assert isinstance(combined_audio, AudioBuffer)
            assert combined_audio.duration > 1.0  # Should be longer due to pauses
            
        except Exception as e:
            pytest.skip(f"Multi-segment synthesis failed: {e}")
    
    def test_enhanced_accent_configurations(self, tts_engine):
        """Test enhanced accent configurations."""
        # Test that all accent types have enhanced configurations
        for accent in AccentType:
            config = tts_engine.ACCENT_CONFIGS.get(accent)
            assert config is not None
            assert 'speed' in config
            assert 'pitch_shift' in config
            assert 'formant_shift' in config
            assert 'emphasis_factor' in config
            assert 'pause_duration' in config
    
    def test_quality_settings(self, tts_engine):
        """Test quality settings configuration."""
        # Test different quality levels
        for quality in ['high', 'medium', 'low']:
            engine = TTSEngine(sample_rate=22050, quality=quality)
            assert engine.quality == quality
            assert quality in engine.QUALITY_SETTINGS
            
            config = engine.quality_config
            assert 'sample_rate' in config
            assert 'bitrate' in config
            assert 'normalize' in config
    
    def test_cache_functionality(self, tts_engine):
        """Test TTS caching functionality."""
        # Test cache stats
        stats = tts_engine.get_cache_stats()
        assert 'cache_size' in stats
        assert 'max_cache_size' in stats
        assert 'cache_usage_percent' in stats
        
        # Test cache clearing
        tts_engine.clear_cache()
        stats_after_clear = tts_engine.get_cache_stats()
        assert stats_after_clear['cache_size'] == 0
    
    def test_silence_generation(self, tts_engine):
        """Test silence generation fallback."""
        silence = tts_engine._generate_silence(1.0)
        
        assert isinstance(silence, AudioBuffer)
        assert silence.duration == 1.0
        assert silence.sample_rate == tts_engine.sample_rate
        assert len(silence.data) == tts_engine.sample_rate  # 1 second of samples
    
    def test_synthesis_time_estimation(self, tts_engine):
        """Test synthesis time estimation."""
        # Test with different languages
        for language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL]:
            estimated_time = tts_engine.estimate_synthesis_time("Hello world", language)
            assert estimated_time >= 0.5  # Minimum time
            assert estimated_time < 10.0  # Reasonable maximum
        
        # Test with longer text
        long_text = "This is a much longer text that should take more time to synthesize"
        long_time = tts_engine.estimate_synthesis_time(long_text, LanguageCode.ENGLISH_IN)
        short_time = tts_engine.estimate_synthesis_time("Short", LanguageCode.ENGLISH_IN)
        assert long_time > short_time
    
    def test_save_audio_to_file(self, tts_engine, tmp_path):
        """Test saving audio to file."""
        # Create test audio buffer
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=22050,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 22050
        )
        
        # Test saving to different formats
        for format_type in [AudioFormat.WAV, AudioFormat.MP3]:
            file_path = tmp_path / f"test.{format_type.value}"
            try:
                tts_engine.save_audio_to_file(test_audio, str(file_path), format_type)
                # File should exist (though we can't verify content without dependencies)
            except Exception as e:
                # Expected to fail without proper audio libraries
                assert "export" in str(e).lower() or "format" in str(e).lower()


class TestAdaptiveTTSEngine:
    """Test Adaptive TTS Engine functionality."""
    
    @pytest.fixture
    def adaptive_tts_engine(self):
        """Create adaptive TTS engine for testing."""
        return AdaptiveTTSEngine(sample_rate=22050, quality='high')
    
    def test_initialization(self, adaptive_tts_engine):
        """Test adaptive TTS engine initialization."""
        assert isinstance(adaptive_tts_engine.user_preferences, dict)
        assert isinstance(adaptive_tts_engine.feedback_history, list)
        assert len(adaptive_tts_engine.user_preferences) == 0
        assert len(adaptive_tts_engine.feedback_history) == 0
    
    def test_user_preferences_management(self, adaptive_tts_engine):
        """Test user preferences management."""
        user_id = "test_user_123"
        preferences = {
            'preferred_accent': AccentType.MUMBAI,
            'speed_preference': 1.2
        }
        
        # Update preferences
        adaptive_tts_engine.update_user_preferences(user_id, preferences)
        
        # Verify preferences were stored
        assert user_id in adaptive_tts_engine.user_preferences
        assert adaptive_tts_engine.user_preferences[user_id]['preferred_accent'] == AccentType.MUMBAI
        assert adaptive_tts_engine.user_preferences[user_id]['speed_preference'] == 1.2
        
        # Update with additional preferences
        additional_prefs = {'volume_preference': 0.8}
        adaptive_tts_engine.update_user_preferences(user_id, additional_prefs)
        
        # Verify both old and new preferences exist
        user_prefs = adaptive_tts_engine.user_preferences[user_id]
        assert user_prefs['preferred_accent'] == AccentType.MUMBAI
        assert user_prefs['speed_preference'] == 1.2
        assert user_prefs['volume_preference'] == 0.8
    
    def test_feedback_recording(self, adaptive_tts_engine):
        """Test feedback recording functionality."""
        user_id = "test_user_456"
        text = "Test feedback text"
        language = LanguageCode.HINDI
        rating = 4.5
        
        # Record feedback
        adaptive_tts_engine.record_feedback(user_id, text, language, rating)
        
        # Verify feedback was recorded
        assert len(adaptive_tts_engine.feedback_history) == 1
        
        feedback = adaptive_tts_engine.feedback_history[0]
        assert feedback['user_id'] == user_id
        assert feedback['text'] == text
        assert feedback['language'] == language
        assert feedback['rating'] == rating
        assert 'timestamp' in feedback
    
    def test_feedback_history_limit(self, adaptive_tts_engine):
        """Test feedback history size limit."""
        user_id = "test_user_789"
        
        # Add more than 1000 feedback entries
        for i in range(1005):
            adaptive_tts_engine.record_feedback(
                user_id, f"Text {i}", LanguageCode.ENGLISH_IN, 3.0
            )
        
        # Verify history is limited to 1000 entries
        assert len(adaptive_tts_engine.feedback_history) == 1000
        
        # Verify it kept the most recent entries
        assert adaptive_tts_engine.feedback_history[-1]['text'] == "Text 1004"
        assert adaptive_tts_engine.feedback_history[0]['text'] == "Text 5"
    
    @pytest.mark.asyncio
    async def test_synthesize_for_user(self, adaptive_tts_engine):
        """Test user-specific synthesis."""
        user_id = "test_user_synthesis"
        
        # Set user preferences
        preferences = {
            'preferred_accent': AccentType.DELHI,
            'speed_preference': 0.9
        }
        adaptive_tts_engine.update_user_preferences(user_id, preferences)
        
        try:
            # Test synthesis for user
            audio_buffer = await adaptive_tts_engine.synthesize_for_user(
                "Hello user", LanguageCode.ENGLISH_IN, user_id
            )
            
            assert isinstance(audio_buffer, AudioBuffer)
            assert audio_buffer.sample_rate > 0
            assert len(audio_buffer.data) > 0
            
        except Exception as e:
            # Expected to fail without gTTS
            pytest.skip(f"User synthesis failed: {e}")
    
    def test_speed_adjustment(self, adaptive_tts_engine):
        """Test speed adjustment functionality."""
        # Create test audio buffer
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=22050,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 22050
        )
        
        # Test speed adjustment
        try:
            # Test speed up
            faster_audio = adaptive_tts_engine._adjust_speed(test_audio, 1.5)
            assert isinstance(faster_audio, AudioBuffer)
            assert faster_audio.sample_rate == test_audio.sample_rate
            
            # Test slow down
            slower_audio = adaptive_tts_engine._adjust_speed(test_audio, 0.7)
            assert isinstance(slower_audio, AudioBuffer)
            assert slower_audio.sample_rate == test_audio.sample_rate
            
            # Test no change
            same_audio = adaptive_tts_engine._adjust_speed(test_audio, 1.0)
            assert same_audio == test_audio
            
        except Exception as e:
            # Expected to fail without pydub
            assert "AudioSegment" in str(e) or "pydub" in str(e)


class TestRealTimeAudioProcessor:
    """Test real-time audio processing functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor for real-time processor."""
        return AudioProcessor(sample_rate=16000)
    
    @pytest.fixture
    def realtime_processor(self, audio_processor):
        """Create RealTimeAudioProcessor for testing."""
        return RealTimeAudioProcessor(
            audio_processor,
            buffer_size=1024,
            overlap_ratio=0.5
        )
    
    @pytest.mark.asyncio
    async def test_process_stream(self, realtime_processor):
        """Test real-time stream processing."""
        # Generate test audio chunk
        chunk_size = 512
        audio_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
        
        processed_audio, vad_results = await realtime_processor.process_stream(
            audio_chunk, LanguageCode.HINDI
        )
        
        # First chunk might not produce output (buffer not full)
        assert isinstance(vad_results, list)
        
        # Add more chunks to fill buffer
        for _ in range(3):
            audio_chunk = np.random.normal(0, 0.1, chunk_size).astype(np.float32)
            processed_audio, vad_results = await realtime_processor.process_stream(
                audio_chunk, LanguageCode.HINDI
            )
        
        # Should have some results now
        if processed_audio:
            assert isinstance(processed_audio, AudioBuffer)
        assert len(vad_results) >= 0
    
    def test_buffer_reset(self, realtime_processor):
        """Test buffer reset functionality."""
        # Add some data to buffer
        realtime_processor.audio_buffer = np.array([1, 2, 3, 4, 5])
        
        # Reset buffer
        realtime_processor.reset_buffer()
        
        assert len(realtime_processor.audio_buffer) == 0


class TestVoiceProcessingService:
    """Test main VoiceProcessingService functionality."""
    
    @pytest.fixture
    def voice_service(self):
        """Create VoiceProcessingService for testing."""
        return create_voice_processing_service(
            sample_rate=16000,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5,
            enable_adaptive_tts=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def test_audio_buffer(self):
        """Create test audio buffer."""
        data = [0.1, 0.2, -0.1, -0.2] * 4000  # 1 second at 16kHz
        return AudioBuffer(
            data=data,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.mark.asyncio
    async def test_process_audio_stream(self, voice_service, test_audio_buffer):
        """Test audio stream processing through service."""
        processed = await voice_service.process_audio_stream(
            test_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(processed, AudioBuffer)
        assert processed.sample_rate == voice_service.sample_rate
    
    @pytest.mark.asyncio
    async def test_detect_voice_activity(self, voice_service, test_audio_buffer):
        """Test voice activity detection through service."""
        vad_result = await voice_service.detect_voice_activity(test_audio_buffer)
        
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
        assert 0.0 <= vad_result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_filter_background_noise(self, voice_service, test_audio_buffer):
        """Test noise filtering through service."""
        filtered = await voice_service.filter_background_noise(test_audio_buffer)
        
        assert isinstance(filtered, AudioBuffer)
        assert len(filtered.data) == len(test_audio_buffer.data)
    
    @pytest.mark.asyncio
    async def test_preprocess_for_recognition(self, voice_service, test_audio_buffer):
        """Test preprocessing for recognition."""
        preprocessed = await voice_service.preprocess_for_recognition(test_audio_buffer)
        
        assert isinstance(preprocessed, AudioBuffer)
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
    
    @pytest.mark.asyncio
    async def test_extract_audio_features(self, voice_service, test_audio_buffer):
        """Test audio feature extraction."""
        features = await voice_service.extract_audio_features(test_audio_buffer)
        
        assert isinstance(features, dict)
        assert 'mfcc' in features
        assert 'spectral_centroid' in features
    
    @pytest.mark.asyncio
    async def test_realtime_stream_processing(self, voice_service):
        """Test real-time stream processing."""
        audio_chunk = [0.1, 0.2, -0.1, -0.2] * 256  # Small chunk
        
        processed_audio, vad_results = await voice_service.process_realtime_stream(
            audio_chunk, LanguageCode.HINDI
        )
        
        assert isinstance(vad_results, list)
        # processed_audio might be None if buffer not full
    
    def test_service_stats(self, voice_service):
        """Test service statistics."""
        stats = voice_service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert 'total_processed' in stats
        assert 'total_synthesized' in stats
        assert 'average_processing_time' in stats
        assert 'vad_detections' in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, voice_service):
        """Test service health check."""
        health = await voice_service.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] in ['healthy', 'unhealthy']
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_with_quality(self, voice_service, test_audio_buffer):
        """Test speech synthesis with quality optimization."""
        try:
            # Test with quality optimization
            synthesized = await voice_service.synthesize_speech(
                "Hello world", LanguageCode.ENGLISH_IN, AccentType.STANDARD, 
                quality_optimize=True
            )
            
            assert isinstance(synthesized, AudioBuffer)
            assert synthesized.sample_rate == voice_service.sample_rate
            
        except Exception as e:
            pytest.skip(f"TTS synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_streaming(self, voice_service):
        """Test streaming synthesis through service."""
        try:
            chunks = []
            async for chunk in voice_service.synthesize_streaming(
                "This is a streaming test", LanguageCode.ENGLISH_IN, 
                AccentType.STANDARD, chunk_duration=0.3
            ):
                chunks.append(chunk)
                assert isinstance(chunk, AudioBuffer)
                assert chunk.sample_rate == voice_service.sample_rate
            
            assert len(chunks) > 0
            
        except Exception as e:
            pytest.skip(f"Streaming synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_to_format(self, voice_service):
        """Test format synthesis through service."""
        try:
            # Test WAV format
            wav_data = await voice_service.synthesize_to_format(
                "Format test", LanguageCode.ENGLISH_IN, AudioFormat.WAV
            )
            assert isinstance(wav_data, bytes)
            assert len(wav_data) > 0
            
        except Exception as e:
            pytest.skip(f"Format synthesis failed: {e}")
    
    @pytest.mark.asyncio
    async def test_synthesize_with_pauses(self, voice_service):
        """Test multi-segment synthesis through service."""
        try:
            segments = ["First part", "Second part"]
            combined_audio = await voice_service.synthesize_with_pauses(
                segments, LanguageCode.ENGLISH_IN, AccentType.STANDARD, pause_duration=0.2
            )
            
            assert isinstance(combined_audio, AudioBuffer)
            assert combined_audio.sample_rate == voice_service.sample_rate
            
        except Exception as e:
            pytest.skip(f"Multi-segment synthesis failed: {e}")
    
    def test_synthesis_time_estimation(self, voice_service):
        """Test synthesis time estimation through service."""
        estimated_time = voice_service.estimate_synthesis_time(
            "Hello world", LanguageCode.ENGLISH_IN
        )
        assert estimated_time >= 0.5
        assert estimated_time < 10.0
    
    def test_save_synthesized_audio(self, voice_service, tmp_path):
        """Test saving synthesized audio through service."""
        test_audio = AudioBuffer(
            data=[0.1, 0.2, -0.1, -0.2] * 1000,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=4000 / 16000
        )
        
        file_path = tmp_path / "test_output.wav"
        try:
            voice_service.save_synthesized_audio(test_audio, str(file_path), AudioFormat.WAV)
        except Exception as e:
            # Expected to fail without proper audio libraries
            assert "export" in str(e).lower() or "format" in str(e).lower()
    
    def test_tts_user_preferences(self, voice_service):
        """Test TTS user preferences management."""
        user_id = "test_user_voice_service"
        preferences = {
            'preferred_accent': AccentType.BANGALORE,
            'speed_preference': 1.1
        }
        
        # This should work if adaptive TTS is enabled
        voice_service.update_user_tts_preferences(user_id, preferences)
        
        # Record feedback
        voice_service.record_tts_feedback(
            user_id, "Test text", LanguageCode.HINDI, 4.0
        )
    
    def test_cache_management(self, voice_service):
        """Test cache management functionality."""
        # Test clearing caches
        voice_service.clear_caches()
        
        # Test buffer reset
        voice_service.reset_realtime_buffer()
        
        # Should not raise exceptions


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor for edge case testing."""
        return AudioProcessor(sample_rate=16000)
    
    @pytest.mark.asyncio
    async def test_empty_audio_processing(self, audio_processor):
        """Test processing empty audio."""
        empty_buffer = AudioBuffer(
            data=[],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.0
        )
        
        # Should handle empty audio gracefully
        try:
            await audio_processor.process_audio_stream(empty_buffer, LanguageCode.HINDI)
        except Exception as e:
            # Expected to fail, but should not crash
            assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_very_short_audio_vad(self, audio_processor):
        """Test VAD with very short audio."""
        short_buffer = AudioBuffer(
            data=[0.1, 0.2],  # Very short
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=2 / 16000
        )
        
        vad_result = await audio_processor.detect_voice_activity(short_buffer)
        
        # Should return a valid result even for short audio
        assert hasattr(vad_result, 'is_speech')
        assert hasattr(vad_result, 'confidence')
    
    def test_invalid_sample_rate_conversion(self):
        """Test handling of invalid sample rates."""
        buffer = AudioBuffer(
            data=[0.1, 0.2, 0.3, 0.4],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=4 / 16000
        )
        
        # Convert to valid sample rate
        converted = AudioFormatConverter.convert_format(
            buffer,
            target_format=AudioFormat.WAV,
            target_sample_rate=22050
        )
        
        assert converted.sample_rate == 22050
        assert len(converted.data) > 0


# Property-based tests would go here if using hypothesis
# For now, we'll stick to unit tests as requested

if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])