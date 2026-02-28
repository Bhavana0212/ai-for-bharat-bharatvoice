<<<<<<< HEAD
"""
Unit tests for the Language Engine Service.

This module tests the multilingual ASR system, language detection,
code-switching detection, and translation capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine import (
    LanguageEngineService,
    MultilingualASREngine,
    create_language_engine_service,
    create_multilingual_asr_engine,
)


class TestMultilingualASREngine:
    """Test cases for the MultilingualASREngine class."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model for testing."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "नमस्ते, आप कैसे हैं?",
            "language": "hi",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "नमस्ते, आप कैसे हैं?",
                    "avg_logprob": -0.5
                }
            ]
        }
        return mock_model
    
    @pytest.fixture
    def sample_audio_buffer(self):
        """Create a sample audio buffer for testing."""
        return AudioBuffer(
            data=[0.1, -0.1, 0.2, -0.2] * 4000,  # 1 second at 16kHz
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.fixture
    def asr_engine(self, mock_whisper_model):
        """Create ASR engine with mocked Whisper model."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model') as mock_load:
            mock_load.return_value = mock_whisper_model
            engine = create_multilingual_asr_engine(
                model_size="base",
                device="cpu",
                enable_language_detection=True
            )
            return engine
    
    @pytest.mark.asyncio
    async def test_recognize_speech_basic(self, asr_engine, sample_audio_buffer):
        """Test basic speech recognition functionality."""
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(sample_audio_buffer)
            
            assert isinstance(result, RecognitionResult)
            assert result.transcribed_text == "नमस्ते, आप कैसे हैं?"
            assert result.detected_language == LanguageCode.HINDI
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_recognize_speech_empty_audio(self, asr_engine):
        """Test recognition with empty audio."""
        empty_audio = AudioBuffer(
            data=[],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.0
        )
        
        result = await asr_engine.recognize_speech(empty_audio)
        
        assert isinstance(result, RecognitionResult)
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_language_hindi(self, asr_engine):
        """Test language detection for Hindi text."""
        hindi_text = "नमस्ते, आप कैसे हैं?"
        
        detected_language = await asr_engine.detect_language(hindi_text)
        
        # Should detect Hindi or fall back to English
        assert detected_language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
    
    @pytest.mark.asyncio
    async def test_detect_language_english(self, asr_engine):
        """Test language detection for English text."""
        english_text = "Hello, how are you?"
        
        detected_language = await asr_engine.detect_language(english_text)
        
        assert detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_language_empty_text(self, asr_engine):
        """Test language detection with empty text."""
        detected_language = await asr_engine.detect_language("")
        
        assert detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_code_switching(self, asr_engine):
        """Test code-switching detection in mixed-language text."""
        mixed_text = "Hello नमस्ते, how are you आप कैसे हैं?"
        
        code_switches = await asr_engine.detect_code_switching(mixed_text)
        
        assert isinstance(code_switches, list)
        # Should detect at least some language switches
        # (exact number depends on detection algorithm)
    
    @pytest.mark.asyncio
    async def test_get_detailed_code_switching_analysis(self, asr_engine):
        """Test detailed code-switching analysis."""
        mixed_text = "Hello नमस्ते world"
        
        result = await asr_engine.get_detailed_code_switching_analysis(mixed_text)
        
        assert hasattr(result, 'segments')
        assert hasattr(result, 'switch_points')
        assert hasattr(result, 'dominant_language')
        assert hasattr(result, 'switching_frequency')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time')
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, asr_engine):
        """Test language transition suggestions."""
        suggestions = await asr_engine.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
    
    @pytest.mark.asyncio
    async def test_translate_text_same_language(self, asr_engine):
        """Test translation when source and target languages are the same."""
        text = "Hello, world!"
        
        result = await asr_engine.translate_text(
            text, LanguageCode.ENGLISH_IN, LanguageCode.ENGLISH_IN
        )
        
        assert result == text
    
    @pytest.mark.asyncio
    async def test_translate_text_different_languages(self, asr_engine):
        """Test translation between different languages."""
        text = "Hello, world!"
        
        result = await asr_engine.translate_text(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Should return some translation (placeholder implementation)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_supported_languages(self, asr_engine):
        """Test getting list of supported languages."""
        languages = asr_engine.get_supported_languages()
        
        assert isinstance(languages, list)
        assert LanguageCode.HINDI in languages
        assert LanguageCode.ENGLISH_IN in languages
        assert LanguageCode.TAMIL in languages
    
    def test_get_model_info(self, asr_engine):
        """Test getting model information."""
        info = asr_engine.get_model_info()
        
        assert isinstance(info, dict)
        assert "whisper_model_size" in info
        assert "device" in info
        assert "supported_languages" in info
        assert info["whisper_model_size"] == "base"
    
    @pytest.mark.asyncio
    async def test_health_check(self, asr_engine):
        """Test ASR engine health check."""
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            health = await asr_engine.health_check()
            
            assert isinstance(health, dict)
            assert "status" in health
            assert health["status"] in ["healthy", "unhealthy"]


class TestLanguageEngineService:
    """Test cases for the LanguageEngineService class."""
    
    @pytest.fixture
    def mock_asr_engine(self):
        """Mock ASR engine for testing."""
        mock_engine = Mock(spec=MultilingualASREngine)
        
        # Mock recognition result
        mock_result = RecognitionResult(
            transcribed_text="Test transcription",
            confidence=0.85,
            detected_language=LanguageCode.ENGLISH_IN,
            code_switching_points=[],
            alternative_transcriptions=[],
            processing_time=0.5
        )
        mock_engine.recognize_speech = AsyncMock(return_value=mock_result)
        mock_engine.detect_language = AsyncMock(return_value=LanguageCode.ENGLISH_IN)
        mock_engine.detect_code_switching = AsyncMock(return_value=[])
        mock_engine.translate_text = AsyncMock(return_value="Translated text")
        mock_engine.get_supported_languages.return_value = [
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL
        ]
        mock_engine.get_model_info.return_value = {"model": "test"}
        mock_engine.health_check = AsyncMock(return_value={"status": "healthy"})
        
        return mock_engine
    
    @pytest.fixture
    def sample_audio_buffer(self):
        """Create a sample audio buffer for testing."""
        return AudioBuffer(
            data=[0.1, -0.1, 0.2, -0.2] * 4000,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.fixture
    def language_service(self, mock_asr_engine):
        """Create language service with mocked ASR engine."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine') as mock_create:
            mock_create.return_value = mock_asr_engine
            service = create_language_engine_service(
                asr_model_size="base",
                device="cpu",
                enable_caching=True
            )
            return service
    
    @pytest.mark.asyncio
    async def test_recognize_speech_basic(self, language_service, sample_audio_buffer):
        """Test basic speech recognition through service."""
        result = await language_service.recognize_speech(sample_audio_buffer)
        
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == "Test transcription"
        assert result.confidence == 0.85
        assert result.detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_recognize_speech_caching(self, language_service, sample_audio_buffer):
        """Test recognition result caching."""
        # First call should miss cache
        result1 = await language_service.recognize_speech(sample_audio_buffer)
        
        # Second call with same audio should hit cache
        result2 = await language_service.recognize_speech(sample_audio_buffer)
        
        assert result1.transcribed_text == result2.transcribed_text
        assert language_service.stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching(self, language_service):
        """Test code-switching detection through service."""
        mixed_text = "Hello नमस्ते world"
        
        result = await language_service.detect_code_switching(mixed_text)
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_detailed_code_switching_analysis(self, language_service):
        """Test detailed code-switching analysis through service."""
        mixed_text = "Hello नमस्ते world"
        
        result = await language_service.get_detailed_code_switching_analysis(mixed_text)
        
        assert isinstance(result, dict)
        assert 'segments' in result
        assert 'switch_points' in result
        assert 'dominant_language' in result
        assert 'switching_frequency' in result
        assert 'confidence' in result
        assert 'processing_time' in result
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, language_service):
        """Test language transition suggestions through service."""
        suggestions = await language_service.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
    
    @pytest.mark.asyncio
    async def test_translate_text(self, language_service):
        """Test text translation through service."""
        result = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result == "Translated text"
    
    @pytest.mark.asyncio
    async def test_translate_text_caching(self, language_service):
        """Test translation result caching."""
        # First translation should miss cache
        result1 = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Second identical translation should hit cache
        result2 = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result1 == result2
        assert language_service.stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_detect_language(self, language_service):
        """Test language detection through service."""
        result = await language_service.detect_language("Hello world")
        
        assert result == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_batch_recognize_speech(self, language_service, sample_audio_buffer):
        """Test batch speech recognition."""
        audio_buffers = [sample_audio_buffer, sample_audio_buffer, sample_audio_buffer]
        
        results = await language_service.batch_recognize_speech(audio_buffers)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RecognitionResult)
            assert result.transcribed_text == "Test transcription"
    
    @pytest.mark.asyncio
    async def test_batch_recognize_speech_empty_list(self, language_service):
        """Test batch recognition with empty list."""
        results = await language_service.batch_recognize_speech([])
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_get_language_confidence_scores(self, language_service, sample_audio_buffer):
        """Test getting language confidence scores."""
        scores = await language_service.get_language_confidence_scores(sample_audio_buffer)
        
        assert isinstance(scores, dict)
        assert LanguageCode.ENGLISH_IN in scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    @pytest.mark.asyncio
    async def test_recognize_with_language_hint(self, language_service, sample_audio_buffer):
        """Test recognition with language hint."""
        result = await language_service.recognize_with_language_hint(
            sample_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == "Test transcription"
    
    def test_get_supported_languages(self, language_service):
        """Test getting supported languages."""
        languages = language_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert LanguageCode.HINDI in languages
        assert LanguageCode.ENGLISH_IN in languages
    
    def test_get_service_stats(self, language_service):
        """Test getting service statistics."""
        stats = language_service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert "total_recognitions" in stats
        assert "total_translations" in stats
        assert "cache_stats" in stats
        assert "asr_engine_info" in stats
    
    def test_clear_caches(self, language_service):
        """Test clearing service caches."""
        # Add something to cache first
        language_service.recognition_cache["test"] = "value"
        language_service.translation_cache["test"] = "value"
        
        language_service.clear_caches()
        
        assert len(language_service.recognition_cache) == 0
        assert len(language_service.translation_cache) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, language_service):
        """Test service health check."""
        health = await language_service.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_multilingual_asr_engine(self):
        """Test ASR engine factory function."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model'):
            engine = create_multilingual_asr_engine(
                model_size="tiny",
                device="cpu",
                enable_language_detection=False
            )
            
            assert isinstance(engine, MultilingualASREngine)
            assert engine.model_size == "tiny"
            assert engine.device == "cpu"
            assert engine.enable_language_detection == False
    
    def test_create_language_engine_service(self):
        """Test language service factory function."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine'):
            service = create_language_engine_service(
                asr_model_size="small",
                device="cuda",
                enable_caching=False
            )
            
            assert isinstance(service, LanguageEngineService)
            assert service.enable_caching == False


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.fixture
    def failing_asr_engine(self):
        """Mock ASR engine that raises exceptions."""
        mock_engine = Mock(spec=MultilingualASREngine)
        mock_engine.recognize_speech = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.detect_language = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.detect_code_switching = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.translate_text = AsyncMock(side_effect=Exception("Test error"))
        return mock_engine
    
    @pytest.fixture
    def failing_language_service(self, failing_asr_engine):
        """Create language service with failing ASR engine."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine') as mock_create:
            mock_create.return_value = failing_asr_engine
            service = create_language_engine_service()
            return service
    
    @pytest.mark.asyncio
    async def test_recognize_speech_error_handling(self, failing_language_service):
        """Test error handling in speech recognition."""
        audio_buffer = AudioBuffer(
            data=[0.1, -0.1],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.1
        )
        
        result = await failing_language_service.recognize_speech(audio_buffer)
        
        # Should return empty result on error
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == ""
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_language_error_handling(self, failing_language_service):
        """Test error handling in language detection."""
        result = await failing_language_service.detect_language("test")
        
        # Should return default language on error
        assert result == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_error_handling(self, failing_language_service):
        """Test error handling in code-switching detection."""
        result = await failing_language_service.detect_code_switching("test")
        
        # Should return empty list on error
        assert result == []
    
    @pytest.mark.asyncio
    async def test_translate_text_error_handling(self, failing_language_service):
        """Test error handling in translation."""
        result = await failing_language_service.translate_text(
            "test", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Should return original text on error
        assert result == "test"


class TestCacheManagement:
    """Test cases for cache management functionality."""
    
    @pytest.fixture
    def language_service_small_cache(self):
        """Create language service with small cache for testing eviction."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine'):
            service = create_language_engine_service(
                cache_size=2,  # Small cache for testing eviction
                enable_caching=True
            )
            return service
    
    def test_cache_eviction(self, language_service_small_cache):
        """Test LRU cache eviction."""
        service = language_service_small_cache
        
        # Fill cache beyond capacity
        service._cache_recognition_result("key1", Mock())
        service._cache_recognition_result("key2", Mock())
        service._cache_recognition_result("key3", Mock())  # Should evict key1
        
        assert len(service.recognition_cache) == 2
        assert "key1" not in service.recognition_cache
        assert "key2" in service.recognition_cache
        assert "key3" in service.recognition_cache
    
    def test_cache_key_generation(self, language_service_small_cache):
        """Test audio cache key generation."""
        service = language_service_small_cache
        
        audio1 = AudioBuffer(
            data=[0.1, 0.2, 0.3],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
        
        audio2 = AudioBuffer(
            data=[0.1, 0.2, 0.4],  # Different data
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
        
        key1 = service._generate_audio_cache_key(audio1)
        key2 = service._generate_audio_cache_key(audio2)
        
        assert key1 != key2  # Different audio should have different keys
        
        # Same audio should have same key
        key1_repeat = service._generate_audio_cache_key(audio1)
        assert key1 == key1_repeat


if __name__ == "__main__":
=======
"""
Unit tests for the Language Engine Service.

This module tests the multilingual ASR system, language detection,
code-switching detection, and translation capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine import (
    LanguageEngineService,
    MultilingualASREngine,
    create_language_engine_service,
    create_multilingual_asr_engine,
)


class TestMultilingualASREngine:
    """Test cases for the MultilingualASREngine class."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model for testing."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "नमस्ते, आप कैसे हैं?",
            "language": "hi",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "नमस्ते, आप कैसे हैं?",
                    "avg_logprob": -0.5
                }
            ]
        }
        return mock_model
    
    @pytest.fixture
    def sample_audio_buffer(self):
        """Create a sample audio buffer for testing."""
        return AudioBuffer(
            data=[0.1, -0.1, 0.2, -0.2] * 4000,  # 1 second at 16kHz
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.fixture
    def asr_engine(self, mock_whisper_model):
        """Create ASR engine with mocked Whisper model."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model') as mock_load:
            mock_load.return_value = mock_whisper_model
            engine = create_multilingual_asr_engine(
                model_size="base",
                device="cpu",
                enable_language_detection=True
            )
            return engine
    
    @pytest.mark.asyncio
    async def test_recognize_speech_basic(self, asr_engine, sample_audio_buffer):
        """Test basic speech recognition functionality."""
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(sample_audio_buffer)
            
            assert isinstance(result, RecognitionResult)
            assert result.transcribed_text == "नमस्ते, आप कैसे हैं?"
            assert result.detected_language == LanguageCode.HINDI
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_recognize_speech_empty_audio(self, asr_engine):
        """Test recognition with empty audio."""
        empty_audio = AudioBuffer(
            data=[],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.0
        )
        
        result = await asr_engine.recognize_speech(empty_audio)
        
        assert isinstance(result, RecognitionResult)
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_language_hindi(self, asr_engine):
        """Test language detection for Hindi text."""
        hindi_text = "नमस्ते, आप कैसे हैं?"
        
        detected_language = await asr_engine.detect_language(hindi_text)
        
        # Should detect Hindi or fall back to English
        assert detected_language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
    
    @pytest.mark.asyncio
    async def test_detect_language_english(self, asr_engine):
        """Test language detection for English text."""
        english_text = "Hello, how are you?"
        
        detected_language = await asr_engine.detect_language(english_text)
        
        assert detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_language_empty_text(self, asr_engine):
        """Test language detection with empty text."""
        detected_language = await asr_engine.detect_language("")
        
        assert detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_code_switching(self, asr_engine):
        """Test code-switching detection in mixed-language text."""
        mixed_text = "Hello नमस्ते, how are you आप कैसे हैं?"
        
        code_switches = await asr_engine.detect_code_switching(mixed_text)
        
        assert isinstance(code_switches, list)
        # Should detect at least some language switches
        # (exact number depends on detection algorithm)
    
    @pytest.mark.asyncio
    async def test_get_detailed_code_switching_analysis(self, asr_engine):
        """Test detailed code-switching analysis."""
        mixed_text = "Hello नमस्ते world"
        
        result = await asr_engine.get_detailed_code_switching_analysis(mixed_text)
        
        assert hasattr(result, 'segments')
        assert hasattr(result, 'switch_points')
        assert hasattr(result, 'dominant_language')
        assert hasattr(result, 'switching_frequency')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time')
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, asr_engine):
        """Test language transition suggestions."""
        suggestions = await asr_engine.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
    
    @pytest.mark.asyncio
    async def test_translate_text_same_language(self, asr_engine):
        """Test translation when source and target languages are the same."""
        text = "Hello, world!"
        
        result = await asr_engine.translate_text(
            text, LanguageCode.ENGLISH_IN, LanguageCode.ENGLISH_IN
        )
        
        assert result == text
    
    @pytest.mark.asyncio
    async def test_translate_text_different_languages(self, asr_engine):
        """Test translation between different languages."""
        text = "Hello, world!"
        
        result = await asr_engine.translate_text(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Should return some translation (placeholder implementation)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_supported_languages(self, asr_engine):
        """Test getting list of supported languages."""
        languages = asr_engine.get_supported_languages()
        
        assert isinstance(languages, list)
        assert LanguageCode.HINDI in languages
        assert LanguageCode.ENGLISH_IN in languages
        assert LanguageCode.TAMIL in languages
    
    def test_get_model_info(self, asr_engine):
        """Test getting model information."""
        info = asr_engine.get_model_info()
        
        assert isinstance(info, dict)
        assert "whisper_model_size" in info
        assert "device" in info
        assert "supported_languages" in info
        assert info["whisper_model_size"] == "base"
    
    @pytest.mark.asyncio
    async def test_health_check(self, asr_engine):
        """Test ASR engine health check."""
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            health = await asr_engine.health_check()
            
            assert isinstance(health, dict)
            assert "status" in health
            assert health["status"] in ["healthy", "unhealthy"]


class TestLanguageEngineService:
    """Test cases for the LanguageEngineService class."""
    
    @pytest.fixture
    def mock_asr_engine(self):
        """Mock ASR engine for testing."""
        mock_engine = Mock(spec=MultilingualASREngine)
        
        # Mock recognition result
        mock_result = RecognitionResult(
            transcribed_text="Test transcription",
            confidence=0.85,
            detected_language=LanguageCode.ENGLISH_IN,
            code_switching_points=[],
            alternative_transcriptions=[],
            processing_time=0.5
        )
        mock_engine.recognize_speech = AsyncMock(return_value=mock_result)
        mock_engine.detect_language = AsyncMock(return_value=LanguageCode.ENGLISH_IN)
        mock_engine.detect_code_switching = AsyncMock(return_value=[])
        mock_engine.translate_text = AsyncMock(return_value="Translated text")
        mock_engine.get_supported_languages.return_value = [
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL
        ]
        mock_engine.get_model_info.return_value = {"model": "test"}
        mock_engine.health_check = AsyncMock(return_value={"status": "healthy"})
        
        return mock_engine
    
    @pytest.fixture
    def sample_audio_buffer(self):
        """Create a sample audio buffer for testing."""
        return AudioBuffer(
            data=[0.1, -0.1, 0.2, -0.2] * 4000,
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
    
    @pytest.fixture
    def language_service(self, mock_asr_engine):
        """Create language service with mocked ASR engine."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine') as mock_create:
            mock_create.return_value = mock_asr_engine
            service = create_language_engine_service(
                asr_model_size="base",
                device="cpu",
                enable_caching=True
            )
            return service
    
    @pytest.mark.asyncio
    async def test_recognize_speech_basic(self, language_service, sample_audio_buffer):
        """Test basic speech recognition through service."""
        result = await language_service.recognize_speech(sample_audio_buffer)
        
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == "Test transcription"
        assert result.confidence == 0.85
        assert result.detected_language == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_recognize_speech_caching(self, language_service, sample_audio_buffer):
        """Test recognition result caching."""
        # First call should miss cache
        result1 = await language_service.recognize_speech(sample_audio_buffer)
        
        # Second call with same audio should hit cache
        result2 = await language_service.recognize_speech(sample_audio_buffer)
        
        assert result1.transcribed_text == result2.transcribed_text
        assert language_service.stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching(self, language_service):
        """Test code-switching detection through service."""
        mixed_text = "Hello नमस्ते world"
        
        result = await language_service.detect_code_switching(mixed_text)
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_detailed_code_switching_analysis(self, language_service):
        """Test detailed code-switching analysis through service."""
        mixed_text = "Hello नमस्ते world"
        
        result = await language_service.get_detailed_code_switching_analysis(mixed_text)
        
        assert isinstance(result, dict)
        assert 'segments' in result
        assert 'switch_points' in result
        assert 'dominant_language' in result
        assert 'switching_frequency' in result
        assert 'confidence' in result
        assert 'processing_time' in result
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, language_service):
        """Test language transition suggestions through service."""
        suggestions = await language_service.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
    
    @pytest.mark.asyncio
    async def test_translate_text(self, language_service):
        """Test text translation through service."""
        result = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result == "Translated text"
    
    @pytest.mark.asyncio
    async def test_translate_text_caching(self, language_service):
        """Test translation result caching."""
        # First translation should miss cache
        result1 = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Second identical translation should hit cache
        result2 = await language_service.translate_text(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result1 == result2
        assert language_service.stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_detect_language(self, language_service):
        """Test language detection through service."""
        result = await language_service.detect_language("Hello world")
        
        assert result == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_batch_recognize_speech(self, language_service, sample_audio_buffer):
        """Test batch speech recognition."""
        audio_buffers = [sample_audio_buffer, sample_audio_buffer, sample_audio_buffer]
        
        results = await language_service.batch_recognize_speech(audio_buffers)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RecognitionResult)
            assert result.transcribed_text == "Test transcription"
    
    @pytest.mark.asyncio
    async def test_batch_recognize_speech_empty_list(self, language_service):
        """Test batch recognition with empty list."""
        results = await language_service.batch_recognize_speech([])
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_get_language_confidence_scores(self, language_service, sample_audio_buffer):
        """Test getting language confidence scores."""
        scores = await language_service.get_language_confidence_scores(sample_audio_buffer)
        
        assert isinstance(scores, dict)
        assert LanguageCode.ENGLISH_IN in scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    @pytest.mark.asyncio
    async def test_recognize_with_language_hint(self, language_service, sample_audio_buffer):
        """Test recognition with language hint."""
        result = await language_service.recognize_with_language_hint(
            sample_audio_buffer, LanguageCode.HINDI
        )
        
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == "Test transcription"
    
    def test_get_supported_languages(self, language_service):
        """Test getting supported languages."""
        languages = language_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert LanguageCode.HINDI in languages
        assert LanguageCode.ENGLISH_IN in languages
    
    def test_get_service_stats(self, language_service):
        """Test getting service statistics."""
        stats = language_service.get_service_stats()
        
        assert isinstance(stats, dict)
        assert "total_recognitions" in stats
        assert "total_translations" in stats
        assert "cache_stats" in stats
        assert "asr_engine_info" in stats
    
    def test_clear_caches(self, language_service):
        """Test clearing service caches."""
        # Add something to cache first
        language_service.recognition_cache["test"] = "value"
        language_service.translation_cache["test"] = "value"
        
        language_service.clear_caches()
        
        assert len(language_service.recognition_cache) == 0
        assert len(language_service.translation_cache) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, language_service):
        """Test service health check."""
        health = await language_service.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_multilingual_asr_engine(self):
        """Test ASR engine factory function."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model'):
            engine = create_multilingual_asr_engine(
                model_size="tiny",
                device="cpu",
                enable_language_detection=False
            )
            
            assert isinstance(engine, MultilingualASREngine)
            assert engine.model_size == "tiny"
            assert engine.device == "cpu"
            assert engine.enable_language_detection == False
    
    def test_create_language_engine_service(self):
        """Test language service factory function."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine'):
            service = create_language_engine_service(
                asr_model_size="small",
                device="cuda",
                enable_caching=False
            )
            
            assert isinstance(service, LanguageEngineService)
            assert service.enable_caching == False


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.fixture
    def failing_asr_engine(self):
        """Mock ASR engine that raises exceptions."""
        mock_engine = Mock(spec=MultilingualASREngine)
        mock_engine.recognize_speech = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.detect_language = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.detect_code_switching = AsyncMock(side_effect=Exception("Test error"))
        mock_engine.translate_text = AsyncMock(side_effect=Exception("Test error"))
        return mock_engine
    
    @pytest.fixture
    def failing_language_service(self, failing_asr_engine):
        """Create language service with failing ASR engine."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine') as mock_create:
            mock_create.return_value = failing_asr_engine
            service = create_language_engine_service()
            return service
    
    @pytest.mark.asyncio
    async def test_recognize_speech_error_handling(self, failing_language_service):
        """Test error handling in speech recognition."""
        audio_buffer = AudioBuffer(
            data=[0.1, -0.1],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=0.1
        )
        
        result = await failing_language_service.recognize_speech(audio_buffer)
        
        # Should return empty result on error
        assert isinstance(result, RecognitionResult)
        assert result.transcribed_text == ""
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_language_error_handling(self, failing_language_service):
        """Test error handling in language detection."""
        result = await failing_language_service.detect_language("test")
        
        # Should return default language on error
        assert result == LanguageCode.ENGLISH_IN
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_error_handling(self, failing_language_service):
        """Test error handling in code-switching detection."""
        result = await failing_language_service.detect_code_switching("test")
        
        # Should return empty list on error
        assert result == []
    
    @pytest.mark.asyncio
    async def test_translate_text_error_handling(self, failing_language_service):
        """Test error handling in translation."""
        result = await failing_language_service.translate_text(
            "test", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Should return original text on error
        assert result == "test"


class TestCacheManagement:
    """Test cases for cache management functionality."""
    
    @pytest.fixture
    def language_service_small_cache(self):
        """Create language service with small cache for testing eviction."""
        with patch('bharatvoice.services.language_engine.service.create_multilingual_asr_engine'):
            service = create_language_engine_service(
                cache_size=2,  # Small cache for testing eviction
                enable_caching=True
            )
            return service
    
    def test_cache_eviction(self, language_service_small_cache):
        """Test LRU cache eviction."""
        service = language_service_small_cache
        
        # Fill cache beyond capacity
        service._cache_recognition_result("key1", Mock())
        service._cache_recognition_result("key2", Mock())
        service._cache_recognition_result("key3", Mock())  # Should evict key1
        
        assert len(service.recognition_cache) == 2
        assert "key1" not in service.recognition_cache
        assert "key2" in service.recognition_cache
        assert "key3" in service.recognition_cache
    
    def test_cache_key_generation(self, language_service_small_cache):
        """Test audio cache key generation."""
        service = language_service_small_cache
        
        audio1 = AudioBuffer(
            data=[0.1, 0.2, 0.3],
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
        
        audio2 = AudioBuffer(
            data=[0.1, 0.2, 0.4],  # Different data
            sample_rate=16000,
            channels=1,
            format=AudioFormat.WAV,
            duration=1.0
        )
        
        key1 = service._generate_audio_cache_key(audio1)
        key2 = service._generate_audio_cache_key(audio2)
        
        assert key1 != key2  # Different audio should have different keys
        
        # Same audio should have same key
        key1_repeat = service._generate_audio_cache_key(audio1)
        assert key1 == key1_repeat


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])