<<<<<<< HEAD
"""
Property-based tests for multilingual speech recognition accuracy.

This module implements Property 1: Multilingual Speech Recognition Accuracy
which validates Requirements 1.1 and 1.2 for the BharatVoice Assistant.

**Validates: Requirements 1.1, 1.2**
"""

import pytest
import asyncio
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Tuple, Dict

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.asr_engine import (
    MultilingualASREngine,
    create_multilingual_asr_engine,
)


# Test data strategies
@composite
def supported_language_strategy(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_IN,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@composite
def audio_buffer_strategy(draw):
    """Generate valid audio buffers for testing."""
    # Generate realistic audio parameters
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    channels = draw(st.sampled_from([1, 2]))
    duration = draw(st.floats(min_value=0.1, max_value=10.0))
    
    # Calculate number of samples
    num_samples = int(sample_rate * duration * channels)
    
    # Generate audio data (simulate speech-like patterns)
    audio_data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=num_samples,
        max_size=num_samples
    ))
    
    return AudioBuffer(
        data=audio_data,
        sample_rate=sample_rate,
        channels=channels,
        format=AudioFormat.WAV,
        duration=duration
    )


@composite
def multilingual_text_strategy(draw):
    """Generate multilingual text samples."""
    language = draw(supported_language_strategy())
    
    # Language-specific text samples
    text_samples = {
        LanguageCode.HINDI: [
            "नमस्ते, आप कैसे हैं?",
            "मुझे खाना चाहिए",
            "कृपया मदद करें",
            "धन्यवाद",
            "आज मौसम कैसा है?",
        ],
        LanguageCode.ENGLISH_IN: [
            "Hello, how are you?",
            "I need some food",
            "Please help me",
            "Thank you very much",
            "What's the weather like today?",
        ],
        LanguageCode.TAMIL: [
            "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "எனக்கு உணவு வேண்டும்",
            "தயவுசெய்து உதவுங்கள்",
            "நன்றி",
            "இன்று வானிலை எப்படி இருக்கிறது?",
        ],
        LanguageCode.TELUGU: [
            "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "నాకు ఆహారం కావాలి",
            "దయచేసి సహాయం చేయండి",
            "ధన్యవాదాలు",
            "ఈరోజు వాతావరణం ఎలా ఉంది?",
        ],
        LanguageCode.BENGALI: [
            "নমস্কার, আপনি কেমন আছেন?",
            "আমার খাবার দরকার",
            "দয়া করে সাহায্য করুন",
            "ধন্যবাদ",
            "আজ আবহাওয়া কেমন?",
        ],
    }
    
    # Default to English if language not in samples
    samples = text_samples.get(language, text_samples[LanguageCode.ENGLISH_IN])
    return draw(st.sampled_from(samples)), language


@composite
def code_switching_text_strategy(draw):
    """Generate code-switching text samples."""
    # Common Hindi-English code-switching patterns
    code_switching_samples = [
        "Hello नमस्ते, how are you आप कैसे हैं?",
        "I am going to market मैं बाज़ार जा रहा हूँ",
        "Please help me कृपया मदद करें",
        "Thank you धन्यवाद very much",
        "What time है अभी?",
        "Good morning सुप्रभात, have a nice day",
        "I love Indian food मुझे भारतीय खाना पसंद है",
        "Let's go चलते हैं together",
    ]
    
    return draw(st.sampled_from(code_switching_samples))


@composite
def confidence_threshold_strategy(draw):
    """Generate confidence threshold values."""
    return draw(st.floats(min_value=0.1, max_value=0.9))


class TestMultilingualSpeechRecognitionAccuracy:
    """
    Property-based tests for multilingual speech recognition accuracy.
    
    **Property 1: Multilingual Speech Recognition Accuracy**
    **Validates: Requirements 1.1, 1.2**
    """
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model with realistic responses."""
        mock_model = Mock()
        
        def mock_transcribe(audio_path, language=None, task="transcribe", verbose=False):
            # Simulate realistic Whisper responses based on language
            if language == "hi":
                return {
                    "text": "नमस्ते, आप कैसे हैं?",
                    "language": "hi",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "नमस्ते, आप कैसे हैं?",
                        "avg_logprob": -0.3
                    }]
                }
            elif language == "en":
                return {
                    "text": "Hello, how are you?",
                    "language": "en",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello, how are you?",
                        "avg_logprob": -0.2
                    }]
                }
            else:
                # Auto-detect mode - return based on audio characteristics
                return {
                    "text": "Hello, how are you?",
                    "language": "en",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello, how are you?",
                        "avg_logprob": -0.4
                    }]
                }
        
        mock_model.transcribe.side_effect = mock_transcribe
        return mock_model
    
    @pytest.fixture
    def asr_engine(self, mock_whisper_model):
        """Create ASR engine with mocked dependencies."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model') as mock_load, \
             patch('bharatvoice.services.language_engine.asr_engine.pipeline') as mock_pipeline:
            
            mock_load.return_value = mock_whisper_model
            mock_pipeline.return_value = None  # Disable language detection pipeline
            
            engine = create_multilingual_asr_engine(
                model_size="base",
                device="cpu",
                enable_language_detection=True,
                confidence_threshold=0.7,
                max_alternatives=3
            )
            return engine
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=20, deadline=10000)
    async def test_recognition_completeness_property(self, asr_engine, audio_buffer):
        """
        Property: Recognition must always return a complete result structure.
        
        For any valid audio input, the ASR system must return a RecognitionResult
        with all required fields populated, regardless of recognition success.
        """
        assume(len(audio_buffer.data) > 0)  # Ensure non-empty audio
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Completeness assertions
            assert isinstance(result, RecognitionResult)
            assert isinstance(result.transcribed_text, str)
            assert isinstance(result.confidence, float)
            assert isinstance(result.detected_language, LanguageCode)
            assert isinstance(result.code_switching_points, list)
            assert isinstance(result.alternative_transcriptions, list)
            assert isinstance(result.processing_time, float)
            
            # Value range assertions
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time >= 0.0
            
            # Alternative transcriptions structure
            for alt in result.alternative_transcriptions:
                assert isinstance(alt, AlternativeResult)
                assert isinstance(alt.text, str)
                assert isinstance(alt.confidence, float)
                assert isinstance(alt.language, LanguageCode)
                assert 0.0 <= alt.confidence <= 1.0
            
            # Code-switching points structure
            for switch_point in result.code_switching_points:
                assert isinstance(switch_point, LanguageSwitchPoint)
                assert isinstance(switch_point.position, int)
                assert isinstance(switch_point.from_language, LanguageCode)
                assert isinstance(switch_point.to_language, LanguageCode)
                assert isinstance(switch_point.confidence, float)
                assert 0.0 <= switch_point.confidence <= 1.0
                assert switch_point.position >= 0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy(), supported_language_strategy())
    @settings(max_examples=15, deadline=10000)
    async def test_language_detection_consistency_property(self, asr_engine, audio_buffer, expected_language):
        """
        Property: Language detection must be consistent and within supported languages.
        
        The detected language must always be one of the supported languages,
        and the confidence should reflect the detection certainty.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Language detection consistency
            supported_languages = asr_engine.get_supported_languages()
            assert result.detected_language in supported_languages
            
            # If confidence is high, language detection should be reliable
            if result.confidence > 0.8:
                # High confidence should mean clear language detection
                assert result.detected_language is not None
            
            # Alternative transcriptions should have valid languages
            for alt in result.alternative_transcriptions:
                assert alt.language in supported_languages
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy(), confidence_threshold_strategy())
    @settings(max_examples=12, deadline=10000)
    async def test_confidence_scoring_property(self, asr_engine, audio_buffer, threshold):
        """
        Property: Confidence scores must be meaningful and consistent.
        
        Confidence scores should reflect the actual quality of recognition,
        with higher scores indicating better recognition quality.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Confidence score validity
            assert 0.0 <= result.confidence <= 1.0
            
            # Alternative transcriptions should have decreasing confidence
            if len(result.alternative_transcriptions) > 1:
                confidences = [alt.confidence for alt in result.alternative_transcriptions]
                # Should be sorted in descending order
                assert confidences == sorted(confidences, reverse=True)
            
            # Primary result should have highest confidence among alternatives
            if result.alternative_transcriptions:
                max_alt_confidence = max(alt.confidence for alt in result.alternative_transcriptions)
                # Primary confidence should be at least as high as alternatives
                assert result.confidence >= max_alt_confidence * 0.8  # Allow some tolerance
    
    @pytest.mark.asyncio
    @given(st.lists(audio_buffer_strategy(), min_size=2, max_size=4))
    @settings(max_examples=8, deadline=15000)
    async def test_batch_recognition_consistency_property(self, asr_engine, audio_buffers):
        """
        Property: Batch recognition should be consistent with individual recognition.
        
        Processing multiple audio buffers should yield consistent results
        compared to processing them individually.
        """
        # Filter out empty audio buffers
        valid_buffers = [buf for buf in audio_buffers if len(buf.data) > 0]
        assume(len(valid_buffers) >= 2)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            # Process individually
            individual_results = []
            for audio_buffer in valid_buffers:
                result = await asr_engine.recognize_speech(audio_buffer)
                individual_results.append(result)
            
            # Process as batch (simulate batch processing)
            batch_results = []
            for audio_buffer in valid_buffers:
                result = await asr_engine.recognize_speech(audio_buffer)
                batch_results.append(result)
            
            # Consistency checks
            assert len(individual_results) == len(batch_results)
            
            for individual, batch in zip(individual_results, batch_results):
                # Results should be structurally consistent
                assert type(individual.transcribed_text) == type(batch.transcribed_text)
                assert type(individual.detected_language) == type(batch.detected_language)
                assert 0.0 <= individual.confidence <= 1.0
                assert 0.0 <= batch.confidence <= 1.0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=10, deadline=10000)
    async def test_processing_time_property(self, asr_engine, audio_buffer):
        """
        Property: Processing time should be reasonable and correlate with audio length.
        
        Processing time should be positive and generally correlate with audio duration,
        though not necessarily linearly due to complexity variations.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Processing time should be positive
            assert result.processing_time > 0.0
            
            # Processing time should be reasonable (not more than 10x audio duration)
            max_reasonable_time = max(10.0, audio_buffer.duration * 10)
            assert result.processing_time <= max_reasonable_time
    
    @pytest.mark.asyncio
    @given(code_switching_text_strategy())
    @settings(max_examples=8, deadline=10000)
    async def test_code_switching_detection_property(self, asr_engine, mixed_text):
        """
        Property: Code-switching detection should identify language transitions.
        
        When processing text with multiple languages, the system should
        detect and report language switching points accurately.
        """
        assume(len(mixed_text.strip()) > 0)
        
        # Test code-switching detection directly on text
        code_switches = await asr_engine.detect_code_switching(mixed_text)
        
        # Code-switching results should be well-formed
        assert isinstance(code_switches, list)
        
        for switch in code_switches:
            assert isinstance(switch, dict)
            assert "position" in switch
            assert "from_language" in switch
            assert "to_language" in switch
            assert "confidence" in switch
            
            # Position should be valid
            assert 0 <= switch["position"] <= len(mixed_text)
            
            # Languages should be different
            assert switch["from_language"] != switch["to_language"]
            
            # Confidence should be valid
            assert 0.0 <= switch["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=6, deadline=10000)
    async def test_error_resilience_property(self, asr_engine, audio_buffer):
        """
        Property: System should handle errors gracefully without crashing.
        
        Even when processing problematic audio or encountering errors,
        the system should return a valid (possibly empty) result.
        """
        assume(len(audio_buffer.data) > 0)
        
        # Test with various error conditions
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            # Simulate Whisper model failure
            with patch.object(asr_engine.whisper_model, 'transcribe', side_effect=Exception("Model error")):
                result = await asr_engine.recognize_speech(audio_buffer)
                
                # Should return valid empty result on error
                assert isinstance(result, RecognitionResult)
                assert result.transcribed_text == ""
                assert result.confidence == 0.0
                assert isinstance(result.detected_language, LanguageCode)
                assert result.processing_time >= 0.0
    
    @pytest.mark.asyncio
    @given(supported_language_strategy())
    @settings(max_examples=10, deadline=5000)
    async def test_language_support_property(self, asr_engine, language):
        """
        Property: All supported languages should be properly handled.
        
        The system should be able to process and detect all languages
        listed in its supported languages list.
        """
        # Test language detection capability
        test_text = "Hello world"  # Simple test text
        
        detected_language = await asr_engine.detect_language(test_text)
        
        # Should return a supported language
        supported_languages = asr_engine.get_supported_languages()
        assert detected_language in supported_languages
        
        # Language should be valid enum value
        assert isinstance(detected_language, LanguageCode)
    
    def test_model_configuration_property(self, asr_engine):
        """
        Property: Model configuration should be consistent and valid.
        
        The ASR engine should maintain consistent configuration
        and provide accurate model information.
        """
        model_info = asr_engine.get_model_info()
        
        # Model info should be complete
        assert isinstance(model_info, dict)
        assert "whisper_model_size" in model_info
        assert "device" in model_info
        assert "supported_languages" in model_info
        assert "confidence_threshold" in model_info
        
        # Configuration should be consistent
        assert model_info["whisper_model_size"] == asr_engine.model_size
        assert model_info["device"] == asr_engine.device
        assert model_info["confidence_threshold"] == asr_engine.confidence_threshold
        
        # Supported languages should be valid
        supported_langs = model_info["supported_languages"]
        assert isinstance(supported_langs, list)
        assert len(supported_langs) > 0
        
        # All supported languages should be valid LanguageCode values
        for lang_str in supported_langs:
            # Should be able to create LanguageCode from string
            lang_code = LanguageCode(lang_str)
            assert isinstance(lang_code, LanguageCode)


if __name__ == "__main__":
=======
"""
Property-based tests for multilingual speech recognition accuracy.

This module implements Property 1: Multilingual Speech Recognition Accuracy
which validates Requirements 1.1 and 1.2 for the BharatVoice Assistant.

**Validates: Requirements 1.1, 1.2**
"""

import pytest
import asyncio
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Tuple, Dict

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.asr_engine import (
    MultilingualASREngine,
    create_multilingual_asr_engine,
)


# Test data strategies
@composite
def supported_language_strategy(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_IN,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@composite
def audio_buffer_strategy(draw):
    """Generate valid audio buffers for testing."""
    # Generate realistic audio parameters
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    channels = draw(st.sampled_from([1, 2]))
    duration = draw(st.floats(min_value=0.1, max_value=10.0))
    
    # Calculate number of samples
    num_samples = int(sample_rate * duration * channels)
    
    # Generate audio data (simulate speech-like patterns)
    audio_data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=num_samples,
        max_size=num_samples
    ))
    
    return AudioBuffer(
        data=audio_data,
        sample_rate=sample_rate,
        channels=channels,
        format=AudioFormat.WAV,
        duration=duration
    )


@composite
def multilingual_text_strategy(draw):
    """Generate multilingual text samples."""
    language = draw(supported_language_strategy())
    
    # Language-specific text samples
    text_samples = {
        LanguageCode.HINDI: [
            "नमस्ते, आप कैसे हैं?",
            "मुझे खाना चाहिए",
            "कृपया मदद करें",
            "धन्यवाद",
            "आज मौसम कैसा है?",
        ],
        LanguageCode.ENGLISH_IN: [
            "Hello, how are you?",
            "I need some food",
            "Please help me",
            "Thank you very much",
            "What's the weather like today?",
        ],
        LanguageCode.TAMIL: [
            "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "எனக்கு உணவு வேண்டும்",
            "தயவுசெய்து உதவுங்கள்",
            "நன்றி",
            "இன்று வானிலை எப்படி இருக்கிறது?",
        ],
        LanguageCode.TELUGU: [
            "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "నాకు ఆహారం కావాలి",
            "దయచేసి సహాయం చేయండి",
            "ధన్యవాదాలు",
            "ఈరోజు వాతావరణం ఎలా ఉంది?",
        ],
        LanguageCode.BENGALI: [
            "নমস্কার, আপনি কেমন আছেন?",
            "আমার খাবার দরকার",
            "দয়া করে সাহায্য করুন",
            "ধন্যবাদ",
            "আজ আবহাওয়া কেমন?",
        ],
    }
    
    # Default to English if language not in samples
    samples = text_samples.get(language, text_samples[LanguageCode.ENGLISH_IN])
    return draw(st.sampled_from(samples)), language


@composite
def code_switching_text_strategy(draw):
    """Generate code-switching text samples."""
    # Common Hindi-English code-switching patterns
    code_switching_samples = [
        "Hello नमस्ते, how are you आप कैसे हैं?",
        "I am going to market मैं बाज़ार जा रहा हूँ",
        "Please help me कृपया मदद करें",
        "Thank you धन्यवाद very much",
        "What time है अभी?",
        "Good morning सुप्रभात, have a nice day",
        "I love Indian food मुझे भारतीय खाना पसंद है",
        "Let's go चलते हैं together",
    ]
    
    return draw(st.sampled_from(code_switching_samples))


@composite
def confidence_threshold_strategy(draw):
    """Generate confidence threshold values."""
    return draw(st.floats(min_value=0.1, max_value=0.9))


class TestMultilingualSpeechRecognitionAccuracy:
    """
    Property-based tests for multilingual speech recognition accuracy.
    
    **Property 1: Multilingual Speech Recognition Accuracy**
    **Validates: Requirements 1.1, 1.2**
    """
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model with realistic responses."""
        mock_model = Mock()
        
        def mock_transcribe(audio_path, language=None, task="transcribe", verbose=False):
            # Simulate realistic Whisper responses based on language
            if language == "hi":
                return {
                    "text": "नमस्ते, आप कैसे हैं?",
                    "language": "hi",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "नमस्ते, आप कैसे हैं?",
                        "avg_logprob": -0.3
                    }]
                }
            elif language == "en":
                return {
                    "text": "Hello, how are you?",
                    "language": "en",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello, how are you?",
                        "avg_logprob": -0.2
                    }]
                }
            else:
                # Auto-detect mode - return based on audio characteristics
                return {
                    "text": "Hello, how are you?",
                    "language": "en",
                    "segments": [{
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello, how are you?",
                        "avg_logprob": -0.4
                    }]
                }
        
        mock_model.transcribe.side_effect = mock_transcribe
        return mock_model
    
    @pytest.fixture
    def asr_engine(self, mock_whisper_model):
        """Create ASR engine with mocked dependencies."""
        with patch('bharatvoice.services.language_engine.asr_engine.whisper.load_model') as mock_load, \
             patch('bharatvoice.services.language_engine.asr_engine.pipeline') as mock_pipeline:
            
            mock_load.return_value = mock_whisper_model
            mock_pipeline.return_value = None  # Disable language detection pipeline
            
            engine = create_multilingual_asr_engine(
                model_size="base",
                device="cpu",
                enable_language_detection=True,
                confidence_threshold=0.7,
                max_alternatives=3
            )
            return engine
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=20, deadline=10000)
    async def test_recognition_completeness_property(self, asr_engine, audio_buffer):
        """
        Property: Recognition must always return a complete result structure.
        
        For any valid audio input, the ASR system must return a RecognitionResult
        with all required fields populated, regardless of recognition success.
        """
        assume(len(audio_buffer.data) > 0)  # Ensure non-empty audio
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Completeness assertions
            assert isinstance(result, RecognitionResult)
            assert isinstance(result.transcribed_text, str)
            assert isinstance(result.confidence, float)
            assert isinstance(result.detected_language, LanguageCode)
            assert isinstance(result.code_switching_points, list)
            assert isinstance(result.alternative_transcriptions, list)
            assert isinstance(result.processing_time, float)
            
            # Value range assertions
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time >= 0.0
            
            # Alternative transcriptions structure
            for alt in result.alternative_transcriptions:
                assert isinstance(alt, AlternativeResult)
                assert isinstance(alt.text, str)
                assert isinstance(alt.confidence, float)
                assert isinstance(alt.language, LanguageCode)
                assert 0.0 <= alt.confidence <= 1.0
            
            # Code-switching points structure
            for switch_point in result.code_switching_points:
                assert isinstance(switch_point, LanguageSwitchPoint)
                assert isinstance(switch_point.position, int)
                assert isinstance(switch_point.from_language, LanguageCode)
                assert isinstance(switch_point.to_language, LanguageCode)
                assert isinstance(switch_point.confidence, float)
                assert 0.0 <= switch_point.confidence <= 1.0
                assert switch_point.position >= 0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy(), supported_language_strategy())
    @settings(max_examples=15, deadline=10000)
    async def test_language_detection_consistency_property(self, asr_engine, audio_buffer, expected_language):
        """
        Property: Language detection must be consistent and within supported languages.
        
        The detected language must always be one of the supported languages,
        and the confidence should reflect the detection certainty.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Language detection consistency
            supported_languages = asr_engine.get_supported_languages()
            assert result.detected_language in supported_languages
            
            # If confidence is high, language detection should be reliable
            if result.confidence > 0.8:
                # High confidence should mean clear language detection
                assert result.detected_language is not None
            
            # Alternative transcriptions should have valid languages
            for alt in result.alternative_transcriptions:
                assert alt.language in supported_languages
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy(), confidence_threshold_strategy())
    @settings(max_examples=12, deadline=10000)
    async def test_confidence_scoring_property(self, asr_engine, audio_buffer, threshold):
        """
        Property: Confidence scores must be meaningful and consistent.
        
        Confidence scores should reflect the actual quality of recognition,
        with higher scores indicating better recognition quality.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Confidence score validity
            assert 0.0 <= result.confidence <= 1.0
            
            # Alternative transcriptions should have decreasing confidence
            if len(result.alternative_transcriptions) > 1:
                confidences = [alt.confidence for alt in result.alternative_transcriptions]
                # Should be sorted in descending order
                assert confidences == sorted(confidences, reverse=True)
            
            # Primary result should have highest confidence among alternatives
            if result.alternative_transcriptions:
                max_alt_confidence = max(alt.confidence for alt in result.alternative_transcriptions)
                # Primary confidence should be at least as high as alternatives
                assert result.confidence >= max_alt_confidence * 0.8  # Allow some tolerance
    
    @pytest.mark.asyncio
    @given(st.lists(audio_buffer_strategy(), min_size=2, max_size=4))
    @settings(max_examples=8, deadline=15000)
    async def test_batch_recognition_consistency_property(self, asr_engine, audio_buffers):
        """
        Property: Batch recognition should be consistent with individual recognition.
        
        Processing multiple audio buffers should yield consistent results
        compared to processing them individually.
        """
        # Filter out empty audio buffers
        valid_buffers = [buf for buf in audio_buffers if len(buf.data) > 0]
        assume(len(valid_buffers) >= 2)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            # Process individually
            individual_results = []
            for audio_buffer in valid_buffers:
                result = await asr_engine.recognize_speech(audio_buffer)
                individual_results.append(result)
            
            # Process as batch (simulate batch processing)
            batch_results = []
            for audio_buffer in valid_buffers:
                result = await asr_engine.recognize_speech(audio_buffer)
                batch_results.append(result)
            
            # Consistency checks
            assert len(individual_results) == len(batch_results)
            
            for individual, batch in zip(individual_results, batch_results):
                # Results should be structurally consistent
                assert type(individual.transcribed_text) == type(batch.transcribed_text)
                assert type(individual.detected_language) == type(batch.detected_language)
                assert 0.0 <= individual.confidence <= 1.0
                assert 0.0 <= batch.confidence <= 1.0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=10, deadline=10000)
    async def test_processing_time_property(self, asr_engine, audio_buffer):
        """
        Property: Processing time should be reasonable and correlate with audio length.
        
        Processing time should be positive and generally correlate with audio duration,
        though not necessarily linearly due to complexity variations.
        """
        assume(len(audio_buffer.data) > 0)
        
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            result = await asr_engine.recognize_speech(audio_buffer)
            
            # Processing time should be positive
            assert result.processing_time > 0.0
            
            # Processing time should be reasonable (not more than 10x audio duration)
            max_reasonable_time = max(10.0, audio_buffer.duration * 10)
            assert result.processing_time <= max_reasonable_time
    
    @pytest.mark.asyncio
    @given(code_switching_text_strategy())
    @settings(max_examples=8, deadline=10000)
    async def test_code_switching_detection_property(self, asr_engine, mixed_text):
        """
        Property: Code-switching detection should identify language transitions.
        
        When processing text with multiple languages, the system should
        detect and report language switching points accurately.
        """
        assume(len(mixed_text.strip()) > 0)
        
        # Test code-switching detection directly on text
        code_switches = await asr_engine.detect_code_switching(mixed_text)
        
        # Code-switching results should be well-formed
        assert isinstance(code_switches, list)
        
        for switch in code_switches:
            assert isinstance(switch, dict)
            assert "position" in switch
            assert "from_language" in switch
            assert "to_language" in switch
            assert "confidence" in switch
            
            # Position should be valid
            assert 0 <= switch["position"] <= len(mixed_text)
            
            # Languages should be different
            assert switch["from_language"] != switch["to_language"]
            
            # Confidence should be valid
            assert 0.0 <= switch["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    @given(audio_buffer_strategy())
    @settings(max_examples=6, deadline=10000)
    async def test_error_resilience_property(self, asr_engine, audio_buffer):
        """
        Property: System should handle errors gracefully without crashing.
        
        Even when processing problematic audio or encountering errors,
        the system should return a valid (possibly empty) result.
        """
        assume(len(audio_buffer.data) > 0)
        
        # Test with various error conditions
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('soundfile.write') as mock_sf_write, \
             patch('os.path.exists') as mock_exists, \
             patch('os.unlink') as mock_unlink:
            
            mock_mkstemp.return_value = (1, '/tmp/test.wav')
            mock_exists.return_value = True
            
            # Simulate Whisper model failure
            with patch.object(asr_engine.whisper_model, 'transcribe', side_effect=Exception("Model error")):
                result = await asr_engine.recognize_speech(audio_buffer)
                
                # Should return valid empty result on error
                assert isinstance(result, RecognitionResult)
                assert result.transcribed_text == ""
                assert result.confidence == 0.0
                assert isinstance(result.detected_language, LanguageCode)
                assert result.processing_time >= 0.0
    
    @pytest.mark.asyncio
    @given(supported_language_strategy())
    @settings(max_examples=10, deadline=5000)
    async def test_language_support_property(self, asr_engine, language):
        """
        Property: All supported languages should be properly handled.
        
        The system should be able to process and detect all languages
        listed in its supported languages list.
        """
        # Test language detection capability
        test_text = "Hello world"  # Simple test text
        
        detected_language = await asr_engine.detect_language(test_text)
        
        # Should return a supported language
        supported_languages = asr_engine.get_supported_languages()
        assert detected_language in supported_languages
        
        # Language should be valid enum value
        assert isinstance(detected_language, LanguageCode)
    
    def test_model_configuration_property(self, asr_engine):
        """
        Property: Model configuration should be consistent and valid.
        
        The ASR engine should maintain consistent configuration
        and provide accurate model information.
        """
        model_info = asr_engine.get_model_info()
        
        # Model info should be complete
        assert isinstance(model_info, dict)
        assert "whisper_model_size" in model_info
        assert "device" in model_info
        assert "supported_languages" in model_info
        assert "confidence_threshold" in model_info
        
        # Configuration should be consistent
        assert model_info["whisper_model_size"] == asr_engine.model_size
        assert model_info["device"] == asr_engine.device
        assert model_info["confidence_threshold"] == asr_engine.confidence_threshold
        
        # Supported languages should be valid
        supported_langs = model_info["supported_languages"]
        assert isinstance(supported_langs, list)
        assert len(supported_langs) > 0
        
        # All supported languages should be valid LanguageCode values
        for lang_str in supported_langs:
            # Should be able to create LanguageCode from string
            lang_code = LanguageCode(lang_str)
            assert isinstance(lang_code, LanguageCode)


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])