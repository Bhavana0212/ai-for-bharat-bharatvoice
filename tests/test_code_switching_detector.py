<<<<<<< HEAD
"""
Unit tests for the Enhanced Code-Switching Detection module.

This module tests the advanced code-switching detection functionality,
including language boundary detection, seamless transition handling,
and mixed-language processing capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from bharatvoice.core.models import LanguageCode, LanguageSwitchPoint
from bharatvoice.services.language_engine.code_switching_detector import (
    EnhancedCodeSwitchingDetector,
    create_enhanced_code_switching_detector,
    LanguageSegment,
    CodeSwitchingResult,
)


class TestEnhancedCodeSwitchingDetector:
    """Test cases for the EnhancedCodeSwitchingDetector class."""
    
    @pytest.fixture
    def mock_transformer_pipeline(self):
        """Mock transformer pipeline for testing."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "hi", "score": 0.9},
            {"label": "en", "score": 0.1}
        ]
        return mock_pipeline
    
    @pytest.fixture
    def detector(self, mock_transformer_pipeline):
        """Create detector with mocked models."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline_func:
            mock_pipeline_func.return_value = mock_transformer_pipeline
            detector = create_enhanced_code_switching_detector(
                device="cpu",
                confidence_threshold=0.7,
                min_segment_length=3,
                enable_word_level_detection=True
            )
            return detector
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_empty_text(self, detector):
        """Test code-switching detection with empty text."""
        result = await detector.detect_code_switching("")
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) == 0
        assert len(result.switch_points) == 0
        assert result.dominant_language == LanguageCode.ENGLISH_IN
        assert result.switching_frequency == 0.0
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_single_language(self, detector):
        """Test detection with single language text."""
        text = "Hello, how are you today?"
        
        result = await detector.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 1
        assert len(result.switch_points) == 0  # No switches in single language
        assert result.switching_frequency == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_mixed_language(self, detector):
        """Test detection with mixed Hindi-English text."""
        text = "Hello नमस्ते, how are you आप कैसे हैं?"
        
        with patch.object(detector, '_detect_language_ensemble') as mock_detect:
            # Mock alternating language detection
            mock_detect.side_effect = [
                (LanguageCode.ENGLISH_IN, 0.8),
                (LanguageCode.HINDI, 0.9),
                (LanguageCode.ENGLISH_IN, 0.8),
                (LanguageCode.HINDI, 0.9)
            ]
            
            result = await detector.detect_code_switching(text)
            
            assert isinstance(result, CodeSwitchingResult)
            assert len(result.segments) >= 2
            assert len(result.switch_points) >= 1  # Should detect switches
            assert result.switching_frequency > 0.0
    
    @pytest.mark.asyncio
    async def test_segment_text_advanced(self, detector):
        """Test advanced text segmentation."""
        text = "Hello world. नमस्ते दुनिया, how are you? आप कैसे हैं।"
        
        segments = await detector._segment_text_advanced(text)
        
        assert isinstance(segments, list)
        assert len(segments) >= 2  # Should split on sentence boundaries
        
        for seg_text, start_pos, end_pos in segments:
            assert isinstance(seg_text, str)
            assert isinstance(start_pos, int)
            assert isinstance(end_pos, int)
            assert start_pos < end_pos
            assert len(seg_text.strip()) >= detector.min_segment_length
    
    @pytest.mark.asyncio
    async def test_segment_by_phrases(self, detector):
        """Test phrase-level segmentation."""
        sentence = "Hello world, नमस्ते दुनिया; how are you"
        base_offset = 0
        
        segments = await detector._segment_by_phrases(sentence, base_offset)
        
        assert isinstance(segments, list)
        assert len(segments) >= 2  # Should split on punctuation
        
        for seg_text, start_pos, end_pos in segments:
            assert start_pos >= base_offset
            assert end_pos > start_pos
    
    @pytest.mark.asyncio
    async def test_detect_intra_phrase_switches(self, detector):
        """Test detection of switches within phrases."""
        phrase = "Hello नमस्ते world आप कैसे हैं how are you"
        base_offset = 0
        
        segments = await detector._detect_intra_phrase_switches(phrase, base_offset)
        
        # Should detect pattern-based switches or return empty list
        assert isinstance(segments, list)
        
        if segments:  # If switches detected
            for seg_text, start_pos, end_pos in segments:
                assert start_pos >= base_offset
                assert end_pos > start_pos
    
    @pytest.mark.asyncio
    async def test_detect_language_ensemble(self, detector):
        """Test ensemble language detection."""
        text = "नमस्ते दुनिया"
        context_language = LanguageCode.HINDI
        
        with patch.object(detector, '_detect_with_transformer_model') as mock_transformer, \
             patch.object(detector, '_detect_with_langdetect') as mock_langdetect, \
             patch.object(detector, '_detect_with_patterns') as mock_patterns:
            
            mock_transformer.return_value = (LanguageCode.HINDI, 0.9)
            mock_langdetect.return_value = (LanguageCode.HINDI, 0.8)
            mock_patterns.return_value = (LanguageCode.HINDI, 0.85)
            
            language, confidence = await detector._detect_language_ensemble(
                text, context_language
            )
            
            assert language == LanguageCode.HINDI
            assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_with_transformer_model(self, detector):
        """Test transformer-based language detection."""
        text = "Hello world"
        mock_model = Mock()
        mock_model.return_value = [
            {"label": "en", "score": 0.95},
            {"label": "hi", "score": 0.05}
        ]
        
        result = await detector._detect_with_transformer_model(
            text, mock_model, 'xlm_roberta'
        )
        
        assert result is not None
        language, confidence = result
        assert language == LanguageCode.ENGLISH_IN
        assert confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_detect_with_langdetect(self, detector):
        """Test langdetect-based language detection."""
        text = "Hello world"
        
        with patch('bharatvoice.services.language_engine.code_switching_detector.detect_langs') as mock_detect:
            mock_lang = Mock()
            mock_lang.lang = "en"
            mock_lang.prob = 0.9
            mock_detect.return_value = [mock_lang]
            
            result = await detector._detect_with_langdetect(text)
            
            assert result is not None
            language, confidence = result
            assert language == LanguageCode.ENGLISH_IN
            assert confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_detect_with_patterns(self, detector):
        """Test pattern-based language detection."""
        # Test Hindi text with Devanagari script
        hindi_text = "नमस्ते दुनिया"
        
        result = await detector._detect_with_patterns(hindi_text)
        
        if result:  # Pattern detection might not always work
            language, confidence = result
            assert language == LanguageCode.HINDI
            assert 0.0 <= confidence <= 1.0
        
        # Test English text
        english_text = "Hello world"
        
        result = await detector._detect_with_patterns(english_text)
        
        if result:
            language, confidence = result
            assert language == LanguageCode.ENGLISH_IN
            assert 0.0 <= confidence <= 1.0
    
    def test_combine_detection_results(self, detector):
        """Test combining multiple detection results."""
        detections = [
            (LanguageCode.HINDI, 0.8),
            (LanguageCode.HINDI, 0.9),
            (LanguageCode.ENGLISH_IN, 0.6)
        ]
        context_language = LanguageCode.HINDI
        
        language, confidence = detector._combine_detection_results(
            detections, context_language
        )
        
        assert language == LanguageCode.HINDI  # Should win by votes
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_word_boundaries(self, detector):
        """Test word boundary extraction."""
        text = "Hello world नमस्ते"
        base_offset = 10
        
        boundaries = detector._extract_word_boundaries(text, base_offset)
        
        assert isinstance(boundaries, list)
        assert len(boundaries) >= 2  # At least "Hello" and "world"
        
        for start, end in boundaries:
            assert start >= base_offset
            assert end > start
    
    @pytest.mark.asyncio
    async def test_refine_language_boundaries(self, detector):
        """Test language boundary refinement."""
        # Create test segments
        segments = [
            LanguageSegment(
                text="Hello",
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[(0, 5)]
            ),
            LanguageSegment(
                text=" world",
                language=LanguageCode.ENGLISH_IN,
                start_pos=5,
                end_pos=11,
                confidence=0.8,
                word_boundaries=[(6, 11)]
            )
        ]
        original_text = "Hello world"
        
        refined = await detector._refine_language_boundaries(segments, original_text)
        
        assert isinstance(refined, list)
        assert len(refined) == 1  # Should merge adjacent same-language segments
        assert refined[0].text == "Hello world"
        assert refined[0].language == LanguageCode.ENGLISH_IN
    
    def test_generate_switch_points(self, detector):
        """Test generation of language switch points."""
        segments = [
            LanguageSegment(
                text="Hello",
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",
                language=LanguageCode.HINDI,
                start_pos=6,
                end_pos=12,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        switch_points = detector._generate_switch_points(segments)
        
        assert isinstance(switch_points, list)
        assert len(switch_points) == 1
        
        switch = switch_points[0]
        assert switch.position == 6
        assert switch.from_language == LanguageCode.ENGLISH_IN
        assert switch.to_language == LanguageCode.HINDI
        assert 0.0 <= switch.confidence <= 1.0
    
    def test_calculate_dominant_language(self, detector):
        """Test dominant language calculation."""
        segments = [
            LanguageSegment(
                text="Hello world",  # 11 characters
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=11,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",  # 6 characters
                language=LanguageCode.HINDI,
                start_pos=12,
                end_pos=18,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        dominant = detector._calculate_dominant_language(segments)
        
        assert dominant == LanguageCode.ENGLISH_IN  # More characters
    
    def test_calculate_switching_frequency(self, detector):
        """Test switching frequency calculation."""
        switch_points = [
            LanguageSwitchPoint(
                position=10,
                from_language=LanguageCode.ENGLISH_IN,
                to_language=LanguageCode.HINDI,
                confidence=0.8
            ),
            LanguageSwitchPoint(
                position=20,
                from_language=LanguageCode.HINDI,
                to_language=LanguageCode.ENGLISH_IN,
                confidence=0.9
            )
        ]
        text_length = 50
        
        frequency = detector._calculate_switching_frequency(switch_points, text_length)
        
        assert frequency == 4.0  # 2 switches per 50 chars = 4 per 100 chars
    
    def test_calculate_overall_confidence(self, detector):
        """Test overall confidence calculation."""
        segments = [
            LanguageSegment(
                text="Hello",  # 5 characters
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",  # 6 characters
                language=LanguageCode.HINDI,
                start_pos=6,
                end_pos=12,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        confidence = detector._calculate_overall_confidence(segments)
        
        # Weighted average: (0.9 * 5 + 0.8 * 6) / 11 ≈ 0.845
        assert 0.8 <= confidence <= 0.9
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, detector):
        """Test language transition suggestions."""
        suggestions = await detector.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
        
        assert isinstance(suggestions['connectors'], list)
        assert isinstance(suggestions['fillers'], list)
        assert isinstance(suggestions['markers'], list)
    
    def test_get_detection_stats(self, detector):
        """Test getting detector statistics."""
        stats = detector.get_detection_stats()
        
        assert isinstance(stats, dict)
        assert 'device' in stats
        assert 'confidence_threshold' in stats
        assert 'min_segment_length' in stats
        assert 'word_level_detection_enabled' in stats
        assert 'primary_detector_loaded' in stats
        assert 'secondary_detector_loaded' in stats
        assert 'tokenizer_loaded' in stats
        assert 'supported_language_pairs' in stats
        
        assert stats['device'] == "cpu"
        assert stats['confidence_threshold'] == 0.7
        assert stats['min_segment_length'] == 3
        assert stats['word_level_detection_enabled'] == True


class TestFactoryFunction:
    """Test cases for the factory function."""
    
    def test_create_enhanced_code_switching_detector(self):
        """Test factory function for creating detector."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline'):
            detector = create_enhanced_code_switching_detector(
                device="cuda",
                confidence_threshold=0.8,
                min_segment_length=5,
                enable_word_level_detection=False
            )
            
            assert isinstance(detector, EnhancedCodeSwitchingDetector)
            assert detector.device == "cuda"
            assert detector.confidence_threshold == 0.8
            assert detector.min_segment_length == 5
            assert detector.enable_word_level_detection == False


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.fixture
    def failing_detector(self):
        """Create detector with failing models."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Model loading failed")
            detector = create_enhanced_code_switching_detector()
            return detector
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_with_model_failure(self, failing_detector):
        """Test code-switching detection when models fail."""
        text = "Hello नमस्ते world"
        
        result = await failing_detector.detect_code_switching(text)
        
        # Should return fallback result
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) == 1  # Fallback single segment
        assert result.confidence == 0.5  # Fallback confidence
    
    @pytest.mark.asyncio
    async def test_detect_language_ensemble_all_fail(self, failing_detector):
        """Test ensemble detection when all methods fail."""
        text = "Hello world"
        
        with patch.object(failing_detector, '_detect_with_transformer_model', return_value=None), \
             patch.object(failing_detector, '_detect_with_langdetect', return_value=None), \
             patch.object(failing_detector, '_detect_with_patterns', return_value=None):
            
            language, confidence = await failing_detector._detect_language_ensemble(
                text, LanguageCode.HINDI
            )
            
            # Should return context language as fallback
            assert language == LanguageCode.HINDI
            assert confidence == 0.5


class TestIntegrationScenarios:
    """Integration test scenarios with realistic data."""
    
    @pytest.fixture
    def detector_with_mocks(self):
        """Create detector with comprehensive mocks."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline:
            mock_model = Mock()
            mock_pipeline.return_value = mock_model
            
            detector = create_enhanced_code_switching_detector()
            
            # Mock language detection to return realistic results
            async def mock_ensemble_detect(text, context):
                if any(char in text for char in "नमस्ते"):
                    return LanguageCode.HINDI, 0.9
                else:
                    return LanguageCode.ENGLISH_IN, 0.8
            
            detector._detect_language_ensemble = mock_ensemble_detect
            
            return detector
    
    @pytest.mark.asyncio
    async def test_realistic_hindi_english_mixing(self, detector_with_mocks):
        """Test realistic Hindi-English code-switching scenario."""
        text = "Hello, मैं आज office जा रहा हूं। How are you?"
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 3  # Should detect multiple segments
        assert len(result.switch_points) >= 2  # Should detect switches
        assert result.switching_frequency > 0
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_complex_multilingual_text(self, detector_with_mocks):
        """Test complex multilingual text processing."""
        text = "Good morning! नमस्ते sir, आज का weather कैसा है? It's quite nice today."
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 4
        assert result.processing_time >= 0.0
        
        # Check that segments cover the entire text
        total_coverage = sum(
            seg.end_pos - seg.start_pos for seg in result.segments
        )
        assert total_coverage > 0
    
    @pytest.mark.asyncio
    async def test_single_word_switches(self, detector_with_mocks):
        """Test detection of single word language switches."""
        text = "I am going to the market आज"
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        # Should handle single word switches appropriately


if __name__ == "__main__":
=======
"""
Unit tests for the Enhanced Code-Switching Detection module.

This module tests the advanced code-switching detection functionality,
including language boundary detection, seamless transition handling,
and mixed-language processing capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from bharatvoice.core.models import LanguageCode, LanguageSwitchPoint
from bharatvoice.services.language_engine.code_switching_detector import (
    EnhancedCodeSwitchingDetector,
    create_enhanced_code_switching_detector,
    LanguageSegment,
    CodeSwitchingResult,
)


class TestEnhancedCodeSwitchingDetector:
    """Test cases for the EnhancedCodeSwitchingDetector class."""
    
    @pytest.fixture
    def mock_transformer_pipeline(self):
        """Mock transformer pipeline for testing."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            {"label": "hi", "score": 0.9},
            {"label": "en", "score": 0.1}
        ]
        return mock_pipeline
    
    @pytest.fixture
    def detector(self, mock_transformer_pipeline):
        """Create detector with mocked models."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline_func:
            mock_pipeline_func.return_value = mock_transformer_pipeline
            detector = create_enhanced_code_switching_detector(
                device="cpu",
                confidence_threshold=0.7,
                min_segment_length=3,
                enable_word_level_detection=True
            )
            return detector
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_empty_text(self, detector):
        """Test code-switching detection with empty text."""
        result = await detector.detect_code_switching("")
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) == 0
        assert len(result.switch_points) == 0
        assert result.dominant_language == LanguageCode.ENGLISH_IN
        assert result.switching_frequency == 0.0
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_single_language(self, detector):
        """Test detection with single language text."""
        text = "Hello, how are you today?"
        
        result = await detector.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 1
        assert len(result.switch_points) == 0  # No switches in single language
        assert result.switching_frequency == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_mixed_language(self, detector):
        """Test detection with mixed Hindi-English text."""
        text = "Hello नमस्ते, how are you आप कैसे हैं?"
        
        with patch.object(detector, '_detect_language_ensemble') as mock_detect:
            # Mock alternating language detection
            mock_detect.side_effect = [
                (LanguageCode.ENGLISH_IN, 0.8),
                (LanguageCode.HINDI, 0.9),
                (LanguageCode.ENGLISH_IN, 0.8),
                (LanguageCode.HINDI, 0.9)
            ]
            
            result = await detector.detect_code_switching(text)
            
            assert isinstance(result, CodeSwitchingResult)
            assert len(result.segments) >= 2
            assert len(result.switch_points) >= 1  # Should detect switches
            assert result.switching_frequency > 0.0
    
    @pytest.mark.asyncio
    async def test_segment_text_advanced(self, detector):
        """Test advanced text segmentation."""
        text = "Hello world. नमस्ते दुनिया, how are you? आप कैसे हैं।"
        
        segments = await detector._segment_text_advanced(text)
        
        assert isinstance(segments, list)
        assert len(segments) >= 2  # Should split on sentence boundaries
        
        for seg_text, start_pos, end_pos in segments:
            assert isinstance(seg_text, str)
            assert isinstance(start_pos, int)
            assert isinstance(end_pos, int)
            assert start_pos < end_pos
            assert len(seg_text.strip()) >= detector.min_segment_length
    
    @pytest.mark.asyncio
    async def test_segment_by_phrases(self, detector):
        """Test phrase-level segmentation."""
        sentence = "Hello world, नमस्ते दुनिया; how are you"
        base_offset = 0
        
        segments = await detector._segment_by_phrases(sentence, base_offset)
        
        assert isinstance(segments, list)
        assert len(segments) >= 2  # Should split on punctuation
        
        for seg_text, start_pos, end_pos in segments:
            assert start_pos >= base_offset
            assert end_pos > start_pos
    
    @pytest.mark.asyncio
    async def test_detect_intra_phrase_switches(self, detector):
        """Test detection of switches within phrases."""
        phrase = "Hello नमस्ते world आप कैसे हैं how are you"
        base_offset = 0
        
        segments = await detector._detect_intra_phrase_switches(phrase, base_offset)
        
        # Should detect pattern-based switches or return empty list
        assert isinstance(segments, list)
        
        if segments:  # If switches detected
            for seg_text, start_pos, end_pos in segments:
                assert start_pos >= base_offset
                assert end_pos > start_pos
    
    @pytest.mark.asyncio
    async def test_detect_language_ensemble(self, detector):
        """Test ensemble language detection."""
        text = "नमस्ते दुनिया"
        context_language = LanguageCode.HINDI
        
        with patch.object(detector, '_detect_with_transformer_model') as mock_transformer, \
             patch.object(detector, '_detect_with_langdetect') as mock_langdetect, \
             patch.object(detector, '_detect_with_patterns') as mock_patterns:
            
            mock_transformer.return_value = (LanguageCode.HINDI, 0.9)
            mock_langdetect.return_value = (LanguageCode.HINDI, 0.8)
            mock_patterns.return_value = (LanguageCode.HINDI, 0.85)
            
            language, confidence = await detector._detect_language_ensemble(
                text, context_language
            )
            
            assert language == LanguageCode.HINDI
            assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_detect_with_transformer_model(self, detector):
        """Test transformer-based language detection."""
        text = "Hello world"
        mock_model = Mock()
        mock_model.return_value = [
            {"label": "en", "score": 0.95},
            {"label": "hi", "score": 0.05}
        ]
        
        result = await detector._detect_with_transformer_model(
            text, mock_model, 'xlm_roberta'
        )
        
        assert result is not None
        language, confidence = result
        assert language == LanguageCode.ENGLISH_IN
        assert confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_detect_with_langdetect(self, detector):
        """Test langdetect-based language detection."""
        text = "Hello world"
        
        with patch('bharatvoice.services.language_engine.code_switching_detector.detect_langs') as mock_detect:
            mock_lang = Mock()
            mock_lang.lang = "en"
            mock_lang.prob = 0.9
            mock_detect.return_value = [mock_lang]
            
            result = await detector._detect_with_langdetect(text)
            
            assert result is not None
            language, confidence = result
            assert language == LanguageCode.ENGLISH_IN
            assert confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_detect_with_patterns(self, detector):
        """Test pattern-based language detection."""
        # Test Hindi text with Devanagari script
        hindi_text = "नमस्ते दुनिया"
        
        result = await detector._detect_with_patterns(hindi_text)
        
        if result:  # Pattern detection might not always work
            language, confidence = result
            assert language == LanguageCode.HINDI
            assert 0.0 <= confidence <= 1.0
        
        # Test English text
        english_text = "Hello world"
        
        result = await detector._detect_with_patterns(english_text)
        
        if result:
            language, confidence = result
            assert language == LanguageCode.ENGLISH_IN
            assert 0.0 <= confidence <= 1.0
    
    def test_combine_detection_results(self, detector):
        """Test combining multiple detection results."""
        detections = [
            (LanguageCode.HINDI, 0.8),
            (LanguageCode.HINDI, 0.9),
            (LanguageCode.ENGLISH_IN, 0.6)
        ]
        context_language = LanguageCode.HINDI
        
        language, confidence = detector._combine_detection_results(
            detections, context_language
        )
        
        assert language == LanguageCode.HINDI  # Should win by votes
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_word_boundaries(self, detector):
        """Test word boundary extraction."""
        text = "Hello world नमस्ते"
        base_offset = 10
        
        boundaries = detector._extract_word_boundaries(text, base_offset)
        
        assert isinstance(boundaries, list)
        assert len(boundaries) >= 2  # At least "Hello" and "world"
        
        for start, end in boundaries:
            assert start >= base_offset
            assert end > start
    
    @pytest.mark.asyncio
    async def test_refine_language_boundaries(self, detector):
        """Test language boundary refinement."""
        # Create test segments
        segments = [
            LanguageSegment(
                text="Hello",
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[(0, 5)]
            ),
            LanguageSegment(
                text=" world",
                language=LanguageCode.ENGLISH_IN,
                start_pos=5,
                end_pos=11,
                confidence=0.8,
                word_boundaries=[(6, 11)]
            )
        ]
        original_text = "Hello world"
        
        refined = await detector._refine_language_boundaries(segments, original_text)
        
        assert isinstance(refined, list)
        assert len(refined) == 1  # Should merge adjacent same-language segments
        assert refined[0].text == "Hello world"
        assert refined[0].language == LanguageCode.ENGLISH_IN
    
    def test_generate_switch_points(self, detector):
        """Test generation of language switch points."""
        segments = [
            LanguageSegment(
                text="Hello",
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",
                language=LanguageCode.HINDI,
                start_pos=6,
                end_pos=12,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        switch_points = detector._generate_switch_points(segments)
        
        assert isinstance(switch_points, list)
        assert len(switch_points) == 1
        
        switch = switch_points[0]
        assert switch.position == 6
        assert switch.from_language == LanguageCode.ENGLISH_IN
        assert switch.to_language == LanguageCode.HINDI
        assert 0.0 <= switch.confidence <= 1.0
    
    def test_calculate_dominant_language(self, detector):
        """Test dominant language calculation."""
        segments = [
            LanguageSegment(
                text="Hello world",  # 11 characters
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=11,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",  # 6 characters
                language=LanguageCode.HINDI,
                start_pos=12,
                end_pos=18,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        dominant = detector._calculate_dominant_language(segments)
        
        assert dominant == LanguageCode.ENGLISH_IN  # More characters
    
    def test_calculate_switching_frequency(self, detector):
        """Test switching frequency calculation."""
        switch_points = [
            LanguageSwitchPoint(
                position=10,
                from_language=LanguageCode.ENGLISH_IN,
                to_language=LanguageCode.HINDI,
                confidence=0.8
            ),
            LanguageSwitchPoint(
                position=20,
                from_language=LanguageCode.HINDI,
                to_language=LanguageCode.ENGLISH_IN,
                confidence=0.9
            )
        ]
        text_length = 50
        
        frequency = detector._calculate_switching_frequency(switch_points, text_length)
        
        assert frequency == 4.0  # 2 switches per 50 chars = 4 per 100 chars
    
    def test_calculate_overall_confidence(self, detector):
        """Test overall confidence calculation."""
        segments = [
            LanguageSegment(
                text="Hello",  # 5 characters
                language=LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=5,
                confidence=0.9,
                word_boundaries=[]
            ),
            LanguageSegment(
                text="नमस्ते",  # 6 characters
                language=LanguageCode.HINDI,
                start_pos=6,
                end_pos=12,
                confidence=0.8,
                word_boundaries=[]
            )
        ]
        
        confidence = detector._calculate_overall_confidence(segments)
        
        # Weighted average: (0.9 * 5 + 0.8 * 6) / 11 ≈ 0.845
        assert 0.8 <= confidence <= 0.9
    
    @pytest.mark.asyncio
    async def test_get_language_transition_suggestions(self, detector):
        """Test language transition suggestions."""
        suggestions = await detector.get_language_transition_suggestions(
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(suggestions, dict)
        assert 'connectors' in suggestions
        assert 'fillers' in suggestions
        assert 'markers' in suggestions
        
        assert isinstance(suggestions['connectors'], list)
        assert isinstance(suggestions['fillers'], list)
        assert isinstance(suggestions['markers'], list)
    
    def test_get_detection_stats(self, detector):
        """Test getting detector statistics."""
        stats = detector.get_detection_stats()
        
        assert isinstance(stats, dict)
        assert 'device' in stats
        assert 'confidence_threshold' in stats
        assert 'min_segment_length' in stats
        assert 'word_level_detection_enabled' in stats
        assert 'primary_detector_loaded' in stats
        assert 'secondary_detector_loaded' in stats
        assert 'tokenizer_loaded' in stats
        assert 'supported_language_pairs' in stats
        
        assert stats['device'] == "cpu"
        assert stats['confidence_threshold'] == 0.7
        assert stats['min_segment_length'] == 3
        assert stats['word_level_detection_enabled'] == True


class TestFactoryFunction:
    """Test cases for the factory function."""
    
    def test_create_enhanced_code_switching_detector(self):
        """Test factory function for creating detector."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline'):
            detector = create_enhanced_code_switching_detector(
                device="cuda",
                confidence_threshold=0.8,
                min_segment_length=5,
                enable_word_level_detection=False
            )
            
            assert isinstance(detector, EnhancedCodeSwitchingDetector)
            assert detector.device == "cuda"
            assert detector.confidence_threshold == 0.8
            assert detector.min_segment_length == 5
            assert detector.enable_word_level_detection == False


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    @pytest.fixture
    def failing_detector(self):
        """Create detector with failing models."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Model loading failed")
            detector = create_enhanced_code_switching_detector()
            return detector
    
    @pytest.mark.asyncio
    async def test_detect_code_switching_with_model_failure(self, failing_detector):
        """Test code-switching detection when models fail."""
        text = "Hello नमस्ते world"
        
        result = await failing_detector.detect_code_switching(text)
        
        # Should return fallback result
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) == 1  # Fallback single segment
        assert result.confidence == 0.5  # Fallback confidence
    
    @pytest.mark.asyncio
    async def test_detect_language_ensemble_all_fail(self, failing_detector):
        """Test ensemble detection when all methods fail."""
        text = "Hello world"
        
        with patch.object(failing_detector, '_detect_with_transformer_model', return_value=None), \
             patch.object(failing_detector, '_detect_with_langdetect', return_value=None), \
             patch.object(failing_detector, '_detect_with_patterns', return_value=None):
            
            language, confidence = await failing_detector._detect_language_ensemble(
                text, LanguageCode.HINDI
            )
            
            # Should return context language as fallback
            assert language == LanguageCode.HINDI
            assert confidence == 0.5


class TestIntegrationScenarios:
    """Integration test scenarios with realistic data."""
    
    @pytest.fixture
    def detector_with_mocks(self):
        """Create detector with comprehensive mocks."""
        with patch('bharatvoice.services.language_engine.code_switching_detector.pipeline') as mock_pipeline:
            mock_model = Mock()
            mock_pipeline.return_value = mock_model
            
            detector = create_enhanced_code_switching_detector()
            
            # Mock language detection to return realistic results
            async def mock_ensemble_detect(text, context):
                if any(char in text for char in "नमस्ते"):
                    return LanguageCode.HINDI, 0.9
                else:
                    return LanguageCode.ENGLISH_IN, 0.8
            
            detector._detect_language_ensemble = mock_ensemble_detect
            
            return detector
    
    @pytest.mark.asyncio
    async def test_realistic_hindi_english_mixing(self, detector_with_mocks):
        """Test realistic Hindi-English code-switching scenario."""
        text = "Hello, मैं आज office जा रहा हूं। How are you?"
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 3  # Should detect multiple segments
        assert len(result.switch_points) >= 2  # Should detect switches
        assert result.switching_frequency > 0
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_complex_multilingual_text(self, detector_with_mocks):
        """Test complex multilingual text processing."""
        text = "Good morning! नमस्ते sir, आज का weather कैसा है? It's quite nice today."
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        assert len(result.segments) >= 4
        assert result.processing_time >= 0.0
        
        # Check that segments cover the entire text
        total_coverage = sum(
            seg.end_pos - seg.start_pos for seg in result.segments
        )
        assert total_coverage > 0
    
    @pytest.mark.asyncio
    async def test_single_word_switches(self, detector_with_mocks):
        """Test detection of single word language switches."""
        text = "I am going to the market आज"
        
        result = await detector_with_mocks.detect_code_switching(text)
        
        assert isinstance(result, CodeSwitchingResult)
        # Should handle single word switches appropriately


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])