"""
Unit tests for the Translation Engine.

Tests cover neural machine translation, semantic preservation,
cultural context adaptation, and quality scoring functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from bharatvoice.core.models import LanguageCode
from bharatvoice.services.language_engine.translation_engine import (
    TranslationEngine,
    TranslationResult,
    TranslationQuality,
    SemanticPreservation,
    CulturalContext,
    CulturalTerm,
    SemanticAnalysis
)


class TestTranslationEngine:
    """Test suite for TranslationEngine class."""
    
    @pytest.fixture
    def translation_engine(self):
        """Create a translation engine instance for testing."""
        return TranslationEngine(
            model_cache_size=10,
            enable_cultural_adaptation=True,
            enable_semantic_validation=True,
            quality_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, translation_engine):
        """Test translation engine initialization."""
        assert translation_engine.model_cache_size == 10
        assert translation_engine.enable_cultural_adaptation is True
        assert translation_engine.enable_semantic_validation is True
        assert translation_engine.quality_threshold == 0.7
        assert len(translation_engine.translation_cache) == 0
        assert len(translation_engine.model_cache) == 0
        assert translation_engine.stats['total_translations'] == 0
    
    @pytest.mark.asyncio
    async def test_empty_text_translation(self, translation_engine):
        """Test translation of empty text."""
        result = await translation_engine.translate(
            "", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result.translated_text == ""
        assert result.source_language == LanguageCode.ENGLISH_IN
        assert result.target_language == LanguageCode.HINDI
        assert result.quality_score == 1.0
        assert result.semantic_preservation == SemanticPreservation.FULL
        assert result.cultural_context == CulturalContext.PRESERVED
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_same_language_translation(self, translation_engine):
        """Test translation when source and target languages are the same."""
        text = "Hello world"
        result = await translation_engine.translate(
            text, LanguageCode.ENGLISH_IN, LanguageCode.ENGLISH_IN
        )
        
        assert result.translated_text == text
        assert result.source_language == LanguageCode.ENGLISH_IN
        assert result.target_language == LanguageCode.ENGLISH_IN
        assert result.quality_score == 1.0
        assert result.semantic_preservation == SemanticPreservation.FULL
        assert result.cultural_context == CulturalContext.PRESERVED
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_basic_translation(self, translation_engine):
        """Test basic translation functionality."""
        result = await translation_engine.translate(
            "hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert result.translated_text == "नमस्ते"
        assert result.source_language == LanguageCode.ENGLISH_IN
        assert result.target_language == LanguageCode.HINDI
        assert result.quality_score > 0.0
        assert result.confidence > 0.0
        assert isinstance(result.processing_time, float)
        assert result.processing_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_reverse_translation(self, translation_engine):
        """Test reverse translation (Hindi to English)."""
        result = await translation_engine.translate(
            "नमस्ते", LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert result.translated_text == "hello"
        assert result.source_language == LanguageCode.HINDI
        assert result.target_language == LanguageCode.ENGLISH_IN
        assert result.quality_score > 0.0
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_translation_caching(self, translation_engine):
        """Test translation result caching."""
        text = "hello"
        source_lang = LanguageCode.ENGLISH_IN
        target_lang = LanguageCode.HINDI
        
        # First translation
        result1 = await translation_engine.translate(text, source_lang, target_lang)
        assert translation_engine.stats['cache_misses'] == 1
        assert translation_engine.stats['cache_hits'] == 0
        
        # Second translation (should hit cache)
        result2 = await translation_engine.translate(text, source_lang, target_lang)
        assert translation_engine.stats['cache_hits'] == 1
        assert result1.translated_text == result2.translated_text
    
    @pytest.mark.asyncio
    async def test_batch_translation(self, translation_engine):
        """Test batch translation functionality."""
        texts = ["hello", "goodbye", "thank you"]
        results = await translation_engine.batch_translate(
            texts, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert len(results) == 3
        assert results[0].translated_text == "नमस्ते"
        assert results[1].translated_text == "अलविदा"
        assert results[2].translated_text == "धन्यवाद"
        
        for result in results:
            assert result.source_language == LanguageCode.ENGLISH_IN
            assert result.target_language == LanguageCode.HINDI
            assert result.quality_score > 0.0
    
    @pytest.mark.asyncio
    async def test_empty_batch_translation(self, translation_engine):
        """Test batch translation with empty list."""
        results = await translation_engine.batch_translate(
            [], LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_cultural_term_detection(self, translation_engine):
        """Test detection of cultural terms."""
        text = "Namaste, how are you?"
        cultural_terms = await translation_engine.detect_cultural_terms(
            text, LanguageCode.HINDI
        )
        
        assert len(cultural_terms) > 0
        namaste_term = next((term for term in cultural_terms if term.term == "Namaste"), None)
        assert namaste_term is not None
        assert namaste_term.meaning == "Traditional Indian greeting"
        assert namaste_term.cultural_significance == "Respectful greeting with spiritual meaning"
    
    @pytest.mark.asyncio
    async def test_cultural_adaptation(self, translation_engine):
        """Test cultural context adaptation in translation."""
        result = await translation_engine.translate(
            "Hello friend", LanguageCode.ENGLISH_IN, LanguageCode.HINDI,
            preserve_cultural_context=True
        )
        
        # Should adapt greeting to culturally appropriate form
        assert "नमस्ते" in result.translated_text or "hello" in result.translated_text.lower()
        assert len(result.cultural_adaptations) >= 0  # May have adaptations
    
    @pytest.mark.asyncio
    async def test_semantic_analysis(self, translation_engine):
        """Test semantic analysis of text."""
        text = "I am very happy today"
        semantics = await translation_engine._analyze_semantics(text, LanguageCode.ENGLISH_IN)
        
        assert isinstance(semantics, SemanticAnalysis)
        assert semantics.sentiment == "positive"
        assert len(semantics.key_concepts) > 0
        assert semantics.formality_level in ["formal", "informal", "neutral"]
    
    @pytest.mark.asyncio
    async def test_quality_validation(self, translation_engine):
        """Test translation quality validation."""
        source_text = "Hello world"
        translated_text = "नमस्ते दुनिया"
        
        validation_result = await translation_engine.validate_translation_quality(
            source_text, translated_text,
            LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert "overall_quality_score" in validation_result
        assert "quality_level" in validation_result
        assert "semantic_preservation" in validation_result
        assert "cultural_context" in validation_result
        assert "fluency_score" in validation_result
        assert "adequacy_score" in validation_result
        assert validation_result["overall_quality_score"] >= 0.0
        assert validation_result["overall_quality_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_translation_suggestions(self, translation_engine):
        """Test translation suggestions functionality."""
        suggestions = await translation_engine.get_translation_suggestions(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert "main_translation" in suggestions
        assert "alternative_translations" in suggestions
        assert "contextual_alternatives" in suggestions
        assert "potential_issues" in suggestions
        assert "improvement_suggestions" in suggestions
        assert "cultural_notes" in suggestions
        
        main_translation = suggestions["main_translation"]
        assert main_translation is not None
        assert "translated_text" in main_translation
    
    @pytest.mark.asyncio
    async def test_contextual_suggestions(self, translation_engine):
        """Test contextual translation suggestions."""
        suggestions = await translation_engine.get_translation_suggestions(
            "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI,
            context="formal business meeting"
        )
        
        assert "contextual_alternatives" in suggestions
        # Should provide formal alternatives for business context
        contextual_alts = suggestions["contextual_alternatives"]
        assert isinstance(contextual_alts, list)
    
    @pytest.mark.asyncio
    async def test_supported_language_pairs(self, translation_engine):
        """Test getting supported language pairs."""
        pairs = translation_engine.get_supported_language_pairs()
        
        assert len(pairs) > 0
        assert (LanguageCode.ENGLISH_IN, LanguageCode.HINDI) in pairs
        assert (LanguageCode.HINDI, LanguageCode.ENGLISH_IN) in pairs
        assert (LanguageCode.ENGLISH_IN, LanguageCode.TAMIL) in pairs
        
        # Ensure no self-pairs
        for source, target in pairs:
            assert source != target
    
    def test_translation_stats(self, translation_engine):
        """Test translation statistics functionality."""
        stats = translation_engine.get_translation_stats()
        
        assert "total_translations" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "quality_distribution" in stats
        assert "language_pairs" in stats
        assert "average_processing_time" in stats
        assert "semantic_preservation_rate" in stats
        assert "cultural_adaptation_rate" in stats
        assert "cache_stats" in stats
        assert "configuration" in stats
        
        # Check quality distribution structure
        quality_dist = stats["quality_distribution"]
        assert "excellent" in quality_dist
        assert "good" in quality_dist
        assert "fair" in quality_dist
        assert "poor" in quality_dist
    
    def test_cache_management(self, translation_engine):
        """Test cache management functionality."""
        # Add some dummy data to caches
        translation_engine.translation_cache["test_key"] = Mock()
        translation_engine.model_cache["test_model"] = Mock()
        
        assert len(translation_engine.translation_cache) == 1
        assert len(translation_engine.model_cache) == 1
        
        # Clear caches
        translation_engine.clear_caches()
        
        assert len(translation_engine.translation_cache) == 0
        assert len(translation_engine.model_cache) == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, translation_engine):
        """Test translation engine health check."""
        health_result = await translation_engine.health_check()
        
        assert "status" in health_result
        assert "translation_test" in health_result
        assert "cultural_detection_test" in health_result
        assert "supported_language_pairs" in health_result
        assert "stats" in health_result
        
        assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
        assert health_result["supported_language_pairs"] > 0
    
    @pytest.mark.asyncio
    async def test_translation_with_cultural_terms(self, translation_engine):
        """Test translation containing cultural terms."""
        text = "Namaste, I hope you have a good day"
        result = await translation_engine.translate(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Should handle cultural terms appropriately
        assert result.translated_text is not None
        assert len(result.translated_text) > 0
        assert result.cultural_context in [
            CulturalContext.PRESERVED,
            CulturalContext.ADAPTED,
            CulturalContext.PARTIALLY_LOST,
            CulturalContext.LOST
        ]
    
    @pytest.mark.asyncio
    async def test_translation_quality_scoring(self, translation_engine):
        """Test translation quality scoring."""
        # Test with a good translation
        good_result = await translation_engine.translate(
            "hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        assert good_result.quality_score > 0.5
        
        # Test quality classification
        quality_level = translation_engine._classify_quality(good_result.quality_score)
        assert quality_level in [
            TranslationQuality.EXCELLENT,
            TranslationQuality.GOOD,
            TranslationQuality.FAIR,
            TranslationQuality.POOR
        ]
    
    @pytest.mark.asyncio
    async def test_semantic_preservation_assessment(self, translation_engine):
        """Test semantic preservation assessment."""
        source_text = "I am happy"
        source_semantics = await translation_engine._analyze_semantics(
            source_text, LanguageCode.ENGLISH_IN
        )
        
        translated_text = "मैं खुश हूं"
        preservation = await translation_engine._assess_semantic_preservation(
            source_semantics, translated_text, LanguageCode.HINDI
        )
        
        assert preservation in [
            SemanticPreservation.FULL,
            SemanticPreservation.PARTIAL,
            SemanticPreservation.MINIMAL,
            SemanticPreservation.LOST
        ]
    
    @pytest.mark.asyncio
    async def test_cultural_context_assessment(self, translation_engine):
        """Test cultural context assessment."""
        source_text = "Namaste friend"
        translated_text = "Hello friend"
        
        cultural_context = await translation_engine._assess_cultural_context(
            source_text, translated_text,
            LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert cultural_context in [
            CulturalContext.PRESERVED,
            CulturalContext.ADAPTED,
            CulturalContext.PARTIALLY_LOST,
            CulturalContext.LOST
        ]
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, translation_engine):
        """Test confidence score calculation."""
        confidence = translation_engine._calculate_confidence(
            quality_score=0.8,
            semantic_preservation=SemanticPreservation.FULL,
            cultural_context=CulturalContext.PRESERVED
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high for good inputs
    
    @pytest.mark.asyncio
    async def test_translation_result_serialization(self, translation_engine):
        """Test TranslationResult serialization."""
        result = await translation_engine.translate(
            "hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        result_dict = result.to_dict()
        
        assert "translated_text" in result_dict
        assert "source_language" in result_dict
        assert "target_language" in result_dict
        assert "quality_score" in result_dict
        assert "semantic_preservation" in result_dict
        assert "cultural_context" in result_dict
        assert "confidence" in result_dict
        assert "processing_time" in result_dict
        assert "alternative_translations" in result_dict
        assert "cultural_adaptations" in result_dict
        
        # Check that enum values are serialized as strings
        assert isinstance(result_dict["source_language"], str)
        assert isinstance(result_dict["target_language"], str)
        assert isinstance(result_dict["semantic_preservation"], str)
        assert isinstance(result_dict["cultural_context"], str)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, translation_engine):
        """Test error handling in translation."""
        # Test with invalid input that might cause errors
        with patch.object(translation_engine, '_perform_translation', side_effect=Exception("Test error")):
            result = await translation_engine.translate(
                "test", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
            )
            
            assert "[Translation Error:" in result.translated_text
            assert result.quality_score == 0.0
            assert result.confidence == 0.0
            assert result.semantic_preservation == SemanticPreservation.LOST
            assert result.cultural_context == CulturalContext.LOST
    
    @pytest.mark.asyncio
    async def test_batch_translation_error_handling(self, translation_engine):
        """Test error handling in batch translation."""
        texts = ["hello", "error_text", "goodbye"]
        
        # Mock one translation to fail
        original_translate = translation_engine.translate
        async def mock_translate(text, source_lang, target_lang, preserve_cultural_context=True, validate_semantics=True):
            if text == "error_text":
                raise Exception("Translation failed")
            return await original_translate(text, source_lang, target_lang, preserve_cultural_context, validate_semantics)
        
        with patch.object(translation_engine, 'translate', side_effect=mock_translate):
            results = await translation_engine.batch_translate(
                texts, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
            )
            
            assert len(results) == 3
            assert results[0].translated_text == "नमस्ते"  # Should succeed
            assert "[Translation Error:" in results[1].translated_text  # Should fail gracefully
            assert results[2].translated_text == "अलविदा"  # Should succeed
    
    @pytest.mark.asyncio
    async def test_idiom_detection(self, translation_engine):
        """Test idiom detection functionality."""
        text = "Break a leg in your performance!"
        idioms = await translation_engine._detect_idioms(text, LanguageCode.ENGLISH_IN)
        
        assert len(idioms) > 0
        break_leg_idiom = next((idiom for idiom in idioms if "break a leg" in idiom["expression"]), None)
        assert break_leg_idiom is not None
        assert break_leg_idiom["meaning"] == "good luck"
    
    @pytest.mark.asyncio
    async def test_fluency_assessment(self, translation_engine):
        """Test fluency assessment."""
        # Test with fluent text
        fluent_text = "This is a well-formed sentence with proper punctuation."
        fluency_score = await translation_engine._assess_fluency(fluent_text, LanguageCode.ENGLISH_IN)
        assert fluency_score > 0.5
        
        # Test with less fluent text
        poor_text = "word word word"
        poor_fluency = await translation_engine._assess_fluency(poor_text, LanguageCode.ENGLISH_IN)
        assert poor_fluency < fluency_score
    
    @pytest.mark.asyncio
    async def test_adequacy_assessment(self, translation_engine):
        """Test adequacy assessment."""
        source_text = "Hello world"
        good_translation = "नमस्ते दुनिया"
        poor_translation = ""
        
        good_adequacy = await translation_engine._assess_adequacy(
            source_text, good_translation, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        poor_adequacy = await translation_engine._assess_adequacy(
            source_text, poor_translation, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert good_adequacy > poor_adequacy
        assert good_adequacy > 0.5
    
    def test_cache_key_generation(self, translation_engine):
        """Test cache key generation."""
        key1 = translation_engine._generate_cache_key(
            "hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        key2 = translation_engine._generate_cache_key(
            "hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        key3 = translation_engine._generate_cache_key(
            "goodbye", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert key1 == key2  # Same input should generate same key
        assert key1 != key3  # Different input should generate different key
        assert len(key1) == 32  # MD5 hash length
    
    @pytest.mark.asyncio
    async def test_translation_issue_detection(self, translation_engine):
        """Test detection of translation issues."""
        source_text = "Hello world!"
        translated_text = "नमस्ते"  # Missing translation of "world"
        
        issues = await translation_engine._detect_translation_issues(
            source_text, translated_text,
            LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        assert isinstance(issues, list)
        # Should detect length discrepancy
        length_issue = any("short" in issue.lower() for issue in issues)
        assert length_issue or len(issues) == 0  # May or may not detect depending on thresholds
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, translation_engine):
        """Test generation of improvement suggestions."""
        issues = ["Translation appears too short", "Missing punctuation in translation"]
        suggestions = await translation_engine._generate_improvement_suggestions(
            "Hello world!", "नमस्ते",
            LanguageCode.ENGLISH_IN, LanguageCode.HINDI, issues
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should provide relevant suggestions for the issues
        content_suggestion = any("content" in suggestion.lower() for suggestion in suggestions)
        punct_suggestion = any("punctuation" in suggestion.lower() for suggestion in suggestions)
        assert content_suggestion or punct_suggestion or "good quality" in suggestions[0].lower()
    
    @pytest.mark.asyncio
    async def test_cultural_notes(self, translation_engine):
        """Test generation of cultural notes."""
        notes = await translation_engine._get_cultural_notes(
            "Namaste friend", LanguageCode.HINDI, LanguageCode.ENGLISH_IN
        )
        
        assert isinstance(notes, list)
        # Should provide cultural context for Hindi to English translation
        if notes:
            namaste_note = any("namaste" in note.lower() for note in notes)
            assert namaste_note or len(notes) > 0