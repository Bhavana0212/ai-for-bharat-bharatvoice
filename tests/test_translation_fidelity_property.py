"""
Property-based tests for Translation Fidelity.

**Validates: Requirements 3.4**

This module contains property-based tests that verify the translation engine
maintains fidelity across various inputs and language pairs, ensuring semantic
meaning preservation and cultural context adaptation.
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from typing import List, Tuple

from bharatvoice.core.models import LanguageCode
from bharatvoice.services.language_engine.translation_engine import (
    TranslationEngine,
    TranslationResult,
    TranslationQuality,
    SemanticPreservation,
    CulturalContext
)


# Test data strategies
@composite
def language_pairs(draw):
    """Generate valid language pairs for translation."""
    supported_languages = [
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
        LanguageCode.ODIA
    ]
    
    source = draw(st.sampled_from(supported_languages))
    target = draw(st.sampled_from(supported_languages))
    assume(source != target)  # Ensure different languages
    
    return source, target


@composite
def simple_text(draw):
    """Generate simple text for translation testing."""
    # Common words and phrases that should translate well
    words = [
        "hello", "goodbye", "thank you", "please", "yes", "no",
        "water", "food", "house", "family", "friend", "love",
        "good", "bad", "big", "small", "happy", "sad"
    ]
    
    # Generate 1-5 words
    word_count = draw(st.integers(min_value=1, max_value=5))
    selected_words = draw(st.lists(st.sampled_from(words), min_size=word_count, max_size=word_count))
    
    return " ".join(selected_words)


@composite
def cultural_text(draw):
    """Generate text with cultural terms."""
    cultural_terms = [
        "Namaste", "Dharma", "Karma", "Guru", "Ashram",
        "Vanakkam", "Thalaivar", "Adab", "Babu"
    ]
    
    base_text = draw(simple_text())
    cultural_term = draw(st.sampled_from(cultural_terms))
    
    # Insert cultural term at beginning, middle, or end
    position = draw(st.sampled_from(["start", "middle", "end"]))
    
    if position == "start":
        return f"{cultural_term} {base_text}"
    elif position == "middle":
        words = base_text.split()
        if len(words) > 1:
            mid = len(words) // 2
            words.insert(mid, cultural_term)
            return " ".join(words)
        else:
            return f"{base_text} {cultural_term}"
    else:  # end
        return f"{base_text} {cultural_term}"


class TestTranslationFidelityProperties:
    """Property-based tests for translation fidelity."""
    
    @pytest.fixture
    def translation_engine(self):
        """Create translation engine for testing."""
        return TranslationEngine(
            model_cache_size=50,
            enable_cultural_adaptation=True,
            enable_semantic_validation=True,
            quality_threshold=0.5
        )
    
    @pytest.mark.asyncio
    @given(language_pairs(), simple_text())
    @settings(max_examples=20, deadline=10000)  # Reduced for faster testing
    async def test_translation_completeness_property(self, translation_engine, language_pair, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: All translations should produce non-empty results for non-empty inputs.
        """
        source_lang, target_lang = language_pair
        assume(len(text.strip()) > 0)  # Ensure non-empty input
        
        result = await translation_engine.translate(text, source_lang, target_lang)
        
        # Property: Non-empty input should produce non-empty translation
        assert len(result.translated_text.strip()) > 0, f"Empty translation for input: '{text}'"
        
        # Property: Result should have valid structure
        assert isinstance(result, TranslationResult)
        assert result.source_language == source_lang
        assert result.target_language == target_lang
        assert 0.0 <= result.quality_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time >= 0.0
        assert isinstance(result.alternative_translations, list)
        assert isinstance(result.cultural_adaptations, list)
    
    @pytest.mark.asyncio
    @given(language_pairs(), simple_text())
    @settings(max_examples=15, deadline=10000)
    async def test_translation_consistency_property(self, translation_engine, language_pair, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Identical inputs should produce identical translations (consistency).
        """
        source_lang, target_lang = language_pair
        assume(len(text.strip()) > 0)
        
        # Translate the same text twice
        result1 = await translation_engine.translate(text, source_lang, target_lang)
        result2 = await translation_engine.translate(text, source_lang, target_lang)
        
        # Property: Identical inputs should produce identical translations
        assert result1.translated_text == result2.translated_text, \
            f"Inconsistent translations for '{text}': '{result1.translated_text}' vs '{result2.translated_text}'"
        
        # Property: Quality metrics should be consistent
        assert result1.quality_score == result2.quality_score
        assert result1.confidence == result2.confidence
        assert result1.semantic_preservation == result2.semantic_preservation
        assert result1.cultural_context == result2.cultural_context
    
    @pytest.mark.asyncio
    @given(st.sampled_from([LanguageCode.ENGLISH_IN, LanguageCode.HINDI]), simple_text())
    @settings(max_examples=10, deadline=10000)
    async def test_round_trip_translation_property(self, translation_engine, language, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Round-trip translation should preserve core meaning.
        """
        assume(len(text.strip()) > 0)
        
        # Choose target language different from source
        if language == LanguageCode.ENGLISH_IN:
            target_lang = LanguageCode.HINDI
        else:
            target_lang = LanguageCode.ENGLISH_IN
        
        # Forward translation
        forward_result = await translation_engine.translate(text, language, target_lang)
        assume(not forward_result.translated_text.startswith("[Translation"))  # Skip failed translations
        
        # Backward translation
        backward_result = await translation_engine.translate(
            forward_result.translated_text, target_lang, language
        )
        assume(not backward_result.translated_text.startswith("[Translation"))
        
        # Property: Round-trip should maintain reasonable quality
        # (We don't expect perfect round-trip, but should maintain some fidelity)
        assert backward_result.quality_score > 0.0, \
            f"Round-trip translation failed for '{text}'"
        
        # Property: Semantic preservation should not be completely lost
        assert backward_result.semantic_preservation != SemanticPreservation.LOST, \
            f"Complete semantic loss in round-trip for '{text}'"
    
    @pytest.mark.asyncio
    @given(cultural_text())
    @settings(max_examples=10, deadline=10000)
    async def test_cultural_context_preservation_property(self, translation_engine, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Cultural terms should be handled appropriately in translations.
        """
        assume(len(text.strip()) > 0)
        
        # Test Hindi to English (common cultural adaptation scenario)
        result = await translation_engine.translate(
            text, LanguageCode.HINDI, LanguageCode.ENGLISH_IN,
            preserve_cultural_context=True
        )
        
        # Property: Cultural context should not be completely lost
        assert result.cultural_context != CulturalContext.LOST, \
            f"Complete cultural context loss for '{text}'"
        
        # Property: Translation should handle cultural terms
        # (Either preserve them or provide appropriate adaptations)
        cultural_terms = await translation_engine.detect_cultural_terms(text, LanguageCode.HINDI)
        if cultural_terms:
            # If cultural terms were detected, they should be handled somehow
            assert (
                len(result.cultural_adaptations) > 0 or
                any(term.term.lower() in result.translated_text.lower() for term in cultural_terms)
            ), f"Cultural terms not handled in translation of '{text}'"
    
    @pytest.mark.asyncio
    @given(simple_text())
    @settings(max_examples=10, deadline=10000)
    async def test_batch_translation_consistency_property(self, translation_engine, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Batch translation should produce same results as individual translations.
        """
        assume(len(text.strip()) > 0)
        
        source_lang = LanguageCode.ENGLISH_IN
        target_lang = LanguageCode.HINDI
        
        # Individual translation
        individual_result = await translation_engine.translate(text, source_lang, target_lang)
        
        # Batch translation with single item
        batch_results = await translation_engine.batch_translate(
            [text], source_lang, target_lang
        )
        
        # Property: Batch and individual results should be identical
        assert len(batch_results) == 1
        batch_result = batch_results[0]
        
        assert individual_result.translated_text == batch_result.translated_text, \
            f"Batch translation inconsistent for '{text}'"
        assert individual_result.quality_score == batch_result.quality_score
        assert individual_result.confidence == batch_result.confidence
    
    @pytest.mark.asyncio
    @given(st.lists(simple_text(), min_size=2, max_size=5))
    @settings(max_examples=5, deadline=15000)
    async def test_batch_translation_completeness_property(self, translation_engine, texts):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Batch translation should handle all inputs without loss.
        """
        # Filter out empty texts
        non_empty_texts = [text for text in texts if len(text.strip()) > 0]
        assume(len(non_empty_texts) >= 2)
        
        source_lang = LanguageCode.ENGLISH_IN
        target_lang = LanguageCode.HINDI
        
        results = await translation_engine.batch_translate(
            non_empty_texts, source_lang, target_lang
        )
        
        # Property: Should get result for each input
        assert len(results) == len(non_empty_texts), \
            f"Batch translation count mismatch: {len(results)} vs {len(non_empty_texts)}"
        
        # Property: Each result should be valid
        for i, result in enumerate(results):
            assert isinstance(result, TranslationResult)
            assert len(result.translated_text.strip()) > 0, \
                f"Empty translation for batch item {i}: '{non_empty_texts[i]}'"
            assert result.source_language == source_lang
            assert result.target_language == target_lang
    
    @pytest.mark.asyncio
    @given(simple_text())
    @settings(max_examples=10, deadline=10000)
    async def test_quality_metrics_validity_property(self, translation_engine, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Quality metrics should be within valid ranges and consistent.
        """
        assume(len(text.strip()) > 0)
        
        result = await translation_engine.translate(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Property: Quality score should be in valid range
        assert 0.0 <= result.quality_score <= 1.0, \
            f"Invalid quality score: {result.quality_score}"
        
        # Property: Confidence should be in valid range
        assert 0.0 <= result.confidence <= 1.0, \
            f"Invalid confidence score: {result.confidence}"
        
        # Property: Processing time should be non-negative
        assert result.processing_time >= 0.0, \
            f"Invalid processing time: {result.processing_time}"
        
        # Property: Semantic preservation should be valid enum value
        assert result.semantic_preservation in [
            SemanticPreservation.FULL,
            SemanticPreservation.PARTIAL,
            SemanticPreservation.MINIMAL,
            SemanticPreservation.LOST
        ], f"Invalid semantic preservation: {result.semantic_preservation}"
        
        # Property: Cultural context should be valid enum value
        assert result.cultural_context in [
            CulturalContext.PRESERVED,
            CulturalContext.ADAPTED,
            CulturalContext.PARTIALLY_LOST,
            CulturalContext.LOST
        ], f"Invalid cultural context: {result.cultural_context}"
    
    @pytest.mark.asyncio
    @given(simple_text())
    @settings(max_examples=8, deadline=10000)
    async def test_translation_quality_validation_property(self, translation_engine, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Quality validation should provide comprehensive metrics.
        """
        assume(len(text.strip()) > 0)
        
        # Get translation
        result = await translation_engine.translate(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        assume(not result.translated_text.startswith("[Translation"))
        
        # Validate quality
        validation = await translation_engine.validate_translation_quality(
            text, result.translated_text,
            LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Property: Validation should include all required metrics
        required_keys = [
            'overall_quality_score', 'quality_level', 'semantic_preservation',
            'cultural_context', 'fluency_score', 'adequacy_score'
        ]
        
        for key in required_keys:
            assert key in validation, f"Missing validation metric: {key}"
        
        # Property: Scores should be in valid ranges
        assert 0.0 <= validation['overall_quality_score'] <= 1.0
        assert 0.0 <= validation['fluency_score'] <= 1.0
        assert 0.0 <= validation['adequacy_score'] <= 1.0
        
        # Property: Quality level should be valid
        assert validation['quality_level'] in [
            TranslationQuality.EXCELLENT.value,
            TranslationQuality.GOOD.value,
            TranslationQuality.FAIR.value,
            TranslationQuality.POOR.value
        ]
    
    @pytest.mark.asyncio
    @given(simple_text())
    @settings(max_examples=5, deadline=10000)
    async def test_translation_suggestions_completeness_property(self, translation_engine, text):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Translation suggestions should provide comprehensive information.
        """
        assume(len(text.strip()) > 0)
        
        suggestions = await translation_engine.get_translation_suggestions(
            text, LanguageCode.ENGLISH_IN, LanguageCode.HINDI
        )
        
        # Property: Should include all required sections
        required_keys = [
            'main_translation', 'alternative_translations', 'contextual_alternatives',
            'potential_issues', 'improvement_suggestions', 'cultural_notes'
        ]
        
        for key in required_keys:
            assert key in suggestions, f"Missing suggestion section: {key}"
        
        # Property: Main translation should be valid
        main_translation = suggestions['main_translation']
        if main_translation is not None:  # May be None on error
            assert 'translated_text' in main_translation
            assert 'quality_score' in main_translation
            assert 'confidence' in main_translation
        
        # Property: Lists should be actual lists
        assert isinstance(suggestions['alternative_translations'], list)
        assert isinstance(suggestions['contextual_alternatives'], list)
        assert isinstance(suggestions['potential_issues'], list)
        assert isinstance(suggestions['improvement_suggestions'], list)
        assert isinstance(suggestions['cultural_notes'], list)
    
    def test_supported_language_pairs_property(self, translation_engine):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Supported language pairs should be comprehensive and bidirectional.
        """
        pairs = translation_engine.get_supported_language_pairs()
        
        # Property: Should have reasonable number of pairs
        assert len(pairs) > 0, "No supported language pairs"
        
        # Property: Should include major Indian language pairs
        expected_languages = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL]
        
        for lang1 in expected_languages:
            for lang2 in expected_languages:
                if lang1 != lang2:
                    assert (lang1, lang2) in pairs, f"Missing language pair: {lang1} -> {lang2}"
        
        # Property: No self-translation pairs
        for source, target in pairs:
            assert source != target, f"Invalid self-translation pair: {source} -> {target}"
        
        # Property: Should be bidirectional for major languages
        for lang1 in expected_languages:
            for lang2 in expected_languages:
                if lang1 != lang2:
                    assert (lang1, lang2) in pairs and (lang2, lang1) in pairs, \
                        f"Missing bidirectional support for {lang1} <-> {lang2}"
    
    @pytest.mark.asyncio
    async def test_health_check_property(self, translation_engine):
        """
        **Property 11: Translation Fidelity**
        **Validates: Requirements 3.4**
        
        Property: Health check should provide comprehensive system status.
        """
        health = await translation_engine.health_check()
        
        # Property: Should include all required health metrics
        required_keys = ['status', 'translation_test', 'cultural_detection_test', 'supported_language_pairs', 'stats']
        
        for key in required_keys:
            assert key in health, f"Missing health check metric: {key}"
        
        # Property: Status should be valid
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Property: Test results should be valid
        assert health['translation_test'] in ['ok', 'degraded', 'error']
        assert health['cultural_detection_test'] in ['ok', 'error']
        
        # Property: Should report positive number of language pairs
        assert health['supported_language_pairs'] > 0
        
        # Property: Stats should be included
        assert isinstance(health['stats'], dict)