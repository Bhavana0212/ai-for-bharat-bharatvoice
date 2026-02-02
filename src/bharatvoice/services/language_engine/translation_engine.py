"""
Translation Engine for BharatVoice Assistant.

This module provides neural machine translation capabilities between Indian languages
with semantic meaning preservation, cultural context preservation, and quality scoring.
"""

import asyncio
import logging
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from bharatvoice.core.models import LanguageCode

logger = logging.getLogger(__name__)


class TranslationQuality(str, Enum):
    """Translation quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class SemanticPreservation(str, Enum):
    """Semantic meaning preservation levels."""
    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    LOST = "lost"


class CulturalContext(str, Enum):
    """Cultural context preservation levels."""
    PRESERVED = "preserved"
    ADAPTED = "adapted"
    PARTIALLY_LOST = "partially_lost"
    LOST = "lost"


@dataclass
class TranslationResult:
    """Result of translation operation."""
    translated_text: str
    source_language: LanguageCode
    target_language: LanguageCode
    quality_score: float
    semantic_preservation: SemanticPreservation
    cultural_context: CulturalContext
    confidence: float
    processing_time: float
    alternative_translations: List[str]
    cultural_adaptations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "translated_text": self.translated_text,
            "source_language": self.source_language.value,
            "target_language": self.target_language.value,
            "quality_score": self.quality_score,
            "semantic_preservation": self.semantic_preservation.value,
            "cultural_context": self.cultural_context.value,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "alternative_translations": self.alternative_translations,
            "cultural_adaptations": self.cultural_adaptations
        }


@dataclass
class CulturalTerm:
    """Cultural term with context information."""
    term: str
    language: LanguageCode
    meaning: str
    cultural_significance: str
    regional_variants: List[str]
    translation_notes: str


@dataclass
class SemanticAnalysis:
    """Semantic analysis of text."""
    key_concepts: List[str]
    sentiment: str
    formality_level: str
    cultural_references: List[str]
    idiomatic_expressions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_concepts": self.key_concepts,
            "sentiment": self.sentiment,
            "formality_level": self.formality_level,
            "cultural_references": self.cultural_references,
            "idiomatic_expressions": self.idiomatic_expressions
        }


class TranslationEngine:
    """
    Neural machine translation engine for Indian languages.
    
    Provides bidirectional translation between Hindi, English, and other Indian languages
    with focus on semantic meaning preservation and cultural context adaptation.
    """
    
    def __init__(
        self,
        model_cache_size: int = 100,
        enable_cultural_adaptation: bool = True,
        enable_semantic_validation: bool = True,
        quality_threshold: float = 0.7
    ):
        """
        Initialize the translation engine.
        
        Args:
            model_cache_size: Maximum number of cached translation models
            enable_cultural_adaptation: Whether to enable cultural context adaptation
            enable_semantic_validation: Whether to validate semantic preservation
            quality_threshold: Minimum quality score for translations
        """
        self.model_cache_size = model_cache_size
        self.enable_cultural_adaptation = enable_cultural_adaptation
        self.enable_semantic_validation = enable_semantic_validation
        self.quality_threshold = quality_threshold
        
        # Translation caches
        self.translation_cache: Dict[str, TranslationResult] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Cultural knowledge base
        self.cultural_terms = self._initialize_cultural_terms()
        self.regional_adaptations = self._initialize_regional_adaptations()
        
        # Translation statistics
        self.stats = {
            'total_translations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'quality_distribution': {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0
            },
            'language_pairs': {},
            'average_processing_time': 0.0,
            'semantic_preservation_rate': 0.0,
            'cultural_adaptation_rate': 0.0
        }
        
        logger.info("TranslationEngine initialized successfully")
    
    async def translate(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> TranslationResult:
        """
        Translate text between languages with quality assessment.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            Translation result with quality metrics
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Input validation
            if not text.strip():
                return self._create_empty_result(source_language, target_language)
            
            if source_language == target_language:
                return self._create_identity_result(text, source_language, start_time)
            
            # Check cache
            cache_key = self._generate_cache_key(text, source_language, target_language)
            if cache_key in self.translation_cache:
                self.stats['cache_hits'] += 1
                logger.debug("Translation found in cache")
                return self.translation_cache[cache_key]
            
            self.stats['cache_misses'] += 1
            
            # Perform semantic analysis of source text
            source_semantics = None
            if validate_semantics and self.enable_semantic_validation:
                source_semantics = await self._analyze_semantics(text, source_language)
            
            # Perform translation
            translated_text = await self._perform_translation(
                text, source_language, target_language
            )
            
            # Apply cultural adaptations
            if preserve_cultural_context and self.enable_cultural_adaptation:
                translated_text, cultural_adaptations = await self._apply_cultural_adaptations(
                    translated_text, source_language, target_language, text
                )
            else:
                cultural_adaptations = []
            
            # Generate alternative translations
            alternatives = await self._generate_alternatives(
                text, source_language, target_language, translated_text
            )
            
            # Calculate quality metrics
            quality_score = await self._calculate_quality_score(
                text, translated_text, source_language, target_language
            )
            
            # Assess semantic preservation
            semantic_preservation = SemanticPreservation.FULL
            if validate_semantics and self.enable_semantic_validation and source_semantics:
                semantic_preservation = await self._assess_semantic_preservation(
                    source_semantics, translated_text, target_language
                )
            
            # Assess cultural context preservation
            cultural_context = await self._assess_cultural_context(
                text, translated_text, source_language, target_language
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                quality_score, semantic_preservation, cultural_context
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create result
            result = TranslationResult(
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                quality_score=quality_score,
                semantic_preservation=semantic_preservation,
                cultural_context=cultural_context,
                confidence=confidence,
                processing_time=processing_time,
                alternative_translations=alternatives,
                cultural_adaptations=cultural_adaptations
            )
            
            # Update statistics
            self._update_translation_stats(result)
            
            # Cache result
            self._cache_translation_result(cache_key, result)
            
            logger.info(
                f"Translation completed: {source_language} -> {target_language}, "
                f"quality={quality_score:.3f}, confidence={confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return self._create_error_result(text, source_language, target_language, str(e))
    
    async def batch_translate(
        self,
        texts: List[str],
        source_language: LanguageCode,
        target_language: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            List of translation results
        """
        try:
            if not texts:
                return []
            
            # Process translations concurrently
            tasks = [
                self.translate(
                    text, source_language, target_language,
                    preserve_cultural_context, validate_semantics
                )
                for text in texts
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error translating text {i}: {result}")
                    processed_results.append(
                        self._create_error_result(
                            texts[i], source_language, target_language, str(result)
                        )
                    )
                else:
                    processed_results.append(result)
            
            logger.info(f"Batch translation completed for {len(texts)} texts")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            return [
                self._create_error_result(text, source_language, target_language, str(e))
                for text in texts
            ]
    
    async def detect_cultural_terms(
        self,
        text: str,
        language: LanguageCode
    ) -> List[CulturalTerm]:
        """
        Detect cultural terms and expressions in text.
        
        Args:
            text: Text to analyze
            language: Language of the text
            
        Returns:
            List of detected cultural terms
        """
        try:
            detected_terms = []
            
            # Get cultural terms for the language
            language_terms = self.cultural_terms.get(language, [])
            
            # Search for cultural terms in text
            text_lower = text.lower()
            for term_data in language_terms:
                term = term_data['term'].lower()
                if term in text_lower:
                    cultural_term = CulturalTerm(
                        term=term_data['term'],
                        language=language,
                        meaning=term_data['meaning'],
                        cultural_significance=term_data['significance'],
                        regional_variants=term_data.get('variants', []),
                        translation_notes=term_data.get('notes', '')
                    )
                    detected_terms.append(cultural_term)
            
            # Detect idiomatic expressions
            idioms = await self._detect_idioms(text, language)
            for idiom in idioms:
                cultural_term = CulturalTerm(
                    term=idiom['expression'],
                    language=language,
                    meaning=idiom['meaning'],
                    cultural_significance='idiomatic expression',
                    regional_variants=idiom.get('variants', []),
                    translation_notes=idiom.get('notes', '')
                )
                detected_terms.append(cultural_term)
            
            logger.debug(f"Detected {len(detected_terms)} cultural terms in text")
            return detected_terms
            
        except Exception as e:
            logger.error(f"Error detecting cultural terms: {e}")
            return []
    
    async def validate_translation_quality(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Validate translation quality with detailed metrics.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_language: Source language
            target_language: Target language
            
        Returns:
            Detailed quality validation results
        """
        try:
            # Calculate various quality metrics
            quality_score = await self._calculate_quality_score(
                source_text, translated_text, source_language, target_language
            )
            
            # Semantic analysis
            source_semantics = await self._analyze_semantics(source_text, source_language)
            semantic_preservation = await self._assess_semantic_preservation(
                source_semantics, translated_text, target_language
            )
            
            # Cultural context assessment
            cultural_context = await self._assess_cultural_context(
                source_text, translated_text, source_language, target_language
            )
            
            # Fluency assessment
            fluency_score = await self._assess_fluency(translated_text, target_language)
            
            # Adequacy assessment
            adequacy_score = await self._assess_adequacy(
                source_text, translated_text, source_language, target_language
            )
            
            # Overall quality classification
            quality_level = self._classify_quality(quality_score)
            
            return {
                'overall_quality_score': quality_score,
                'quality_level': quality_level.value,
                'semantic_preservation': semantic_preservation.value,
                'cultural_context': cultural_context.value,
                'fluency_score': fluency_score,
                'adequacy_score': adequacy_score,
                'source_semantics': source_semantics.to_dict() if source_semantics else None,
                'recommendations': self._generate_quality_recommendations(
                    quality_score, semantic_preservation, cultural_context
                )
            }
            
        except Exception as e:
            logger.error(f"Error validating translation quality: {e}")
            return {
                'overall_quality_score': 0.0,
                'quality_level': TranslationQuality.POOR.value,
                'error': str(e)
            }
    
    async def get_translation_suggestions(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get translation suggestions with alternatives and improvements.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            context: Optional context for better translation
            
        Returns:
            Translation suggestions and alternatives
        """
        try:
            # Perform main translation
            main_result = await self.translate(
                text, source_language, target_language
            )
            
            # Generate contextual alternatives if context provided
            contextual_alternatives = []
            if context:
                contextual_alternatives = await self._generate_contextual_alternatives(
                    text, source_language, target_language, context
                )
            
            # Detect potential issues
            issues = await self._detect_translation_issues(
                text, main_result.translated_text, source_language, target_language
            )
            
            # Generate improvement suggestions
            improvements = await self._generate_improvement_suggestions(
                text, main_result.translated_text, source_language, target_language, issues
            )
            
            return {
                'main_translation': main_result.to_dict(),
                'alternative_translations': main_result.alternative_translations,
                'contextual_alternatives': contextual_alternatives,
                'potential_issues': issues,
                'improvement_suggestions': improvements,
                'cultural_notes': await self._get_cultural_notes(
                    text, source_language, target_language
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting translation suggestions: {e}")
            return {
                'error': str(e),
                'main_translation': None,
                'alternative_translations': [],
                'contextual_alternatives': [],
                'potential_issues': [],
                'improvement_suggestions': [],
                'cultural_notes': []
            }
    
    def get_supported_language_pairs(self) -> List[Tuple[LanguageCode, LanguageCode]]:
        """
        Get list of supported language pairs for translation.
        
        Returns:
            List of supported (source, target) language pairs
        """
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
        
        # Generate all possible pairs (bidirectional)
        pairs = []
        for source in supported_languages:
            for target in supported_languages:
                if source != target:
                    pairs.append((source, target))
        
        return pairs
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """
        Get translation engine statistics.
        
        Returns:
            Dictionary with translation statistics
        """
        stats = self.stats.copy()
        
        # Add cache statistics
        stats['cache_stats'] = {
            'translation_cache_size': len(self.translation_cache),
            'model_cache_size': len(self.model_cache),
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            )
        }
        
        # Add configuration info
        stats['configuration'] = {
            'cultural_adaptation_enabled': self.enable_cultural_adaptation,
            'semantic_validation_enabled': self.enable_semantic_validation,
            'quality_threshold': self.quality_threshold,
            'model_cache_size': self.model_cache_size
        }
        
        return stats
    
    def clear_caches(self):
        """Clear all translation caches."""
        self.translation_cache.clear()
        self.model_cache.clear()
        logger.info("Cleared translation engine caches")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the translation engine.
        
        Returns:
            Health check result
        """
        try:
            # Test basic translation functionality
            test_result = await self.translate(
                "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
            )
            
            translation_status = "ok" if test_result.confidence > 0.5 else "degraded"
            
            # Test cultural term detection
            cultural_terms = await self.detect_cultural_terms(
                "Namaste", LanguageCode.HINDI
            )
            cultural_detection_status = "ok" if len(cultural_terms) >= 0 else "error"
            
            overall_status = (
                "healthy" if (
                    translation_status == "ok" and
                    cultural_detection_status == "ok"
                ) else "degraded"
            )
            
            return {
                "status": overall_status,
                "translation_test": translation_status,
                "cultural_detection_test": cultural_detection_status,
                "supported_language_pairs": len(self.get_supported_language_pairs()),
                "stats": self.get_translation_stats()
            }
            
        except Exception as e:
            logger.error(f"Translation engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_translation_stats()
            }
    # Private helper methods
    
    def _initialize_cultural_terms(self) -> Dict[LanguageCode, List[Dict[str, Any]]]:
        """Initialize cultural terms database."""
        return {
            LanguageCode.HINDI: [
                {
                    'term': 'Namaste',
                    'meaning': 'Traditional Indian greeting',
                    'significance': 'Respectful greeting with spiritual meaning',
                    'variants': ['Namaskar', 'Namaskaram'],
                    'notes': 'Used with palms together gesture'
                },
                {
                    'term': 'Dharma',
                    'meaning': 'Righteous duty or way of life',
                    'significance': 'Core concept in Hindu philosophy',
                    'variants': ['Dharm'],
                    'notes': 'Context-dependent translation needed'
                },
                {
                    'term': 'Karma',
                    'meaning': 'Action and its consequences',
                    'significance': 'Fundamental concept of cause and effect',
                    'variants': ['Karm'],
                    'notes': 'Often misunderstood in Western contexts'
                },
                {
                    'term': 'Guru',
                    'meaning': 'Teacher or spiritual guide',
                    'significance': 'Revered figure in Indian culture',
                    'variants': ['Guruji'],
                    'notes': 'More than just teacher - spiritual significance'
                },
                {
                    'term': 'Ashram',
                    'meaning': 'Spiritual retreat or hermitage',
                    'significance': 'Place of spiritual learning',
                    'variants': ['Ashrama'],
                    'notes': 'Different from Western retreat centers'
                }
            ],
            LanguageCode.TAMIL: [
                {
                    'term': 'Vanakkam',
                    'meaning': 'Traditional Tamil greeting',
                    'significance': 'Respectful greeting in Tamil culture',
                    'variants': ['Vanakam'],
                    'notes': 'Equivalent to Namaste in Hindi'
                },
                {
                    'term': 'Thalaivar',
                    'meaning': 'Leader or respected person',
                    'significance': 'Term of respect in Tamil culture',
                    'variants': ['Thalaiva'],
                    'notes': 'Often used for film stars and politicians'
                }
            ],
            LanguageCode.BENGALI: [
                {
                    'term': 'Adab',
                    'meaning': 'Traditional Bengali greeting',
                    'significance': 'Respectful greeting in Bengali culture',
                    'variants': ['Aadab'],
                    'notes': 'Used in formal contexts'
                },
                {
                    'term': 'Babu',
                    'meaning': 'Respectful address for men',
                    'significance': 'Traditional term of respect',
                    'variants': ['Babumoshai'],
                    'notes': 'Context-dependent usage'
                }
            ]
        }
    
    def _initialize_regional_adaptations(self) -> Dict[str, Dict[str, str]]:
        """Initialize regional adaptation rules."""
        return {
            'currency': {
                'dollar': 'rupee',
                'cents': 'paise',
                'USD': 'INR'
            },
            'measurements': {
                'fahrenheit': 'celsius',
                'miles': 'kilometers',
                'feet': 'meters',
                'pounds': 'kilograms'
            },
            'time_format': {
                '12_hour': '24_hour',
                'AM/PM': '24_hour_format'
            },
            'cultural_references': {
                'thanksgiving': 'diwali',
                'christmas_dinner': 'festival_feast',
                'baseball': 'cricket'
            }
        }
    
    def _generate_cache_key(
        self, 
        text: str, 
        source_language: LanguageCode, 
        target_language: LanguageCode
    ) -> str:
        """Generate cache key for translation."""
        content = f"{text}|{source_language.value}|{target_language.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_empty_result(
        self, 
        source_language: LanguageCode, 
        target_language: LanguageCode
    ) -> TranslationResult:
        """Create result for empty input."""
        return TranslationResult(
            translated_text="",
            source_language=source_language,
            target_language=target_language,
            quality_score=1.0,
            semantic_preservation=SemanticPreservation.FULL,
            cultural_context=CulturalContext.PRESERVED,
            confidence=1.0,
            processing_time=0.0,
            alternative_translations=[],
            cultural_adaptations=[]
        )
    
    def _create_identity_result(
        self, 
        text: str, 
        language: LanguageCode, 
        start_time: float
    ) -> TranslationResult:
        """Create result for same source and target language."""
        processing_time = asyncio.get_event_loop().time() - start_time
        return TranslationResult(
            translated_text=text,
            source_language=language,
            target_language=language,
            quality_score=1.0,
            semantic_preservation=SemanticPreservation.FULL,
            cultural_context=CulturalContext.PRESERVED,
            confidence=1.0,
            processing_time=processing_time,
            alternative_translations=[],
            cultural_adaptations=[]
        )
    
    def _create_error_result(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        error: str
    ) -> TranslationResult:
        """Create result for translation error."""
        return TranslationResult(
            translated_text=f"[Translation Error: {error}]",
            source_language=source_language,
            target_language=target_language,
            quality_score=0.0,
            semantic_preservation=SemanticPreservation.LOST,
            cultural_context=CulturalContext.LOST,
            confidence=0.0,
            processing_time=0.0,
            alternative_translations=[],
            cultural_adaptations=[]
        )
    
    async def _analyze_semantics(
        self, 
        text: str, 
        language: LanguageCode
    ) -> SemanticAnalysis:
        """Analyze semantic content of text."""
        try:
            # Extract key concepts (simplified implementation)
            key_concepts = []
            words = text.lower().split()
            
            # Identify important nouns and verbs (simplified)
            important_words = [word for word in words if len(word) > 3]
            key_concepts = important_words[:5]  # Top 5 concepts
            
            # Determine sentiment (simplified)
            positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'wonderful']
            negative_words = ['bad', 'terrible', 'hate', 'sad', 'awful', 'horrible']
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Determine formality level
            formal_indicators = ['please', 'kindly', 'respectfully', 'sir', 'madam']
            informal_indicators = ['hey', 'hi', 'yeah', 'ok', 'cool']
            
            formal_count = sum(1 for word in words if word in formal_indicators)
            informal_count = sum(1 for word in words if word in informal_indicators)
            
            if formal_count > informal_count:
                formality_level = 'formal'
            elif informal_count > formal_count:
                formality_level = 'informal'
            else:
                formality_level = 'neutral'
            
            # Detect cultural references
            cultural_references = []
            cultural_terms = self.cultural_terms.get(language, [])
            for term_data in cultural_terms:
                if term_data['term'].lower() in text.lower():
                    cultural_references.append(term_data['term'])
            
            # Detect idiomatic expressions (simplified)
            idiomatic_expressions = []
            common_idioms = {
                'break a leg': 'good luck',
                'piece of cake': 'easy task',
                'hit the nail on the head': 'exactly right'
            }
            
            for idiom in common_idioms:
                if idiom in text.lower():
                    idiomatic_expressions.append(idiom)
            
            return SemanticAnalysis(
                key_concepts=key_concepts,
                sentiment=sentiment,
                formality_level=formality_level,
                cultural_references=cultural_references,
                idiomatic_expressions=idiomatic_expressions
            )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return SemanticAnalysis(
                key_concepts=[],
                sentiment='neutral',
                formality_level='neutral',
                cultural_references=[],
                idiomatic_expressions=[]
            )
    
    async def _perform_translation(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> str:
        """Perform the actual neural machine translation."""
        try:
            # This is a simplified implementation
            # In a real system, this would use a neural translation model
            # like Google Translate API, Azure Translator, or a local model
            
            # For demonstration, we'll use a simple rule-based approach
            # with some common translations
            
            simple_translations = {
                (LanguageCode.ENGLISH_IN, LanguageCode.HINDI): {
                    'hello': 'नमस्ते',
                    'goodbye': 'अलविदा',
                    'thank you': 'धन्यवाद',
                    'please': 'कृपया',
                    'yes': 'हाँ',
                    'no': 'नहीं',
                    'water': 'पानी',
                    'food': 'खाना',
                    'house': 'घर',
                    'family': 'परिवार'
                },
                (LanguageCode.HINDI, LanguageCode.ENGLISH_IN): {
                    'नमस्ते': 'hello',
                    'अलविदा': 'goodbye',
                    'धन्यवाद': 'thank you',
                    'कृपया': 'please',
                    'हाँ': 'yes',
                    'नहीं': 'no',
                    'पानी': 'water',
                    'खाना': 'food',
                    'घर': 'house',
                    'परिवार': 'family'
                },
                (LanguageCode.ENGLISH_IN, LanguageCode.TAMIL): {
                    'hello': 'வணக்கம்',
                    'goodbye': 'போய் வருகிறேன்',
                    'thank you': 'நன்றி',
                    'please': 'தயவுசெய்து',
                    'yes': 'ஆம்',
                    'no': 'இல்லை'
                }
            }
            
            # Get translation dictionary for language pair
            translation_dict = simple_translations.get(
                (source_language, target_language), {}
            )
            
            # Try direct translation first
            text_lower = text.lower().strip()
            if text_lower in translation_dict:
                return translation_dict[text_lower]
            
            # For longer texts, try word-by-word translation
            words = text.split()
            translated_words = []
            
            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                if word_lower in translation_dict:
                    translated_words.append(translation_dict[word_lower])
                else:
                    # Keep original word if no translation found
                    translated_words.append(word)
            
            if translated_words:
                return ' '.join(translated_words)
            
            # Fallback: return original text with note
            return f"[Translation needed: {text}]"
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return f"[Translation error: {str(e)}]"
    
    async def _apply_cultural_adaptations(
        self,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        original_text: str
    ) -> Tuple[str, List[str]]:
        """Apply cultural context adaptations to translation."""
        try:
            adaptations = []
            adapted_text = translated_text
            
            # Apply regional adaptations
            for category, adaptations_dict in self.regional_adaptations.items():
                for source_term, target_term in adaptations_dict.items():
                    if source_term.lower() in adapted_text.lower():
                        adapted_text = re.sub(
                            re.escape(source_term), 
                            target_term, 
                            adapted_text, 
                            flags=re.IGNORECASE
                        )
                        adaptations.append(f"Adapted {source_term} to {target_term}")
            
            # Handle cultural greetings
            if source_language == LanguageCode.ENGLISH_IN and target_language == LanguageCode.HINDI:
                if 'hello' in original_text.lower():
                    adapted_text = adapted_text.replace('hello', 'नमस्ते')
                    adaptations.append("Adapted greeting to culturally appropriate form")
            
            # Handle formal/informal address
            if 'you' in original_text.lower() and target_language == LanguageCode.HINDI:
                # In Hindi, formal 'you' is 'आप' and informal is 'तुम'
                # Default to formal for respectful communication
                adapted_text = adapted_text.replace('you', 'आप')
                adaptations.append("Used formal address for respectful communication")
            
            return adapted_text, adaptations
            
        except Exception as e:
            logger.error(f"Error applying cultural adaptations: {e}")
            return translated_text, []
    
    async def _generate_alternatives(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        primary_translation: str
    ) -> List[str]:
        """Generate alternative translations."""
        try:
            alternatives = []
            
            # Generate variations based on formality
            if 'please' in text.lower():
                formal_alt = primary_translation.replace('please', 'kindly')
                if formal_alt != primary_translation:
                    alternatives.append(formal_alt)
            
            # Generate variations based on regional preferences
            if target_language == LanguageCode.HINDI:
                # Add Devanagari script alternatives
                if 'hello' in primary_translation.lower():
                    alternatives.extend(['नमस्कार', 'आदाब'])
            
            # Limit to top 3 alternatives
            return alternatives[:3]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    async def _calculate_quality_score(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> float:
        """Calculate translation quality score."""
        try:
            score = 0.0
            
            # Basic checks
            if not translated_text or '[Translation' in translated_text:
                return 0.0
            
            # Length similarity (translations should be reasonably similar in length)
            length_ratio = min(len(translated_text), len(source_text)) / max(len(translated_text), len(source_text))
            length_score = length_ratio * 0.3
            
            # Character diversity (good translations have diverse characters)
            unique_chars = len(set(translated_text.lower()))
            diversity_score = min(unique_chars / 20, 1.0) * 0.2
            
            # Word count similarity
            source_words = len(source_text.split())
            translated_words = len(translated_text.split())
            word_ratio = min(translated_words, source_words) / max(translated_words, source_words)
            word_score = word_ratio * 0.3
            
            # Completeness (no error messages)
            completeness_score = 0.2 if '[Translation' not in translated_text else 0.0
            
            score = length_score + diversity_score + word_score + completeness_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    async def _assess_semantic_preservation(
        self,
        source_semantics: SemanticAnalysis,
        translated_text: str,
        target_language: LanguageCode
    ) -> SemanticPreservation:
        """Assess how well semantic meaning is preserved."""
        try:
            # Analyze translated text semantics
            target_semantics = await self._analyze_semantics(translated_text, target_language)
            
            # Compare key concepts
            common_concepts = set(source_semantics.key_concepts) & set(target_semantics.key_concepts)
            concept_preservation = len(common_concepts) / max(len(source_semantics.key_concepts), 1)
            
            # Compare sentiment
            sentiment_match = source_semantics.sentiment == target_semantics.sentiment
            
            # Compare formality
            formality_match = source_semantics.formality_level == target_semantics.formality_level
            
            # Calculate overall preservation score
            preservation_score = (
                concept_preservation * 0.5 +
                (1.0 if sentiment_match else 0.0) * 0.3 +
                (1.0 if formality_match else 0.0) * 0.2
            )
            
            if preservation_score >= 0.8:
                return SemanticPreservation.FULL
            elif preservation_score >= 0.6:
                return SemanticPreservation.PARTIAL
            elif preservation_score >= 0.3:
                return SemanticPreservation.MINIMAL
            else:
                return SemanticPreservation.LOST
                
        except Exception as e:
            logger.error(f"Error assessing semantic preservation: {e}")
            return SemanticPreservation.PARTIAL
    
    async def _assess_cultural_context(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> CulturalContext:
        """Assess cultural context preservation."""
        try:
            # Detect cultural terms in source
            source_cultural_terms = await self.detect_cultural_terms(source_text, source_language)
            
            # Check if cultural terms are appropriately handled
            cultural_preservation_score = 0.0
            
            if not source_cultural_terms:
                # No cultural terms to preserve
                return CulturalContext.PRESERVED
            
            preserved_count = 0
            for term in source_cultural_terms:
                # Check if term is translated or preserved appropriately
                if (term.term.lower() in translated_text.lower() or 
                    any(variant.lower() in translated_text.lower() for variant in term.regional_variants)):
                    preserved_count += 1
            
            cultural_preservation_score = preserved_count / len(source_cultural_terms)
            
            if cultural_preservation_score >= 0.8:
                return CulturalContext.PRESERVED
            elif cultural_preservation_score >= 0.5:
                return CulturalContext.ADAPTED
            elif cultural_preservation_score >= 0.2:
                return CulturalContext.PARTIALLY_LOST
            else:
                return CulturalContext.LOST
                
        except Exception as e:
            logger.error(f"Error assessing cultural context: {e}")
            return CulturalContext.ADAPTED
    
    def _calculate_confidence(
        self,
        quality_score: float,
        semantic_preservation: SemanticPreservation,
        cultural_context: CulturalContext
    ) -> float:
        """Calculate overall confidence score."""
        try:
            # Map semantic preservation to score
            semantic_scores = {
                SemanticPreservation.FULL: 1.0,
                SemanticPreservation.PARTIAL: 0.7,
                SemanticPreservation.MINIMAL: 0.4,
                SemanticPreservation.LOST: 0.0
            }
            
            # Map cultural context to score
            cultural_scores = {
                CulturalContext.PRESERVED: 1.0,
                CulturalContext.ADAPTED: 0.8,
                CulturalContext.PARTIALLY_LOST: 0.5,
                CulturalContext.LOST: 0.2
            }
            
            semantic_score = semantic_scores.get(semantic_preservation, 0.5)
            cultural_score = cultural_scores.get(cultural_context, 0.5)
            
            # Weighted average
            confidence = (
                quality_score * 0.4 +
                semantic_score * 0.4 +
                cultural_score * 0.2
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _update_translation_stats(self, result: TranslationResult):
        """Update translation statistics."""
        try:
            self.stats['total_translations'] += 1
            
            # Update quality distribution
            quality_level = self._classify_quality(result.quality_score)
            self.stats['quality_distribution'][quality_level.value] += 1
            
            # Update language pairs
            pair_key = f"{result.source_language.value}->{result.target_language.value}"
            self.stats['language_pairs'][pair_key] = self.stats['language_pairs'].get(pair_key, 0) + 1
            
            # Update average processing time
            total_time = self.stats['average_processing_time'] * (self.stats['total_translations'] - 1)
            self.stats['average_processing_time'] = (total_time + result.processing_time) / self.stats['total_translations']
            
            # Update semantic preservation rate
            semantic_score = 1.0 if result.semantic_preservation == SemanticPreservation.FULL else 0.5
            total_semantic = self.stats['semantic_preservation_rate'] * (self.stats['total_translations'] - 1)
            self.stats['semantic_preservation_rate'] = (total_semantic + semantic_score) / self.stats['total_translations']
            
            # Update cultural adaptation rate
            cultural_score = 1.0 if result.cultural_context in [CulturalContext.PRESERVED, CulturalContext.ADAPTED] else 0.0
            total_cultural = self.stats['cultural_adaptation_rate'] * (self.stats['total_translations'] - 1)
            self.stats['cultural_adaptation_rate'] = (total_cultural + cultural_score) / self.stats['total_translations']
            
        except Exception as e:
            logger.error(f"Error updating translation stats: {e}")
    
    def _cache_translation_result(self, cache_key: str, result: TranslationResult):
        """Cache translation result."""
        try:
            # Implement LRU cache behavior
            if len(self.translation_cache) >= self.model_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.translation_cache))
                del self.translation_cache[oldest_key]
            
            self.translation_cache[cache_key] = result
            
        except Exception as e:
            logger.error(f"Error caching translation result: {e}")
    
    def _classify_quality(self, quality_score: float) -> TranslationQuality:
        """Classify quality score into quality level."""
        if quality_score >= 0.8:
            return TranslationQuality.EXCELLENT
        elif quality_score >= 0.6:
            return TranslationQuality.GOOD
        elif quality_score >= 0.4:
            return TranslationQuality.FAIR
        else:
            return TranslationQuality.POOR
    
    async def _detect_idioms(self, text: str, language: LanguageCode) -> List[Dict[str, Any]]:
        """Detect idiomatic expressions in text."""
        try:
            idioms = []
            
            # Common English idioms
            english_idioms = {
                'break a leg': {'meaning': 'good luck', 'variants': ['break your leg']},
                'piece of cake': {'meaning': 'very easy', 'variants': ['easy as pie']},
                'hit the nail on the head': {'meaning': 'exactly right', 'variants': []},
                'spill the beans': {'meaning': 'reveal a secret', 'variants': []},
                'cost an arm and a leg': {'meaning': 'very expensive', 'variants': []}
            }
            
            # Common Hindi idioms
            hindi_idioms = {
                'आँखों का तारा': {'meaning': 'beloved person', 'variants': []},
                'हाथ पर हाथ रखकर बैठना': {'meaning': 'to sit idle', 'variants': []},
                'पहाड़ टूट पड़ना': {'meaning': 'great calamity', 'variants': []}
            }
            
            idiom_dict = {}
            if language == LanguageCode.ENGLISH_IN:
                idiom_dict = english_idioms
            elif language == LanguageCode.HINDI:
                idiom_dict = hindi_idioms
            
            text_lower = text.lower()
            for idiom, data in idiom_dict.items():
                if idiom.lower() in text_lower:
                    idioms.append({
                        'expression': idiom,
                        'meaning': data['meaning'],
                        'variants': data['variants']
                    })
            
            return idioms
            
        except Exception as e:
            logger.error(f"Error detecting idioms: {e}")
            return []
    
    async def _assess_fluency(self, text: str, language: LanguageCode) -> float:
        """Assess fluency of translated text."""
        try:
            # Simple fluency assessment based on text characteristics
            score = 0.0
            
            # Check for basic sentence structure
            sentences = text.split('.')
            if len(sentences) > 0:
                score += 0.2
            
            # Check for reasonable word length distribution
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if 3 <= avg_word_length <= 8:  # Reasonable average word length
                    score += 0.3
            
            # Check for punctuation usage
            if any(punct in text for punct in '.,!?;:'):
                score += 0.2
            
            # Check for capitalization
            if text and text[0].isupper():
                score += 0.1
            
            # Check for no error markers
            if '[Translation' not in text:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing fluency: {e}")
            return 0.5
    
    async def _assess_adequacy(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> float:
        """Assess adequacy of translation (completeness)."""
        try:
            # Simple adequacy assessment
            score = 0.0
            
            # Check if translation is not empty
            if translated_text.strip():
                score += 0.3
            
            # Check if translation is not just an error message
            if '[Translation' not in translated_text:
                score += 0.3
            
            # Check length similarity (adequate translations should have reasonable length)
            if source_text and translated_text:
                length_ratio = len(translated_text) / len(source_text)
                if 0.5 <= length_ratio <= 2.0:  # Reasonable length ratio
                    score += 0.4
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing adequacy: {e}")
            return 0.5
    
    def _generate_quality_recommendations(
        self,
        quality_score: float,
        semantic_preservation: SemanticPreservation,
        cultural_context: CulturalContext
    ) -> List[str]:
        """Generate recommendations for improving translation quality."""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Consider using a more advanced translation model")
            recommendations.append("Review source text for clarity and complexity")
        
        if semantic_preservation in [SemanticPreservation.MINIMAL, SemanticPreservation.LOST]:
            recommendations.append("Focus on preserving key concepts and meaning")
            recommendations.append("Consider breaking complex sentences into simpler parts")
        
        if cultural_context in [CulturalContext.PARTIALLY_LOST, CulturalContext.LOST]:
            recommendations.append("Add cultural context adaptation")
            recommendations.append("Consider regional variations and cultural nuances")
        
        if not recommendations:
            recommendations.append("Translation quality is good - no specific improvements needed")
        
        return recommendations
    
    async def _generate_contextual_alternatives(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        context: str
    ) -> List[str]:
        """Generate contextual alternative translations."""
        try:
            alternatives = []
            
            # Analyze context for formality level
            if 'formal' in context.lower() or 'business' in context.lower():
                # Generate more formal alternatives
                formal_translation = await self._perform_translation(
                    f"Respectfully, {text}", source_language, target_language
                )
                if formal_translation != text:
                    alternatives.append(formal_translation)
            
            if 'casual' in context.lower() or 'friend' in context.lower():
                # Generate more casual alternatives
                casual_translation = await self._perform_translation(
                    f"Hey, {text}", source_language, target_language
                )
                if casual_translation != text:
                    alternatives.append(casual_translation)
            
            return alternatives[:3]  # Limit to 3 alternatives
            
        except Exception as e:
            logger.error(f"Error generating contextual alternatives: {e}")
            return []
    
    async def _detect_translation_issues(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> List[str]:
        """Detect potential issues in translation."""
        issues = []
        
        try:
            # Check for untranslated text
            if '[Translation' in translated_text:
                issues.append("Contains untranslated segments")
            
            # Check for length discrepancy
            length_ratio = len(translated_text) / max(len(source_text), 1)
            if length_ratio < 0.3:
                issues.append("Translation appears too short")
            elif length_ratio > 3.0:
                issues.append("Translation appears too long")
            
            # Check for missing punctuation
            source_punct = set(char for char in source_text if char in '.,!?;:')
            trans_punct = set(char for char in translated_text if char in '.,!?;:')
            if source_punct and not trans_punct:
                issues.append("Missing punctuation in translation")
            
            # Check for cultural terms that might need special handling
            cultural_terms = await self.detect_cultural_terms(source_text, source_language)
            if cultural_terms:
                for term in cultural_terms:
                    if term.term.lower() not in translated_text.lower():
                        issues.append(f"Cultural term '{term.term}' may need special handling")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error detecting translation issues: {e}")
            return ["Error analyzing translation for issues"]
    
    async def _generate_improvement_suggestions(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        issues: List[str]
    ) -> List[str]:
        """Generate suggestions for improving translation."""
        suggestions = []
        
        try:
            for issue in issues:
                if "untranslated segments" in issue:
                    suggestions.append("Use a more comprehensive translation model")
                elif "too short" in issue:
                    suggestions.append("Ensure all source content is translated")
                elif "too long" in issue:
                    suggestions.append("Review for unnecessary additions or repetitions")
                elif "missing punctuation" in issue:
                    suggestions.append("Preserve punctuation from source text")
                elif "cultural term" in issue:
                    suggestions.append("Consider cultural adaptation or explanation")
            
            if not suggestions:
                suggestions.append("Translation appears to be of good quality")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return ["Unable to generate improvement suggestions"]
    
    async def _get_cultural_notes(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> List[str]:
        """Get cultural notes for translation."""
        notes = []
        
        try:
            # Detect cultural terms and provide notes
            cultural_terms = await self.detect_cultural_terms(text, source_language)
            
            for term in cultural_terms:
                if term.translation_notes:
                    notes.append(f"{term.term}: {term.translation_notes}")
            
            # Add general cultural notes based on language pair
            if source_language == LanguageCode.ENGLISH_IN and target_language == LanguageCode.HINDI:
                notes.append("Consider using formal address (आप) for respectful communication")
                notes.append("Hindi speakers may prefer indirect communication styles")
            
            return notes
            
        except Exception as e:
            logger.error(f"Error getting cultural notes: {e}")
            return []