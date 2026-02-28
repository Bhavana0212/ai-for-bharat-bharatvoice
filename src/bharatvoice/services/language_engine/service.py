<<<<<<< HEAD
"""
Language Engine Service implementation for BharatVoice Assistant.

This module provides the main language processing service that integrates
multilingual ASR, code-switching detection, translation, and language adaptation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from bharatvoice.core.interfaces import LanguageEngine
from bharatvoice.core.models import (
    AudioBuffer,
    LanguageCode,
    RecognitionResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.asr_engine import (
    MultilingualASREngine,
    create_multilingual_asr_engine,
)
from bharatvoice.services.language_engine.translation_engine import (
    TranslationEngine,
    TranslationResult,
)

logger = logging.getLogger(__name__)


class LanguageEngineService(LanguageEngine):
    """
    Main language engine service that coordinates multilingual processing.
    
    This service provides a unified interface for:
    - Multilingual speech recognition
    - Language detection and switching
    - Code-switching detection
    - Translation between Indian languages
    - Regional accent adaptation
    """
    
    def __init__(
        self,
        asr_model_size: str = "base",
        device: str = "cpu",
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_language_detection: bool = True
    ):
        """
        Initialize the language engine service.
        
        Args:
            asr_model_size: Whisper model size for ASR
            device: Device to run inference on
            enable_caching: Whether to enable result caching
            cache_size: Maximum number of cached results
            enable_language_detection: Whether to enable automatic language detection
        """
        self.device = device
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize ASR engine with proper error handling
        try:
            self.asr_engine = create_multilingual_asr_engine(
                model_size=asr_model_size,
                device=device,
                enable_language_detection=enable_language_detection
            )
            logger.info("ASR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine: {e}")
            # Create a fallback ASR engine with minimal configuration
            try:
                self.asr_engine = create_multilingual_asr_engine(
                    model_size="tiny",
                    device="cpu",
                    enable_language_detection=False
                )
                logger.info("Fallback ASR engine initialized with minimal configuration")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback ASR engine: {fallback_e}")
                raise RuntimeError("Could not initialize any ASR engine configuration") from fallback_e
        
        # Initialize translation engine with proper error handling
        try:
            self.translation_engine = TranslationEngine(
                model_cache_size=cache_size,
                enable_cultural_adaptation=True,
                enable_semantic_validation=True,
                quality_threshold=0.7
            )
            logger.info("Translation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation engine: {e}")
            # Create a fallback translation engine with minimal configuration
            try:
                self.translation_engine = TranslationEngine(
                    model_cache_size=100,
                    enable_cultural_adaptation=False,
                    enable_semantic_validation=False,
                    quality_threshold=0.5
                )
                logger.info("Fallback translation engine initialized with minimal configuration")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback translation engine: {fallback_e}")
                raise RuntimeError("Could not initialize any translation engine configuration") from fallback_e
        
        # Initialize caching with proper structure if enabled
        if enable_caching:
            self.recognition_cache = {}
            self.translation_cache = {}
            self.accent_adaptation_cache = {}
            logger.info(f"Caching enabled with size limit: {cache_size}")
        else:
            self.recognition_cache = None
            self.translation_cache = None
            self.accent_adaptation_cache = None
            logger.info("Caching disabled")
        
        # Service statistics
        self.stats = {
            'total_recognitions': 0,
            'total_translations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_recognition_time': 0.0,
            'language_distribution': {},
            'code_switching_detections': 0
        }
        
        logger.info("LanguageEngineService initialized successfully")
    
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input with caching support.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Generate cache key if caching is enabled
            cache_key = None
            if self.enable_caching:
                cache_key = self._generate_audio_cache_key(audio)
                
                # Check cache first
                if cache_key in self.recognition_cache:
                    self.stats['cache_hits'] += 1
                    logger.debug("Recognition result found in cache")
                    return self.recognition_cache[cache_key]
                else:
                    self.stats['cache_misses'] += 1
            
            # Perform speech recognition
            result = await self.asr_engine.recognize_speech(audio)
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_recognition_stats(result, processing_time)
            
            # Cache result if caching is enabled
            if self.enable_caching and cache_key:
                self._cache_recognition_result(cache_key, result)
            
            logger.info(
                f"Speech recognition completed: '{result.transcribed_text[:50]}...' "
                f"({result.detected_language}, confidence={result.confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            # Return empty result on error
            return RecognitionResult(
                transcribed_text="",
                confidence=0.0,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.0
            )
    
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        try:
            if not text.strip():
                return []
            
            # Use ASR engine's enhanced code-switching detection
            code_switches = await self.asr_engine.detect_code_switching(text)
            
            # Update statistics
            if code_switches:
                self.stats['code_switching_detections'] += 1
            
            logger.debug(f"Detected {len(code_switches)} code-switching points in text")
            return code_switches
            
        except Exception as e:
            logger.error(f"Error in code-switching detection: {e}")
            return []
    
    async def get_detailed_code_switching_analysis(
        self, 
        text: str, 
        context_language: Optional[LanguageCode] = None
    ) -> Dict[str, any]:
        """
        Get detailed code-switching analysis with enhanced features.
        
        Args:
            text: Input text to analyze
            context_language: Context language for better detection
            
        Returns:
            Detailed code-switching analysis result
        """
        try:
            if not text.strip():
                return {
                    "segments": [],
                    "switch_points": [],
                    "dominant_language": LanguageCode.ENGLISH_IN.value,
                    "switching_frequency": 0.0,
                    "confidence": 1.0,
                    "processing_time": 0.0
                }
            
            # Get detailed analysis from ASR engine
            result = await self.asr_engine.get_detailed_code_switching_analysis(
                text, context_language
            )
            
            # Convert to serializable format
            return {
                "segments": [
                    {
                        "text": seg.text,
                        "language": seg.language.value,
                        "start_pos": seg.start_pos,
                        "end_pos": seg.end_pos,
                        "confidence": seg.confidence,
                        "word_boundaries": seg.word_boundaries
                    }
                    for seg in result.segments
                ],
                "switch_points": [
                    {
                        "position": sp.position,
                        "from_language": sp.from_language.value,
                        "to_language": sp.to_language.value,
                        "confidence": sp.confidence
                    }
                    for sp in result.switch_points
                ],
                "dominant_language": result.dominant_language.value,
                "switching_frequency": result.switching_frequency,
                "confidence": result.confidence,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in detailed code-switching analysis: {e}")
            return {
                "segments": [],
                "switch_points": [],
                "dominant_language": LanguageCode.ENGLISH_IN.value,
                "switching_frequency": 0.0,
                "confidence": 0.0,
                "processing_time": 0.0
            }
    
    async def get_language_transition_suggestions(
        self, 
        from_language: LanguageCode, 
        to_language: LanguageCode
    ) -> Dict[str, List[str]]:
        """
        Get suggestions for smooth language transitions.
        
        Args:
            from_language: Source language
            to_language: Target language
            
        Returns:
            Dictionary with transition suggestions
        """
        try:
            return await self.asr_engine.get_language_transition_suggestions(
                from_language, to_language
            )
        except Exception as e:
            logger.error(f"Error getting transition suggestions: {e}")
            return {
                'connectors': ['that is', 'I mean'],
                'fillers': ['okay', 'so', 'well'],
                'markers': []
            }
    
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> TranslationResult:
        """
        Translate text between languages with quality assessment.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            Translation result with quality metrics
        """
        try:
            if not text.strip():
                return TranslationResult(
                    translated_text="",
                    source_language=source_lang,
                    target_language=target_lang,
                    quality_score=1.0,
                    semantic_preservation="full",
                    cultural_context="preserved",
                    confidence=1.0,
                    processing_time=0.0,
                    alternative_translations=[],
                    cultural_adaptations=[]
                )
            
            # Use the translation engine for comprehensive translation
            result = await self.translation_engine.translate(
                text, source_lang, target_lang,
                preserve_cultural_context, validate_semantics
            )
            
            # Update statistics
            self.stats['total_translations'] += 1
            
            logger.info(
                f"Translation completed: {source_lang} -> {target_lang}, "
                f"quality={result.quality_score:.3f}, confidence={result.confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            # Return error result
            return TranslationResult(
                translated_text=f"[Translation Error: {str(e)}]",
                source_language=source_lang,
                target_language=target_lang,
                quality_score=0.0,
                semantic_preservation="lost",
                cultural_context="lost",
                confidence=0.0,
                processing_time=0.0,
                alternative_translations=[],
                cultural_adaptations=[]
            )
    
    async def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the primary language of input text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Detected language code
        """
        try:
            if not text.strip():
                return LanguageCode.ENGLISH_IN
            
            detected_language = await self.asr_engine.detect_language(text)
            
            logger.debug(f"Language detected: {detected_language} for text: '{text[:50]}...'")
            return detected_language
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return LanguageCode.ENGLISH_IN
    
    async def adapt_to_regional_accent(
        self, 
        model_id: str, 
        accent_data: Dict[str, any]
    ) -> str:
        """
        Adapt language model to regional accent.
        
        Args:
            model_id: Base model identifier
            accent_data: Regional accent adaptation data
            
        Returns:
            Adapted model identifier
        """
        try:
            # Check cache first
            cache_key = f"{model_id}_{hash(str(sorted(accent_data.items())))}"
            if self.accent_adaptation_cache and cache_key in self.accent_adaptation_cache:
                logger.debug(f"Using cached accent adaptation: {cache_key}")
                return self.accent_adaptation_cache[cache_key]
            
            # Extract accent information
            region = accent_data.get("region", "standard")
            language = accent_data.get("language", "en")
            accent_samples = accent_data.get("samples", [])
            
            # Use ASR engine's accent adaptation
            adapted_model_id = await self.asr_engine.adapt_to_regional_accent(
                model_id, accent_data
            )
            
            # Cache the result
            if self.accent_adaptation_cache:
                if len(self.accent_adaptation_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.accent_adaptation_cache))
                    del self.accent_adaptation_cache[oldest_key]
                
                self.accent_adaptation_cache[cache_key] = adapted_model_id
            
            logger.info(f"Model adapted for regional accent: {model_id} -> {adapted_model_id} (region: {region})")
            return adapted_model_id
            
        except Exception as e:
            logger.error(f"Error in accent adaptation: {e}")
            return model_id
    
    async def recognize_with_language_hint(
        self, 
        audio: AudioBuffer, 
        language_hint: LanguageCode
    ) -> RecognitionResult:
        """
        Recognize speech with a language hint for improved accuracy.
        
        Args:
            audio: Audio buffer containing speech
            language_hint: Suggested language for recognition
            
        Returns:
            Speech recognition result
        """
        try:
            # For now, use the standard recognition
            # In production, this could bias the recognition towards the hinted language
            result = await self.recognize_speech(audio)
            
            # If confidence is low and language doesn't match hint, 
            # we could retry with language-specific models
            if (result.confidence < 0.7 and 
                result.detected_language != language_hint):
                
                logger.info(
                    f"Low confidence with language mismatch. "
                    f"Detected: {result.detected_language}, Hint: {language_hint}"
                )
                
                # Could implement language-specific retry logic here
            
            return result
            
        except Exception as e:
            logger.error(f"Error in recognition with language hint: {e}")
            return await self.recognize_speech(audio)
    
    async def batch_recognize_speech(
        self, 
        audio_buffers: List[AudioBuffer]
    ) -> List[RecognitionResult]:
        """
        Recognize speech from multiple audio buffers in batch.
        
        Args:
            audio_buffers: List of audio buffers to process
            
        Returns:
            List of recognition results
        """
        try:
            if not audio_buffers:
                return []
            
            # Process all audio buffers concurrently
            tasks = [self.recognize_speech(audio) for audio in audio_buffers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing audio buffer {i}: {result}")
                    # Create empty result for failed processing
                    processed_results.append(RecognitionResult(
                        transcribed_text="",
                        confidence=0.0,
                        detected_language=LanguageCode.ENGLISH_IN,
                        code_switching_points=[],
                        alternative_transcriptions=[],
                        processing_time=0.0
                    ))
                else:
                    processed_results.append(result)
            
            logger.info(f"Batch recognition completed for {len(audio_buffers)} audio buffers")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch speech recognition: {e}")
            return []
    
    async def get_language_confidence_scores(
        self, 
        audio: AudioBuffer
    ) -> Dict[LanguageCode, float]:
        """
        Get confidence scores for all supported languages.
        
        Args:
            audio: Audio buffer to analyze
            
        Returns:
            Dictionary mapping language codes to confidence scores
        """
        try:
            # Perform recognition to get primary result
            result = await self.recognize_speech(audio)
            
            # Initialize confidence scores
            confidence_scores = {}
            
            # Set primary language confidence
            confidence_scores[result.detected_language] = result.confidence
            
            # Add alternative language confidences
            for alt in result.alternative_transcriptions:
                confidence_scores[alt.language] = alt.confidence
            
            # Fill in remaining supported languages with low confidence
            supported_languages = self.asr_engine.get_supported_languages()
            for lang in supported_languages:
                if lang not in confidence_scores:
                    confidence_scores[lang] = 0.1  # Low default confidence
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error getting language confidence scores: {e}")
            return {}
    
    def _generate_audio_cache_key(self, audio: AudioBuffer) -> str:
        """
        Generate cache key for audio buffer.
        
        Args:
            audio: Audio buffer
            
        Returns:
            Cache key string
        """
        # Create hash based on audio characteristics
        import hashlib
        
        # Use sample of audio data for hashing (to avoid hashing large arrays)
        sample_size = min(1000, len(audio.data))
        audio_sample = audio.data[:sample_size]
        
        key_data = f"{audio.sample_rate}:{audio.channels}:{audio.duration}:{hash(tuple(audio_sample))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_recognition_result(self, cache_key: str, result: RecognitionResult):
        """
        Cache recognition result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Recognition result to cache
        """
        if len(self.recognition_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.recognition_cache))
            del self.recognition_cache[oldest_key]
        
        self.recognition_cache[cache_key] = result
    
    async def batch_translate_texts(
        self,
        texts: List[str],
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            List of translation results
        """
        try:
            return await self.translation_engine.batch_translate(
                texts, source_lang, target_lang,
                preserve_cultural_context, validate_semantics
            )
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            return []
    
    async def validate_translation_quality(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> Dict[str, any]:
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
            return await self.translation_engine.validate_translation_quality(
                source_text, translated_text, source_language, target_language
            )
        except Exception as e:
            logger.error(f"Error validating translation quality: {e}")
            return {
                'overall_quality_score': 0.0,
                'quality_level': 'poor',
                'error': str(e)
            }
    
    async def get_translation_suggestions(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        context: Optional[str] = None
    ) -> Dict[str, any]:
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
            return await self.translation_engine.get_translation_suggestions(
                text, source_language, target_language, context
            )
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
    
    async def detect_cultural_terms(
        self,
        text: str,
        language: LanguageCode
    ) -> List[Dict[str, any]]:
        """
        Detect cultural terms and expressions in text.
        
        Args:
            text: Text to analyze
            language: Language of the text
            
        Returns:
            List of detected cultural terms
        """
        try:
            cultural_terms = await self.translation_engine.detect_cultural_terms(text, language)
            
            # Convert to serializable format
            return [
                {
                    'term': term.term,
                    'language': term.language.value,
                    'meaning': term.meaning,
                    'cultural_significance': term.cultural_significance,
                    'regional_variants': term.regional_variants,
                    'translation_notes': term.translation_notes
                }
                for term in cultural_terms
            ]
        except Exception as e:
            logger.error(f"Error detecting cultural terms: {e}")
            return []
    
    def get_supported_translation_pairs(self) -> List[Tuple[LanguageCode, LanguageCode]]:
        """
        Get list of supported language pairs for translation.
        
        Returns:
            List of supported (source, target) language pairs
        """
        return self.translation_engine.get_supported_language_pairs()
    
    def get_translation_stats(self) -> Dict[str, any]:
        """
        Get translation engine statistics.
        
        Returns:
            Dictionary with translation statistics
        """
        return self.translation_engine.get_translation_stats()
    
    def clear_translation_caches(self):
        """Clear translation engine caches."""
        self.translation_engine.clear_caches()
        logger.info("Cleared translation engine caches")
    
    def _cache_translation_result(self, cache_key: str, result: str):
        """
        Cache translation result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Translation result to cache
        """
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[cache_key] = result
        """
        Cache translation result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Translation result to cache
        """
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[cache_key] = result
    
    def _update_recognition_stats(self, result: RecognitionResult, processing_time: float):
        """
        Update recognition statistics.
        
        Args:
            result: Recognition result
            processing_time: Processing time in seconds
        """
        self.stats['total_recognitions'] += 1
        
        # Update average processing time
        total = self.stats['total_recognitions']
        current_avg = self.stats['average_recognition_time']
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.stats['average_recognition_time'] = new_avg
        
        # Update language distribution
        lang = result.detected_language.value
        if lang not in self.stats['language_distribution']:
            self.stats['language_distribution'][lang] = 0
        self.stats['language_distribution'][lang] += 1
    
    def get_supported_languages(self) -> List[LanguageCode]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return self.asr_engine.get_supported_languages()
    
    def get_service_stats(self) -> Dict[str, any]:
        """
        Get language engine service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        stats = self.stats.copy()
        
        # Add cache statistics
        if self.enable_caching:
            stats['cache_stats'] = {
                'recognition_cache_size': len(self.recognition_cache) if self.recognition_cache else 0,
                'translation_cache_size': len(self.translation_cache) if self.translation_cache else 0,
                'cache_hit_rate': (
                    self.stats['cache_hits'] / 
                    max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                )
            }
        
        # Add ASR engine info
        stats['asr_engine_info'] = self.asr_engine.get_model_info()
        
        # Add translation engine stats
        stats['translation_engine_stats'] = self.translation_engine.get_translation_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all internal caches."""
        try:
            if self.recognition_cache:
                self.recognition_cache.clear()
                logger.debug("Recognition cache cleared")
            if self.translation_cache:
                self.translation_cache.clear()
                logger.debug("Translation cache cleared")
            if self.accent_adaptation_cache:
                self.accent_adaptation_cache.clear()
                logger.debug("Accent adaptation cache cleared")
            
            # Clear translation engine caches
            if hasattr(self.translation_engine, 'clear_caches'):
                self.translation_engine.clear_caches()
                logger.debug("Translation engine caches cleared")
            
            logger.info("All language engine service caches cleared")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    async def get_engine_health_status(self) -> Dict[str, any]:
        """
        Get detailed health status of all engine components.
        
        Returns:
            Detailed health status information
        """
        try:
            health_status = {
                "overall_status": "healthy",
                "components": {},
                "errors": []
            }
            
            # Check ASR engine health
            try:
                if hasattr(self.asr_engine, 'health_check'):
                    asr_health = await self.asr_engine.health_check()
                    health_status["components"]["asr_engine"] = asr_health
                    if asr_health.get("status") != "healthy":
                        health_status["overall_status"] = "degraded"
                else:
                    health_status["components"]["asr_engine"] = {"status": "unknown"}
            except Exception as e:
                health_status["components"]["asr_engine"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"ASR engine health check failed: {e}")
                health_status["overall_status"] = "unhealthy"
            
            # Check translation engine health
            try:
                if hasattr(self.translation_engine, 'health_check'):
                    translation_health = await self.translation_engine.health_check()
                    health_status["components"]["translation_engine"] = translation_health
                    if translation_health.get("status") != "healthy":
                        health_status["overall_status"] = "degraded"
                else:
                    health_status["components"]["translation_engine"] = {"status": "unknown"}
            except Exception as e:
                health_status["components"]["translation_engine"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"Translation engine health check failed: {e}")
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"
            
            # Check cache status
            if self.enable_caching:
                cache_status = {
                    "recognition_cache_size": len(self.recognition_cache) if self.recognition_cache else 0,
                    "translation_cache_size": len(self.translation_cache) if self.translation_cache else 0,
                    "accent_cache_size": len(self.accent_adaptation_cache) if self.accent_adaptation_cache else 0,
                    "cache_limit": self.cache_size
                }
                health_status["components"]["caching"] = {"status": "enabled", "details": cache_status}
            else:
                health_status["components"]["caching"] = {"status": "disabled"}
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting engine health status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "components": {},
                "errors": [f"Health status check failed: {e}"]
            }
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the language engine service.
        
        Returns:
            Health check result
        """
        try:
            # Check ASR engine health
            asr_health = await self.asr_engine.health_check()
            
            # Check translation engine health
            translation_health = await self.translation_engine.health_check()
            
            # Test basic functionality
            test_audio = AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence at 16kHz
                sample_rate=16000,
                channels=1,
                duration=0.1
            )
            
            # Test recognition
            recognition_result = await self.recognize_speech(test_audio)
            recognition_status = "ok" if recognition_result is not None else "error"
            
            # Test language detection
            lang_detection_result = await self.detect_language("Hello नमस्ते")
            lang_detection_status = "ok" if lang_detection_result is not None else "error"
            
            # Test translation
            translation_result = await self.translate_text(
                "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
            )
            translation_status = "ok" if translation_result.confidence > 0.0 else "error"
            
            overall_status = (
                "healthy" if (
                    asr_health["status"] == "healthy" and
                    translation_health["status"] == "healthy" and
                    recognition_status == "ok" and
                    lang_detection_status == "ok" and
                    translation_status == "ok"
                ) else "unhealthy"
            )
            
            return {
                "status": overall_status,
                "asr_engine": asr_health,
                "translation_engine": translation_health,
                "recognition_test": recognition_status,
                "language_detection_test": lang_detection_status,
                "translation_test": translation_status,
                "service_stats": self.get_service_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_stats": self.get_service_stats()
            }


# Factory function for creating language engine service
def create_language_engine_service(
    asr_model_size: str = "base",
    device: str = "cpu",
    enable_caching: bool = True,
    cache_size: int = 1000,
    enable_language_detection: bool = True
) -> LanguageEngineService:
    """
    Factory function to create a language engine service instance.
    
    Args:
        asr_model_size: Whisper model size for ASR
        device: Device to run inference on
        enable_caching: Whether to enable result caching
        cache_size: Maximum number of cached results
        enable_language_detection: Whether to enable automatic language detection
        
    Returns:
        Configured LanguageEngineService instance
    """
    return LanguageEngineService(
        asr_model_size=asr_model_size,
        device=device,
        enable_caching=enable_caching,
        cache_size=cache_size,
        enable_language_detection=enable_language_detection
=======
"""
Language Engine Service implementation for BharatVoice Assistant.

This module provides the main language processing service that integrates
multilingual ASR, code-switching detection, translation, and language adaptation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from bharatvoice.core.interfaces import LanguageEngine
from bharatvoice.core.models import (
    AudioBuffer,
    LanguageCode,
    RecognitionResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.asr_engine import (
    MultilingualASREngine,
    create_multilingual_asr_engine,
)
from bharatvoice.services.language_engine.translation_engine import (
    TranslationEngine,
    TranslationResult,
)

logger = logging.getLogger(__name__)


class LanguageEngineService(LanguageEngine):
    """
    Main language engine service that coordinates multilingual processing.
    
    This service provides a unified interface for:
    - Multilingual speech recognition
    - Language detection and switching
    - Code-switching detection
    - Translation between Indian languages
    - Regional accent adaptation
    """
    
    def __init__(
        self,
        asr_model_size: str = "base",
        device: str = "cpu",
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_language_detection: bool = True
    ):
        """
        Initialize the language engine service.
        
        Args:
            asr_model_size: Whisper model size for ASR
            device: Device to run inference on
            enable_caching: Whether to enable result caching
            cache_size: Maximum number of cached results
            enable_language_detection: Whether to enable automatic language detection
        """
        self.device = device
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize ASR engine with proper error handling
        try:
            self.asr_engine = create_multilingual_asr_engine(
                model_size=asr_model_size,
                device=device,
                enable_language_detection=enable_language_detection
            )
            logger.info("ASR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine: {e}")
            # Create a fallback ASR engine with minimal configuration
            try:
                self.asr_engine = create_multilingual_asr_engine(
                    model_size="tiny",
                    device="cpu",
                    enable_language_detection=False
                )
                logger.info("Fallback ASR engine initialized with minimal configuration")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback ASR engine: {fallback_e}")
                raise RuntimeError("Could not initialize any ASR engine configuration") from fallback_e
        
        # Initialize translation engine with proper error handling
        try:
            self.translation_engine = TranslationEngine(
                model_cache_size=cache_size,
                enable_cultural_adaptation=True,
                enable_semantic_validation=True,
                quality_threshold=0.7
            )
            logger.info("Translation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize translation engine: {e}")
            # Create a fallback translation engine with minimal configuration
            try:
                self.translation_engine = TranslationEngine(
                    model_cache_size=100,
                    enable_cultural_adaptation=False,
                    enable_semantic_validation=False,
                    quality_threshold=0.5
                )
                logger.info("Fallback translation engine initialized with minimal configuration")
            except Exception as fallback_e:
                logger.error(f"Failed to initialize fallback translation engine: {fallback_e}")
                raise RuntimeError("Could not initialize any translation engine configuration") from fallback_e
        
        # Initialize caching with proper structure if enabled
        if enable_caching:
            self.recognition_cache = {}
            self.translation_cache = {}
            self.accent_adaptation_cache = {}
            logger.info(f"Caching enabled with size limit: {cache_size}")
        else:
            self.recognition_cache = None
            self.translation_cache = None
            self.accent_adaptation_cache = None
            logger.info("Caching disabled")
        
        # Service statistics
        self.stats = {
            'total_recognitions': 0,
            'total_translations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_recognition_time': 0.0,
            'language_distribution': {},
            'code_switching_detections': 0
        }
        
        logger.info("LanguageEngineService initialized successfully")
    
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input with caching support.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Generate cache key if caching is enabled
            cache_key = None
            if self.enable_caching:
                cache_key = self._generate_audio_cache_key(audio)
                
                # Check cache first
                if cache_key in self.recognition_cache:
                    self.stats['cache_hits'] += 1
                    logger.debug("Recognition result found in cache")
                    return self.recognition_cache[cache_key]
                else:
                    self.stats['cache_misses'] += 1
            
            # Perform speech recognition
            result = await self.asr_engine.recognize_speech(audio)
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_recognition_stats(result, processing_time)
            
            # Cache result if caching is enabled
            if self.enable_caching and cache_key:
                self._cache_recognition_result(cache_key, result)
            
            logger.info(
                f"Speech recognition completed: '{result.transcribed_text[:50]}...' "
                f"({result.detected_language}, confidence={result.confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            # Return empty result on error
            return RecognitionResult(
                transcribed_text="",
                confidence=0.0,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.0
            )
    
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        try:
            if not text.strip():
                return []
            
            # Use ASR engine's enhanced code-switching detection
            code_switches = await self.asr_engine.detect_code_switching(text)
            
            # Update statistics
            if code_switches:
                self.stats['code_switching_detections'] += 1
            
            logger.debug(f"Detected {len(code_switches)} code-switching points in text")
            return code_switches
            
        except Exception as e:
            logger.error(f"Error in code-switching detection: {e}")
            return []
    
    async def get_detailed_code_switching_analysis(
        self, 
        text: str, 
        context_language: Optional[LanguageCode] = None
    ) -> Dict[str, any]:
        """
        Get detailed code-switching analysis with enhanced features.
        
        Args:
            text: Input text to analyze
            context_language: Context language for better detection
            
        Returns:
            Detailed code-switching analysis result
        """
        try:
            if not text.strip():
                return {
                    "segments": [],
                    "switch_points": [],
                    "dominant_language": LanguageCode.ENGLISH_IN.value,
                    "switching_frequency": 0.0,
                    "confidence": 1.0,
                    "processing_time": 0.0
                }
            
            # Get detailed analysis from ASR engine
            result = await self.asr_engine.get_detailed_code_switching_analysis(
                text, context_language
            )
            
            # Convert to serializable format
            return {
                "segments": [
                    {
                        "text": seg.text,
                        "language": seg.language.value,
                        "start_pos": seg.start_pos,
                        "end_pos": seg.end_pos,
                        "confidence": seg.confidence,
                        "word_boundaries": seg.word_boundaries
                    }
                    for seg in result.segments
                ],
                "switch_points": [
                    {
                        "position": sp.position,
                        "from_language": sp.from_language.value,
                        "to_language": sp.to_language.value,
                        "confidence": sp.confidence
                    }
                    for sp in result.switch_points
                ],
                "dominant_language": result.dominant_language.value,
                "switching_frequency": result.switching_frequency,
                "confidence": result.confidence,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in detailed code-switching analysis: {e}")
            return {
                "segments": [],
                "switch_points": [],
                "dominant_language": LanguageCode.ENGLISH_IN.value,
                "switching_frequency": 0.0,
                "confidence": 0.0,
                "processing_time": 0.0
            }
    
    async def get_language_transition_suggestions(
        self, 
        from_language: LanguageCode, 
        to_language: LanguageCode
    ) -> Dict[str, List[str]]:
        """
        Get suggestions for smooth language transitions.
        
        Args:
            from_language: Source language
            to_language: Target language
            
        Returns:
            Dictionary with transition suggestions
        """
        try:
            return await self.asr_engine.get_language_transition_suggestions(
                from_language, to_language
            )
        except Exception as e:
            logger.error(f"Error getting transition suggestions: {e}")
            return {
                'connectors': ['that is', 'I mean'],
                'fillers': ['okay', 'so', 'well'],
                'markers': []
            }
    
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> TranslationResult:
        """
        Translate text between languages with quality assessment.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            Translation result with quality metrics
        """
        try:
            if not text.strip():
                return TranslationResult(
                    translated_text="",
                    source_language=source_lang,
                    target_language=target_lang,
                    quality_score=1.0,
                    semantic_preservation="full",
                    cultural_context="preserved",
                    confidence=1.0,
                    processing_time=0.0,
                    alternative_translations=[],
                    cultural_adaptations=[]
                )
            
            # Use the translation engine for comprehensive translation
            result = await self.translation_engine.translate(
                text, source_lang, target_lang,
                preserve_cultural_context, validate_semantics
            )
            
            # Update statistics
            self.stats['total_translations'] += 1
            
            logger.info(
                f"Translation completed: {source_lang} -> {target_lang}, "
                f"quality={result.quality_score:.3f}, confidence={result.confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            # Return error result
            return TranslationResult(
                translated_text=f"[Translation Error: {str(e)}]",
                source_language=source_lang,
                target_language=target_lang,
                quality_score=0.0,
                semantic_preservation="lost",
                cultural_context="lost",
                confidence=0.0,
                processing_time=0.0,
                alternative_translations=[],
                cultural_adaptations=[]
            )
    
    async def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the primary language of input text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Detected language code
        """
        try:
            if not text.strip():
                return LanguageCode.ENGLISH_IN
            
            detected_language = await self.asr_engine.detect_language(text)
            
            logger.debug(f"Language detected: {detected_language} for text: '{text[:50]}...'")
            return detected_language
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return LanguageCode.ENGLISH_IN
    
    async def adapt_to_regional_accent(
        self, 
        model_id: str, 
        accent_data: Dict[str, any]
    ) -> str:
        """
        Adapt language model to regional accent.
        
        Args:
            model_id: Base model identifier
            accent_data: Regional accent adaptation data
            
        Returns:
            Adapted model identifier
        """
        try:
            # Check cache first
            cache_key = f"{model_id}_{hash(str(sorted(accent_data.items())))}"
            if self.accent_adaptation_cache and cache_key in self.accent_adaptation_cache:
                logger.debug(f"Using cached accent adaptation: {cache_key}")
                return self.accent_adaptation_cache[cache_key]
            
            # Extract accent information
            region = accent_data.get("region", "standard")
            language = accent_data.get("language", "en")
            accent_samples = accent_data.get("samples", [])
            
            # Use ASR engine's accent adaptation
            adapted_model_id = await self.asr_engine.adapt_to_regional_accent(
                model_id, accent_data
            )
            
            # Cache the result
            if self.accent_adaptation_cache:
                if len(self.accent_adaptation_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.accent_adaptation_cache))
                    del self.accent_adaptation_cache[oldest_key]
                
                self.accent_adaptation_cache[cache_key] = adapted_model_id
            
            logger.info(f"Model adapted for regional accent: {model_id} -> {adapted_model_id} (region: {region})")
            return adapted_model_id
            
        except Exception as e:
            logger.error(f"Error in accent adaptation: {e}")
            return model_id
    
    async def recognize_with_language_hint(
        self, 
        audio: AudioBuffer, 
        language_hint: LanguageCode
    ) -> RecognitionResult:
        """
        Recognize speech with a language hint for improved accuracy.
        
        Args:
            audio: Audio buffer containing speech
            language_hint: Suggested language for recognition
            
        Returns:
            Speech recognition result
        """
        try:
            # For now, use the standard recognition
            # In production, this could bias the recognition towards the hinted language
            result = await self.recognize_speech(audio)
            
            # If confidence is low and language doesn't match hint, 
            # we could retry with language-specific models
            if (result.confidence < 0.7 and 
                result.detected_language != language_hint):
                
                logger.info(
                    f"Low confidence with language mismatch. "
                    f"Detected: {result.detected_language}, Hint: {language_hint}"
                )
                
                # Could implement language-specific retry logic here
            
            return result
            
        except Exception as e:
            logger.error(f"Error in recognition with language hint: {e}")
            return await self.recognize_speech(audio)
    
    async def batch_recognize_speech(
        self, 
        audio_buffers: List[AudioBuffer]
    ) -> List[RecognitionResult]:
        """
        Recognize speech from multiple audio buffers in batch.
        
        Args:
            audio_buffers: List of audio buffers to process
            
        Returns:
            List of recognition results
        """
        try:
            if not audio_buffers:
                return []
            
            # Process all audio buffers concurrently
            tasks = [self.recognize_speech(audio) for audio in audio_buffers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing audio buffer {i}: {result}")
                    # Create empty result for failed processing
                    processed_results.append(RecognitionResult(
                        transcribed_text="",
                        confidence=0.0,
                        detected_language=LanguageCode.ENGLISH_IN,
                        code_switching_points=[],
                        alternative_transcriptions=[],
                        processing_time=0.0
                    ))
                else:
                    processed_results.append(result)
            
            logger.info(f"Batch recognition completed for {len(audio_buffers)} audio buffers")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch speech recognition: {e}")
            return []
    
    async def get_language_confidence_scores(
        self, 
        audio: AudioBuffer
    ) -> Dict[LanguageCode, float]:
        """
        Get confidence scores for all supported languages.
        
        Args:
            audio: Audio buffer to analyze
            
        Returns:
            Dictionary mapping language codes to confidence scores
        """
        try:
            # Perform recognition to get primary result
            result = await self.recognize_speech(audio)
            
            # Initialize confidence scores
            confidence_scores = {}
            
            # Set primary language confidence
            confidence_scores[result.detected_language] = result.confidence
            
            # Add alternative language confidences
            for alt in result.alternative_transcriptions:
                confidence_scores[alt.language] = alt.confidence
            
            # Fill in remaining supported languages with low confidence
            supported_languages = self.asr_engine.get_supported_languages()
            for lang in supported_languages:
                if lang not in confidence_scores:
                    confidence_scores[lang] = 0.1  # Low default confidence
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error getting language confidence scores: {e}")
            return {}
    
    def _generate_audio_cache_key(self, audio: AudioBuffer) -> str:
        """
        Generate cache key for audio buffer.
        
        Args:
            audio: Audio buffer
            
        Returns:
            Cache key string
        """
        # Create hash based on audio characteristics
        import hashlib
        
        # Use sample of audio data for hashing (to avoid hashing large arrays)
        sample_size = min(1000, len(audio.data))
        audio_sample = audio.data[:sample_size]
        
        key_data = f"{audio.sample_rate}:{audio.channels}:{audio.duration}:{hash(tuple(audio_sample))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_recognition_result(self, cache_key: str, result: RecognitionResult):
        """
        Cache recognition result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Recognition result to cache
        """
        if len(self.recognition_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.recognition_cache))
            del self.recognition_cache[oldest_key]
        
        self.recognition_cache[cache_key] = result
    
    async def batch_translate_texts(
        self,
        texts: List[str],
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        preserve_cultural_context: bool = True,
        validate_semantics: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_cultural_context: Whether to preserve cultural context
            validate_semantics: Whether to validate semantic preservation
            
        Returns:
            List of translation results
        """
        try:
            return await self.translation_engine.batch_translate(
                texts, source_lang, target_lang,
                preserve_cultural_context, validate_semantics
            )
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            return []
    
    async def validate_translation_quality(
        self,
        source_text: str,
        translated_text: str,
        source_language: LanguageCode,
        target_language: LanguageCode
    ) -> Dict[str, any]:
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
            return await self.translation_engine.validate_translation_quality(
                source_text, translated_text, source_language, target_language
            )
        except Exception as e:
            logger.error(f"Error validating translation quality: {e}")
            return {
                'overall_quality_score': 0.0,
                'quality_level': 'poor',
                'error': str(e)
            }
    
    async def get_translation_suggestions(
        self,
        text: str,
        source_language: LanguageCode,
        target_language: LanguageCode,
        context: Optional[str] = None
    ) -> Dict[str, any]:
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
            return await self.translation_engine.get_translation_suggestions(
                text, source_language, target_language, context
            )
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
    
    async def detect_cultural_terms(
        self,
        text: str,
        language: LanguageCode
    ) -> List[Dict[str, any]]:
        """
        Detect cultural terms and expressions in text.
        
        Args:
            text: Text to analyze
            language: Language of the text
            
        Returns:
            List of detected cultural terms
        """
        try:
            cultural_terms = await self.translation_engine.detect_cultural_terms(text, language)
            
            # Convert to serializable format
            return [
                {
                    'term': term.term,
                    'language': term.language.value,
                    'meaning': term.meaning,
                    'cultural_significance': term.cultural_significance,
                    'regional_variants': term.regional_variants,
                    'translation_notes': term.translation_notes
                }
                for term in cultural_terms
            ]
        except Exception as e:
            logger.error(f"Error detecting cultural terms: {e}")
            return []
    
    def get_supported_translation_pairs(self) -> List[Tuple[LanguageCode, LanguageCode]]:
        """
        Get list of supported language pairs for translation.
        
        Returns:
            List of supported (source, target) language pairs
        """
        return self.translation_engine.get_supported_language_pairs()
    
    def get_translation_stats(self) -> Dict[str, any]:
        """
        Get translation engine statistics.
        
        Returns:
            Dictionary with translation statistics
        """
        return self.translation_engine.get_translation_stats()
    
    def clear_translation_caches(self):
        """Clear translation engine caches."""
        self.translation_engine.clear_caches()
        logger.info("Cleared translation engine caches")
    
    def _cache_translation_result(self, cache_key: str, result: str):
        """
        Cache translation result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Translation result to cache
        """
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[cache_key] = result
        """
        Cache translation result with LRU eviction.
        
        Args:
            cache_key: Cache key
            result: Translation result to cache
        """
        if len(self.translation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, could implement proper LRU)
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[cache_key] = result
    
    def _update_recognition_stats(self, result: RecognitionResult, processing_time: float):
        """
        Update recognition statistics.
        
        Args:
            result: Recognition result
            processing_time: Processing time in seconds
        """
        self.stats['total_recognitions'] += 1
        
        # Update average processing time
        total = self.stats['total_recognitions']
        current_avg = self.stats['average_recognition_time']
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.stats['average_recognition_time'] = new_avg
        
        # Update language distribution
        lang = result.detected_language.value
        if lang not in self.stats['language_distribution']:
            self.stats['language_distribution'][lang] = 0
        self.stats['language_distribution'][lang] += 1
    
    def get_supported_languages(self) -> List[LanguageCode]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return self.asr_engine.get_supported_languages()
    
    def get_service_stats(self) -> Dict[str, any]:
        """
        Get language engine service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        stats = self.stats.copy()
        
        # Add cache statistics
        if self.enable_caching:
            stats['cache_stats'] = {
                'recognition_cache_size': len(self.recognition_cache) if self.recognition_cache else 0,
                'translation_cache_size': len(self.translation_cache) if self.translation_cache else 0,
                'cache_hit_rate': (
                    self.stats['cache_hits'] / 
                    max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
                )
            }
        
        # Add ASR engine info
        stats['asr_engine_info'] = self.asr_engine.get_model_info()
        
        # Add translation engine stats
        stats['translation_engine_stats'] = self.translation_engine.get_translation_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all internal caches."""
        try:
            if self.recognition_cache:
                self.recognition_cache.clear()
                logger.debug("Recognition cache cleared")
            if self.translation_cache:
                self.translation_cache.clear()
                logger.debug("Translation cache cleared")
            if self.accent_adaptation_cache:
                self.accent_adaptation_cache.clear()
                logger.debug("Accent adaptation cache cleared")
            
            # Clear translation engine caches
            if hasattr(self.translation_engine, 'clear_caches'):
                self.translation_engine.clear_caches()
                logger.debug("Translation engine caches cleared")
            
            logger.info("All language engine service caches cleared")
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    async def get_engine_health_status(self) -> Dict[str, any]:
        """
        Get detailed health status of all engine components.
        
        Returns:
            Detailed health status information
        """
        try:
            health_status = {
                "overall_status": "healthy",
                "components": {},
                "errors": []
            }
            
            # Check ASR engine health
            try:
                if hasattr(self.asr_engine, 'health_check'):
                    asr_health = await self.asr_engine.health_check()
                    health_status["components"]["asr_engine"] = asr_health
                    if asr_health.get("status") != "healthy":
                        health_status["overall_status"] = "degraded"
                else:
                    health_status["components"]["asr_engine"] = {"status": "unknown"}
            except Exception as e:
                health_status["components"]["asr_engine"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"ASR engine health check failed: {e}")
                health_status["overall_status"] = "unhealthy"
            
            # Check translation engine health
            try:
                if hasattr(self.translation_engine, 'health_check'):
                    translation_health = await self.translation_engine.health_check()
                    health_status["components"]["translation_engine"] = translation_health
                    if translation_health.get("status") != "healthy":
                        health_status["overall_status"] = "degraded"
                else:
                    health_status["components"]["translation_engine"] = {"status": "unknown"}
            except Exception as e:
                health_status["components"]["translation_engine"] = {"status": "error", "error": str(e)}
                health_status["errors"].append(f"Translation engine health check failed: {e}")
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"
            
            # Check cache status
            if self.enable_caching:
                cache_status = {
                    "recognition_cache_size": len(self.recognition_cache) if self.recognition_cache else 0,
                    "translation_cache_size": len(self.translation_cache) if self.translation_cache else 0,
                    "accent_cache_size": len(self.accent_adaptation_cache) if self.accent_adaptation_cache else 0,
                    "cache_limit": self.cache_size
                }
                health_status["components"]["caching"] = {"status": "enabled", "details": cache_status}
            else:
                health_status["components"]["caching"] = {"status": "disabled"}
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting engine health status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "components": {},
                "errors": [f"Health status check failed: {e}"]
            }
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the language engine service.
        
        Returns:
            Health check result
        """
        try:
            # Check ASR engine health
            asr_health = await self.asr_engine.health_check()
            
            # Check translation engine health
            translation_health = await self.translation_engine.health_check()
            
            # Test basic functionality
            test_audio = AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence at 16kHz
                sample_rate=16000,
                channels=1,
                duration=0.1
            )
            
            # Test recognition
            recognition_result = await self.recognize_speech(test_audio)
            recognition_status = "ok" if recognition_result is not None else "error"
            
            # Test language detection
            lang_detection_result = await self.detect_language("Hello नमस्ते")
            lang_detection_status = "ok" if lang_detection_result is not None else "error"
            
            # Test translation
            translation_result = await self.translate_text(
                "Hello", LanguageCode.ENGLISH_IN, LanguageCode.HINDI
            )
            translation_status = "ok" if translation_result.confidence > 0.0 else "error"
            
            overall_status = (
                "healthy" if (
                    asr_health["status"] == "healthy" and
                    translation_health["status"] == "healthy" and
                    recognition_status == "ok" and
                    lang_detection_status == "ok" and
                    translation_status == "ok"
                ) else "unhealthy"
            )
            
            return {
                "status": overall_status,
                "asr_engine": asr_health,
                "translation_engine": translation_health,
                "recognition_test": recognition_status,
                "language_detection_test": lang_detection_status,
                "translation_test": translation_status,
                "service_stats": self.get_service_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_stats": self.get_service_stats()
            }


# Factory function for creating language engine service
def create_language_engine_service(
    asr_model_size: str = "base",
    device: str = "cpu",
    enable_caching: bool = True,
    cache_size: int = 1000,
    enable_language_detection: bool = True
) -> LanguageEngineService:
    """
    Factory function to create a language engine service instance.
    
    Args:
        asr_model_size: Whisper model size for ASR
        device: Device to run inference on
        enable_caching: Whether to enable result caching
        cache_size: Maximum number of cached results
        enable_language_detection: Whether to enable automatic language detection
        
    Returns:
        Configured LanguageEngineService instance
    """
    return LanguageEngineService(
        asr_model_size=asr_model_size,
        device=device,
        enable_caching=enable_caching,
        cache_size=cache_size,
        enable_language_detection=enable_language_detection
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )