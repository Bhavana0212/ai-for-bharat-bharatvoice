"""
Enhanced Code-Switching Detection Module for BharatVoice Assistant.

This module implements advanced code-switching detection using multiple language
identification models, boundary detection, and seamless language transition handling
for mixed-language processing within single utterances.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
import re
from dataclasses import dataclass

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect_langs, LangDetectError

from bharatvoice.core.models import LanguageCode, LanguageSwitchPoint

logger = logging.getLogger(__name__)


@dataclass
class LanguageSegment:
    """Represents a segment of text in a specific language."""
    text: str
    language: LanguageCode
    start_pos: int
    end_pos: int
    confidence: float
    word_boundaries: List[Tuple[int, int]]  # (start, end) positions of words


@dataclass
class CodeSwitchingResult:
    """Result of code-switching detection analysis."""
    segments: List[LanguageSegment]
    switch_points: List[LanguageSwitchPoint]
    dominant_language: LanguageCode
    switching_frequency: float  # switches per 100 characters
    confidence: float
    processing_time: float


class EnhancedCodeSwitchingDetector:
    """
    Enhanced code-switching detector using multiple language identification models.
    
    Features:
    - Multi-model ensemble for improved accuracy
    - Word-level and phrase-level boundary detection
    - Confidence scoring for language switches
    - Support for Indian language mixing patterns
    - Contextual language transition handling
    """
    
    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.7,
        min_segment_length: int = 3,
        enable_word_level_detection: bool = True
    ):
        """
        Initialize the enhanced code-switching detector.
        
        Args:
            device: Device to run inference on ("cpu" or "cuda")
            confidence_threshold: Minimum confidence for language detection
            min_segment_length: Minimum character length for language segments
            enable_word_level_detection: Enable word-level language detection
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_segment_length = min_segment_length
        self.enable_word_level_detection = enable_word_level_detection
        
        # Language models for detection
        self.primary_detector = None
        self.secondary_detector = None
        self.tokenizer = None
        
        # Language mapping for different models
        self.language_mappings = {
            # XLM-RoBERTa language detection model mappings
            'xlm_roberta': {
                'hi': LanguageCode.HINDI,
                'en': LanguageCode.ENGLISH_IN,
                'ta': LanguageCode.TAMIL,
                'te': LanguageCode.TELUGU,
                'bn': LanguageCode.BENGALI,
                'mr': LanguageCode.MARATHI,
                'gu': LanguageCode.GUJARATI,
                'kn': LanguageCode.KANNADA,
                'ml': LanguageCode.MALAYALAM,
                'pa': LanguageCode.PUNJABI,
                'or': LanguageCode.ODIA,
            },
            # langdetect mappings
            'langdetect': {
                'hi': LanguageCode.HINDI,
                'en': LanguageCode.ENGLISH_IN,
                'ta': LanguageCode.TAMIL,
                'te': LanguageCode.TELUGU,
                'bn': LanguageCode.BENGALI,
                'mr': LanguageCode.MARATHI,
                'gu': LanguageCode.GUJARATI,
                'kn': LanguageCode.KANNADA,
                'ml': LanguageCode.MALAYALAM,
                'pa': LanguageCode.PUNJABI,
            }
        }
        
        # Common code-switching patterns in Indian languages
        self.switching_patterns = {
            'hindi_english': [
                r'\b(the|and|or|but|so|because|that|this|is|are|was|were)\b',
                r'\b(में|का|की|के|है|हैं|था|थे|को|से|पर)\b',
            ],
            'tamil_english': [
                r'\b(the|and|or|but|so|because|that|this|is|are|was|were)\b',
                r'\b(இல்|ஆக|என்|அந்த|இந்த|உள்ள|இருக்க)\b',
            ],
            'telugu_english': [
                r'\b(the|and|or|but|so|because|that|this|is|are|was|were)\b',
                r'\b(లో|గా|అని|ఆ|ఈ|ఉన్న|ఉంది)\b',
            ]
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("EnhancedCodeSwitchingDetector initialized successfully")
    
    def _initialize_models(self):
        """Initialize language detection models."""
        try:
            # Primary model: XLM-RoBERTa based language detection
            logger.info("Loading primary language detection model...")
            self.primary_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Secondary model: FastText-based language detection (fallback)
            logger.info("Loading secondary language detection model...")
            try:
                self.secondary_detector = pipeline(
                    "text-classification",
                    model="facebook/fasttext-language-identification",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Failed to load secondary model: {e}")
                self.secondary_detector = None
            
            # Load tokenizer for word-level analysis
            if self.enable_word_level_detection:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "papluca/xlm-roberta-base-language-detection"
                )
            
            logger.info("Language detection models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize language detection models: {e}")
            # Fallback to basic detection
            self.primary_detector = None
            self.secondary_detector = None
            self.tokenizer = None
    
    async def detect_code_switching(
        self, 
        text: str, 
        context_language: Optional[LanguageCode] = None
    ) -> CodeSwitchingResult:
        """
        Detect code-switching in text with enhanced analysis.
        
        Args:
            text: Input text potentially containing multiple languages
            context_language: Context language for better detection
            
        Returns:
            Comprehensive code-switching analysis result
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not text.strip():
                return CodeSwitchingResult(
                    segments=[],
                    switch_points=[],
                    dominant_language=LanguageCode.ENGLISH_IN,
                    switching_frequency=0.0,
                    confidence=1.0,
                    processing_time=0.0
                )
            
            # Step 1: Segment text into analyzable units
            segments = await self._segment_text_advanced(text)
            
            # Step 2: Detect language for each segment
            language_segments = await self._detect_segment_languages(
                segments, context_language
            )
            
            # Step 3: Refine boundaries and merge adjacent same-language segments
            refined_segments = await self._refine_language_boundaries(
                language_segments, text
            )
            
            # Step 4: Generate switch points
            switch_points = self._generate_switch_points(refined_segments)
            
            # Step 5: Calculate statistics
            dominant_language = self._calculate_dominant_language(refined_segments)
            switching_frequency = self._calculate_switching_frequency(
                switch_points, len(text)
            )
            overall_confidence = self._calculate_overall_confidence(refined_segments)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = CodeSwitchingResult(
                segments=refined_segments,
                switch_points=switch_points,
                dominant_language=dominant_language,
                switching_frequency=switching_frequency,
                confidence=overall_confidence,
                processing_time=processing_time
            )
            
            logger.debug(
                f"Code-switching detection completed: {len(switch_points)} switches, "
                f"dominant={dominant_language}, frequency={switching_frequency:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in code-switching detection: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return fallback result
            return CodeSwitchingResult(
                segments=[LanguageSegment(
                    text=text,
                    language=context_language or LanguageCode.ENGLISH_IN,
                    start_pos=0,
                    end_pos=len(text),
                    confidence=0.5,
                    word_boundaries=[]
                )],
                switch_points=[],
                dominant_language=context_language or LanguageCode.ENGLISH_IN,
                switching_frequency=0.0,
                confidence=0.5,
                processing_time=processing_time
            )
    
    async def _segment_text_advanced(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Advanced text segmentation for language detection.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of (segment_text, start_pos, end_pos) tuples
        """
        segments = []
        
        # Multi-level segmentation approach
        
        # Level 1: Sentence boundaries
        sentence_pattern = r'[.!?।॥]+\s*'
        sentences = re.split(sentence_pattern, text)
        
        current_pos = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence_start = text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)
            
            # Level 2: Phrase boundaries within sentences
            phrase_segments = await self._segment_by_phrases(
                sentence, sentence_start
            )
            segments.extend(phrase_segments)
            
            current_pos = sentence_end
        
        # Filter out very short segments
        filtered_segments = [
            (seg_text, start, end) for seg_text, start, end in segments
            if len(seg_text.strip()) >= self.min_segment_length
        ]
        
        return filtered_segments
    
    async def _segment_by_phrases(
        self, 
        sentence: str, 
        base_offset: int
    ) -> List[Tuple[str, int, int]]:
        """
        Segment sentence into phrases based on punctuation and patterns.
        
        Args:
            sentence: Sentence to segment
            base_offset: Base character offset in original text
            
        Returns:
            List of phrase segments
        """
        # Split on commas, semicolons, and other phrase boundaries
        phrase_pattern = r'[,;:\-–—]+\s*'
        phrases = re.split(phrase_pattern, sentence)
        
        segments = []
        current_pos = 0
        
        for phrase in phrases:
            if not phrase.strip():
                continue
            
            phrase_start = sentence.find(phrase, current_pos)
            if phrase_start == -1:
                continue
            
            phrase_end = phrase_start + len(phrase)
            
            # Further split long phrases if they contain obvious language switches
            sub_segments = await self._detect_intra_phrase_switches(
                phrase, base_offset + phrase_start
            )
            
            if sub_segments:
                segments.extend(sub_segments)
            else:
                segments.append((
                    phrase,
                    base_offset + phrase_start,
                    base_offset + phrase_end
                ))
            
            current_pos = phrase_end
        
        return segments
    
    async def _detect_intra_phrase_switches(
        self, 
        phrase: str, 
        base_offset: int
    ) -> List[Tuple[str, int, int]]:
        """
        Detect language switches within a single phrase.
        
        Args:
            phrase: Phrase to analyze
            base_offset: Base character offset
            
        Returns:
            List of sub-segments if switches detected, empty list otherwise
        """
        if not self.enable_word_level_detection or len(phrase) < 20:
            return []
        
        # Use pattern matching to detect obvious switches
        segments = []
        
        # Look for common switching patterns
        for pattern_type, patterns in self.switching_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, phrase, re.IGNORECASE))
                if matches:
                    # Split around pattern matches
                    last_end = 0
                    for match in matches:
                        # Add segment before match
                        if match.start() > last_end:
                            pre_text = phrase[last_end:match.start()].strip()
                            if pre_text:
                                segments.append((
                                    pre_text,
                                    base_offset + last_end,
                                    base_offset + match.start()
                                ))
                        
                        # Add the match itself
                        match_text = match.group().strip()
                        if match_text:
                            segments.append((
                                match_text,
                                base_offset + match.start(),
                                base_offset + match.end()
                            ))
                        
                        last_end = match.end()
                    
                    # Add remaining text
                    if last_end < len(phrase):
                        remaining_text = phrase[last_end:].strip()
                        if remaining_text:
                            segments.append((
                                remaining_text,
                                base_offset + last_end,
                                base_offset + len(phrase)
                            ))
                    
                    return segments
        
        return []
    
    async def _detect_segment_languages(
        self, 
        segments: List[Tuple[str, int, int]], 
        context_language: Optional[LanguageCode]
    ) -> List[LanguageSegment]:
        """
        Detect language for each text segment using ensemble approach.
        
        Args:
            segments: List of text segments to analyze
            context_language: Context language for bias
            
        Returns:
            List of language segments with detected languages
        """
        language_segments = []
        
        for seg_text, start_pos, end_pos in segments:
            # Detect language using ensemble approach
            detected_language, confidence = await self._detect_language_ensemble(
                seg_text, context_language
            )
            
            # Extract word boundaries
            word_boundaries = self._extract_word_boundaries(seg_text, start_pos)
            
            language_segment = LanguageSegment(
                text=seg_text,
                language=detected_language,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=confidence,
                word_boundaries=word_boundaries
            )
            
            language_segments.append(language_segment)
        
        return language_segments
    
    async def _detect_language_ensemble(
        self, 
        text: str, 
        context_language: Optional[LanguageCode]
    ) -> Tuple[LanguageCode, float]:
        """
        Detect language using ensemble of multiple models.
        
        Args:
            text: Text to analyze
            context_language: Context language for bias
            
        Returns:
            Tuple of (detected_language, confidence)
        """
        detections = []
        
        # Primary model detection
        if self.primary_detector:
            try:
                primary_result = await self._detect_with_transformer_model(
                    text, self.primary_detector, 'xlm_roberta'
                )
                if primary_result:
                    detections.append(primary_result)
            except Exception as e:
                logger.warning(f"Primary model detection failed: {e}")
        
        # Secondary model detection
        if self.secondary_detector:
            try:
                secondary_result = await self._detect_with_transformer_model(
                    text, self.secondary_detector, 'xlm_roberta'
                )
                if secondary_result:
                    detections.append(secondary_result)
            except Exception as e:
                logger.warning(f"Secondary model detection failed: {e}")
        
        # Fallback to langdetect
        try:
            langdetect_result = await self._detect_with_langdetect(text)
            if langdetect_result:
                detections.append(langdetect_result)
        except Exception as e:
            logger.warning(f"Langdetect detection failed: {e}")
        
        # Pattern-based detection for Indian languages
        pattern_result = await self._detect_with_patterns(text)
        if pattern_result:
            detections.append(pattern_result)
        
        # Ensemble decision
        if not detections:
            return context_language or LanguageCode.ENGLISH_IN, 0.5
        
        # Weight and combine results
        final_language, final_confidence = self._combine_detection_results(
            detections, context_language
        )
        
        return final_language, final_confidence
    
    async def _detect_with_transformer_model(
        self, 
        text: str, 
        model, 
        model_type: str
    ) -> Optional[Tuple[LanguageCode, float]]:
        """
        Detect language using transformer-based model.
        
        Args:
            text: Text to analyze
            model: Transformer model pipeline
            model_type: Type of model for mapping
            
        Returns:
            Detection result or None if failed
        """
        try:
            # Run detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, model, text)
            
            if not results or not isinstance(results, list):
                return None
            
            # Get top prediction
            top_result = max(results, key=lambda x: x['score'])
            detected_lang = top_result['label'].lower()
            confidence = top_result['score']
            
            # Map to our language codes
            language_mapping = self.language_mappings.get(model_type, {})
            if detected_lang in language_mapping:
                return language_mapping[detected_lang], confidence
            
            return None
            
        except Exception as e:
            logger.warning(f"Transformer model detection error: {e}")
            return None
    
    async def _detect_with_langdetect(
        self, 
        text: str
    ) -> Optional[Tuple[LanguageCode, float]]:
        """
        Detect language using langdetect library.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detection result or None if failed
        """
        try:
            lang_probs = detect_langs(text)
            if not lang_probs:
                return None
            
            top_detection = lang_probs[0]
            detected_lang = top_detection.lang
            confidence = top_detection.prob
            
            # Map to our language codes
            language_mapping = self.language_mappings.get('langdetect', {})
            if detected_lang in language_mapping:
                return language_mapping[detected_lang], confidence
            
            return None
            
        except LangDetectError:
            return None
        except Exception as e:
            logger.warning(f"Langdetect error: {e}")
            return None
    
    async def _detect_with_patterns(
        self, 
        text: str
    ) -> Optional[Tuple[LanguageCode, float]]:
        """
        Detect language using script and pattern analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detection result or None if no clear pattern
        """
        # Script-based detection patterns
        script_patterns = {
            LanguageCode.HINDI: r'[\u0900-\u097F]+',  # Devanagari
            LanguageCode.TAMIL: r'[\u0B80-\u0BFF]+',  # Tamil
            LanguageCode.TELUGU: r'[\u0C00-\u0C7F]+',  # Telugu
            LanguageCode.BENGALI: r'[\u0980-\u09FF]+',  # Bengali
            LanguageCode.GUJARATI: r'[\u0A80-\u0AFF]+',  # Gujarati
            LanguageCode.KANNADA: r'[\u0C80-\u0CFF]+',  # Kannada
            LanguageCode.MALAYALAM: r'[\u0D00-\u0D7F]+',  # Malayalam
            LanguageCode.PUNJABI: r'[\u0A00-\u0A7F]+',  # Gurmukhi
            LanguageCode.ODIA: r'[\u0B00-\u0B7F]+',  # Odia
        }
        
        # Count characters for each script
        script_counts = {}
        total_chars = len(text)
        
        for language, pattern in script_patterns.items():
            matches = re.findall(pattern, text)
            char_count = sum(len(match) for match in matches)
            if char_count > 0:
                script_counts[language] = char_count / total_chars
        
        # Check for English (Latin script)
        english_pattern = r'[a-zA-Z]+'
        english_matches = re.findall(english_pattern, text)
        english_count = sum(len(match) for match in english_matches)
        if english_count > 0:
            script_counts[LanguageCode.ENGLISH_IN] = english_count / total_chars
        
        if not script_counts:
            return None
        
        # Return language with highest script presence
        dominant_language = max(script_counts, key=script_counts.get)
        confidence = script_counts[dominant_language]
        
        # Only return if confidence is reasonable
        if confidence >= 0.3:
            return dominant_language, min(confidence, 0.9)
        
        return None
    
    def _combine_detection_results(
        self, 
        detections: List[Tuple[LanguageCode, float]], 
        context_language: Optional[LanguageCode]
    ) -> Tuple[LanguageCode, float]:
        """
        Combine multiple detection results using weighted voting.
        
        Args:
            detections: List of (language, confidence) tuples
            context_language: Context language for bias
            
        Returns:
            Final (language, confidence) result
        """
        if not detections:
            return context_language or LanguageCode.ENGLISH_IN, 0.5
        
        if len(detections) == 1:
            return detections[0]
        
        # Weight votes by confidence
        language_votes = {}
        total_weight = 0.0
        
        for language, confidence in detections:
            if language not in language_votes:
                language_votes[language] = 0.0
            
            # Apply context bias
            weight = confidence
            if context_language and language == context_language:
                weight *= 1.2  # Boost context language
            
            language_votes[language] += weight
            total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            for language in language_votes:
                language_votes[language] /= total_weight
        
        # Select language with highest vote
        final_language = max(language_votes, key=language_votes.get)
        final_confidence = language_votes[final_language]
        
        return final_language, final_confidence
    
    def _extract_word_boundaries(
        self, 
        text: str, 
        base_offset: int
    ) -> List[Tuple[int, int]]:
        """
        Extract word boundaries from text.
        
        Args:
            text: Text to analyze
            base_offset: Base character offset
            
        Returns:
            List of (start, end) word boundary positions
        """
        # Simple word boundary extraction using regex
        word_pattern = r'\b\w+\b'
        boundaries = []
        
        for match in re.finditer(word_pattern, text):
            start = base_offset + match.start()
            end = base_offset + match.end()
            boundaries.append((start, end))
        
        return boundaries
    
    async def _refine_language_boundaries(
        self, 
        segments: List[LanguageSegment], 
        original_text: str
    ) -> List[LanguageSegment]:
        """
        Refine language boundaries and merge adjacent same-language segments.
        
        Args:
            segments: Initial language segments
            original_text: Original input text
            
        Returns:
            Refined language segments
        """
        if not segments:
            return []
        
        # Sort segments by position
        segments.sort(key=lambda x: x.start_pos)
        
        # Merge adjacent segments with same language
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Check if segments are adjacent and same language
            if (current_segment.language == next_segment.language and
                abs(current_segment.end_pos - next_segment.start_pos) <= 5):
                
                # Merge segments
                merged_text = original_text[
                    current_segment.start_pos:next_segment.end_pos
                ]
                merged_confidence = (
                    current_segment.confidence + next_segment.confidence
                ) / 2
                
                merged_boundaries = (
                    current_segment.word_boundaries + 
                    next_segment.word_boundaries
                )
                
                current_segment = LanguageSegment(
                    text=merged_text,
                    language=current_segment.language,
                    start_pos=current_segment.start_pos,
                    end_pos=next_segment.end_pos,
                    confidence=merged_confidence,
                    word_boundaries=merged_boundaries
                )
            else:
                # Add current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        return merged_segments
    
    def _generate_switch_points(
        self, 
        segments: List[LanguageSegment]
    ) -> List[LanguageSwitchPoint]:
        """
        Generate language switch points from segments.
        
        Args:
            segments: Language segments
            
        Returns:
            List of language switch points
        """
        switch_points = []
        
        for i in range(1, len(segments)):
            prev_segment = segments[i - 1]
            curr_segment = segments[i]
            
            if prev_segment.language != curr_segment.language:
                # Calculate switch confidence based on segment confidences
                switch_confidence = min(
                    prev_segment.confidence, 
                    curr_segment.confidence
                )
                
                switch_point = LanguageSwitchPoint(
                    position=curr_segment.start_pos,
                    from_language=prev_segment.language,
                    to_language=curr_segment.language,
                    confidence=switch_confidence
                )
                
                switch_points.append(switch_point)
        
        return switch_points
    
    def _calculate_dominant_language(
        self, 
        segments: List[LanguageSegment]
    ) -> LanguageCode:
        """
        Calculate the dominant language based on character count.
        
        Args:
            segments: Language segments
            
        Returns:
            Dominant language code
        """
        if not segments:
            return LanguageCode.ENGLISH_IN
        
        language_counts = {}
        
        for segment in segments:
            language = segment.language
            char_count = len(segment.text)
            
            if language not in language_counts:
                language_counts[language] = 0
            language_counts[language] += char_count
        
        return max(language_counts, key=language_counts.get)
    
    def _calculate_switching_frequency(
        self, 
        switch_points: List[LanguageSwitchPoint], 
        text_length: int
    ) -> float:
        """
        Calculate code-switching frequency per 100 characters.
        
        Args:
            switch_points: Language switch points
            text_length: Total text length
            
        Returns:
            Switching frequency
        """
        if text_length == 0:
            return 0.0
        
        return (len(switch_points) / text_length) * 100
    
    def _calculate_overall_confidence(
        self, 
        segments: List[LanguageSegment]
    ) -> float:
        """
        Calculate overall confidence for the detection result.
        
        Args:
            segments: Language segments
            
        Returns:
            Overall confidence score
        """
        if not segments:
            return 0.0
        
        # Weight confidence by segment length
        total_weighted_confidence = 0.0
        total_length = 0
        
        for segment in segments:
            length = len(segment.text)
            total_weighted_confidence += segment.confidence * length
            total_length += length
        
        if total_length == 0:
            return 0.0
        
        return total_weighted_confidence / total_length
    
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
        # Common transition phrases for different language pairs
        transitions = {
            (LanguageCode.HINDI, LanguageCode.ENGLISH_IN): {
                'connectors': ['यानी', 'मतलब', 'that is', 'I mean'],
                'fillers': ['अच्छा', 'okay', 'so', 'well'],
                'markers': ['English में कहें तो', 'in English']
            },
            (LanguageCode.ENGLISH_IN, LanguageCode.HINDI): {
                'connectors': ['that is', 'I mean', 'यानी', 'मतलब'],
                'fillers': ['okay', 'so', 'अच्छा', 'well'],
                'markers': ['Hindi में कहें तो', 'in Hindi']
            },
            (LanguageCode.TAMIL, LanguageCode.ENGLISH_IN): {
                'connectors': ['அதாவது', 'that is', 'I mean'],
                'fillers': ['சரி', 'okay', 'so'],
                'markers': ['English-ல் சொன்னால்', 'in English']
            }
        }
        
        transition_key = (from_language, to_language)
        return transitions.get(transition_key, {
            'connectors': ['that is', 'I mean'],
            'fillers': ['okay', 'so', 'well'],
            'markers': []
        })
    
    def get_detection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the detector configuration.
        
        Returns:
            Detector statistics
        """
        return {
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'min_segment_length': self.min_segment_length,
            'word_level_detection_enabled': self.enable_word_level_detection,
            'primary_detector_loaded': self.primary_detector is not None,
            'secondary_detector_loaded': self.secondary_detector is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'supported_language_pairs': len(self.switching_patterns),
        }


# Factory function for creating enhanced code-switching detector
def create_enhanced_code_switching_detector(
    device: str = "cpu",
    confidence_threshold: float = 0.7,
    min_segment_length: int = 3,
    enable_word_level_detection: bool = True
) -> EnhancedCodeSwitchingDetector:
    """
    Factory function to create an enhanced code-switching detector.
    
    Args:
        device: Device to run inference on ("cpu" or "cuda")
        confidence_threshold: Minimum confidence for language detection
        min_segment_length: Minimum character length for language segments
        enable_word_level_detection: Enable word-level language detection
        
    Returns:
        Configured EnhancedCodeSwitchingDetector instance
    """
    return EnhancedCodeSwitchingDetector(
        device=device,
        confidence_threshold=confidence_threshold,
        min_segment_length=min_segment_length,
        enable_word_level_detection=enable_word_level_detection
    )