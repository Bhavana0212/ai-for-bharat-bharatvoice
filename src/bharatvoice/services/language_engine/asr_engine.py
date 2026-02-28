<<<<<<< HEAD
"""
Multilingual Automatic Speech Recognition (ASR) Engine for BharatVoice Assistant.

This module implements a comprehensive ASR system supporting Hindi, English, and
regional Indian languages with advanced features like confidence scoring,
alternative transcriptions, and language detection.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import tempfile
import os

import numpy as np
import whisper
from langdetect import detect, LangDetectError
from transformers import pipeline

from bharatvoice.core.interfaces import LanguageEngine
from bharatvoice.core.models import (
    AudioBuffer,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.code_switching_detector import (
    EnhancedCodeSwitchingDetector,
    create_enhanced_code_switching_detector,
    CodeSwitchingResult,
)

logger = logging.getLogger(__name__)


class MultilingualASREngine(LanguageEngine):
    """
    Multilingual ASR engine supporting Indian languages with advanced features.
    
    Features:
    - Whisper-based speech recognition for high accuracy
    - Support for 10+ Indian languages
    - Confidence scoring and alternative transcriptions
    - Language detection and code-switching detection
    - Regional accent adaptation
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        enable_language_detection: bool = True,
        confidence_threshold: float = 0.7,
        max_alternatives: int = 3
    ):
        """
        Initialize the multilingual ASR engine.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run inference on ("cpu" or "cuda")
            enable_language_detection: Whether to enable automatic language detection
            confidence_threshold: Minimum confidence threshold for results
            max_alternatives: Maximum number of alternative transcriptions
        """
        self.model_size = model_size
        self.device = device
        self.enable_language_detection = enable_language_detection
        self.confidence_threshold = confidence_threshold
        self.max_alternatives = max_alternatives
        
        # Initialize Whisper model
        self.whisper_model = None
        self.language_detector = None
        
        # Initialize enhanced code-switching detector
        self.code_switching_detector = None
        
        # Language mapping for Whisper
        self.whisper_language_map = {
            LanguageCode.HINDI: "hi",
            LanguageCode.ENGLISH_IN: "en",
            LanguageCode.TAMIL: "ta",
            LanguageCode.TELUGU: "te",
            LanguageCode.BENGALI: "bn",
            LanguageCode.MARATHI: "mr",
            LanguageCode.GUJARATI: "gu",
            LanguageCode.KANNADA: "kn",
            LanguageCode.MALAYALAM: "ml",
            LanguageCode.PUNJABI: "pa",
            LanguageCode.ODIA: "or",
        }
        
        # Reverse mapping for language detection
        self.language_code_map = {v: k for k, v in self.whisper_language_map.items()}
        
        # Language-specific confidence adjustments
        self.language_confidence_factors = {
            LanguageCode.HINDI: 1.0,
            LanguageCode.ENGLISH_IN: 0.95,  # Slightly lower due to accent variations
            LanguageCode.TAMIL: 0.9,
            LanguageCode.TELUGU: 0.9,
            LanguageCode.BENGALI: 0.85,
            LanguageCode.MARATHI: 0.85,
            LanguageCode.GUJARATI: 0.8,
            LanguageCode.KANNADA: 0.8,
            LanguageCode.MALAYALAM: 0.8,
            LanguageCode.PUNJABI: 0.85,
            LanguageCode.ODIA: 0.8,
        }
        
        # Initialize components
        self._initialize_models()
        
        logger.info(f"MultilingualASREngine initialized with model_size={model_size}")
    
    def _initialize_models(self):
        """Initialize the ASR models and components."""
        try:
            # Load Whisper model with proper error handling
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.whisper_model = whisper.load_model(self.model_size, device=self.device)
                logger.info(f"Whisper model '{self.model_size}' loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{self.model_size}': {e}")
                # Try fallback to smaller model
                if self.model_size != "tiny":
                    logger.info("Attempting fallback to 'tiny' model...")
                    try:
                        self.whisper_model = whisper.load_model("tiny", device="cpu")
                        self.model_size = "tiny"
                        self.device = "cpu"
                        logger.info("Successfully loaded fallback 'tiny' model on CPU")
                    except Exception as fallback_e:
                        logger.error(f"Fallback model loading also failed: {fallback_e}")
                        raise
                else:
                    raise
            
            # Initialize language detection pipeline if enabled
            if self.enable_language_detection:
                try:
                    self.language_detector = pipeline(
                        "text-classification",
                        model="papluca/xlm-roberta-base-language-detection",
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("Language detection pipeline initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to load language detection model: {e}")
                    logger.info("Language detection will use fallback langdetect library")
                    self.language_detector = None
            
            # Initialize enhanced code-switching detector
            try:
                logger.info("Initializing enhanced code-switching detector...")
                self.code_switching_detector = create_enhanced_code_switching_detector(
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                    min_segment_length=3,
                    enable_word_level_detection=True
                )
                logger.info("Enhanced code-switching detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced code-switching detector: {e}")
                logger.info("Code-switching detection will use basic fallback implementation")
                self.code_switching_detector = None
            
            logger.info("ASR models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR models: {e}")
            raise
    
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input with multilingual support.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Convert audio buffer to temporary file for Whisper
            temp_audio_path = await self._prepare_audio_for_whisper(audio)
            
            try:
                # Perform speech recognition
                result = await self._transcribe_with_whisper(temp_audio_path)
                
                # Extract primary transcription and language
                primary_text = result["text"].strip()
                detected_language = self._map_whisper_language_to_code(result.get("language", "en"))
                
                # Calculate confidence score
                confidence = self._calculate_confidence(result, detected_language)
                
                # Generate alternative transcriptions
                alternatives = await self._generate_alternatives(temp_audio_path, primary_text)
                
                # Detect code-switching points
                code_switching_points = await self._detect_code_switching_in_result(
                    primary_text, detected_language
                )
                
                # Calculate processing time
                processing_time = asyncio.get_event_loop().time() - start_time
                
                recognition_result = RecognitionResult(
                    transcribed_text=primary_text,
                    confidence=confidence,
                    detected_language=detected_language,
                    code_switching_points=code_switching_points,
                    alternative_transcriptions=alternatives,
                    processing_time=processing_time
                )
                
                logger.info(
                    f"Speech recognition completed: '{primary_text[:50]}...' "
                    f"({detected_language}, confidence={confidence:.3f})"
                )
                
                return recognition_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
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
            
            # Try using the transformer-based language detector first
            if self.language_detector:
                try:
                    result = self.language_detector(text)
                    if result and len(result) > 0:
                        detected_lang = result[0]["label"].lower()
                        # Map to our language codes
                        if detected_lang in self.language_code_map:
                            return self.language_code_map[detected_lang]
                except Exception as e:
                    logger.warning(f"Transformer language detection failed: {e}")
            
            # Fallback to langdetect
            try:
                detected = detect(text)
                if detected in self.language_code_map:
                    return self.language_code_map[detected]
            except LangDetectError:
                logger.warning("Language detection failed, defaulting to English")
            
            # Default fallback
            return LanguageCode.ENGLISH_IN
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return LanguageCode.ENGLISH_IN
    
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text using enhanced detection.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        try:
            if not text.strip():
                return []
            
            # Use enhanced code-switching detector if available
            if self.code_switching_detector:
                result = await self.code_switching_detector.detect_code_switching(text)
                
                # Convert to legacy format for compatibility
                code_switches = []
                for switch_point in result.switch_points:
                    code_switches.append({
                        "position": switch_point.position,
                        "from_language": switch_point.from_language,
                        "to_language": switch_point.to_language,
                        "confidence": switch_point.confidence,
                        "segment": self._get_segment_at_position(
                            text, switch_point.position, result.segments
                        )
                    })
                
                logger.debug(
                    f"Enhanced detection: {len(code_switches)} switches, "
                    f"dominant={result.dominant_language}, "
                    f"frequency={result.switching_frequency:.2f}"
                )
                
                return code_switches
            
            # Fallback to basic detection if enhanced detector not available
            return await self._basic_code_switching_detection(text)
            
        except Exception as e:
            logger.error(f"Error in code-switching detection: {e}")
            return []
    
    def _get_segment_at_position(self, text: str, position: int, segments) -> str:
        """
        Get the text segment at a specific position.
        
        Args:
            text: Original text
            position: Character position
            segments: Language segments
            
        Returns:
            Text segment at position
        """
        for segment in segments:
            if segment.start_pos <= position < segment.end_pos:
                return segment.text
        
        # Fallback: return a small context around position
        start = max(0, position - 10)
        end = min(len(text), position + 10)
        return text[start:end]
    
    async def _basic_code_switching_detection(self, text: str) -> List[Dict[str, any]]:
        """
        Basic code-switching detection (fallback method).
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        # Split text into sentences/segments
        segments = self._segment_text(text)
        
        code_switches = []
        current_language = None
        position = 0
        
        for segment in segments:
            segment_text = segment.strip()
            if not segment_text:
                position += len(segment)
                continue
            
            # Detect language of current segment
            segment_language = await self.detect_language(segment_text)
            
            # Check for language switch
            if current_language and current_language != segment_language:
                code_switches.append({
                    "position": position,
                    "from_language": current_language,
                    "to_language": segment_language,
                    "confidence": 0.8,  # Simplified confidence
                    "segment": segment_text
                })
            
            current_language = segment_language
            position += len(segment)
        
        logger.debug(f"Basic detection: {len(code_switches)} code-switching points")
        return code_switches
    
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode
    ) -> str:
        """
        Translate text between languages.
        
        Note: This is a placeholder implementation. In production, this would
        integrate with translation services like Google Translate or custom models.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            # Placeholder implementation
            # In production, integrate with translation services
            logger.info(f"Translation requested: {source_lang} -> {target_lang}")
            
            # For now, return the original text with a note
            if source_lang == target_lang:
                return text
            
            # Simple placeholder logic
            return f"[Translated from {source_lang} to {target_lang}] {text}"
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return text
    
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
            # Placeholder for accent adaptation
            # In production, this would fine-tune models with regional data
            logger.info(f"Accent adaptation requested for model {model_id}")
            
            region = accent_data.get("region", "standard")
            adapted_model_id = f"{model_id}_adapted_{region}"
            
            logger.info(f"Created adapted model: {adapted_model_id}")
            return adapted_model_id
            
        except Exception as e:
            logger.error(f"Error in accent adaptation: {e}")
            return model_id
    
    async def _prepare_audio_for_whisper(self, audio: AudioBuffer) -> str:
        """
        Prepare audio buffer for Whisper processing.
        
        Args:
            audio: Audio buffer to prepare
            
        Returns:
            Path to temporary audio file
        """
        temp_fd = None
        temp_path = None
        
        try:
            # Create temporary file with proper cleanup
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="bharatvoice_")
            
            # Convert audio data to numpy array
            if hasattr(audio, 'numpy_array'):
                audio_array = audio.numpy_array
            else:
                # Convert from list to numpy array if needed
                audio_array = np.array(audio.data, dtype=np.float32)
            
            # Ensure audio is in the right format for Whisper (16kHz, mono)
            target_sample_rate = 16000
            
            if audio.sample_rate != target_sample_rate:
                try:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=audio.sample_rate, 
                        target_sr=target_sample_rate
                    )
                    logger.debug(f"Resampled audio from {audio.sample_rate}Hz to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("librosa not available for resampling, using original sample rate")
                    target_sample_rate = audio.sample_rate
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
                logger.debug("Converted stereo audio to mono")
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
                logger.debug("Normalized audio to prevent clipping")
            
            # Write audio to temporary file
            try:
                import soundfile as sf
                sf.write(temp_path, audio_array, target_sample_rate)
                logger.debug(f"Audio written to temporary file: {temp_path}")
            except ImportError:
                # Fallback to basic WAV writing if soundfile not available
                logger.warning("soundfile not available, using basic WAV writing")
                self._write_wav_file(temp_path, audio_array, target_sample_rate)
            
            return temp_path
            
        except Exception as e:
            # Clean up on error
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            logger.error(f"Failed to prepare audio for Whisper: {e}")
            raise
        finally:
            # Close file descriptor if still open
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
    
    def _write_wav_file(self, filepath: str, audio_data: np.ndarray, sample_rate: int):
        """
        Basic WAV file writing fallback when soundfile is not available.
        
        Args:
            filepath: Path to write the WAV file
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
        """
        import struct
        import wave
        
        # Convert float32 to int16
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    async def _transcribe_with_whisper(self, audio_path: str) -> Dict[str, any]:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Whisper transcription result
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if self.whisper_model is None:
                raise RuntimeError("Whisper model not initialized")
            
            # Run Whisper transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def transcribe_sync():
                return self.whisper_model.transcribe(
                    audio_path,
                    language=None,  # Auto-detect language
                    task="transcribe",
                    verbose=False,
                    fp16=False,  # Use fp32 for better compatibility
                    temperature=0.0,  # Deterministic output
                    best_of=1,  # Single pass for speed
                    beam_size=1,  # Single beam for speed
                    patience=1.0,
                    length_penalty=1.0,
                    suppress_tokens="-1",  # Don't suppress any tokens
                    initial_prompt=None,
                    condition_on_previous_text=True,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
            
            result = await loop.run_in_executor(None, transcribe_sync)
            
            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"Invalid Whisper result type: {type(result)}")
                return {"text": "", "language": "en", "segments": []}
            
            # Ensure required fields exist
            result.setdefault("text", "")
            result.setdefault("language", "en")
            result.setdefault("segments", [])
            
            logger.debug(f"Whisper transcription completed: '{result['text'][:100]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {"text": "", "language": "en", "segments": []}
    
    def _map_whisper_language_to_code(self, whisper_lang: str) -> LanguageCode:
        """
        Map Whisper language code to our LanguageCode enum.
        
        Args:
            whisper_lang: Whisper language code
            
        Returns:
            Corresponding LanguageCode
        """
        return self.language_code_map.get(whisper_lang, LanguageCode.ENGLISH_IN)
    
    def _calculate_confidence(self, whisper_result: Dict[str, any], language: LanguageCode) -> float:
        """
        Calculate confidence score for transcription result.
        
        Args:
            whisper_result: Whisper transcription result
            language: Detected language
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Extract segment-level confidence scores if available
            segments = whisper_result.get("segments", [])
            text = whisper_result.get("text", "").strip()
            
            if not text:
                return 0.0
            
            if segments:
                # Calculate average confidence from segments
                total_confidence = 0.0
                total_duration = 0.0
                
                for segment in segments:
                    # Whisper provides probability scores for tokens
                    segment_confidence = 0.5  # Default confidence
                    
                    # Use average log probability as confidence proxy
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence (rough approximation)
                        log_prob = segment["avg_logprob"]
                        # Clamp log_prob to reasonable range
                        log_prob = max(-3.0, min(0.0, log_prob))
                        segment_confidence = max(0.0, min(1.0, np.exp(log_prob)))
                    
                    # Use no_speech_prob if available (lower is better)
                    if "no_speech_prob" in segment:
                        no_speech_prob = segment["no_speech_prob"]
                        speech_confidence = 1.0 - no_speech_prob
                        segment_confidence = (segment_confidence + speech_confidence) / 2
                    
                    # Weight by segment duration
                    duration = max(0.1, segment.get("end", 0) - segment.get("start", 0))
                    total_confidence += segment_confidence * duration
                    total_duration += duration
                
                if total_duration > 0:
                    avg_confidence = total_confidence / total_duration
                else:
                    avg_confidence = 0.5
            else:
                # Fallback confidence calculation based on text characteristics
                text_length = len(text)
                if text_length > 0:
                    # Longer text generally indicates better recognition
                    length_factor = min(1.0, text_length / 50)  # Normalize to 50 chars
                    # Check for common recognition artifacts
                    artifact_penalty = 0.0
                    if text.count("...") > 2:
                        artifact_penalty += 0.2
                    if text.count("[") > 0 or text.count("]") > 0:
                        artifact_penalty += 0.1
                    
                    avg_confidence = max(0.1, min(0.9, 0.5 + length_factor * 0.3 - artifact_penalty))
                else:
                    avg_confidence = 0.0
            
            # Apply language-specific confidence factor
            language_factor = self.language_confidence_factors.get(language, 0.8)
            final_confidence = avg_confidence * language_factor
            
            # Additional quality checks
            if text:
                # Penalize very short transcriptions
                if len(text) < 3:
                    final_confidence *= 0.5
                
                # Penalize transcriptions with too many repeated characters
                if len(set(text.replace(" ", ""))) < len(text) * 0.3:
                    final_confidence *= 0.7
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _generate_alternatives(
        self, 
        audio_path: str, 
        primary_text: str
    ) -> List[AlternativeResult]:
        """
        Generate alternative transcriptions.
        
        Args:
            audio_path: Path to audio file
            primary_text: Primary transcription result
            
        Returns:
            List of alternative transcription results
        """
        try:
            alternatives = []
            
            if not os.path.exists(audio_path) or self.whisper_model is None:
                return alternatives
            
            # Try different language hints for alternatives
            language_candidates = [
                LanguageCode.HINDI, 
                LanguageCode.ENGLISH_IN, 
                LanguageCode.TAMIL,
                LanguageCode.BENGALI,
                LanguageCode.MARATHI
            ]
            
            loop = asyncio.get_event_loop()
            
            for lang_code in language_candidates:
                if len(alternatives) >= self.max_alternatives:
                    break
                
                try:
                    # Transcribe with specific language hint
                    whisper_lang = self.whisper_language_map.get(lang_code, "en")
                    
                    def transcribe_with_lang():
                        return self.whisper_model.transcribe(
                            audio_path,
                            language=whisper_lang,
                            task="transcribe",
                            verbose=False,
                            fp16=False,
                            temperature=0.2,  # Slightly higher temperature for diversity
                            best_of=2,  # Try 2 candidates
                            beam_size=2,  # Use beam search for alternatives
                            patience=1.0,
                            length_penalty=1.0,
                            suppress_tokens="-1",
                            initial_prompt=None,
                            condition_on_previous_text=False,  # Don't condition for alternatives
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                    
                    result = await loop.run_in_executor(None, transcribe_with_lang)
                    
                    alt_text = result.get("text", "").strip()
                    
                    # Only add if different from primary and not empty
                    if alt_text and alt_text != primary_text and len(alt_text) > 2:
                        # Check if this alternative is significantly different
                        similarity = self._calculate_text_similarity(primary_text, alt_text)
                        if similarity < 0.8:  # Only include if less than 80% similar
                            confidence = self._calculate_confidence(result, lang_code)
                            
                            alternatives.append(AlternativeResult(
                                text=alt_text,
                                confidence=confidence,
                                language=lang_code
                            ))
                
                except Exception as e:
                    logger.warning(f"Failed to generate alternative for {lang_code}: {e}")
                    continue
            
            # Try temperature-based alternatives if we don't have enough
            if len(alternatives) < self.max_alternatives:
                try:
                    def transcribe_with_temp():
                        return self.whisper_model.transcribe(
                            audio_path,
                            language=None,  # Auto-detect
                            task="transcribe",
                            verbose=False,
                            fp16=False,
                            temperature=0.5,  # Higher temperature for diversity
                            best_of=3,
                            beam_size=3,
                            patience=1.0,
                            length_penalty=1.0,
                            suppress_tokens="-1",
                            initial_prompt=None,
                            condition_on_previous_text=False,
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                    
                    result = await loop.run_in_executor(None, transcribe_with_temp)
                    alt_text = result.get("text", "").strip()
                    
                    if alt_text and alt_text != primary_text and len(alt_text) > 2:
                        similarity = self._calculate_text_similarity(primary_text, alt_text)
                        if similarity < 0.8:
                            detected_lang = self._map_whisper_language_to_code(result.get("language", "en"))
                            confidence = self._calculate_confidence(result, detected_lang)
                            
                            alternatives.append(AlternativeResult(
                                text=alt_text,
                                confidence=confidence,
                                language=detected_lang
                            ))
                
                except Exception as e:
                    logger.warning(f"Failed to generate temperature-based alternative: {e}")
            
            # Sort by confidence (highest first)
            alternatives.sort(key=lambda x: x.confidence, reverse=True)
            
            return alternatives[:self.max_alternatives]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    async def _detect_code_switching_in_result(
        self, 
        text: str, 
        primary_language: LanguageCode
    ) -> List[LanguageSwitchPoint]:
        """
        Detect code-switching points in transcription result.
        
        Args:
            text: Transcribed text
            primary_language: Primary detected language
            
        Returns:
            List of language switch points
        """
        try:
            if not text.strip():
                return []
            
            # Get code-switching detection results
            code_switches = await self.detect_code_switching(text)
            
            # Convert to LanguageSwitchPoint objects
            switch_points = []
            
            for switch in code_switches:
                switch_point = LanguageSwitchPoint(
                    position=switch["position"],
                    from_language=switch["from_language"],
                    to_language=switch["to_language"],
                    confidence=switch["confidence"]
                )
                switch_points.append(switch_point)
            
            return switch_points
            
        except Exception as e:
            logger.error(f"Error detecting code-switching in result: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple character-based similarity
            text1_clean = text1.lower().replace(" ", "")
            text2_clean = text2.lower().replace(" ", "")
            
            if text1_clean == text2_clean:
                return 1.0
            
            # Calculate Levenshtein distance-based similarity
            def levenshtein_distance(s1: str, s2: str) -> int:
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(text1_clean, text2_clean)
            max_len = max(len(text1_clean), len(text2_clean))
            
            if max_len == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_len)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.5
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into smaller units for language detection.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments
        """
        # Simple sentence-based segmentation
        # In production, this could use more sophisticated NLP tools
        
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?।]+', text)
        
        # Also split on commas and other punctuation for finer granularity
        segments = []
        for sentence in sentences:
            if sentence.strip():
                # Further split on commas
                sub_segments = re.split(r'[,;]+', sentence)
                segments.extend([seg.strip() for seg in sub_segments if seg.strip()])
        
        return segments
    
    def get_supported_languages(self) -> List[LanguageCode]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.whisper_language_map.keys())
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded models.
        
        Returns:
            Model information dictionary
        """
        info = {
            "whisper_model_size": self.model_size,
            "device": self.device,
            "supported_languages": [lang.value for lang in self.get_supported_languages()],
            "language_detection_enabled": self.enable_language_detection,
            "confidence_threshold": self.confidence_threshold,
            "max_alternatives": self.max_alternatives
        }
        
        # Add enhanced code-switching detector info
        if self.code_switching_detector:
            info["enhanced_code_switching"] = self.code_switching_detector.get_detection_stats()
        else:
            info["enhanced_code_switching"] = {"enabled": False}
        
        return info
    
    async def get_detailed_code_switching_analysis(
        self, 
        text: str, 
        context_language: Optional[LanguageCode] = None
    ) -> CodeSwitchingResult:
        """
        Get detailed code-switching analysis using enhanced detector.
        
        Args:
            text: Input text to analyze
            context_language: Context language for better detection
            
        Returns:
            Detailed code-switching analysis result
        """
        if self.code_switching_detector:
            return await self.code_switching_detector.detect_code_switching(
                text, context_language
            )
        else:
            # Fallback to basic analysis
            basic_switches = await self._basic_code_switching_detection(text)
            
            # Convert to CodeSwitchingResult format
            from bharatvoice.services.language_engine.code_switching_detector import (
                CodeSwitchingResult, LanguageSegment
            )
            
            # Create basic segments
            segments = [LanguageSegment(
                text=text,
                language=context_language or LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=len(text),
                confidence=0.8,
                word_boundaries=[]
            )]
            
            # Convert basic switches to LanguageSwitchPoint format
            switch_points = []
            for switch in basic_switches:
                switch_point = LanguageSwitchPoint(
                    position=switch["position"],
                    from_language=switch["from_language"],
                    to_language=switch["to_language"],
                    confidence=switch["confidence"]
                )
                switch_points.append(switch_point)
            
            return CodeSwitchingResult(
                segments=segments,
                switch_points=switch_points,
                dominant_language=context_language or LanguageCode.ENGLISH_IN,
                switching_frequency=len(switch_points) / max(1, len(text)) * 100,
                confidence=0.8,
                processing_time=0.0
            )
    
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
        if self.code_switching_detector:
            return await self.code_switching_detector.get_language_transition_suggestions(
                from_language, to_language
            )
        else:
            # Basic fallback suggestions
            return {
                'connectors': ['that is', 'I mean', 'यानी', 'मतलब'],
                'fillers': ['okay', 'so', 'अच्छा', 'well'],
                'markers': []
            }
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the ASR engine.
        
        Returns:
            Health check result
        """
        try:
            # Check if models are loaded
            whisper_status = "ok" if self.whisper_model is not None else "error"
            lang_detector_status = "ok" if self.language_detector is not None else "disabled"
            
            # Test basic functionality with dummy data
            test_audio = AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence at 16kHz
                sample_rate=16000,
                channels=1,
                duration=0.1
            )
            
            # Quick recognition test
            test_result = await self.recognize_speech(test_audio)
            recognition_status = "ok" if test_result is not None else "error"
            
            return {
                "status": "healthy" if whisper_status == "ok" else "unhealthy",
                "whisper_model": whisper_status,
                "language_detector": lang_detector_status,
                "recognition_test": recognition_status,
                "model_info": self.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_info": self.get_model_info()
            }


# Factory function for creating ASR engine
def create_multilingual_asr_engine(
    model_size: str = "base",
    device: str = "cpu",
    enable_language_detection: bool = True,
    confidence_threshold: float = 0.7,
    max_alternatives: int = 3
) -> MultilingualASREngine:
    """
    Factory function to create a multilingual ASR engine instance.
    
    Args:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Device to run inference on ("cpu" or "cuda")
        enable_language_detection: Whether to enable automatic language detection
        confidence_threshold: Minimum confidence threshold for results
        max_alternatives: Maximum number of alternative transcriptions
        
    Returns:
        Configured MultilingualASREngine instance
    """
    return MultilingualASREngine(
        model_size=model_size,
        device=device,
        enable_language_detection=enable_language_detection,
        confidence_threshold=confidence_threshold,
        max_alternatives=max_alternatives
=======
"""
Multilingual Automatic Speech Recognition (ASR) Engine for BharatVoice Assistant.

This module implements a comprehensive ASR system supporting Hindi, English, and
regional Indian languages with advanced features like confidence scoring,
alternative transcriptions, and language detection.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import tempfile
import os

import numpy as np
import whisper
from langdetect import detect, LangDetectError
from transformers import pipeline

from bharatvoice.core.interfaces import LanguageEngine
from bharatvoice.core.models import (
    AudioBuffer,
    LanguageCode,
    RecognitionResult,
    AlternativeResult,
    LanguageSwitchPoint,
)
from bharatvoice.services.language_engine.code_switching_detector import (
    EnhancedCodeSwitchingDetector,
    create_enhanced_code_switching_detector,
    CodeSwitchingResult,
)

logger = logging.getLogger(__name__)


class MultilingualASREngine(LanguageEngine):
    """
    Multilingual ASR engine supporting Indian languages with advanced features.
    
    Features:
    - Whisper-based speech recognition for high accuracy
    - Support for 10+ Indian languages
    - Confidence scoring and alternative transcriptions
    - Language detection and code-switching detection
    - Regional accent adaptation
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        enable_language_detection: bool = True,
        confidence_threshold: float = 0.7,
        max_alternatives: int = 3
    ):
        """
        Initialize the multilingual ASR engine.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run inference on ("cpu" or "cuda")
            enable_language_detection: Whether to enable automatic language detection
            confidence_threshold: Minimum confidence threshold for results
            max_alternatives: Maximum number of alternative transcriptions
        """
        self.model_size = model_size
        self.device = device
        self.enable_language_detection = enable_language_detection
        self.confidence_threshold = confidence_threshold
        self.max_alternatives = max_alternatives
        
        # Initialize Whisper model
        self.whisper_model = None
        self.language_detector = None
        
        # Initialize enhanced code-switching detector
        self.code_switching_detector = None
        
        # Language mapping for Whisper
        self.whisper_language_map = {
            LanguageCode.HINDI: "hi",
            LanguageCode.ENGLISH_IN: "en",
            LanguageCode.TAMIL: "ta",
            LanguageCode.TELUGU: "te",
            LanguageCode.BENGALI: "bn",
            LanguageCode.MARATHI: "mr",
            LanguageCode.GUJARATI: "gu",
            LanguageCode.KANNADA: "kn",
            LanguageCode.MALAYALAM: "ml",
            LanguageCode.PUNJABI: "pa",
            LanguageCode.ODIA: "or",
        }
        
        # Reverse mapping for language detection
        self.language_code_map = {v: k for k, v in self.whisper_language_map.items()}
        
        # Language-specific confidence adjustments
        self.language_confidence_factors = {
            LanguageCode.HINDI: 1.0,
            LanguageCode.ENGLISH_IN: 0.95,  # Slightly lower due to accent variations
            LanguageCode.TAMIL: 0.9,
            LanguageCode.TELUGU: 0.9,
            LanguageCode.BENGALI: 0.85,
            LanguageCode.MARATHI: 0.85,
            LanguageCode.GUJARATI: 0.8,
            LanguageCode.KANNADA: 0.8,
            LanguageCode.MALAYALAM: 0.8,
            LanguageCode.PUNJABI: 0.85,
            LanguageCode.ODIA: 0.8,
        }
        
        # Initialize components
        self._initialize_models()
        
        logger.info(f"MultilingualASREngine initialized with model_size={model_size}")
    
    def _initialize_models(self):
        """Initialize the ASR models and components."""
        try:
            # Load Whisper model with proper error handling
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.whisper_model = whisper.load_model(self.model_size, device=self.device)
                logger.info(f"Whisper model '{self.model_size}' loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{self.model_size}': {e}")
                # Try fallback to smaller model
                if self.model_size != "tiny":
                    logger.info("Attempting fallback to 'tiny' model...")
                    try:
                        self.whisper_model = whisper.load_model("tiny", device="cpu")
                        self.model_size = "tiny"
                        self.device = "cpu"
                        logger.info("Successfully loaded fallback 'tiny' model on CPU")
                    except Exception as fallback_e:
                        logger.error(f"Fallback model loading also failed: {fallback_e}")
                        raise
                else:
                    raise
            
            # Initialize language detection pipeline if enabled
            if self.enable_language_detection:
                try:
                    self.language_detector = pipeline(
                        "text-classification",
                        model="papluca/xlm-roberta-base-language-detection",
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("Language detection pipeline initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to load language detection model: {e}")
                    logger.info("Language detection will use fallback langdetect library")
                    self.language_detector = None
            
            # Initialize enhanced code-switching detector
            try:
                logger.info("Initializing enhanced code-switching detector...")
                self.code_switching_detector = create_enhanced_code_switching_detector(
                    device=self.device,
                    confidence_threshold=self.confidence_threshold,
                    min_segment_length=3,
                    enable_word_level_detection=True
                )
                logger.info("Enhanced code-switching detector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced code-switching detector: {e}")
                logger.info("Code-switching detection will use basic fallback implementation")
                self.code_switching_detector = None
            
            logger.info("ASR models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR models: {e}")
            raise
    
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input with multilingual support.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Convert audio buffer to temporary file for Whisper
            temp_audio_path = await self._prepare_audio_for_whisper(audio)
            
            try:
                # Perform speech recognition
                result = await self._transcribe_with_whisper(temp_audio_path)
                
                # Extract primary transcription and language
                primary_text = result["text"].strip()
                detected_language = self._map_whisper_language_to_code(result.get("language", "en"))
                
                # Calculate confidence score
                confidence = self._calculate_confidence(result, detected_language)
                
                # Generate alternative transcriptions
                alternatives = await self._generate_alternatives(temp_audio_path, primary_text)
                
                # Detect code-switching points
                code_switching_points = await self._detect_code_switching_in_result(
                    primary_text, detected_language
                )
                
                # Calculate processing time
                processing_time = asyncio.get_event_loop().time() - start_time
                
                recognition_result = RecognitionResult(
                    transcribed_text=primary_text,
                    confidence=confidence,
                    detected_language=detected_language,
                    code_switching_points=code_switching_points,
                    alternative_transcriptions=alternatives,
                    processing_time=processing_time
                )
                
                logger.info(
                    f"Speech recognition completed: '{primary_text[:50]}...' "
                    f"({detected_language}, confidence={confidence:.3f})"
                )
                
                return recognition_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
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
            
            # Try using the transformer-based language detector first
            if self.language_detector:
                try:
                    result = self.language_detector(text)
                    if result and len(result) > 0:
                        detected_lang = result[0]["label"].lower()
                        # Map to our language codes
                        if detected_lang in self.language_code_map:
                            return self.language_code_map[detected_lang]
                except Exception as e:
                    logger.warning(f"Transformer language detection failed: {e}")
            
            # Fallback to langdetect
            try:
                detected = detect(text)
                if detected in self.language_code_map:
                    return self.language_code_map[detected]
            except LangDetectError:
                logger.warning("Language detection failed, defaulting to English")
            
            # Default fallback
            return LanguageCode.ENGLISH_IN
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return LanguageCode.ENGLISH_IN
    
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text using enhanced detection.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        try:
            if not text.strip():
                return []
            
            # Use enhanced code-switching detector if available
            if self.code_switching_detector:
                result = await self.code_switching_detector.detect_code_switching(text)
                
                # Convert to legacy format for compatibility
                code_switches = []
                for switch_point in result.switch_points:
                    code_switches.append({
                        "position": switch_point.position,
                        "from_language": switch_point.from_language,
                        "to_language": switch_point.to_language,
                        "confidence": switch_point.confidence,
                        "segment": self._get_segment_at_position(
                            text, switch_point.position, result.segments
                        )
                    })
                
                logger.debug(
                    f"Enhanced detection: {len(code_switches)} switches, "
                    f"dominant={result.dominant_language}, "
                    f"frequency={result.switching_frequency:.2f}"
                )
                
                return code_switches
            
            # Fallback to basic detection if enhanced detector not available
            return await self._basic_code_switching_detection(text)
            
        except Exception as e:
            logger.error(f"Error in code-switching detection: {e}")
            return []
    
    def _get_segment_at_position(self, text: str, position: int, segments) -> str:
        """
        Get the text segment at a specific position.
        
        Args:
            text: Original text
            position: Character position
            segments: Language segments
            
        Returns:
            Text segment at position
        """
        for segment in segments:
            if segment.start_pos <= position < segment.end_pos:
                return segment.text
        
        # Fallback: return a small context around position
        start = max(0, position - 10)
        end = min(len(text), position + 10)
        return text[start:end]
    
    async def _basic_code_switching_detection(self, text: str) -> List[Dict[str, any]]:
        """
        Basic code-switching detection (fallback method).
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        # Split text into sentences/segments
        segments = self._segment_text(text)
        
        code_switches = []
        current_language = None
        position = 0
        
        for segment in segments:
            segment_text = segment.strip()
            if not segment_text:
                position += len(segment)
                continue
            
            # Detect language of current segment
            segment_language = await self.detect_language(segment_text)
            
            # Check for language switch
            if current_language and current_language != segment_language:
                code_switches.append({
                    "position": position,
                    "from_language": current_language,
                    "to_language": segment_language,
                    "confidence": 0.8,  # Simplified confidence
                    "segment": segment_text
                })
            
            current_language = segment_language
            position += len(segment)
        
        logger.debug(f"Basic detection: {len(code_switches)} code-switching points")
        return code_switches
    
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode
    ) -> str:
        """
        Translate text between languages.
        
        Note: This is a placeholder implementation. In production, this would
        integrate with translation services like Google Translate or custom models.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            # Placeholder implementation
            # In production, integrate with translation services
            logger.info(f"Translation requested: {source_lang} -> {target_lang}")
            
            # For now, return the original text with a note
            if source_lang == target_lang:
                return text
            
            # Simple placeholder logic
            return f"[Translated from {source_lang} to {target_lang}] {text}"
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return text
    
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
            # Placeholder for accent adaptation
            # In production, this would fine-tune models with regional data
            logger.info(f"Accent adaptation requested for model {model_id}")
            
            region = accent_data.get("region", "standard")
            adapted_model_id = f"{model_id}_adapted_{region}"
            
            logger.info(f"Created adapted model: {adapted_model_id}")
            return adapted_model_id
            
        except Exception as e:
            logger.error(f"Error in accent adaptation: {e}")
            return model_id
    
    async def _prepare_audio_for_whisper(self, audio: AudioBuffer) -> str:
        """
        Prepare audio buffer for Whisper processing.
        
        Args:
            audio: Audio buffer to prepare
            
        Returns:
            Path to temporary audio file
        """
        temp_fd = None
        temp_path = None
        
        try:
            # Create temporary file with proper cleanup
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="bharatvoice_")
            
            # Convert audio data to numpy array
            if hasattr(audio, 'numpy_array'):
                audio_array = audio.numpy_array
            else:
                # Convert from list to numpy array if needed
                audio_array = np.array(audio.data, dtype=np.float32)
            
            # Ensure audio is in the right format for Whisper (16kHz, mono)
            target_sample_rate = 16000
            
            if audio.sample_rate != target_sample_rate:
                try:
                    import librosa
                    audio_array = librosa.resample(
                        audio_array, 
                        orig_sr=audio.sample_rate, 
                        target_sr=target_sample_rate
                    )
                    logger.debug(f"Resampled audio from {audio.sample_rate}Hz to {target_sample_rate}Hz")
                except ImportError:
                    logger.warning("librosa not available for resampling, using original sample rate")
                    target_sample_rate = audio.sample_rate
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
                logger.debug("Converted stereo audio to mono")
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
                logger.debug("Normalized audio to prevent clipping")
            
            # Write audio to temporary file
            try:
                import soundfile as sf
                sf.write(temp_path, audio_array, target_sample_rate)
                logger.debug(f"Audio written to temporary file: {temp_path}")
            except ImportError:
                # Fallback to basic WAV writing if soundfile not available
                logger.warning("soundfile not available, using basic WAV writing")
                self._write_wav_file(temp_path, audio_array, target_sample_rate)
            
            return temp_path
            
        except Exception as e:
            # Clean up on error
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            logger.error(f"Failed to prepare audio for Whisper: {e}")
            raise
        finally:
            # Close file descriptor if still open
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
    
    def _write_wav_file(self, filepath: str, audio_data: np.ndarray, sample_rate: int):
        """
        Basic WAV file writing fallback when soundfile is not available.
        
        Args:
            filepath: Path to write the WAV file
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz
        """
        import struct
        import wave
        
        # Convert float32 to int16
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    async def _transcribe_with_whisper(self, audio_path: str) -> Dict[str, any]:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Whisper transcription result
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if self.whisper_model is None:
                raise RuntimeError("Whisper model not initialized")
            
            # Run Whisper transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def transcribe_sync():
                return self.whisper_model.transcribe(
                    audio_path,
                    language=None,  # Auto-detect language
                    task="transcribe",
                    verbose=False,
                    fp16=False,  # Use fp32 for better compatibility
                    temperature=0.0,  # Deterministic output
                    best_of=1,  # Single pass for speed
                    beam_size=1,  # Single beam for speed
                    patience=1.0,
                    length_penalty=1.0,
                    suppress_tokens="-1",  # Don't suppress any tokens
                    initial_prompt=None,
                    condition_on_previous_text=True,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
            
            result = await loop.run_in_executor(None, transcribe_sync)
            
            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"Invalid Whisper result type: {type(result)}")
                return {"text": "", "language": "en", "segments": []}
            
            # Ensure required fields exist
            result.setdefault("text", "")
            result.setdefault("language", "en")
            result.setdefault("segments", [])
            
            logger.debug(f"Whisper transcription completed: '{result['text'][:100]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {"text": "", "language": "en", "segments": []}
    
    def _map_whisper_language_to_code(self, whisper_lang: str) -> LanguageCode:
        """
        Map Whisper language code to our LanguageCode enum.
        
        Args:
            whisper_lang: Whisper language code
            
        Returns:
            Corresponding LanguageCode
        """
        return self.language_code_map.get(whisper_lang, LanguageCode.ENGLISH_IN)
    
    def _calculate_confidence(self, whisper_result: Dict[str, any], language: LanguageCode) -> float:
        """
        Calculate confidence score for transcription result.
        
        Args:
            whisper_result: Whisper transcription result
            language: Detected language
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Extract segment-level confidence scores if available
            segments = whisper_result.get("segments", [])
            text = whisper_result.get("text", "").strip()
            
            if not text:
                return 0.0
            
            if segments:
                # Calculate average confidence from segments
                total_confidence = 0.0
                total_duration = 0.0
                
                for segment in segments:
                    # Whisper provides probability scores for tokens
                    segment_confidence = 0.5  # Default confidence
                    
                    # Use average log probability as confidence proxy
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence (rough approximation)
                        log_prob = segment["avg_logprob"]
                        # Clamp log_prob to reasonable range
                        log_prob = max(-3.0, min(0.0, log_prob))
                        segment_confidence = max(0.0, min(1.0, np.exp(log_prob)))
                    
                    # Use no_speech_prob if available (lower is better)
                    if "no_speech_prob" in segment:
                        no_speech_prob = segment["no_speech_prob"]
                        speech_confidence = 1.0 - no_speech_prob
                        segment_confidence = (segment_confidence + speech_confidence) / 2
                    
                    # Weight by segment duration
                    duration = max(0.1, segment.get("end", 0) - segment.get("start", 0))
                    total_confidence += segment_confidence * duration
                    total_duration += duration
                
                if total_duration > 0:
                    avg_confidence = total_confidence / total_duration
                else:
                    avg_confidence = 0.5
            else:
                # Fallback confidence calculation based on text characteristics
                text_length = len(text)
                if text_length > 0:
                    # Longer text generally indicates better recognition
                    length_factor = min(1.0, text_length / 50)  # Normalize to 50 chars
                    # Check for common recognition artifacts
                    artifact_penalty = 0.0
                    if text.count("...") > 2:
                        artifact_penalty += 0.2
                    if text.count("[") > 0 or text.count("]") > 0:
                        artifact_penalty += 0.1
                    
                    avg_confidence = max(0.1, min(0.9, 0.5 + length_factor * 0.3 - artifact_penalty))
                else:
                    avg_confidence = 0.0
            
            # Apply language-specific confidence factor
            language_factor = self.language_confidence_factors.get(language, 0.8)
            final_confidence = avg_confidence * language_factor
            
            # Additional quality checks
            if text:
                # Penalize very short transcriptions
                if len(text) < 3:
                    final_confidence *= 0.5
                
                # Penalize transcriptions with too many repeated characters
                if len(set(text.replace(" ", ""))) < len(text) * 0.3:
                    final_confidence *= 0.7
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _generate_alternatives(
        self, 
        audio_path: str, 
        primary_text: str
    ) -> List[AlternativeResult]:
        """
        Generate alternative transcriptions.
        
        Args:
            audio_path: Path to audio file
            primary_text: Primary transcription result
            
        Returns:
            List of alternative transcription results
        """
        try:
            alternatives = []
            
            if not os.path.exists(audio_path) or self.whisper_model is None:
                return alternatives
            
            # Try different language hints for alternatives
            language_candidates = [
                LanguageCode.HINDI, 
                LanguageCode.ENGLISH_IN, 
                LanguageCode.TAMIL,
                LanguageCode.BENGALI,
                LanguageCode.MARATHI
            ]
            
            loop = asyncio.get_event_loop()
            
            for lang_code in language_candidates:
                if len(alternatives) >= self.max_alternatives:
                    break
                
                try:
                    # Transcribe with specific language hint
                    whisper_lang = self.whisper_language_map.get(lang_code, "en")
                    
                    def transcribe_with_lang():
                        return self.whisper_model.transcribe(
                            audio_path,
                            language=whisper_lang,
                            task="transcribe",
                            verbose=False,
                            fp16=False,
                            temperature=0.2,  # Slightly higher temperature for diversity
                            best_of=2,  # Try 2 candidates
                            beam_size=2,  # Use beam search for alternatives
                            patience=1.0,
                            length_penalty=1.0,
                            suppress_tokens="-1",
                            initial_prompt=None,
                            condition_on_previous_text=False,  # Don't condition for alternatives
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                    
                    result = await loop.run_in_executor(None, transcribe_with_lang)
                    
                    alt_text = result.get("text", "").strip()
                    
                    # Only add if different from primary and not empty
                    if alt_text and alt_text != primary_text and len(alt_text) > 2:
                        # Check if this alternative is significantly different
                        similarity = self._calculate_text_similarity(primary_text, alt_text)
                        if similarity < 0.8:  # Only include if less than 80% similar
                            confidence = self._calculate_confidence(result, lang_code)
                            
                            alternatives.append(AlternativeResult(
                                text=alt_text,
                                confidence=confidence,
                                language=lang_code
                            ))
                
                except Exception as e:
                    logger.warning(f"Failed to generate alternative for {lang_code}: {e}")
                    continue
            
            # Try temperature-based alternatives if we don't have enough
            if len(alternatives) < self.max_alternatives:
                try:
                    def transcribe_with_temp():
                        return self.whisper_model.transcribe(
                            audio_path,
                            language=None,  # Auto-detect
                            task="transcribe",
                            verbose=False,
                            fp16=False,
                            temperature=0.5,  # Higher temperature for diversity
                            best_of=3,
                            beam_size=3,
                            patience=1.0,
                            length_penalty=1.0,
                            suppress_tokens="-1",
                            initial_prompt=None,
                            condition_on_previous_text=False,
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                    
                    result = await loop.run_in_executor(None, transcribe_with_temp)
                    alt_text = result.get("text", "").strip()
                    
                    if alt_text and alt_text != primary_text and len(alt_text) > 2:
                        similarity = self._calculate_text_similarity(primary_text, alt_text)
                        if similarity < 0.8:
                            detected_lang = self._map_whisper_language_to_code(result.get("language", "en"))
                            confidence = self._calculate_confidence(result, detected_lang)
                            
                            alternatives.append(AlternativeResult(
                                text=alt_text,
                                confidence=confidence,
                                language=detected_lang
                            ))
                
                except Exception as e:
                    logger.warning(f"Failed to generate temperature-based alternative: {e}")
            
            # Sort by confidence (highest first)
            alternatives.sort(key=lambda x: x.confidence, reverse=True)
            
            return alternatives[:self.max_alternatives]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    async def _detect_code_switching_in_result(
        self, 
        text: str, 
        primary_language: LanguageCode
    ) -> List[LanguageSwitchPoint]:
        """
        Detect code-switching points in transcription result.
        
        Args:
            text: Transcribed text
            primary_language: Primary detected language
            
        Returns:
            List of language switch points
        """
        try:
            if not text.strip():
                return []
            
            # Get code-switching detection results
            code_switches = await self.detect_code_switching(text)
            
            # Convert to LanguageSwitchPoint objects
            switch_points = []
            
            for switch in code_switches:
                switch_point = LanguageSwitchPoint(
                    position=switch["position"],
                    from_language=switch["from_language"],
                    to_language=switch["to_language"],
                    confidence=switch["confidence"]
                )
                switch_points.append(switch_point)
            
            return switch_points
            
        except Exception as e:
            logger.error(f"Error detecting code-switching in result: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple character-based similarity
            text1_clean = text1.lower().replace(" ", "")
            text2_clean = text2.lower().replace(" ", "")
            
            if text1_clean == text2_clean:
                return 1.0
            
            # Calculate Levenshtein distance-based similarity
            def levenshtein_distance(s1: str, s2: str) -> int:
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(text1_clean, text2_clean)
            max_len = max(len(text1_clean), len(text2_clean))
            
            if max_len == 0:
                return 1.0
            
            similarity = 1.0 - (distance / max_len)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.5
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into smaller units for language detection.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments
        """
        # Simple sentence-based segmentation
        # In production, this could use more sophisticated NLP tools
        
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?।]+', text)
        
        # Also split on commas and other punctuation for finer granularity
        segments = []
        for sentence in sentences:
            if sentence.strip():
                # Further split on commas
                sub_segments = re.split(r'[,;]+', sentence)
                segments.extend([seg.strip() for seg in sub_segments if seg.strip()])
        
        return segments
    
    def get_supported_languages(self) -> List[LanguageCode]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.whisper_language_map.keys())
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded models.
        
        Returns:
            Model information dictionary
        """
        info = {
            "whisper_model_size": self.model_size,
            "device": self.device,
            "supported_languages": [lang.value for lang in self.get_supported_languages()],
            "language_detection_enabled": self.enable_language_detection,
            "confidence_threshold": self.confidence_threshold,
            "max_alternatives": self.max_alternatives
        }
        
        # Add enhanced code-switching detector info
        if self.code_switching_detector:
            info["enhanced_code_switching"] = self.code_switching_detector.get_detection_stats()
        else:
            info["enhanced_code_switching"] = {"enabled": False}
        
        return info
    
    async def get_detailed_code_switching_analysis(
        self, 
        text: str, 
        context_language: Optional[LanguageCode] = None
    ) -> CodeSwitchingResult:
        """
        Get detailed code-switching analysis using enhanced detector.
        
        Args:
            text: Input text to analyze
            context_language: Context language for better detection
            
        Returns:
            Detailed code-switching analysis result
        """
        if self.code_switching_detector:
            return await self.code_switching_detector.detect_code_switching(
                text, context_language
            )
        else:
            # Fallback to basic analysis
            basic_switches = await self._basic_code_switching_detection(text)
            
            # Convert to CodeSwitchingResult format
            from bharatvoice.services.language_engine.code_switching_detector import (
                CodeSwitchingResult, LanguageSegment
            )
            
            # Create basic segments
            segments = [LanguageSegment(
                text=text,
                language=context_language or LanguageCode.ENGLISH_IN,
                start_pos=0,
                end_pos=len(text),
                confidence=0.8,
                word_boundaries=[]
            )]
            
            # Convert basic switches to LanguageSwitchPoint format
            switch_points = []
            for switch in basic_switches:
                switch_point = LanguageSwitchPoint(
                    position=switch["position"],
                    from_language=switch["from_language"],
                    to_language=switch["to_language"],
                    confidence=switch["confidence"]
                )
                switch_points.append(switch_point)
            
            return CodeSwitchingResult(
                segments=segments,
                switch_points=switch_points,
                dominant_language=context_language or LanguageCode.ENGLISH_IN,
                switching_frequency=len(switch_points) / max(1, len(text)) * 100,
                confidence=0.8,
                processing_time=0.0
            )
    
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
        if self.code_switching_detector:
            return await self.code_switching_detector.get_language_transition_suggestions(
                from_language, to_language
            )
        else:
            # Basic fallback suggestions
            return {
                'connectors': ['that is', 'I mean', 'यानी', 'मतलब'],
                'fillers': ['okay', 'so', 'अच्छा', 'well'],
                'markers': []
            }
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the ASR engine.
        
        Returns:
            Health check result
        """
        try:
            # Check if models are loaded
            whisper_status = "ok" if self.whisper_model is not None else "error"
            lang_detector_status = "ok" if self.language_detector is not None else "disabled"
            
            # Test basic functionality with dummy data
            test_audio = AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence at 16kHz
                sample_rate=16000,
                channels=1,
                duration=0.1
            )
            
            # Quick recognition test
            test_result = await self.recognize_speech(test_audio)
            recognition_status = "ok" if test_result is not None else "error"
            
            return {
                "status": "healthy" if whisper_status == "ok" else "unhealthy",
                "whisper_model": whisper_status,
                "language_detector": lang_detector_status,
                "recognition_test": recognition_status,
                "model_info": self.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_info": self.get_model_info()
            }


# Factory function for creating ASR engine
def create_multilingual_asr_engine(
    model_size: str = "base",
    device: str = "cpu",
    enable_language_detection: bool = True,
    confidence_threshold: float = 0.7,
    max_alternatives: int = 3
) -> MultilingualASREngine:
    """
    Factory function to create a multilingual ASR engine instance.
    
    Args:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Device to run inference on ("cpu" or "cuda")
        enable_language_detection: Whether to enable automatic language detection
        confidence_threshold: Minimum confidence threshold for results
        max_alternatives: Maximum number of alternative transcriptions
        
    Returns:
        Configured MultilingualASREngine instance
    """
    return MultilingualASREngine(
        model_size=model_size,
        device=device,
        enable_language_detection=enable_language_detection,
        confidence_threshold=confidence_threshold,
        max_alternatives=max_alternatives
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )