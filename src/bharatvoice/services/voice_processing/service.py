"""
Voice Processing Service implementation for BharatVoice Assistant.

This module provides the main voice processing service that integrates
audio processing, voice activity detection, and text-to-speech synthesis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from bharatvoice.core.interfaces import AudioProcessor as AudioProcessorInterface
from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    VoiceActivityResult,
)
from bharatvoice.services.voice_processing.audio_processor import (
    AudioFormatConverter,
    AudioProcessor,
    RealTimeAudioProcessor,
)
from bharatvoice.services.voice_processing.tts_engine import (
    AdaptiveTTSEngine,
    TTSEngine,
)

logger = logging.getLogger(__name__)


class VoiceProcessingService(AudioProcessorInterface):
    """
    Main voice processing service that coordinates audio processing and TTS.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        noise_reduction_factor: float = 0.5,
        enable_adaptive_tts: bool = True
    ):
        """
        Initialize voice processing service.
        
        Args:
            sample_rate: Audio sample rate in Hz
            vad_aggressiveness: VAD aggressiveness level (0-3)
            noise_reduction_factor: Noise reduction strength (0.0-1.0)
            enable_adaptive_tts: Whether to use adaptive TTS engine
        """
        self.sample_rate = sample_rate
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            vad_aggressiveness=vad_aggressiveness,
            noise_reduction_factor=noise_reduction_factor
        )
        
        # Initialize TTS engine
        if enable_adaptive_tts:
            self.tts_engine = AdaptiveTTSEngine(sample_rate=22050, quality='high')
        else:
            self.tts_engine = TTSEngine(sample_rate=22050, quality='high')
        
        # Initialize real-time processor
        self.realtime_processor = RealTimeAudioProcessor(
            self.audio_processor,
            buffer_size=1024,
            overlap_ratio=0.5
        )
        
        # Format converter
        self.format_converter = AudioFormatConverter()
        
        # Service state
        self.is_initialized = True
        self.processing_stats = {
            'total_processed': 0,
            'total_synthesized': 0,
            'average_processing_time': 0.0,
            'vad_detections': 0
        }
        
        logger.info("VoiceProcessingService initialized successfully")
    
    async def process_audio_stream(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio stream with language-specific optimizations.
        
        Args:
            audio_data: Input audio buffer
            language: Target language for processing
            
        Returns:
            Processed audio buffer
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Process audio using the audio processor
            processed_audio = await self.audio_processor.process_audio_stream(
                audio_data, language
            )
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_processing_stats(processing_time)
            
            logger.debug(f"Processed audio stream in {processing_time:.3f}s")
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error in process_audio_stream: {e}")
            raise
    
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        try:
            vad_result = await self.audio_processor.detect_voice_activity(audio_frame)
            
            # Update statistics
            if vad_result.is_speech:
                self.processing_stats['vad_detections'] += 1
            
            return vad_result
            
        except Exception as e:
            logger.error(f"Error in detect_voice_activity: {e}")
            raise
    
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD,
        quality_optimize: bool = True
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            quality_optimize: Whether to apply quality optimization
            
        Returns:
            Synthesized audio buffer
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Synthesize speech using TTS engine
            synthesized_audio = await self.tts_engine.synthesize_speech(
                text, language, accent, quality_optimize=quality_optimize
            )
            
            # Convert to target sample rate if needed
            if synthesized_audio.sample_rate != self.sample_rate:
                synthesized_audio = self.format_converter.convert_format(
                    synthesized_audio,
                    target_format=AudioFormat.WAV,
                    target_sample_rate=self.sample_rate
                )
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.processing_stats['total_synthesized'] += 1
            
            logger.debug(f"Synthesized speech in {processing_time:.3f}s")
            return synthesized_audio
            
        except Exception as e:
            logger.error(f"Error in synthesize_speech: {e}")
            raise
    
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise from audio data.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        try:
            return await self.audio_processor.filter_background_noise(audio_data)
            
        except Exception as e:
            logger.error(f"Error in filter_background_noise: {e}")
            raise
    
    async def process_realtime_stream(
        self, 
        audio_chunk: List[float], 
        language: LanguageCode
    ) -> Tuple[Optional[AudioBuffer], List[VoiceActivityResult]]:
        """
        Process real-time audio stream chunk.
        
        Args:
            audio_chunk: New audio chunk as list of floats
            language: Target language for processing
            
        Returns:
            Tuple of (processed_audio_buffer, vad_results)
        """
        try:
            import numpy as np
            audio_array = np.array(audio_chunk, dtype=np.float32)
            
            return await self.realtime_processor.process_stream(audio_array, language)
            
        except Exception as e:
            logger.error(f"Error in process_realtime_stream: {e}")
            return None, []
    
    async def synthesize_for_user(
        self, 
        text: str, 
        language: LanguageCode, 
        user_id: str,
        accent: Optional[AccentType] = None
    ) -> AudioBuffer:
        """
        Synthesize speech adapted to user preferences (if adaptive TTS is enabled).
        
        Args:
            text: Text to synthesize
            language: Target language
            user_id: User identifier
            accent: Optional accent override
            
        Returns:
            User-adapted synthesized audio buffer
        """
        try:
            if isinstance(self.tts_engine, AdaptiveTTSEngine):
                return await self.tts_engine.synthesize_for_user(
                    text, language, user_id, accent
                )
            else:
                # Fall back to regular synthesis
                accent = accent or AccentType.STANDARD
                return await self.synthesize_speech(text, language, accent)
                
        except Exception as e:
            logger.error(f"Error in synthesize_for_user: {e}")
            raise
    
    async def preprocess_for_recognition(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Preprocess audio for speech recognition.
        
        Args:
            audio_data: Input audio buffer
            
        Returns:
            Preprocessed audio buffer optimized for ASR
        """
        try:
            # Apply noise filtering first
            filtered_audio = await self.filter_background_noise(audio_data)
            
            # Convert to optimal format for ASR
            preprocessed_audio = self.format_converter.preprocess_for_recognition(
                filtered_audio
            )
            
            logger.debug("Preprocessed audio for recognition")
            return preprocessed_audio
            
        except Exception as e:
            logger.error(f"Error in preprocess_for_recognition: {e}")
            raise
    
    async def extract_audio_features(self, audio_data: AudioBuffer) -> Dict[str, any]:
        """
        Extract audio features for analysis.
        
        Args:
            audio_data: Input audio buffer
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = self.format_converter.extract_features(audio_data)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_features = {}
            for key, value in features.items():
                if hasattr(value, 'tolist'):
                    serializable_features[key] = value.tolist()
                else:
                    serializable_features[key] = value
            
            logger.debug("Extracted audio features")
            return serializable_features
            
        except Exception as e:
            logger.error(f"Error in extract_audio_features: {e}")
            raise
    
    def update_user_tts_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, any]
    ):
        """
        Update user preferences for TTS synthesis.
        
        Args:
            user_id: User identifier
            preferences: User preferences dictionary
        """
        if isinstance(self.tts_engine, AdaptiveTTSEngine):
            self.tts_engine.update_user_preferences(user_id, preferences)
            logger.info(f"Updated TTS preferences for user {user_id}")
        else:
            logger.warning("Adaptive TTS not enabled, cannot update user preferences")
    
    def record_tts_feedback(
        self, 
        user_id: str, 
        text: str, 
        language: LanguageCode,
        rating: float,
        feedback_type: str = "general"
    ):
        """
        Record user feedback for TTS quality.
        
        Args:
            user_id: User identifier
            text: Synthesized text
            language: Language used
            rating: User rating (0.0 to 5.0)
            feedback_type: Type of feedback
        """
        if isinstance(self.tts_engine, AdaptiveTTSEngine):
            self.tts_engine.record_feedback(
                user_id, text, language, rating, feedback_type
            )
            logger.info(f"Recorded TTS feedback from user {user_id}")
        else:
            logger.warning("Adaptive TTS not enabled, cannot record feedback")
    
    async def synthesize_streaming(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD,
        chunk_duration: float = 0.5
    ):
        """
        Synthesize speech with streaming output for real-time playback.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            chunk_duration: Duration of each audio chunk in seconds
            
        Yields:
            Audio chunks as AudioBuffer objects
        """
        try:
            async for chunk in self.tts_engine.synthesize_streaming(
                text, language, accent, chunk_duration
            ):
                # Convert to target sample rate if needed
                if chunk.sample_rate != self.sample_rate:
                    chunk = self.format_converter.convert_format(
                        chunk,
                        target_format=AudioFormat.WAV,
                        target_sample_rate=self.sample_rate
                    )
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            raise
    
    async def synthesize_to_format(
        self, 
        text: str, 
        language: LanguageCode, 
        output_format: AudioFormat,
        accent: AccentType = AccentType.STANDARD,
        bitrate: str = "128k"
    ) -> bytes:
        """
        Synthesize speech and return in specified audio format.
        
        Args:
            text: Text to synthesize
            language: Target language
            output_format: Desired output format
            accent: Regional accent type
            bitrate: Audio bitrate for compressed formats
            
        Returns:
            Audio data as bytes in specified format
        """
        try:
            return await self.tts_engine.synthesize_to_format(
                text, language, output_format, accent, bitrate
            )
            
        except Exception as e:
            logger.error(f"Error in format synthesis: {e}")
            raise
    
    async def synthesize_with_pauses(
        self, 
        text_segments: List[str], 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD,
        pause_duration: float = 0.5
    ) -> AudioBuffer:
        """
        Synthesize multiple text segments with pauses between them.
        
        Args:
            text_segments: List of text segments to synthesize
            language: Target language
            accent: Regional accent type
            pause_duration: Duration of pauses between segments in seconds
            
        Returns:
            Combined audio buffer with pauses
        """
        try:
            combined_audio = await self.tts_engine.synthesize_with_pauses(
                text_segments, language, accent, pause_duration
            )
            
            # Convert to target sample rate if needed
            if combined_audio.sample_rate != self.sample_rate:
                combined_audio = self.format_converter.convert_format(
                    combined_audio,
                    target_format=AudioFormat.WAV,
                    target_sample_rate=self.sample_rate
                )
            
            return combined_audio
            
        except Exception as e:
            logger.error(f"Error in multi-segment synthesis: {e}")
            raise
    
    def save_synthesized_audio(
        self, 
        audio_buffer: AudioBuffer, 
        file_path: str, 
        format: AudioFormat = AudioFormat.WAV
    ):
        """
        Save synthesized audio buffer to file.
        
        Args:
            audio_buffer: Audio buffer to save
            file_path: Output file path
            format: Audio format for output file
        """
        try:
            self.tts_engine.save_audio_to_file(audio_buffer, file_path, format)
            
        except Exception as e:
            logger.error(f"Error saving synthesized audio: {e}")
            raise
    
    def estimate_synthesis_time(self, text: str, language: LanguageCode) -> float:
        """
        Estimate synthesis time for given text.
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Estimated synthesis time in seconds
        """
        return self.tts_engine.estimate_synthesis_time(text, language)
    
    def reset_realtime_buffer(self):
        """Reset the real-time processing buffer."""
        self.realtime_processor.reset_buffer()
        logger.debug("Reset real-time processing buffer")
    
    def get_service_stats(self) -> Dict[str, any]:
        """
        Get voice processing service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        stats = self.processing_stats.copy()
        
        # Add TTS cache stats if available
        if hasattr(self.tts_engine, 'get_cache_stats'):
            stats['tts_cache'] = self.tts_engine.get_cache_stats()
        
        return stats
    
    def clear_caches(self):
        """Clear all internal caches."""
        if hasattr(self.tts_engine, 'clear_cache'):
            self.tts_engine.clear_cache()
        
        logger.info("Cleared voice processing service caches")
    
    async def health_check(self) -> Dict[str, any]:
        """
        Perform health check of the voice processing service.
        
        Returns:
            Health check result
        """
        try:
            # Test basic audio processing
            test_audio = AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence at 16kHz
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
            
            # Test VAD
            vad_result = await self.detect_voice_activity(test_audio)
            
            # Test TTS
            tts_result = await self.synthesize_speech(
                "Test", LanguageCode.ENGLISH_IN, AccentType.STANDARD
            )
            
            return {
                'status': 'healthy',
                'audio_processor': 'ok',
                'vad': 'ok' if vad_result else 'error',
                'tts': 'ok' if tts_result else 'error',
                'stats': self.get_service_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'stats': self.get_service_stats()
            }
    
    def _update_processing_stats(self, processing_time: float):
        """
        Update processing statistics.
        
        Args:
            processing_time: Processing time in seconds
        """
        self.processing_stats['total_processed'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['average_processing_time']
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.processing_stats['average_processing_time'] = new_avg


# Factory function for creating voice processing service
def create_voice_processing_service(
    sample_rate: int = 16000,
    vad_aggressiveness: int = 2,
    noise_reduction_factor: float = 0.5,
    enable_adaptive_tts: bool = True
) -> VoiceProcessingService:
    """
    Factory function to create a voice processing service instance.
    
    Args:
        sample_rate: Audio sample rate in Hz
        vad_aggressiveness: VAD aggressiveness level (0-3)
        noise_reduction_factor: Noise reduction strength (0.0-1.0)
        enable_adaptive_tts: Whether to use adaptive TTS engine
        
    Returns:
        Configured VoiceProcessingService instance
    """
    return VoiceProcessingService(
        sample_rate=sample_rate,
        vad_aggressiveness=vad_aggressiveness,
        noise_reduction_factor=noise_reduction_factor,
        enable_adaptive_tts=enable_adaptive_tts
    )