"""
Text-to-Speech (TTS) engine implementation for BharatVoice Assistant.

This module provides TTS synthesis with support for Indian languages and regional accents.
Enhanced with quality optimization, streaming capabilities, and advanced accent adaptation.
"""

import asyncio
import io
import logging
import tempfile
import wave
from typing import Dict, List, Optional, Generator, Tuple

import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from scipy import signal
from scipy.io import wavfile

from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech engine with support for Indian languages and accents.
    Enhanced with quality optimization, streaming, and advanced accent adaptation.
    """
    
    # Language mapping for gTTS
    LANGUAGE_MAPPING = {
        LanguageCode.HINDI: 'hi',
        LanguageCode.ENGLISH_IN: 'en',
        LanguageCode.TAMIL: 'ta',
        LanguageCode.TELUGU: 'te',
        LanguageCode.BENGALI: 'bn',
        LanguageCode.MARATHI: 'mr',
        LanguageCode.GUJARATI: 'gu',
        LanguageCode.KANNADA: 'kn',
        LanguageCode.MALAYALAM: 'ml',
        LanguageCode.PUNJABI: 'pa',
        LanguageCode.ODIA: 'or',
    }
    
    # Enhanced regional accent configurations with more parameters
    ACCENT_CONFIGS = {
        AccentType.STANDARD: {
            'speed': 1.0, 
            'pitch_shift': 0, 
            'formant_shift': 0.0,
            'emphasis_factor': 1.0,
            'pause_duration': 1.0
        },
        AccentType.NORTH_INDIAN: {
            'speed': 0.95, 
            'pitch_shift': -2, 
            'formant_shift': -0.05,
            'emphasis_factor': 1.1,
            'pause_duration': 1.2
        },
        AccentType.SOUTH_INDIAN: {
            'speed': 1.05, 
            'pitch_shift': 1, 
            'formant_shift': 0.03,
            'emphasis_factor': 0.9,
            'pause_duration': 0.8
        },
        AccentType.WEST_INDIAN: {
            'speed': 1.0, 
            'pitch_shift': -1, 
            'formant_shift': -0.02,
            'emphasis_factor': 1.05,
            'pause_duration': 0.9
        },
        AccentType.EAST_INDIAN: {
            'speed': 0.9, 
            'pitch_shift': 0, 
            'formant_shift': 0.02,
            'emphasis_factor': 0.95,
            'pause_duration': 1.1
        },
        AccentType.MUMBAI: {
            'speed': 1.1, 
            'pitch_shift': -1, 
            'formant_shift': -0.03,
            'emphasis_factor': 1.15,
            'pause_duration': 0.7
        },
        AccentType.DELHI: {
            'speed': 0.95, 
            'pitch_shift': -2, 
            'formant_shift': -0.04,
            'emphasis_factor': 1.2,
            'pause_duration': 1.3
        },
        AccentType.BANGALORE: {
            'speed': 1.05, 
            'pitch_shift': 1, 
            'formant_shift': 0.04,
            'emphasis_factor': 0.85,
            'pause_duration': 0.8
        },
        AccentType.CHENNAI: {
            'speed': 1.0, 
            'pitch_shift': 2, 
            'formant_shift': 0.05,
            'emphasis_factor': 0.9,
            'pause_duration': 0.9
        },
        AccentType.KOLKATA: {
            'speed': 0.9, 
            'pitch_shift': 0, 
            'formant_shift': 0.03,
            'emphasis_factor': 0.95,
            'pause_duration': 1.2
        },
    }
    
    # Quality optimization settings
    QUALITY_SETTINGS = {
        'high': {
            'sample_rate': 22050,
            'bitrate': '128k',
            'normalize': True,
            'compress': True,
            'noise_gate': True,
            'eq_boost': True
        },
        'medium': {
            'sample_rate': 16000,
            'bitrate': '96k',
            'normalize': True,
            'compress': False,
            'noise_gate': False,
            'eq_boost': False
        },
        'low': {
            'sample_rate': 8000,
            'bitrate': '64k',
            'normalize': False,
            'compress': False,
            'noise_gate': False,
            'eq_boost': False
        }
    }
    
    def __init__(self, sample_rate: int = 22050, quality: str = 'high'):
        """
        Initialize TTS engine.
        
        Args:
            sample_rate: Target sample rate for synthesized audio
            quality: Quality setting ('high', 'medium', 'low')
        """
        self.sample_rate = sample_rate
        self.quality = quality
        self.quality_config = self.QUALITY_SETTINGS.get(quality, self.QUALITY_SETTINGS['high'])
        self.cache: Dict[str, AudioBuffer] = {}
        self.max_cache_size = 100
        
        # Streaming configuration
        self.chunk_size = 1024  # Samples per chunk for streaming
        self.streaming_enabled = True
        
        logger.info(f"TTSEngine initialized with sample_rate={sample_rate}, quality={quality}")
    
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD,
        use_cache: bool = True,
        quality_optimize: bool = True
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            use_cache: Whether to use caching for repeated text
            quality_optimize: Whether to apply quality optimization
            
        Returns:
            Synthesized audio buffer
        """
        try:
            # Create cache key
            cache_key = f"{text}_{language}_{accent}_{quality_optimize}"
            
            # Check cache first
            if use_cache and cache_key in self.cache:
                logger.debug(f"Using cached TTS for: {text[:50]}...")
                return self.cache[cache_key]
            
            # Validate language support
            if language not in self.LANGUAGE_MAPPING:
                logger.warning(f"Language {language} not supported, falling back to English")
                language = LanguageCode.ENGLISH_IN
            
            # Synthesize speech
            audio_buffer = await self._synthesize_with_gtts(text, language, accent, quality_optimize)
            
            # Cache result if enabled
            if use_cache:
                self._add_to_cache(cache_key, audio_buffer)
            
            logger.info(f"Synthesized speech for text: '{text[:50]}...' in {language}")
            return audio_buffer
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            # Return silence as fallback
            return self._generate_silence(1.0)
    
    async def _synthesize_with_gtts(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType,
        quality_optimize: bool = True
    ) -> AudioBuffer:
        """
        Synthesize speech using Google Text-to-Speech with quality optimization.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            quality_optimize: Whether to apply quality optimization
            
        Returns:
            Synthesized audio buffer
        """
        # Get language code for gTTS
        gtts_lang = self.LANGUAGE_MAPPING[language]
        
        # Handle Indian English specifically
        tld = 'co.in' if language == LanguageCode.ENGLISH_IN else 'com'
        
        # Create gTTS object with optimized settings
        tts = gTTS(
            text=text, 
            lang=gtts_lang, 
            tld=tld, 
            slow=False,
            lang_check=True
        )
        
        # Generate audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            tts.save(temp_file.name)
            
            # Load audio with pydub
            audio_segment = AudioSegment.from_mp3(temp_file.name)
            
            # Apply quality optimization if enabled
            if quality_optimize:
                audio_segment = self._apply_quality_optimization(audio_segment)
            
            # Apply accent modifications
            audio_segment = self._apply_enhanced_accent_modifications(audio_segment, accent)
            
            # Convert to target sample rate and format
            target_sample_rate = self.quality_config['sample_rate']
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)
            audio_segment = audio_segment.set_channels(1)  # Mono
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize audio
            if len(audio_array) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize
            
            # Create AudioBuffer
            duration = len(audio_array) / target_sample_rate
            
            return AudioBuffer(
                data=audio_array.tolist(),
                sample_rate=target_sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
    
    def _apply_quality_optimization(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Apply quality optimization to audio segment.
        
        Args:
            audio_segment: Input audio segment
            
        Returns:
            Quality-optimized audio segment
        """
        try:
            # Apply normalization
            if self.quality_config['normalize']:
                audio_segment = normalize(audio_segment)
            
            # Apply dynamic range compression
            if self.quality_config['compress']:
                audio_segment = compress_dynamic_range(audio_segment)
            
            # Apply noise gate (simple implementation)
            if self.quality_config['noise_gate']:
                audio_segment = self._apply_noise_gate(audio_segment)
            
            # Apply EQ boost for clarity
            if self.quality_config['eq_boost']:
                audio_segment = self._apply_eq_boost(audio_segment)
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Quality optimization failed: {e}")
            return audio_segment
    
    def _apply_noise_gate(self, audio_segment: AudioSegment, threshold_db: float = -40.0) -> AudioSegment:
        """
        Apply simple noise gate to reduce background noise.
        
        Args:
            audio_segment: Input audio segment
            threshold_db: Noise gate threshold in dB
            
        Returns:
            Noise-gated audio segment
        """
        try:
            # Convert to numpy array for processing
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Calculate RMS energy in sliding windows
            window_size = int(0.01 * audio_segment.frame_rate)  # 10ms windows
            
            for i in range(0, len(samples) - window_size, window_size):
                window = samples[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_db = 20 * np.log10(rms + 1e-10)
                
                # Apply gate
                if rms_db < threshold_db:
                    samples[i:i + window_size] *= 0.1  # Reduce by 20dB
            
            # Convert back to AudioSegment
            samples = (samples * 32767).astype(np.int16)
            return AudioSegment(
                samples.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=2,
                channels=1
            )
            
        except Exception as e:
            logger.warning(f"Noise gate failed: {e}")
            return audio_segment
    
    def _apply_eq_boost(self, audio_segment: AudioSegment) -> AudioSegment:
        """
        Apply EQ boost for speech clarity.
        
        Args:
            audio_segment: Input audio segment
            
        Returns:
            EQ-boosted audio segment
        """
        try:
            # Simple high-frequency boost for clarity
            # This is a simplified implementation - in production, use proper EQ
            return audio_segment + 2  # Boost by 2dB
            
        except Exception as e:
            logger.warning(f"EQ boost failed: {e}")
            return audio_segment
    
    def _apply_enhanced_accent_modifications(
        self, 
        audio_segment: AudioSegment, 
        accent: AccentType
    ) -> AudioSegment:
        """
        Apply enhanced accent-specific modifications to audio.
        
        Args:
            audio_segment: Input audio segment
            accent: Target accent type
            
        Returns:
            Modified audio segment with enhanced accent processing
        """
        if accent not in self.ACCENT_CONFIGS:
            return audio_segment
        
        config = self.ACCENT_CONFIGS[accent]
        
        try:
            # Apply speed modification
            if config['speed'] != 1.0:
                audio_segment = self._apply_speed_change(audio_segment, config['speed'])
            
            # Apply pitch shift
            if config['pitch_shift'] != 0:
                audio_segment = self._apply_pitch_shift(audio_segment, config['pitch_shift'])
            
            # Apply formant shift for more natural accent
            if config.get('formant_shift', 0.0) != 0.0:
                audio_segment = self._apply_formant_shift(audio_segment, config['formant_shift'])
            
            # Apply emphasis factor
            if config.get('emphasis_factor', 1.0) != 1.0:
                audio_segment = self._apply_emphasis(audio_segment, config['emphasis_factor'])
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Accent modification failed for {accent}: {e}")
            return audio_segment
    
    def _apply_speed_change(self, audio_segment: AudioSegment, speed_factor: float) -> AudioSegment:
        """
        Apply speed change while preserving pitch.
        
        Args:
            audio_segment: Input audio segment
            speed_factor: Speed multiplication factor
            
        Returns:
            Speed-modified audio segment
        """
        try:
            # Use pydub's speedup method which preserves pitch better
            if speed_factor > 1.0:
                # Speed up
                return audio_segment.speedup(playback_speed=speed_factor)
            elif speed_factor < 1.0:
                # Slow down by changing frame rate temporarily
                new_frame_rate = int(audio_segment.frame_rate * speed_factor)
                slowed = audio_segment._spawn(
                    audio_segment.raw_data,
                    overrides={"frame_rate": new_frame_rate}
                )
                return slowed.set_frame_rate(audio_segment.frame_rate)
            else:
                return audio_segment
                
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return audio_segment
    
    def _apply_pitch_shift(self, audio_segment: AudioSegment, semitones: int) -> AudioSegment:
        """
        Apply pitch shift using improved algorithm.
        
        Args:
            audio_segment: Input audio segment
            semitones: Pitch shift in semitones
            
        Returns:
            Pitch-shifted audio segment
        """
        try:
            if semitones == 0:
                return audio_segment
            
            # Convert to numpy array for processing
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio_segment.frame_rate
            
            # Apply pitch shift using phase vocoder approach (simplified)
            shift_factor = 2.0 ** (semitones / 12.0)
            
            # Simple pitch shift by resampling (not perfect but functional)
            new_length = int(len(samples) / shift_factor)
            if new_length > 0:
                shifted_samples = signal.resample(samples, new_length)
                
                # Pad or truncate to original length
                if len(shifted_samples) < len(samples):
                    # Pad with zeros
                    padded = np.zeros(len(samples))
                    padded[:len(shifted_samples)] = shifted_samples
                    shifted_samples = padded
                else:
                    # Truncate
                    shifted_samples = shifted_samples[:len(samples)]
                
                # Convert back to AudioSegment
                shifted_samples = (shifted_samples * 32767).astype(np.int16)
                return AudioSegment(
                    shifted_samples.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio_segment
    
    def _apply_formant_shift(self, audio_segment: AudioSegment, shift_factor: float) -> AudioSegment:
        """
        Apply formant shifting for more natural accent adaptation.
        
        Args:
            audio_segment: Input audio segment
            shift_factor: Formant shift factor (-1.0 to 1.0)
            
        Returns:
            Formant-shifted audio segment
        """
        try:
            # This is a simplified formant shift implementation
            # In production, use more sophisticated formant analysis and synthesis
            
            if abs(shift_factor) < 0.01:
                return audio_segment
            
            # Apply a simple spectral shift as approximation
            # Real formant shifting requires complex spectral analysis
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Apply a simple filter to approximate formant shift
            if shift_factor > 0:
                # Boost higher frequencies slightly
                audio_segment = audio_segment + 1
            else:
                # Boost lower frequencies slightly
                audio_segment = audio_segment.low_pass_filter(3000)
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Formant shift failed: {e}")
            return audio_segment
    
    def _apply_emphasis(self, audio_segment: AudioSegment, emphasis_factor: float) -> AudioSegment:
        """
        Apply emphasis modification for accent characteristics.
        
        Args:
            audio_segment: Input audio segment
            emphasis_factor: Emphasis multiplication factor
            
        Returns:
            Emphasis-modified audio segment
        """
        try:
            if abs(emphasis_factor - 1.0) < 0.01:
                return audio_segment
            
            # Apply dynamic range modification
            if emphasis_factor > 1.0:
                # Increase dynamic range
                return audio_segment + int(20 * np.log10(emphasis_factor))
            else:
                # Decrease dynamic range
                return audio_segment - int(20 * np.log10(1.0 / emphasis_factor))
                
        except Exception as e:
            logger.warning(f"Emphasis modification failed: {e}")
            return audio_segment
    
    async def synthesize_streaming(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD,
        chunk_duration: float = 0.5
    ) -> Generator[AudioBuffer, None, None]:
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
            # First synthesize the complete audio
            complete_audio = await self.synthesize_speech(text, language, accent, use_cache=False)
            
            # Split into chunks for streaming
            chunk_size = int(chunk_duration * complete_audio.sample_rate)
            audio_data = complete_audio.data
            
            for i in range(0, len(audio_data), chunk_size):
                chunk_data = audio_data[i:i + chunk_size]
                
                if len(chunk_data) > 0:
                    chunk_duration_actual = len(chunk_data) / complete_audio.sample_rate
                    
                    chunk_buffer = AudioBuffer(
                        data=chunk_data,
                        sample_rate=complete_audio.sample_rate,
                        channels=complete_audio.channels,
                        format=complete_audio.format,
                        duration=chunk_duration_actual
                    )
                    
                    yield chunk_buffer
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            # Yield silence as fallback
            silence = self._generate_silence(chunk_duration)
            yield silence
    
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
            # Synthesize audio
            audio_buffer = await self.synthesize_speech(text, language, accent)
            
            # Convert to AudioSegment
            audio_array = np.array(audio_buffer.data, dtype=np.float32)
            audio_array = (audio_array * 32767).astype(np.int16)
            
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=audio_buffer.sample_rate,
                sample_width=2,
                channels=audio_buffer.channels
            )
            
            # Export to desired format
            output_buffer = io.BytesIO()
            
            if output_format == AudioFormat.WAV:
                audio_segment.export(output_buffer, format="wav")
            elif output_format == AudioFormat.MP3:
                audio_segment.export(output_buffer, format="mp3", bitrate=bitrate)
            elif output_format == AudioFormat.FLAC:
                audio_segment.export(output_buffer, format="flac")
            elif output_format == AudioFormat.OGG:
                audio_segment.export(output_buffer, format="ogg", codec="libvorbis")
            else:
                # Default to WAV
                audio_segment.export(output_buffer, format="wav")
            
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error in format conversion: {e}")
            return b""
    
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
            combined_audio_data = []
            sample_rate = self.quality_config['sample_rate']
            
            for i, segment in enumerate(text_segments):
                if segment.strip():  # Skip empty segments
                    # Synthesize segment
                    segment_audio = await self.synthesize_speech(segment, language, accent)
                    combined_audio_data.extend(segment_audio.data)
                    
                    # Add pause between segments (except after last segment)
                    if i < len(text_segments) - 1:
                        pause_samples = int(pause_duration * sample_rate)
                        combined_audio_data.extend([0.0] * pause_samples)
            
            # Create combined audio buffer
            total_duration = len(combined_audio_data) / sample_rate
            
            return AudioBuffer(
                data=combined_audio_data,
                sample_rate=sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=total_duration
            )
            
        except Exception as e:
            logger.error(f"Error in multi-segment synthesis: {e}")
            return self._generate_silence(1.0)
    
    def save_audio_to_file(self, audio_buffer: AudioBuffer, file_path: str, format: AudioFormat = AudioFormat.WAV):
        """
        Save audio buffer to file.
        
        Args:
            audio_buffer: Audio buffer to save
            file_path: Output file path
            format: Audio format for output file
        """
        try:
            # Convert to AudioSegment
            audio_array = np.array(audio_buffer.data, dtype=np.float32)
            audio_array = (audio_array * 32767).astype(np.int16)
            
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=audio_buffer.sample_rate,
                sample_width=2,
                channels=audio_buffer.channels
            )
            
            # Export to file
            if format == AudioFormat.WAV:
                audio_segment.export(file_path, format="wav")
            elif format == AudioFormat.MP3:
                audio_segment.export(file_path, format="mp3", bitrate=self.quality_config['bitrate'])
            elif format == AudioFormat.FLAC:
                audio_segment.export(file_path, format="flac")
            elif format == AudioFormat.OGG:
                audio_segment.export(file_path, format="ogg", codec="libvorbis")
            
            logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving audio to file: {e}")
    
    def estimate_synthesis_time(self, text: str, language: LanguageCode) -> float:
        """
        Estimate synthesis time for given text.
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Estimated synthesis time in seconds
        """
        # Simple estimation based on text length and language
        base_time_per_char = 0.01  # 10ms per character
        
        # Language-specific multipliers
        language_multipliers = {
            LanguageCode.HINDI: 1.2,
            LanguageCode.ENGLISH_IN: 1.0,
            LanguageCode.TAMIL: 1.3,
            LanguageCode.TELUGU: 1.3,
            LanguageCode.BENGALI: 1.2,
            LanguageCode.MARATHI: 1.2,
            LanguageCode.GUJARATI: 1.2,
            LanguageCode.KANNADA: 1.3,
            LanguageCode.MALAYALAM: 1.3,
            LanguageCode.PUNJABI: 1.1,
            LanguageCode.ODIA: 1.2,
        }
        
        multiplier = language_multipliers.get(language, 1.0)
        estimated_time = len(text) * base_time_per_char * multiplier
        
        return max(estimated_time, 0.5)  # Minimum 0.5 seconds
    
    def _add_to_cache(self, key: str, audio_buffer: AudioBuffer):
        """
        Add audio buffer to cache with size management.
        
        Args:
            key: Cache key
            audio_buffer: Audio buffer to cache
        """
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove first entry (FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = audio_buffer
        logger.debug(f"Added to TTS cache: {key}")
    
    def _generate_silence(self, duration: float) -> AudioBuffer:
        """
        Generate silence audio buffer.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Silence audio buffer
        """
        samples = int(self.sample_rate * duration)
        silence_data = np.zeros(samples, dtype=np.float32)
        
        return AudioBuffer(
            data=silence_data.tolist(),
            sample_rate=self.sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
    
    async def synthesize_with_emotion(
        self, 
        text: str, 
        language: LanguageCode, 
        emotion: str = "neutral",
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech with emotional tone (future enhancement).
        
        Args:
            text: Text to synthesize
            language: Target language
            emotion: Emotional tone (neutral, happy, sad, etc.)
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer with emotional tone
        """
        # For now, just call regular synthesis
        # In future, this could integrate with emotional TTS models
        logger.info(f"Synthesizing with emotion: {emotion}")
        return await self.synthesize_speech(text, language, accent)
    
    async def synthesize_ssml(
        self, 
        ssml_text: str, 
        language: LanguageCode,
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech from SSML markup (future enhancement).
        
        Args:
            ssml_text: SSML markup text
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        # For now, strip SSML tags and synthesize plain text
        # In future, this could parse SSML and apply appropriate modifications
        import re
        plain_text = re.sub(r'<[^>]+>', '', ssml_text)
        
        logger.info("Synthesizing SSML (simplified)")
        return await self.synthesize_speech(plain_text, language, accent)
    
    def clear_cache(self):
        """Clear the TTS cache."""
        self.cache.clear()
        logger.info("TTS cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_usage_percent': int((len(self.cache) / self.max_cache_size) * 100)
        }


class AdaptiveTTSEngine(TTSEngine):
    """
    Adaptive TTS engine that learns from user preferences and feedback.
    """
    
    def __init__(self, sample_rate: int = 22050, quality: str = 'high'):
        """
        Initialize adaptive TTS engine.
        
        Args:
            sample_rate: Target sample rate for synthesized audio
            quality: Quality setting ('high', 'medium', 'low')
        """
        super().__init__(sample_rate, quality)
        self.user_preferences: Dict[str, Dict[str, any]] = {}
        self.feedback_history: List[Dict[str, any]] = []
        
    async def synthesize_for_user(
        self, 
        text: str, 
        language: LanguageCode, 
        user_id: str,
        accent: Optional[AccentType] = None
    ) -> AudioBuffer:
        """
        Synthesize speech adapted to user preferences.
        
        Args:
            text: Text to synthesize
            language: Target language
            user_id: User identifier
            accent: Optional accent override
            
        Returns:
            User-adapted synthesized audio buffer
        """
        # Get user preferences
        user_prefs = self.user_preferences.get(user_id, {})
        
        # Determine accent based on user preference or default
        if accent is None:
            accent = user_prefs.get('preferred_accent', AccentType.STANDARD)
        
        # Apply user-specific speed adjustment
        speed_adjustment = user_prefs.get('speed_preference', 1.0)
        
        # Synthesize with user preferences
        audio_buffer = await self.synthesize_speech(text, language, accent)
        
        # Apply speed adjustment if needed
        if speed_adjustment != 1.0:
            audio_buffer = self._adjust_speed(audio_buffer, speed_adjustment)
        
        logger.debug(f"Synthesized speech for user {user_id} with preferences")
        return audio_buffer
    
    def update_user_preferences(
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
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
        logger.info(f"Updated TTS preferences for user {user_id}")
    
    def record_feedback(
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
        feedback_entry = {
            'user_id': user_id,
            'text': text,
            'language': language,
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Keep only recent feedback (last 1000 entries)
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
        
        logger.info(f"Recorded TTS feedback from user {user_id}: rating={rating}")
    
    def _adjust_speed(self, audio_buffer: AudioBuffer, speed_factor: float) -> AudioBuffer:
        """
        Adjust playback speed of audio buffer.
        
        Args:
            audio_buffer: Input audio buffer
            speed_factor: Speed adjustment factor (1.0 = normal)
            
        Returns:
            Speed-adjusted audio buffer
        """
        if speed_factor == 1.0:
            return audio_buffer
        
        # Convert to AudioSegment for speed adjustment
        audio_array = np.array(audio_buffer.data, dtype=np.float32)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=audio_buffer.sample_rate,
            sample_width=2,
            channels=audio_buffer.channels
        )
        
        # Adjust speed
        new_frame_rate = int(audio_segment.frame_rate * speed_factor)
        speed_adjusted = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={"frame_rate": new_frame_rate}
        )
        speed_adjusted = speed_adjusted.set_frame_rate(audio_buffer.sample_rate)
        
        # Convert back to AudioBuffer
        adjusted_array = np.array(speed_adjusted.get_array_of_samples(), dtype=np.float32)
        adjusted_array = adjusted_array / np.max(np.abs(adjusted_array))
        
        return AudioBuffer(
            data=adjusted_array.tolist(),
            sample_rate=audio_buffer.sample_rate,
            channels=audio_buffer.channels,
            format=audio_buffer.format,
            duration=len(adjusted_array) / audio_buffer.sample_rate
        )