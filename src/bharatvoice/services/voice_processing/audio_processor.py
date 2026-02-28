<<<<<<< HEAD
"""
Audio processing implementation for BharatVoice Assistant.

This module implements the AudioProcessor interface with real-time stream processing,
voice activity detection, background noise filtering, and audio format conversion.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import webrtcvad
from scipy import signal
from scipy.signal import butter, filtfilt

from bharatvoice.core.interfaces import AudioProcessor as AudioProcessorInterface
from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    VoiceActivityResult,
)

logger = logging.getLogger(__name__)


class AudioProcessor(AudioProcessorInterface):
    """
    Audio processor with real-time stream processing capabilities.
    
    Implements voice activity detection, noise filtering, and audio preprocessing
    optimized for Indian languages and accents.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        noise_reduction_factor: float = 0.5,
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration for VAD in milliseconds
            vad_aggressiveness: VAD aggressiveness level (0-3)
            noise_reduction_factor: Noise reduction strength (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.noise_reduction_factor = noise_reduction_factor
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Audio processing parameters
        self.preemphasis_coeff = 0.97
        self.window_size = 2048
        self.hop_length = 512
        
        # Noise profile for spectral subtraction
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_profile_frames = 10  # Number of frames to estimate noise
        
        logger.info(
            f"AudioProcessor initialized with sample_rate={sample_rate}, "
            f"frame_duration={frame_duration_ms}ms, vad_aggressiveness={vad_aggressiveness}"
        )
    
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
            # Convert to numpy array for processing
            audio_array = audio_data.numpy_array
            
            # Resample if necessary
            if audio_data.sample_rate != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=audio_data.sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Apply language-specific preprocessing
            processed_audio = await self._apply_language_specific_processing(
                audio_array, language
            )
            
            # Apply preemphasis filter
            processed_audio = self._apply_preemphasis(processed_audio)
            
            # Normalize audio
            processed_audio = self._normalize_audio(processed_audio)
            
            # Create processed audio buffer
            processed_buffer = AudioBuffer(
                data=processed_audio.tolist(),
                sample_rate=self.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(processed_audio) / self.sample_rate
            )
            
            logger.debug(f"Processed audio stream for language {language}")
            return processed_buffer
            
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            raise
    
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame using WebRTC VAD.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        try:
            # Convert to required format for WebRTC VAD
            audio_array = audio_frame.numpy_array
            
            # Resample to 16kHz if necessary (WebRTC VAD requirement)
            if audio_frame.sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=audio_frame.sample_rate, 
                    target_sr=16000
                )
            
            # Convert to 16-bit PCM
            audio_pcm = (audio_array * 32767).astype(np.int16)
            
            # Ensure frame size is compatible with WebRTC VAD
            frame_samples = int(16000 * self.frame_duration_ms / 1000)
            
            if len(audio_pcm) < frame_samples:
                # Pad with zeros if frame is too short
                audio_pcm = np.pad(audio_pcm, (0, frame_samples - len(audio_pcm)))
            elif len(audio_pcm) > frame_samples:
                # Truncate if frame is too long
                audio_pcm = audio_pcm[:frame_samples]
            
            # Perform VAD
            is_speech = self.vad.is_speech(audio_pcm.tobytes(), 16000)
            
            # Calculate additional metrics
            energy_level = float(np.mean(audio_array ** 2))
            confidence = self._calculate_vad_confidence(audio_array, is_speech)
            
            # Estimate speech boundaries (simplified)
            start_time = 0.0
            end_time = audio_frame.duration
            
            result = VoiceActivityResult(
                is_speech=is_speech,
                confidence=confidence,
                start_time=start_time,
                end_time=end_time,
                energy_level=energy_level
            )
            
            logger.debug(f"VAD result: speech={is_speech}, confidence={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            # Return default result on error
            return VoiceActivityResult(
                is_speech=False,
                confidence=0.0,
                start_time=0.0,
                end_time=audio_frame.duration,
                energy_level=0.0
            )
    
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Note: This is a placeholder implementation. In production, this would
        integrate with TTS engines like gTTS or Coqui TTS.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        try:
            # Placeholder: Generate silence for now
            # In production, this would call actual TTS engine
            duration = max(1.0, len(text) * 0.1)  # Rough estimate
            samples = int(self.sample_rate * duration)
            
            # Generate placeholder audio (silence with slight noise)
            audio_data = np.random.normal(0, 0.001, samples).astype(np.float32)
            
            synthesized_buffer = AudioBuffer(
                data=audio_data.tolist(),
                sample_rate=self.sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
            logger.info(f"Synthesized speech for text: '{text[:50]}...' in {language}")
            return synthesized_buffer
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            raise
    
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise using spectral subtraction.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        try:
            audio_array = audio_data.numpy_array
            
            # Apply spectral subtraction
            filtered_audio = await self._spectral_subtraction(audio_array)
            
            # Apply additional noise reduction filters
            filtered_audio = self._apply_noise_reduction_filters(filtered_audio)
            
            # Create filtered audio buffer
            filtered_buffer = AudioBuffer(
                data=filtered_audio.tolist(),
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(filtered_audio) / audio_data.sample_rate
            )
            
            logger.debug("Applied background noise filtering")
            return filtered_buffer
            
        except Exception as e:
            logger.error(f"Error filtering background noise: {e}")
            raise
    
    async def _apply_language_specific_processing(
        self, 
        audio: np.ndarray, 
        language: LanguageCode
    ) -> np.ndarray:
        """
        Apply language-specific audio processing optimizations.
        
        Args:
            audio: Input audio array
            language: Target language
            
        Returns:
            Processed audio array
        """
        # Language-specific frequency emphasis
        if language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]:
            # Emphasize mid-frequencies for Hindi and Indian English
            audio = self._apply_frequency_emphasis(audio, 1000, 3000, 1.2)
        elif language in [LanguageCode.TAMIL, LanguageCode.TELUGU]:
            # Emphasize higher frequencies for Dravidian languages
            audio = self._apply_frequency_emphasis(audio, 1500, 4000, 1.1)
        elif language in [LanguageCode.BENGALI, LanguageCode.MARATHI]:
            # Balanced frequency response for these languages
            audio = self._apply_frequency_emphasis(audio, 800, 3500, 1.15)
        
        return audio
    
    def _apply_frequency_emphasis(
        self, 
        audio: np.ndarray, 
        low_freq: float, 
        high_freq: float, 
        gain: float
    ) -> np.ndarray:
        """
        Apply frequency emphasis to audio signal.
        
        Args:
            audio: Input audio array
            low_freq: Lower frequency bound
            high_freq: Upper frequency bound
            gain: Gain factor for the frequency band
            
        Returns:
            Processed audio array
        """
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter and mix with original
        filtered = filtfilt(b, a, audio)
        emphasized = audio + (filtered * (gain - 1))
        
        return emphasized
    
    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis filter to audio signal.
        
        Args:
            audio: Input audio array
            
        Returns:
            Preemphasized audio array
        """
        return np.append(audio[0], audio[1:] - self.preemphasis_coeff * audio[:-1])
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal to prevent clipping.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95  # Leave some headroom
        return audio
    
    async def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            audio: Input audio array
            
        Returns:
            Noise-reduced audio array
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise profile from first few frames if not available
        if self.noise_profile is None:
            noise_frames = min(self.noise_profile_frames, magnitude.shape[1])
            self.noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        alpha = self.noise_reduction_factor
        subtracted_magnitude = magnitude - alpha * self.noise_profile
        
        # Ensure magnitude doesn't go below a minimum threshold
        min_magnitude = 0.1 * magnitude
        subtracted_magnitude = np.maximum(subtracted_magnitude, min_magnitude)
        
        # Reconstruct signal
        enhanced_stft = subtracted_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(
            enhanced_stft, 
            hop_length=self.hop_length, 
            length=len(audio)
        )
        
        return enhanced_audio
    
    def _apply_noise_reduction_filters(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply additional noise reduction filters.
        
        Args:
            audio: Input audio array
            
        Returns:
            Filtered audio array
        """
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 80 / nyquist  # 80 Hz cutoff
        
        b, a = butter(4, low_cutoff, btype='high')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _calculate_vad_confidence(self, audio: np.ndarray, is_speech: bool) -> float:
        """
        Calculate confidence score for VAD result.
        
        Args:
            audio: Audio array
            is_speech: VAD result
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate spectral features for confidence estimation
        energy = np.mean(audio ** 2)
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio))))
        
        # Simple heuristic for confidence calculation
        if is_speech:
            # Higher energy and moderate ZCR indicate confident speech detection
            energy_confidence = min(1.0, energy * 1000)  # Scale energy
            zcr_confidence = 1.0 - min(1.0, zero_crossing_rate * 10)  # Lower ZCR is better for speech
            confidence = (energy_confidence + zcr_confidence) / 2
        else:
            # Lower energy indicates confident non-speech detection
            energy_confidence = 1.0 - min(1.0, energy * 1000)
            confidence = energy_confidence
        
        return max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95


class AudioFormatConverter:
    """Utility class for audio format conversion and preprocessing."""
    
    @staticmethod
    def convert_format(
        audio_buffer: AudioBuffer, 
        target_format: AudioFormat,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None
    ) -> AudioBuffer:
        """
        Convert audio buffer to target format and specifications.
        
        Args:
            audio_buffer: Input audio buffer
            target_format: Target audio format
            target_sample_rate: Target sample rate (optional)
            target_channels: Target number of channels (optional)
            
        Returns:
            Converted audio buffer
        """
        audio_array = audio_buffer.numpy_array
        sample_rate = audio_buffer.sample_rate
        channels = audio_buffer.channels
        
        # Resample if needed
        if target_sample_rate and target_sample_rate != sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate
        
        # Convert channels if needed
        if target_channels and target_channels != channels:
            if channels == 1 and target_channels == 2:
                # Mono to stereo
                audio_array = np.stack([audio_array, audio_array])
            elif channels == 2 and target_channels == 1:
                # Stereo to mono
                audio_array = np.mean(audio_array, axis=0)
            channels = target_channels
        
        return AudioBuffer(
            data=audio_array.tolist(),
            sample_rate=sample_rate,
            channels=channels,
            format=target_format,
            duration=len(audio_array) / sample_rate
        )
    
    @staticmethod
    def preprocess_for_recognition(audio_buffer: AudioBuffer) -> AudioBuffer:
        """
        Preprocess audio for speech recognition.
        
        Args:
            audio_buffer: Input audio buffer
            
        Returns:
            Preprocessed audio buffer optimized for ASR
        """
        # Convert to mono 16kHz (standard for most ASR systems)
        return AudioFormatConverter.convert_format(
            audio_buffer,
            target_format=AudioFormat.WAV,
            target_sample_rate=16000,
            target_channels=1
        )
    
    @staticmethod
    def extract_features(audio_buffer: AudioBuffer) -> Dict[str, np.ndarray]:
        """
        Extract audio features for analysis.
        
        Args:
            audio_buffer: Input audio buffer
            
        Returns:
            Dictionary of extracted features
        """
        audio_array = audio_buffer.numpy_array
        sample_rate = audio_buffer.sample_rate
        
        features = {}
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(
            y=audio_array, 
            sr=sample_rate, 
            n_mfcc=13
        )
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio_array, 
            sr=sample_rate
        )
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio_array, 
            sr=sample_rate
        )
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_array)
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio_array, 
            sr=sample_rate
        )
        
        return features


class RealTimeAudioProcessor:
    """Real-time audio stream processor for continuous audio processing."""
    
    def __init__(
        self, 
        audio_processor: AudioProcessor,
        buffer_size: int = 1024,
        overlap_ratio: float = 0.5
    ):
        """
        Initialize real-time audio processor.
        
        Args:
            audio_processor: AudioProcessor instance
            buffer_size: Size of audio buffer for processing
            overlap_ratio: Overlap ratio between consecutive buffers
        """
        self.audio_processor = audio_processor
        self.buffer_size = buffer_size
        self.overlap_size = int(buffer_size * overlap_ratio)
        self.audio_buffer = np.array([])
        self.is_processing = False
        
    async def process_stream(
        self, 
        audio_chunk: np.ndarray, 
        language: LanguageCode
    ) -> Tuple[Optional[AudioBuffer], List[VoiceActivityResult]]:
        """
        Process incoming audio chunk in real-time.
        
        Args:
            audio_chunk: New audio chunk to process
            language: Target language for processing
            
        Returns:
            Tuple of (processed_audio_buffer, vad_results)
        """
        # Add new chunk to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        processed_audio = None
        vad_results = []
        
        # Process complete buffers
        while len(self.audio_buffer) >= self.buffer_size:
            # Extract buffer for processing
            current_buffer = self.audio_buffer[:self.buffer_size]
            
            # Create AudioBuffer object
            audio_buffer_obj = AudioBuffer(
                data=current_buffer.tolist(),
                sample_rate=self.audio_processor.sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=len(current_buffer) / self.audio_processor.sample_rate
            )
            
            # Process audio
            processed_buffer = await self.audio_processor.process_audio_stream(
                audio_buffer_obj, language
            )
            
            # Detect voice activity
            vad_result = await self.audio_processor.detect_voice_activity(audio_buffer_obj)
            vad_results.append(vad_result)
            
            # Update processed audio
            if processed_audio is None:
                processed_audio = processed_buffer
            else:
                # Concatenate with overlap handling
                processed_audio = self._concatenate_with_overlap(
                    processed_audio, processed_buffer
                )
            
            # Move buffer forward with overlap
            self.audio_buffer = self.audio_buffer[self.buffer_size - self.overlap_size:]
        
        return processed_audio, vad_results
    
    def _concatenate_with_overlap(
        self, 
        buffer1: AudioBuffer, 
        buffer2: AudioBuffer
    ) -> AudioBuffer:
        """
        Concatenate two audio buffers with overlap handling.
        
        Args:
            buffer1: First audio buffer
            buffer2: Second audio buffer
            
        Returns:
            Concatenated audio buffer
        """
        # Simple concatenation for now
        # In production, this would handle overlap-add properly
        combined_data = buffer1.data + buffer2.data
        combined_duration = buffer1.duration + buffer2.duration
        
        return AudioBuffer(
            data=combined_data,
            sample_rate=buffer1.sample_rate,
            channels=buffer1.channels,
            format=buffer1.format,
            duration=combined_duration
        )
    
    def reset_buffer(self):
        """Reset the internal audio buffer."""
=======
"""
Audio processing implementation for BharatVoice Assistant.

This module implements the AudioProcessor interface with real-time stream processing,
voice activity detection, background noise filtering, and audio format conversion.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import webrtcvad
from scipy import signal
from scipy.signal import butter, filtfilt

from bharatvoice.core.interfaces import AudioProcessor as AudioProcessorInterface
from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    VoiceActivityResult,
)

logger = logging.getLogger(__name__)


class AudioProcessor(AudioProcessorInterface):
    """
    Audio processor with real-time stream processing capabilities.
    
    Implements voice activity detection, noise filtering, and audio preprocessing
    optimized for Indian languages and accents.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        noise_reduction_factor: float = 0.5,
    ):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration for VAD in milliseconds
            vad_aggressiveness: VAD aggressiveness level (0-3)
            noise_reduction_factor: Noise reduction strength (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.noise_reduction_factor = noise_reduction_factor
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Audio processing parameters
        self.preemphasis_coeff = 0.97
        self.window_size = 2048
        self.hop_length = 512
        
        # Noise profile for spectral subtraction
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_profile_frames = 10  # Number of frames to estimate noise
        
        logger.info(
            f"AudioProcessor initialized with sample_rate={sample_rate}, "
            f"frame_duration={frame_duration_ms}ms, vad_aggressiveness={vad_aggressiveness}"
        )
    
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
            # Convert to numpy array for processing
            audio_array = audio_data.numpy_array
            
            # Resample if necessary
            if audio_data.sample_rate != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=audio_data.sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Apply language-specific preprocessing
            processed_audio = await self._apply_language_specific_processing(
                audio_array, language
            )
            
            # Apply preemphasis filter
            processed_audio = self._apply_preemphasis(processed_audio)
            
            # Normalize audio
            processed_audio = self._normalize_audio(processed_audio)
            
            # Create processed audio buffer
            processed_buffer = AudioBuffer(
                data=processed_audio.tolist(),
                sample_rate=self.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(processed_audio) / self.sample_rate
            )
            
            logger.debug(f"Processed audio stream for language {language}")
            return processed_buffer
            
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            raise
    
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame using WebRTC VAD.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        try:
            # Convert to required format for WebRTC VAD
            audio_array = audio_frame.numpy_array
            
            # Resample to 16kHz if necessary (WebRTC VAD requirement)
            if audio_frame.sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=audio_frame.sample_rate, 
                    target_sr=16000
                )
            
            # Convert to 16-bit PCM
            audio_pcm = (audio_array * 32767).astype(np.int16)
            
            # Ensure frame size is compatible with WebRTC VAD
            frame_samples = int(16000 * self.frame_duration_ms / 1000)
            
            if len(audio_pcm) < frame_samples:
                # Pad with zeros if frame is too short
                audio_pcm = np.pad(audio_pcm, (0, frame_samples - len(audio_pcm)))
            elif len(audio_pcm) > frame_samples:
                # Truncate if frame is too long
                audio_pcm = audio_pcm[:frame_samples]
            
            # Perform VAD
            is_speech = self.vad.is_speech(audio_pcm.tobytes(), 16000)
            
            # Calculate additional metrics
            energy_level = float(np.mean(audio_array ** 2))
            confidence = self._calculate_vad_confidence(audio_array, is_speech)
            
            # Estimate speech boundaries (simplified)
            start_time = 0.0
            end_time = audio_frame.duration
            
            result = VoiceActivityResult(
                is_speech=is_speech,
                confidence=confidence,
                start_time=start_time,
                end_time=end_time,
                energy_level=energy_level
            )
            
            logger.debug(f"VAD result: speech={is_speech}, confidence={confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            # Return default result on error
            return VoiceActivityResult(
                is_speech=False,
                confidence=0.0,
                start_time=0.0,
                end_time=audio_frame.duration,
                energy_level=0.0
            )
    
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Note: This is a placeholder implementation. In production, this would
        integrate with TTS engines like gTTS or Coqui TTS.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        try:
            # Placeholder: Generate silence for now
            # In production, this would call actual TTS engine
            duration = max(1.0, len(text) * 0.1)  # Rough estimate
            samples = int(self.sample_rate * duration)
            
            # Generate placeholder audio (silence with slight noise)
            audio_data = np.random.normal(0, 0.001, samples).astype(np.float32)
            
            synthesized_buffer = AudioBuffer(
                data=audio_data.tolist(),
                sample_rate=self.sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
            logger.info(f"Synthesized speech for text: '{text[:50]}...' in {language}")
            return synthesized_buffer
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            raise
    
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise using spectral subtraction.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        try:
            audio_array = audio_data.numpy_array
            
            # Apply spectral subtraction
            filtered_audio = await self._spectral_subtraction(audio_array)
            
            # Apply additional noise reduction filters
            filtered_audio = self._apply_noise_reduction_filters(filtered_audio)
            
            # Create filtered audio buffer
            filtered_buffer = AudioBuffer(
                data=filtered_audio.tolist(),
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(filtered_audio) / audio_data.sample_rate
            )
            
            logger.debug("Applied background noise filtering")
            return filtered_buffer
            
        except Exception as e:
            logger.error(f"Error filtering background noise: {e}")
            raise
    
    async def _apply_language_specific_processing(
        self, 
        audio: np.ndarray, 
        language: LanguageCode
    ) -> np.ndarray:
        """
        Apply language-specific audio processing optimizations.
        
        Args:
            audio: Input audio array
            language: Target language
            
        Returns:
            Processed audio array
        """
        # Language-specific frequency emphasis
        if language in [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]:
            # Emphasize mid-frequencies for Hindi and Indian English
            audio = self._apply_frequency_emphasis(audio, 1000, 3000, 1.2)
        elif language in [LanguageCode.TAMIL, LanguageCode.TELUGU]:
            # Emphasize higher frequencies for Dravidian languages
            audio = self._apply_frequency_emphasis(audio, 1500, 4000, 1.1)
        elif language in [LanguageCode.BENGALI, LanguageCode.MARATHI]:
            # Balanced frequency response for these languages
            audio = self._apply_frequency_emphasis(audio, 800, 3500, 1.15)
        
        return audio
    
    def _apply_frequency_emphasis(
        self, 
        audio: np.ndarray, 
        low_freq: float, 
        high_freq: float, 
        gain: float
    ) -> np.ndarray:
        """
        Apply frequency emphasis to audio signal.
        
        Args:
            audio: Input audio array
            low_freq: Lower frequency bound
            high_freq: Upper frequency bound
            gain: Gain factor for the frequency band
            
        Returns:
            Processed audio array
        """
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter and mix with original
        filtered = filtfilt(b, a, audio)
        emphasized = audio + (filtered * (gain - 1))
        
        return emphasized
    
    def _apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis filter to audio signal.
        
        Args:
            audio: Input audio array
            
        Returns:
            Preemphasized audio array
        """
        return np.append(audio[0], audio[1:] - self.preemphasis_coeff * audio[:-1])
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal to prevent clipping.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95  # Leave some headroom
        return audio
    
    async def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            audio: Input audio array
            
        Returns:
            Noise-reduced audio array
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise profile from first few frames if not available
        if self.noise_profile is None:
            noise_frames = min(self.noise_profile_frames, magnitude.shape[1])
            self.noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Apply spectral subtraction
        alpha = self.noise_reduction_factor
        subtracted_magnitude = magnitude - alpha * self.noise_profile
        
        # Ensure magnitude doesn't go below a minimum threshold
        min_magnitude = 0.1 * magnitude
        subtracted_magnitude = np.maximum(subtracted_magnitude, min_magnitude)
        
        # Reconstruct signal
        enhanced_stft = subtracted_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(
            enhanced_stft, 
            hop_length=self.hop_length, 
            length=len(audio)
        )
        
        return enhanced_audio
    
    def _apply_noise_reduction_filters(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply additional noise reduction filters.
        
        Args:
            audio: Input audio array
            
        Returns:
            Filtered audio array
        """
        # High-pass filter to remove low-frequency noise
        nyquist = self.sample_rate / 2
        low_cutoff = 80 / nyquist  # 80 Hz cutoff
        
        b, a = butter(4, low_cutoff, btype='high')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _calculate_vad_confidence(self, audio: np.ndarray, is_speech: bool) -> float:
        """
        Calculate confidence score for VAD result.
        
        Args:
            audio: Audio array
            is_speech: VAD result
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate spectral features for confidence estimation
        energy = np.mean(audio ** 2)
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio))))
        
        # Simple heuristic for confidence calculation
        if is_speech:
            # Higher energy and moderate ZCR indicate confident speech detection
            energy_confidence = min(1.0, energy * 1000)  # Scale energy
            zcr_confidence = 1.0 - min(1.0, zero_crossing_rate * 10)  # Lower ZCR is better for speech
            confidence = (energy_confidence + zcr_confidence) / 2
        else:
            # Lower energy indicates confident non-speech detection
            energy_confidence = 1.0 - min(1.0, energy * 1000)
            confidence = energy_confidence
        
        return max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95


class AudioFormatConverter:
    """Utility class for audio format conversion and preprocessing."""
    
    @staticmethod
    def convert_format(
        audio_buffer: AudioBuffer, 
        target_format: AudioFormat,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None
    ) -> AudioBuffer:
        """
        Convert audio buffer to target format and specifications.
        
        Args:
            audio_buffer: Input audio buffer
            target_format: Target audio format
            target_sample_rate: Target sample rate (optional)
            target_channels: Target number of channels (optional)
            
        Returns:
            Converted audio buffer
        """
        audio_array = audio_buffer.numpy_array
        sample_rate = audio_buffer.sample_rate
        channels = audio_buffer.channels
        
        # Resample if needed
        if target_sample_rate and target_sample_rate != sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sample_rate, 
                target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate
        
        # Convert channels if needed
        if target_channels and target_channels != channels:
            if channels == 1 and target_channels == 2:
                # Mono to stereo
                audio_array = np.stack([audio_array, audio_array])
            elif channels == 2 and target_channels == 1:
                # Stereo to mono
                audio_array = np.mean(audio_array, axis=0)
            channels = target_channels
        
        return AudioBuffer(
            data=audio_array.tolist(),
            sample_rate=sample_rate,
            channels=channels,
            format=target_format,
            duration=len(audio_array) / sample_rate
        )
    
    @staticmethod
    def preprocess_for_recognition(audio_buffer: AudioBuffer) -> AudioBuffer:
        """
        Preprocess audio for speech recognition.
        
        Args:
            audio_buffer: Input audio buffer
            
        Returns:
            Preprocessed audio buffer optimized for ASR
        """
        # Convert to mono 16kHz (standard for most ASR systems)
        return AudioFormatConverter.convert_format(
            audio_buffer,
            target_format=AudioFormat.WAV,
            target_sample_rate=16000,
            target_channels=1
        )
    
    @staticmethod
    def extract_features(audio_buffer: AudioBuffer) -> Dict[str, np.ndarray]:
        """
        Extract audio features for analysis.
        
        Args:
            audio_buffer: Input audio buffer
            
        Returns:
            Dictionary of extracted features
        """
        audio_array = audio_buffer.numpy_array
        sample_rate = audio_buffer.sample_rate
        
        features = {}
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(
            y=audio_array, 
            sr=sample_rate, 
            n_mfcc=13
        )
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio_array, 
            sr=sample_rate
        )
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio_array, 
            sr=sample_rate
        )
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_array)
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio_array, 
            sr=sample_rate
        )
        
        return features


class RealTimeAudioProcessor:
    """Real-time audio stream processor for continuous audio processing."""
    
    def __init__(
        self, 
        audio_processor: AudioProcessor,
        buffer_size: int = 1024,
        overlap_ratio: float = 0.5
    ):
        """
        Initialize real-time audio processor.
        
        Args:
            audio_processor: AudioProcessor instance
            buffer_size: Size of audio buffer for processing
            overlap_ratio: Overlap ratio between consecutive buffers
        """
        self.audio_processor = audio_processor
        self.buffer_size = buffer_size
        self.overlap_size = int(buffer_size * overlap_ratio)
        self.audio_buffer = np.array([])
        self.is_processing = False
        
    async def process_stream(
        self, 
        audio_chunk: np.ndarray, 
        language: LanguageCode
    ) -> Tuple[Optional[AudioBuffer], List[VoiceActivityResult]]:
        """
        Process incoming audio chunk in real-time.
        
        Args:
            audio_chunk: New audio chunk to process
            language: Target language for processing
            
        Returns:
            Tuple of (processed_audio_buffer, vad_results)
        """
        # Add new chunk to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        processed_audio = None
        vad_results = []
        
        # Process complete buffers
        while len(self.audio_buffer) >= self.buffer_size:
            # Extract buffer for processing
            current_buffer = self.audio_buffer[:self.buffer_size]
            
            # Create AudioBuffer object
            audio_buffer_obj = AudioBuffer(
                data=current_buffer.tolist(),
                sample_rate=self.audio_processor.sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=len(current_buffer) / self.audio_processor.sample_rate
            )
            
            # Process audio
            processed_buffer = await self.audio_processor.process_audio_stream(
                audio_buffer_obj, language
            )
            
            # Detect voice activity
            vad_result = await self.audio_processor.detect_voice_activity(audio_buffer_obj)
            vad_results.append(vad_result)
            
            # Update processed audio
            if processed_audio is None:
                processed_audio = processed_buffer
            else:
                # Concatenate with overlap handling
                processed_audio = self._concatenate_with_overlap(
                    processed_audio, processed_buffer
                )
            
            # Move buffer forward with overlap
            self.audio_buffer = self.audio_buffer[self.buffer_size - self.overlap_size:]
        
        return processed_audio, vad_results
    
    def _concatenate_with_overlap(
        self, 
        buffer1: AudioBuffer, 
        buffer2: AudioBuffer
    ) -> AudioBuffer:
        """
        Concatenate two audio buffers with overlap handling.
        
        Args:
            buffer1: First audio buffer
            buffer2: Second audio buffer
            
        Returns:
            Concatenated audio buffer
        """
        # Simple concatenation for now
        # In production, this would handle overlap-add properly
        combined_data = buffer1.data + buffer2.data
        combined_duration = buffer1.duration + buffer2.duration
        
        return AudioBuffer(
            data=combined_data,
            sample_rate=buffer1.sample_rate,
            channels=buffer1.channels,
            format=buffer1.format,
            duration=combined_duration
        )
    
    def reset_buffer(self):
        """Reset the internal audio buffer."""
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        self.audio_buffer = np.array([])