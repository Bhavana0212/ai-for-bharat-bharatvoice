"""
Property-based tests for Audio Processing.

**Property 3: Noise Resilience**
**Validates: Requirements 1.4**

This module contains property-based tests that verify the audio processing
system's resilience to various types of noise and maintains audio quality
under different noise conditions.
"""

import asyncio
import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from typing import Tuple

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
)
from bharatvoice.services.voice_processing.audio_processor import AudioProcessor


# Strategy generators for property-based testing

@composite
def audio_signal_strategy(draw):
    """Generate realistic audio signals for testing."""
    # Audio parameters
    duration = draw(st.floats(min_value=0.1, max_value=2.0))
    sample_rate = draw(st.sampled_from([16000, 22050, 44100]))
    frequency = draw(st.floats(min_value=100, max_value=4000))
    amplitude = draw(st.floats(min_value=0.1, max_value=0.8))
    
    # Generate sine wave
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return AudioBuffer(
        data=signal.tolist(),
        sample_rate=sample_rate,
        channels=1,
        format=AudioFormat.WAV,
        duration=duration
    )


@composite
def noise_strategy(draw):
    """Generate different types of noise for testing."""
    noise_type = draw(st.sampled_from(['white', 'pink', 'brown', 'gaussian']))
    amplitude = draw(st.floats(min_value=0.01, max_value=0.3))
    samples = draw(st.integers(min_value=1600, max_value=88200))  # 0.1 to 2 seconds at 44.1kHz
    
    if noise_type == 'white':
        # White noise - equal power across all frequencies
        noise = np.random.normal(0, amplitude, samples)
    elif noise_type == 'pink':
        # Pink noise - 1/f power spectrum (simplified)
        white_noise = np.random.normal(0, 1, samples)
        # Apply simple pink filter (approximation)
        noise = np.cumsum(white_noise) * amplitude / 10
    elif noise_type == 'brown':
        # Brown noise - 1/f^2 power spectrum (simplified)
        white_noise = np.random.normal(0, 1, samples)
        noise = np.cumsum(np.cumsum(white_noise)) * amplitude / 100
    else:  # gaussian
        # Gaussian noise
        noise = np.random.normal(0, amplitude, samples)
    
    return noise.astype(np.float32)


@composite
def noisy_audio_strategy(draw):
    """Generate audio with added noise for testing."""
    clean_audio = draw(audio_signal_strategy())
    noise = draw(noise_strategy())
    
    # Ensure noise and audio have same length
    audio_samples = len(clean_audio.data)
    if len(noise) > audio_samples:
        noise = noise[:audio_samples]
    elif len(noise) < audio_samples:
        # Repeat noise to match audio length
        repeats = (audio_samples // len(noise)) + 1
        noise = np.tile(noise, repeats)[:audio_samples]
    
    # Add noise to clean audio
    clean_signal = np.array(clean_audio.data)
    noisy_signal = clean_signal + noise
    
    return AudioBuffer(
        data=noisy_signal.tolist(),
        sample_rate=clean_audio.sample_rate,
        channels=clean_audio.channels,
        format=clean_audio.format,
        duration=clean_audio.duration
    ), clean_audio


@composite
def snr_strategy(draw):
    """Generate Signal-to-Noise Ratio values for testing."""
    return draw(st.floats(min_value=-10.0, max_value=30.0))  # dB


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')
    return snr_db


def calculate_audio_quality_metrics(audio: np.ndarray) -> dict:
    """Calculate basic audio quality metrics."""
    return {
        'rms_energy': np.sqrt(np.mean(audio ** 2)),
        'peak_amplitude': np.max(np.abs(audio)),
        'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(audio)))),
        'dynamic_range': np.max(audio) - np.min(audio)
    }


class TestAudioProcessingNoiseResilienceProperties:
    """Property-based tests for audio processing noise resilience."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance for testing."""
        return AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5
        )
    
    @pytest.mark.asyncio
    @given(noisy_audio_strategy())
    @settings(max_examples=25, deadline=8000)
    async def test_noise_filtering_preserves_signal_structure(self, audio_processor, noisy_clean_pair):
        """
        Property: Noise filtering should preserve the basic structure of the signal.
        
        **Validates: Requirements 1.4**
        
        The filtered audio should maintain similar duration, sample rate, and
        basic signal characteristics while reducing noise.
        """
        noisy_audio, clean_audio = noisy_clean_pair
        
        # Skip if audio is too short or has extreme values
        assume(len(noisy_audio.data) >= 160)  # At least 10ms at 16kHz
        assume(np.max(np.abs(noisy_audio.data)) < 10.0)  # Reasonable amplitude
        
        try:
            # Apply noise filtering
            filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
            
            # Property 1: Duration preservation
            assert abs(filtered_audio.duration - noisy_audio.duration) < 0.01, \
                "Noise filtering should preserve audio duration"
            
            # Property 2: Sample rate preservation
            assert filtered_audio.sample_rate == noisy_audio.sample_rate, \
                "Noise filtering should preserve sample rate"
            
            # Property 3: Channel count preservation
            assert filtered_audio.channels == noisy_audio.channels, \
                "Noise filtering should preserve channel count"
            
            # Property 4: Format preservation
            assert filtered_audio.format == noisy_audio.format, \
                "Noise filtering should preserve audio format"
            
            # Property 5: Signal length preservation
            assert len(filtered_audio.data) == len(noisy_audio.data), \
                "Noise filtering should preserve signal length"
            
            # Property 6: Amplitude bounds preservation
            filtered_signal = np.array(filtered_audio.data)
            assert np.all(np.abs(filtered_signal) <= 1.0), \
                "Filtered audio should remain within valid amplitude bounds"
            
        except Exception as e:
            pytest.fail(f"Noise filtering failed on valid input: {e}")
    
    @pytest.mark.asyncio
    @given(noisy_audio_strategy())
    @settings(max_examples=20, deadline=10000)
    async def test_noise_reduction_effectiveness(self, audio_processor, noisy_clean_pair):
        """
        Property: Noise filtering should reduce noise while preserving signal quality.
        
        **Validates: Requirements 1.4**
        
        The filtered audio should have better SNR than the original noisy audio
        when compared to the clean signal.
        """
        noisy_audio, clean_audio = noisy_clean_pair
        
        # Skip edge cases
        assume(len(noisy_audio.data) >= 1600)  # At least 0.1 seconds at 16kHz
        assume(np.std(noisy_audio.data) > 0.001)  # Has some variation
        assume(np.max(np.abs(noisy_audio.data)) < 5.0)  # Reasonable amplitude
        
        try:
            # Apply noise filtering
            filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
            
            # Convert to numpy arrays
            clean_signal = np.array(clean_audio.data)
            noisy_signal = np.array(noisy_audio.data)
            filtered_signal = np.array(filtered_audio.data)
            
            # Ensure all signals have the same length
            min_length = min(len(clean_signal), len(noisy_signal), len(filtered_signal))
            clean_signal = clean_signal[:min_length]
            noisy_signal = noisy_signal[:min_length]
            filtered_signal = filtered_signal[:min_length]
            
            # Calculate noise components
            original_noise = noisy_signal - clean_signal
            filtered_noise = filtered_signal - clean_signal
            
            # Property 1: Noise reduction effectiveness
            original_noise_power = np.mean(original_noise ** 2)
            filtered_noise_power = np.mean(filtered_noise ** 2)
            
            if original_noise_power > 0.001:  # Only test if there's significant noise
                noise_reduction_ratio = filtered_noise_power / original_noise_power
                assert noise_reduction_ratio <= 1.0, \
                    "Noise filtering should not increase noise power"
                
                # Should achieve some noise reduction (at least 10%)
                assert noise_reduction_ratio < 0.9, \
                    "Noise filtering should achieve meaningful noise reduction"
            
            # Property 2: Signal preservation
            clean_signal_power = np.mean(clean_signal ** 2)
            if clean_signal_power > 0.001:  # Only test if there's significant signal
                signal_preservation = np.corrcoef(clean_signal, filtered_signal)[0, 1]
                
                # Filtered signal should be reasonably correlated with clean signal
                if not np.isnan(signal_preservation):
                    assert signal_preservation > 0.3, \
                        "Filtered signal should maintain correlation with clean signal"
            
        except Exception as e:
            pytest.fail(f"Noise reduction test failed: {e}")
    
    @pytest.mark.asyncio
    @given(audio_signal_strategy(), st.floats(min_value=0.01, max_value=0.5))
    @settings(max_examples=15, deadline=8000)
    async def test_noise_filtering_stability(self, audio_processor, clean_audio, noise_level):
        """
        Property: Noise filtering should be stable across different noise levels.
        
        **Validates: Requirements 1.4**
        
        The filtering process should not introduce artifacts or instability
        regardless of the input noise level.
        """
        assume(len(clean_audio.data) >= 800)  # At least 50ms at 16kHz
        assume(np.std(clean_audio.data) > 0.001)  # Has some variation
        
        try:
            # Add controlled noise
            clean_signal = np.array(clean_audio.data)
            noise = np.random.normal(0, noise_level, len(clean_signal))
            noisy_signal = clean_signal + noise
            
            noisy_audio = AudioBuffer(
                data=noisy_signal.tolist(),
                sample_rate=clean_audio.sample_rate,
                channels=clean_audio.channels,
                format=clean_audio.format,
                duration=clean_audio.duration
            )
            
            # Apply noise filtering
            filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
            filtered_signal = np.array(filtered_audio.data)
            
            # Property 1: No NaN or infinite values
            assert np.all(np.isfinite(filtered_signal)), \
                "Filtered audio should not contain NaN or infinite values"
            
            # Property 2: Reasonable amplitude bounds
            assert np.max(np.abs(filtered_signal)) < 10.0, \
                "Filtered audio should have reasonable amplitude bounds"
            
            # Property 3: No excessive amplification
            max_input_amplitude = np.max(np.abs(noisy_signal))
            max_output_amplitude = np.max(np.abs(filtered_signal))
            
            if max_input_amplitude > 0.001:
                amplification_factor = max_output_amplitude / max_input_amplitude
                assert amplification_factor < 5.0, \
                    "Noise filtering should not excessively amplify the signal"
            
            # Property 4: Signal continuity (no sudden jumps)
            if len(filtered_signal) > 1:
                signal_diff = np.abs(np.diff(filtered_signal))
                max_jump = np.max(signal_diff)
                signal_range = np.max(filtered_signal) - np.min(filtered_signal)
                
                if signal_range > 0.001:
                    relative_max_jump = max_jump / signal_range
                    assert relative_max_jump < 0.5, \
                        "Filtered signal should not have excessive discontinuities"
            
        except Exception as e:
            pytest.fail(f"Noise filtering stability test failed: {e}")
    
    @pytest.mark.asyncio
    @given(st.lists(audio_signal_strategy(), min_size=2, max_size=4))
    @settings(max_examples=10, deadline=12000)
    async def test_noise_filtering_consistency(self, audio_processor, audio_list):
        """
        Property: Noise filtering should be consistent across similar inputs.
        
        **Validates: Requirements 1.4**
        
        Similar audio inputs should produce similar filtering results,
        demonstrating consistent behavior of the noise reduction algorithm.
        """
        assume(all(len(audio.data) >= 800 for audio in audio_list))
        assume(all(np.std(audio.data) > 0.001 for audio in audio_list))
        
        try:
            filtered_results = []
            
            # Apply same noise level to all audio samples
            noise_level = 0.1
            
            for audio in audio_list:
                # Add consistent noise
                clean_signal = np.array(audio.data)
                noise = np.random.normal(0, noise_level, len(clean_signal))
                noisy_signal = clean_signal + noise
                
                noisy_audio = AudioBuffer(
                    data=noisy_signal.tolist(),
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                    format=audio.format,
                    duration=audio.duration
                )
                
                # Apply filtering
                filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
                filtered_results.append(filtered_audio)
            
            # Property: Consistent filtering behavior
            for i in range(len(filtered_results)):
                filtered_signal = np.array(filtered_results[i].data)
                
                # Each result should be valid
                assert np.all(np.isfinite(filtered_signal)), \
                    f"Filtered result {i} should not contain invalid values"
                
                # Calculate quality metrics
                metrics = calculate_audio_quality_metrics(filtered_signal)
                
                # Metrics should be within reasonable ranges
                assert 0.0 <= metrics['rms_energy'] <= 2.0, \
                    f"RMS energy should be reasonable for result {i}"
                assert 0.0 <= metrics['peak_amplitude'] <= 2.0, \
                    f"Peak amplitude should be reasonable for result {i}"
                assert 0.0 <= metrics['zero_crossing_rate'] <= 2.0, \
                    f"Zero crossing rate should be reasonable for result {i}"
            
        except Exception as e:
            pytest.fail(f"Noise filtering consistency test failed: {e}")
    
    @pytest.mark.asyncio
    @given(audio_signal_strategy())
    @settings(max_examples=15, deadline=6000)
    async def test_clean_audio_preservation(self, audio_processor, clean_audio):
        """
        Property: Clean audio should pass through noise filtering with minimal alteration.
        
        **Validates: Requirements 1.4**
        
        When processing already clean audio (low noise), the filtering should
        preserve the original signal quality without introducing artifacts.
        """
        assume(len(clean_audio.data) >= 800)  # At least 50ms at 16kHz
        assume(np.std(clean_audio.data) > 0.001)  # Has some variation
        assume(np.max(np.abs(clean_audio.data)) < 1.0)  # Reasonable amplitude
        
        try:
            # Apply noise filtering to clean audio
            filtered_audio = await audio_processor.filter_background_noise(clean_audio)
            
            # Convert to numpy arrays
            original_signal = np.array(clean_audio.data)
            filtered_signal = np.array(filtered_audio.data)
            
            # Ensure same length
            min_length = min(len(original_signal), len(filtered_signal))
            original_signal = original_signal[:min_length]
            filtered_signal = filtered_signal[:min_length]
            
            # Property 1: High correlation with original
            if np.std(original_signal) > 0.001 and np.std(filtered_signal) > 0.001:
                correlation = np.corrcoef(original_signal, filtered_signal)[0, 1]
                if not np.isnan(correlation):
                    assert correlation > 0.7, \
                        "Clean audio should maintain high correlation after filtering"
            
            # Property 2: Similar energy levels
            original_energy = np.mean(original_signal ** 2)
            filtered_energy = np.mean(filtered_signal ** 2)
            
            if original_energy > 0.001:
                energy_ratio = filtered_energy / original_energy
                assert 0.3 <= energy_ratio <= 3.0, \
                    "Energy levels should be preserved for clean audio"
            
            # Property 3: No excessive distortion
            mse = np.mean((original_signal - filtered_signal) ** 2)
            signal_power = np.mean(original_signal ** 2)
            
            if signal_power > 0.001:
                snr_preservation = 10 * np.log10(signal_power / (mse + 1e-10))
                assert snr_preservation > 10.0, \
                    "Clean audio should maintain good SNR after filtering"
            
        except Exception as e:
            pytest.fail(f"Clean audio preservation test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])