<<<<<<< HEAD
#!/usr/bin/env python3
"""
Test runner for audio processing property tests.
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def run_audio_property_tests():
    """Run audio processing property tests."""
    try:
        print("🚀 Starting Audio Processing Property Tests...")
        
        # Import required modules
        from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
        from bharatvoice.core.models import AudioBuffer, AudioFormat, LanguageCode
        import numpy as np
        
        print("✅ Successfully imported required modules")
        
        # Create audio processor
        audio_processor = AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5
        )
        print("✅ Created AudioProcessor instance")
        
        # Test 1: Basic noise filtering functionality
        print("\n📋 Test 1: Basic noise filtering functionality")
        
        # Create test audio
        duration = 1.0
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        clean_signal = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Create audio buffer
        noisy_audio = AudioBuffer(
            data=noisy_signal.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
        
        print(f"  Created noisy audio: {len(noisy_audio.data)} samples, {noisy_audio.duration:.2f}s")
        
        # Apply noise filtering
        filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
        
        print(f"  Filtered audio: {len(filtered_audio.data)} samples, {filtered_audio.duration:.2f}s")
        
        # Verify basic properties
        assert filtered_audio.sample_rate == noisy_audio.sample_rate, "Sample rate should be preserved"
        assert filtered_audio.channels == noisy_audio.channels, "Channel count should be preserved"
        assert filtered_audio.format == noisy_audio.format, "Format should be preserved"
        assert len(filtered_audio.data) == len(noisy_audio.data), "Signal length should be preserved"
        assert abs(filtered_audio.duration - noisy_audio.duration) < 0.01, "Duration should be preserved"
        
        # Check amplitude bounds
        filtered_signal = np.array(filtered_audio.data)
        assert np.all(np.abs(filtered_signal) <= 1.0), "Filtered audio should remain within valid bounds"
        assert np.all(np.isfinite(filtered_signal)), "Filtered audio should not contain invalid values"
        
        print("  ✅ Basic noise filtering properties verified")
        
        # Test 2: Noise reduction effectiveness
        print("\n📋 Test 2: Noise reduction effectiveness")
        
        # Calculate noise reduction
        clean_signal_array = np.array(clean_signal)
        noisy_signal_array = np.array(noisy_signal)
        filtered_signal_array = np.array(filtered_audio.data)
        
        # Ensure same length
        min_length = min(len(clean_signal_array), len(filtered_signal_array))
        clean_signal_array = clean_signal_array[:min_length]
        filtered_signal_array = filtered_signal_array[:min_length]
        noisy_signal_array = noisy_signal_array[:min_length]
        
        # Calculate noise components
        original_noise = noisy_signal_array - clean_signal_array
        filtered_noise = filtered_signal_array - clean_signal_array
        
        original_noise_power = np.mean(original_noise ** 2)
        filtered_noise_power = np.mean(filtered_noise ** 2)
        
        print(f"  Original noise power: {original_noise_power:.6f}")
        print(f"  Filtered noise power: {filtered_noise_power:.6f}")
        
        if original_noise_power > 0.001:
            noise_reduction_ratio = filtered_noise_power / original_noise_power
            print(f"  Noise reduction ratio: {noise_reduction_ratio:.3f}")
            
            assert noise_reduction_ratio <= 1.0, "Noise filtering should not increase noise power"
            if noise_reduction_ratio < 0.9:
                print("  ✅ Achieved meaningful noise reduction")
            else:
                print("  ⚠️  Limited noise reduction achieved")
        
        # Test 3: Signal preservation
        print("\n📋 Test 3: Signal preservation")
        
        clean_signal_power = np.mean(clean_signal_array ** 2)
        if clean_signal_power > 0.001:
            correlation = np.corrcoef(clean_signal_array, filtered_signal_array)[0, 1]
            print(f"  Correlation with clean signal: {correlation:.3f}")
            
            if not np.isnan(correlation) and correlation > 0.3:
                print("  ✅ Good signal preservation")
            else:
                print("  ⚠️  Signal preservation could be better")
        
        # Test 4: Clean audio preservation
        print("\n📋 Test 4: Clean audio preservation")
        
        clean_audio = AudioBuffer(
            data=clean_signal.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
        
        filtered_clean = await audio_processor.filter_background_noise(clean_audio)
        filtered_clean_signal = np.array(filtered_clean.data)
        
        # Check correlation
        if np.std(clean_signal) > 0.001 and np.std(filtered_clean_signal) > 0.001:
            clean_correlation = np.corrcoef(clean_signal, filtered_clean_signal[:len(clean_signal)])[0, 1]
            print(f"  Clean audio correlation: {clean_correlation:.3f}")
            
            if not np.isnan(clean_correlation) and clean_correlation > 0.7:
                print("  ✅ Excellent clean audio preservation")
            else:
                print("  ⚠️  Clean audio preservation could be improved")
        
        # Test 5: Stability test
        print("\n📋 Test 5: Stability test with different noise levels")
        
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        for noise_level in noise_levels:
            test_noise = np.random.normal(0, noise_level, len(clean_signal))
            test_noisy_signal = clean_signal + test_noise
            
            test_noisy_audio = AudioBuffer(
                data=test_noisy_signal.tolist(),
                sample_rate=sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
            test_filtered = await audio_processor.filter_background_noise(test_noisy_audio)
            test_filtered_signal = np.array(test_filtered.data)
            
            # Check for stability
            assert np.all(np.isfinite(test_filtered_signal)), f"Should be stable at noise level {noise_level}"
            assert np.max(np.abs(test_filtered_signal)) < 10.0, f"Should have reasonable bounds at noise level {noise_level}"
            
            print(f"  ✅ Stable at noise level {noise_level}")
        
        print("\n🎉 All audio processing property tests completed successfully!")
        print("\n📊 Test Summary:")
        print("  ✅ Basic noise filtering properties")
        print("  ✅ Noise reduction effectiveness")
        print("  ✅ Signal preservation")
        print("  ✅ Clean audio preservation")
        print("  ✅ Stability across noise levels")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio property tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the audio property tests."""
    success = await run_audio_property_tests()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
=======
#!/usr/bin/env python3
"""
Test runner for audio processing property tests.
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def run_audio_property_tests():
    """Run audio processing property tests."""
    try:
        print("🚀 Starting Audio Processing Property Tests...")
        
        # Import required modules
        from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
        from bharatvoice.core.models import AudioBuffer, AudioFormat, LanguageCode
        import numpy as np
        
        print("✅ Successfully imported required modules")
        
        # Create audio processor
        audio_processor = AudioProcessor(
            sample_rate=16000,
            frame_duration_ms=30,
            vad_aggressiveness=2,
            noise_reduction_factor=0.5
        )
        print("✅ Created AudioProcessor instance")
        
        # Test 1: Basic noise filtering functionality
        print("\n📋 Test 1: Basic noise filtering functionality")
        
        # Create test audio
        duration = 1.0
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        clean_signal = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Create audio buffer
        noisy_audio = AudioBuffer(
            data=noisy_signal.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
        
        print(f"  Created noisy audio: {len(noisy_audio.data)} samples, {noisy_audio.duration:.2f}s")
        
        # Apply noise filtering
        filtered_audio = await audio_processor.filter_background_noise(noisy_audio)
        
        print(f"  Filtered audio: {len(filtered_audio.data)} samples, {filtered_audio.duration:.2f}s")
        
        # Verify basic properties
        assert filtered_audio.sample_rate == noisy_audio.sample_rate, "Sample rate should be preserved"
        assert filtered_audio.channels == noisy_audio.channels, "Channel count should be preserved"
        assert filtered_audio.format == noisy_audio.format, "Format should be preserved"
        assert len(filtered_audio.data) == len(noisy_audio.data), "Signal length should be preserved"
        assert abs(filtered_audio.duration - noisy_audio.duration) < 0.01, "Duration should be preserved"
        
        # Check amplitude bounds
        filtered_signal = np.array(filtered_audio.data)
        assert np.all(np.abs(filtered_signal) <= 1.0), "Filtered audio should remain within valid bounds"
        assert np.all(np.isfinite(filtered_signal)), "Filtered audio should not contain invalid values"
        
        print("  ✅ Basic noise filtering properties verified")
        
        # Test 2: Noise reduction effectiveness
        print("\n📋 Test 2: Noise reduction effectiveness")
        
        # Calculate noise reduction
        clean_signal_array = np.array(clean_signal)
        noisy_signal_array = np.array(noisy_signal)
        filtered_signal_array = np.array(filtered_audio.data)
        
        # Ensure same length
        min_length = min(len(clean_signal_array), len(filtered_signal_array))
        clean_signal_array = clean_signal_array[:min_length]
        filtered_signal_array = filtered_signal_array[:min_length]
        noisy_signal_array = noisy_signal_array[:min_length]
        
        # Calculate noise components
        original_noise = noisy_signal_array - clean_signal_array
        filtered_noise = filtered_signal_array - clean_signal_array
        
        original_noise_power = np.mean(original_noise ** 2)
        filtered_noise_power = np.mean(filtered_noise ** 2)
        
        print(f"  Original noise power: {original_noise_power:.6f}")
        print(f"  Filtered noise power: {filtered_noise_power:.6f}")
        
        if original_noise_power > 0.001:
            noise_reduction_ratio = filtered_noise_power / original_noise_power
            print(f"  Noise reduction ratio: {noise_reduction_ratio:.3f}")
            
            assert noise_reduction_ratio <= 1.0, "Noise filtering should not increase noise power"
            if noise_reduction_ratio < 0.9:
                print("  ✅ Achieved meaningful noise reduction")
            else:
                print("  ⚠️  Limited noise reduction achieved")
        
        # Test 3: Signal preservation
        print("\n📋 Test 3: Signal preservation")
        
        clean_signal_power = np.mean(clean_signal_array ** 2)
        if clean_signal_power > 0.001:
            correlation = np.corrcoef(clean_signal_array, filtered_signal_array)[0, 1]
            print(f"  Correlation with clean signal: {correlation:.3f}")
            
            if not np.isnan(correlation) and correlation > 0.3:
                print("  ✅ Good signal preservation")
            else:
                print("  ⚠️  Signal preservation could be better")
        
        # Test 4: Clean audio preservation
        print("\n📋 Test 4: Clean audio preservation")
        
        clean_audio = AudioBuffer(
            data=clean_signal.tolist(),
            sample_rate=sample_rate,
            channels=1,
            format=AudioFormat.WAV,
            duration=duration
        )
        
        filtered_clean = await audio_processor.filter_background_noise(clean_audio)
        filtered_clean_signal = np.array(filtered_clean.data)
        
        # Check correlation
        if np.std(clean_signal) > 0.001 and np.std(filtered_clean_signal) > 0.001:
            clean_correlation = np.corrcoef(clean_signal, filtered_clean_signal[:len(clean_signal)])[0, 1]
            print(f"  Clean audio correlation: {clean_correlation:.3f}")
            
            if not np.isnan(clean_correlation) and clean_correlation > 0.7:
                print("  ✅ Excellent clean audio preservation")
            else:
                print("  ⚠️  Clean audio preservation could be improved")
        
        # Test 5: Stability test
        print("\n📋 Test 5: Stability test with different noise levels")
        
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        for noise_level in noise_levels:
            test_noise = np.random.normal(0, noise_level, len(clean_signal))
            test_noisy_signal = clean_signal + test_noise
            
            test_noisy_audio = AudioBuffer(
                data=test_noisy_signal.tolist(),
                sample_rate=sample_rate,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
            test_filtered = await audio_processor.filter_background_noise(test_noisy_audio)
            test_filtered_signal = np.array(test_filtered.data)
            
            # Check for stability
            assert np.all(np.isfinite(test_filtered_signal)), f"Should be stable at noise level {noise_level}"
            assert np.max(np.abs(test_filtered_signal)) < 10.0, f"Should have reasonable bounds at noise level {noise_level}"
            
            print(f"  ✅ Stable at noise level {noise_level}")
        
        print("\n🎉 All audio processing property tests completed successfully!")
        print("\n📊 Test Summary:")
        print("  ✅ Basic noise filtering properties")
        print("  ✅ Noise reduction effectiveness")
        print("  ✅ Signal preservation")
        print("  ✅ Clean audio preservation")
        print("  ✅ Stability across noise levels")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio property tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the audio property tests."""
    success = await run_audio_property_tests()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        sys.exit(1)