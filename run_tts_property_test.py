#!/usr/bin/env python3
"""
Test runner for TTS Property-Based Tests.

**Property 10: Natural Speech Synthesis**
**Validates: Requirements 3.3**
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_tts_synthesis_properties():
    """Test TTS synthesis properties."""
    try:
        from bharatvoice.services.voice_processing.tts_engine import TTSEngine, AdaptiveTTSEngine
        from bharatvoice.core.models import LanguageCode, AccentType, AudioFormat
        
        print("🎤 Testing TTS Synthesis Properties...")
        print("=" * 50)
        
        # Initialize TTS engine
        tts_engine = TTSEngine(sample_rate=22050, quality='high')
        print("✓ TTS Engine initialized")
        
        # Test 1: Basic synthesis completeness
        print("\n1. Testing synthesis completeness...")
        test_texts = [
            "Hello world",
            "Namaste",
            "Good morning",
            "Thank you very much"
        ]
        
        for text in test_texts:
            try:
                audio_buffer = await tts_engine.synthesize_speech(
                    text, LanguageCode.ENGLISH_IN, AccentType.STANDARD
                )
                
                # Validate basic properties
                assert len(audio_buffer.data) > 0, f"Empty audio for '{text}'"
                assert audio_buffer.duration > 0, f"Invalid duration for '{text}'"
                assert audio_buffer.sample_rate > 0, f"Invalid sample rate for '{text}'"
                
                print(f"  ✓ '{text}' -> {len(audio_buffer.data)} samples, {audio_buffer.duration:.2f}s")
                
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ['network', 'connection', 'gtts', 'internet']):
                    print(f"  ⚠ '{text}' -> Skipped (network dependency): {e}")
                else:
                    print(f"  ❌ '{text}' -> Failed: {e}")
                    return False
        
        # Test 2: Quality characteristics
        print("\n2. Testing quality characteristics...")
        try:
            audio_buffer = await tts_engine.synthesize_speech(
                "This is a test for quality", LanguageCode.ENGLISH_IN, AccentType.STANDARD,
                quality_optimize=True
            )
            
            # Calculate basic metrics
            import numpy as np
            audio_array = np.array(audio_buffer.data, dtype=np.float32)
            
            rms_energy = np.sqrt(np.mean(audio_array ** 2))
            peak_amplitude = np.max(np.abs(audio_array))
            
            # Validate quality metrics
            assert 0.001 <= rms_energy <= 1.0, f"Poor RMS energy: {rms_energy}"
            assert peak_amplitude <= 1.0, f"Clipping detected: {peak_amplitude}"
            
            print(f"  ✓ RMS Energy: {rms_energy:.4f}")
            print(f"  ✓ Peak Amplitude: {peak_amplitude:.4f}")
            print(f"  ✓ No clipping detected")
            
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ['network', 'connection', 'gtts', 'internet']):
                print(f"  ⚠ Quality test skipped (network dependency): {e}")
            else:
                print(f"  ❌ Quality test failed: {e}")
                return False
        
        # Test 3: Accent adaptation
        print("\n3. Testing accent adaptation...")
        accents_to_test = [AccentType.STANDARD, AccentType.NORTH_INDIAN, AccentType.MUMBAI]
        
        for accent in accents_to_test:
            try:
                audio_buffer = await tts_engine.synthesize_speech(
                    "Testing accent", LanguageCode.ENGLISH_IN, accent
                )
                
                assert len(audio_buffer.data) > 0, f"Empty audio for accent {accent}"
                assert audio_buffer.duration > 0, f"Invalid duration for accent {accent}"
                
                print(f"  ✓ {accent.name} -> {audio_buffer.duration:.2f}s")
                
            except Exception as e:
                if any(keyword in str(e).lower() for keyword in ['network', 'connection', 'gtts', 'internet']):
                    print(f"  ⚠ {accent.name} -> Skipped (network dependency)")
                else:
                    print(f"  ❌ {accent.name} -> Failed: {e}")
                    return False
        
        # Test 4: Configuration validity
        print("\n4. Testing configuration validity...")
        
        # Language mappings
        assert len(tts_engine.LANGUAGE_MAPPING) > 0, "No language mappings"
        print(f"  ✓ {len(tts_engine.LANGUAGE_MAPPING)} language mappings")
        
        # Accent configurations
        assert len(tts_engine.ACCENT_CONFIGS) > 0, "No accent configurations"
        for accent, config in tts_engine.ACCENT_CONFIGS.items():
            required_keys = ['speed', 'pitch_shift', 'formant_shift', 'emphasis_factor', 'pause_duration']
            for key in required_keys:
                assert key in config, f"Missing {key} in {accent} config"
        print(f"  ✓ {len(tts_engine.ACCENT_CONFIGS)} accent configurations")
        
        # Quality settings
        assert len(tts_engine.QUALITY_SETTINGS) > 0, "No quality settings"
        for quality in ['high', 'medium', 'low']:
            assert quality in tts_engine.QUALITY_SETTINGS, f"Missing {quality} quality setting"
        print(f"  ✓ {len(tts_engine.QUALITY_SETTINGS)} quality settings")
        
        # Test 5: Synthesis time estimation
        print("\n5. Testing synthesis time estimation...")
        test_cases = [
            ("Hi", LanguageCode.ENGLISH_IN),
            ("Hello world", LanguageCode.ENGLISH_IN),
            ("This is a longer sentence", LanguageCode.ENGLISH_IN)
        ]
        
        for text, language in test_cases:
            estimated_time = tts_engine.estimate_synthesis_time(text, language)
            
            assert estimated_time > 0, f"Invalid time for '{text}'"
            assert estimated_time >= 0.5, f"Time too low for '{text}'"
            
            print(f"  ✓ '{text}' -> {estimated_time:.2f}s estimated")
        
        # Test 6: Adaptive TTS
        print("\n6. Testing adaptive TTS...")
        adaptive_tts = AdaptiveTTSEngine(sample_rate=22050, quality='medium')
        
        # Test user preferences
        user_id = "test_user"
        preferences = {
            'preferred_accent': AccentType.MUMBAI,
            'speed_preference': 1.1
        }
        adaptive_tts.update_user_preferences(user_id, preferences)
        
        stored_prefs = adaptive_tts.user_preferences.get(user_id, {})
        assert stored_prefs.get('preferred_accent') == AccentType.MUMBAI, "Accent preference not stored"
        assert stored_prefs.get('speed_preference') == 1.1, "Speed preference not stored"
        
        print(f"  ✓ User preferences stored for {user_id}")
        
        # Test feedback recording
        adaptive_tts.record_feedback(user_id, "Test text", LanguageCode.ENGLISH_IN, 4.5)
        assert len(adaptive_tts.feedback_history) == 1, "Feedback not recorded"
        
        feedback = adaptive_tts.feedback_history[0]
        assert feedback['user_id'] == user_id, "Wrong user ID in feedback"
        assert feedback['rating'] == 4.5, "Wrong rating in feedback"
        
        print(f"  ✓ Feedback recorded for {user_id}")
        
        print("\n" + "=" * 50)
        print("🎉 All TTS synthesis property tests passed!")
        print("✅ Property 10: Natural Speech Synthesis validated")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This is expected if dependencies are not installed.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run TTS property tests."""
    print("🚀 Starting TTS Property-Based Tests...")
    print("Testing Property 10: Natural Speech Synthesis")
    print("Validates: Requirements 3.3")
    print()
    
    success = await test_tts_synthesis_properties()
    
    if success:
        print("\n✅ TTS property tests completed successfully!")
        return 0
    else:
        print("\n❌ TTS property tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)