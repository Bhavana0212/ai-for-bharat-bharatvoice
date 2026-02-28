"""
Property-based tests for Accessibility Support.

**Property 18: Accessibility Support**
**Validates: Requirements 5.1, 5.2, 5.3**

This module tests that the BharatVoice Assistant provides comprehensive
accessibility support for users with different needs and abilities.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.accessibility import (
    AccessibilityManager,
    AccessibilitySettings,
    VolumeLevel,
    SpeechRate,
    InteractionMode,
    AccessibilityMode,
    VoiceGuidedHelp,
    VisualIndicator,
    get_accessibility_manager
)


logger = structlog.get_logger(__name__)


# Test data generators
@st.composite
def generate_volume_level(draw):
    """Generate volume levels."""
    return draw(st.sampled_from(list(VolumeLevel)))


@st.composite
def generate_speech_rate(draw):
    """Generate speech rates."""
    return draw(st.sampled_from(list(SpeechRate)))


@st.composite
def generate_interaction_mode(draw):
    """Generate interaction modes."""
    return draw(st.sampled_from(list(InteractionMode)))


@st.composite
def generate_accessibility_mode(draw):
    """Generate accessibility modes."""
    return draw(st.sampled_from(list(AccessibilityMode)))


@st.composite
def generate_language_code(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_INDIA,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@st.composite
def generate_accessibility_settings(draw):
    """Generate accessibility settings."""
    return {
        "volume_level": draw(generate_volume_level()),
        "speech_rate": draw(generate_speech_rate()),
        "listening_timeout": draw(st.floats(min_value=5.0, max_value=120.0)),
        "max_recognition_attempts": draw(st.integers(min_value=1, max_value=10)),
        "interaction_mode": draw(generate_interaction_mode()),
        "accessibility_mode": draw(generate_accessibility_mode()),
        "preferred_language": draw(generate_language_code()),
        "enable_visual_indicators": draw(st.booleans()),
        "enable_voice_guided_help": draw(st.booleans()),
        "enable_confirmation_prompts": draw(st.booleans()),
        "enable_detailed_feedback": draw(st.booleans()),
        "enable_audio_descriptions": draw(st.booleans()),
        "enable_tutorial_mode": draw(st.booleans()),
    }


@st.composite
def generate_help_topics(draw):
    """Generate help topics."""
    topics = [
        "basic_commands",
        "voice_commands", 
        "language_switching",
        "volume_control",
        "accessibility",
        "offline_mode",
        "troubleshooting"
    ]
    return draw(st.sampled_from(topics))


class AccessibilityTestStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for accessibility manager.
    """
    
    def __init__(self):
        super().__init__()
        self.manager = AccessibilityManager()
        self.initial_settings = self.manager.settings
        self.settings_history = []
    
    @rule(settings=generate_accessibility_settings())
    def update_settings(self, settings):
        """Update accessibility settings."""
        self.settings_history.append(dict(settings))
        self.manager.update_settings(settings)
    
    @rule(direction=st.sampled_from(["up", "down", "mute", "max"]))
    def adjust_volume(self, direction):
        """Adjust volume level."""
        old_level = self.manager.settings.volume_level
        new_level = self.manager.adjust_volume(direction)
        
        # Verify volume change is logical
        if direction == "up" and old_level.value < VolumeLevel.MAXIMUM.value:
            assert new_level.value == old_level.value + 1
        elif direction == "down" and old_level.value > VolumeLevel.MUTE.value:
            assert new_level.value == old_level.value - 1
        elif direction == "mute":
            assert new_level == VolumeLevel.MUTE
        elif direction == "max":
            assert new_level == VolumeLevel.MAXIMUM
    
    @rule(mode=generate_interaction_mode())
    def switch_mode(self, mode):
        """Switch interaction mode."""
        old_mode = self.manager.current_mode
        new_mode = self.manager.switch_interaction_mode(mode)
        assert new_mode == mode
        assert self.manager.settings.interaction_mode == mode
    
    @rule(topic=generate_help_topics())
    def request_help(self, topic):
        """Request help on a topic."""
        help_text = self.manager.provide_help(topic)
        assert isinstance(help_text, str)
        assert len(help_text) > 0
    
    @invariant()
    def settings_consistency(self):
        """Settings should remain consistent."""
        settings = self.manager.settings
        
        # Volume level should be valid
        assert isinstance(settings.volume_level, VolumeLevel)
        
        # Speech rate should be valid
        assert isinstance(settings.speech_rate, SpeechRate)
        
        # Timeouts should be reasonable
        assert 5.0 <= settings.listening_timeout <= 300.0
        assert 1 <= settings.max_recognition_attempts <= 20
        
        # Language should be supported
        assert isinstance(settings.preferred_language, LanguageCode)
    
    @invariant()
    def state_consistency(self):
        """Manager state should be consistent."""
        # Current mode should match settings
        assert self.manager.current_mode == self.manager.settings.interaction_mode
        
        # Recognition attempts should be non-negative
        assert self.manager.recognition_attempts >= 0
        
        # Tutorial step should be reasonable
        if self.manager.settings.enable_tutorial_mode:
            total_steps = self.manager.voice_help.get_total_tutorial_steps()
            assert 0 <= self.manager.current_tutorial_step <= total_steps


@pytest.mark.asyncio
class TestAccessibilitySupport:
    """Test accessibility support compliance."""
    
    @pytest.fixture
    def accessibility_manager(self):
        """Create accessibility manager."""
        return AccessibilityManager()
    
    @given(
        volume_level=generate_volume_level(),
        speech_rate=generate_speech_rate(),
        language=generate_language_code()
    )
    @settings(max_examples=50, deadline=10000)
    async def test_adjustable_audio_settings(self, accessibility_manager, volume_level, speech_rate, language):
        """
        **Property 18: Accessibility Support**
        **Validates: Requirements 5.1**
        
        Property: Audio settings should be adjustable and affect speech synthesis.
        """
        # Update settings
        settings = {
            "volume_level": volume_level,
            "speech_rate": speech_rate,
            "preferred_language": language
        }
        accessibility_manager.update_settings(settings)
        
        # Property 1: Settings should be applied correctly
        assert accessibility_manager.settings.volume_level == volume_level, \
            f"Volume level not applied: expected {volume_level}, got {accessibility_manager.settings.volume_level}"
        
        assert accessibility_manager.settings.speech_rate == speech_rate, \
            f"Speech rate not applied: expected {speech_rate}, got {accessibility_manager.settings.speech_rate}"
        
        assert accessibility_manager.settings.preferred_language == language, \
            f"Language not applied: expected {language}, got {accessibility_manager.settings.preferred_language}"
        
        # Property 2: Speech configuration should reflect settings
        test_text = "This is a test message for accessibility."
        speech_config = accessibility_manager.start_speaking(test_text)
        
        assert speech_config["rate"] == speech_rate.value, \
            f"Speech rate not configured: expected {speech_rate.value}, got {speech_config['rate']}"
        
        expected_volume = volume_level.value / VolumeLevel.MAXIMUM.value
        assert abs(speech_config["volume"] - expected_volume) < 0.01, \
            f"Volume not configured: expected {expected_volume}, got {speech_config['volume']}"
        
        assert speech_config["language"] == language.value, \
            f"Language not configured: expected {language.value}, got {speech_config['language']}"
        
        # Property 3: Volume adjustments should work correctly
        old_level = accessibility_manager.settings.volume_level
        
        # Test volume up
        if old_level.value < VolumeLevel.MAXIMUM.value:
            new_level = accessibility_manager.adjust_volume("up")
            assert new_level.value == old_level.value + 1, \
                f"Volume up failed: expected {old_level.value + 1}, got {new_level.value}"
        
        # Test volume down
        current_level = accessibility_manager.settings.volume_level
        if current_level.value > VolumeLevel.MUTE.value:
            new_level = accessibility_manager.adjust_volume("down")
            assert new_level.value == current_level.value - 1, \
                f"Volume down failed: expected {current_level.value - 1}, got {new_level.value}"
        
        # Property 4: Mute should work
        muted_level = accessibility_manager.adjust_volume("mute")
        assert muted_level == VolumeLevel.MUTE, \
            f"Mute failed: expected {VolumeLevel.MUTE}, got {muted_level}"
        
        accessibility_manager.stop_speaking()
    
    @given(
        listening_timeout=st.floats(min_value=10.0, max_value=120.0),
        max_attempts=st.integers(min_value=2, max_value=8),
        enable_confirmation=st.booleans()
    )
    @settings(max_examples=30, deadline=15000)
    async def test_extended_interaction_support(self, accessibility_manager, listening_timeout, max_attempts, enable_confirmation):
        """
        **Property 18: Accessibility Support**
        **Validates: Requirements 5.2**
        
        Property: System should support extended interaction times and multiple attempts.
        """
        # Configure extended interaction settings
        settings = {
            "listening_timeout": listening_timeout,
            "max_recognition_attempts": max_attempts,
            "enable_confirmation_prompts": enable_confirmation
        }
        accessibility_manager.update_settings(settings)
        
        # Property 1: Settings should be applied
        assert accessibility_manager.settings.listening_timeout == listening_timeout, \
            f"Listening timeout not applied: expected {listening_timeout}, got {accessibility_manager.settings.listening_timeout}"
        
        assert accessibility_manager.settings.max_recognition_attempts == max_attempts, \
            f"Max attempts not applied: expected {max_attempts}, got {accessibility_manager.settings.max_recognition_attempts}"
        
        assert accessibility_manager.settings.enable_confirmation_prompts == enable_confirmation, \
            f"Confirmation prompts not applied: expected {enable_confirmation}, got {accessibility_manager.settings.enable_confirmation_prompts}"
        
        # Property 2: Listening should use configured timeout
        accessibility_manager.start_listening()
        assert accessibility_manager.is_listening, "Should be in listening state"
        
        # Simulate listening with custom timeout
        accessibility_manager.start_listening(timeout=listening_timeout)
        # The timeout should be respected (we can't easily test the actual timeout without waiting)
        
        accessibility_manager.stop_listening()
        assert not accessibility_manager.is_listening, "Should stop listening"
        
        # Property 3: Recognition failure handling should respect max attempts
        accessibility_manager.recognition_attempts = 0
        
        for attempt in range(max_attempts - 1):
            should_retry = accessibility_manager.handle_recognition_failure()
            assert should_retry, f"Should retry on attempt {attempt + 1}"
            assert accessibility_manager.recognition_attempts == attempt + 1, \
                f"Attempt count incorrect: expected {attempt + 1}, got {accessibility_manager.recognition_attempts}"
        
        # Final attempt should not allow retry
        should_retry = accessibility_manager.handle_recognition_failure()
        assert not should_retry, "Should not retry after max attempts"
        assert accessibility_manager.recognition_attempts == max_attempts, \
            f"Final attempt count incorrect: expected {max_attempts}, got {accessibility_manager.recognition_attempts}"
        
        # Property 4: System should provide appropriate feedback
        status_message = accessibility_manager.get_status_message()
        assert isinstance(status_message, str), "Status message should be string"
        assert len(status_message) > 0, "Status message should not be empty"
    
    @given(
        interaction_mode=generate_interaction_mode(),
        enable_visual=st.booleans(),
        enable_audio_desc=st.booleans()
    )
    @settings(max_examples=40, deadline=10000)
    async def test_mode_switching_seamless(self, accessibility_manager, interaction_mode, enable_visual, enable_audio_desc):
        """
        **Property 18: Accessibility Support**
        **Validates: Requirements 5.3**
        
        Property: Mode switching should be seamless and maintain user context.
        """
        # Configure accessibility features
        settings = {
            "interaction_mode": interaction_mode,
            "enable_visual_indicators": enable_visual,
            "enable_audio_descriptions": enable_audio_desc
        }
        accessibility_manager.update_settings(settings)
        
        # Property 1: Mode should be switched correctly
        old_mode = accessibility_manager.current_mode
        new_mode = accessibility_manager.switch_interaction_mode(interaction_mode)
        
        assert new_mode == interaction_mode, \
            f"Mode not switched: expected {interaction_mode}, got {new_mode}"
        
        assert accessibility_manager.current_mode == interaction_mode, \
            f"Current mode not updated: expected {interaction_mode}, got {accessibility_manager.current_mode}"
        
        assert accessibility_manager.settings.interaction_mode == interaction_mode, \
            f"Settings not updated: expected {interaction_mode}, got {accessibility_manager.settings.interaction_mode}"
        
        # Property 2: Visual indicators should work when enabled
        if enable_visual:
            indicator = accessibility_manager.visual_indicators.get_indicator("listening")
            assert isinstance(indicator, dict), "Visual indicator should be dictionary"
            assert "color" in indicator, "Indicator should have color"
            assert "animation" in indicator, "Indicator should have animation"
            assert "text" in indicator, "Indicator should have text"
            
            # Test different states
            states = ["listening", "processing", "speaking", "error", "success"]
            for state in states:
                state_indicator = accessibility_manager.visual_indicators.get_indicator(state)
                assert isinstance(state_indicator, dict), f"Indicator for {state} should be dictionary"
                assert len(state_indicator.get("text", "")) > 0, f"Indicator text for {state} should not be empty"
        
        # Property 3: Audio descriptions should be configured
        speech_config = accessibility_manager.start_speaking("Test message")
        assert speech_config["enable_audio_descriptions"] == enable_audio_desc, \
            f"Audio descriptions not configured: expected {enable_audio_desc}, got {speech_config['enable_audio_descriptions']}"
        
        accessibility_manager.stop_speaking()
        
        # Property 4: Mode switching should preserve other settings
        original_volume = accessibility_manager.settings.volume_level
        original_language = accessibility_manager.settings.preferred_language
        
        # Switch to different mode
        other_modes = [mode for mode in InteractionMode if mode != interaction_mode]
        if other_modes:
            other_mode = other_modes[0]
            accessibility_manager.switch_interaction_mode(other_mode)
            
            # Other settings should be preserved
            assert accessibility_manager.settings.volume_level == original_volume, \
                "Volume should be preserved during mode switch"
            
            assert accessibility_manager.settings.preferred_language == original_language, \
                "Language should be preserved during mode switch"
    
    @given(
        language=generate_language_code(),
        topic=generate_help_topics()
    )
    @settings(max_examples=50, deadline=15000)
    async def test_voice_guided_help_system(self, accessibility_manager, language, topic):
        """
        **Property 18: Accessibility Support**
        **Validates: Requirements 5.1, 5.3**
        
        Property: Voice-guided help should provide comprehensive assistance in user's language.
        """
        # Configure language
        settings = {
            "preferred_language": language,
            "enable_voice_guided_help": True
        }
        accessibility_manager.update_settings(settings)
        
        # Property 1: Help system should use correct language
        assert accessibility_manager.voice_help.language == language, \
            f"Help language not updated: expected {language}, got {accessibility_manager.voice_help.language}"
        
        # Property 2: Help should be available for topics
        help_text = accessibility_manager.provide_help(topic)
        assert isinstance(help_text, str), "Help text should be string"
        assert len(help_text) > 0, "Help text should not be empty"
        
        # Property 3: Help should be contextually appropriate
        if topic in ["basic_commands", "voice_commands"]:
            # Basic help should mention core functionality
            help_lower = help_text.lower()
            assert any(word in help_lower for word in ["speak", "voice", "command", "ask"]), \
                f"Basic help should mention voice functionality: {help_text}"
        
        # Property 4: General help should work
        general_help = accessibility_manager.provide_help()
        assert isinstance(general_help, str), "General help should be string"
        assert len(general_help) > 0, "General help should not be empty"
        
        # Property 5: Tutorial system should work
        if accessibility_manager.settings.enable_voice_guided_help:
            tutorial_start = accessibility_manager.start_tutorial()
            assert isinstance(tutorial_start, str), "Tutorial start should be string"
            assert len(tutorial_start) > 0, "Tutorial start should not be empty"
            
            # Should enable tutorial mode
            assert accessibility_manager.settings.enable_tutorial_mode, \
                "Tutorial mode should be enabled after starting tutorial"
            
            # Should track tutorial progress
            assert accessibility_manager.current_tutorial_step == 0, \
                f"Tutorial step should be 0 at start: {accessibility_manager.current_tutorial_step}"
            
            # Next step should work
            next_step = accessibility_manager.get_next_tutorial_step()
            assert isinstance(next_step, str), "Next tutorial step should be string"
            assert accessibility_manager.current_tutorial_step == 1, \
                f"Tutorial step should advance: {accessibility_manager.current_tutorial_step}"
    
    @given(
        accessibility_mode=generate_accessibility_mode(),
        settings=generate_accessibility_settings()
    )
    @settings(max_examples=30, deadline=20000)
    async def test_accessibility_presets_effectiveness(self, accessibility_manager, accessibility_mode, settings):
        """
        **Property 18: Accessibility Support**
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        Property: Accessibility presets should configure appropriate settings for different user needs.
        """
        # Apply custom settings first
        accessibility_manager.update_settings(settings)
        
        # Property 1: Settings should be applied correctly
        for key, value in settings.items():
            if hasattr(accessibility_manager.settings, key):
                current_value = getattr(accessibility_manager.settings, key)
                assert current_value == value, \
                    f"Setting {key} not applied: expected {value}, got {current_value}"
        
        # Property 2: Accessibility mode should influence behavior
        mode_settings = {"accessibility_mode": accessibility_mode}
        accessibility_manager.update_settings(mode_settings)
        
        assert accessibility_manager.settings.accessibility_mode == accessibility_mode, \
            f"Accessibility mode not applied: expected {accessibility_mode}, got {accessibility_manager.settings.accessibility_mode}"
        
        # Property 3: Different modes should have appropriate characteristics
        if accessibility_mode == AccessibilityMode.HEARING_IMPAIRED:
            # Should emphasize visual feedback
            assert accessibility_manager.settings.enable_visual_indicators or \
                   accessibility_manager.settings.interaction_mode == InteractionMode.TEXT_ONLY, \
                   "Hearing impaired mode should enable visual features"
        
        elif accessibility_mode == AccessibilityMode.VISION_IMPAIRED:
            # Should emphasize audio feedback
            assert accessibility_manager.settings.enable_voice_guided_help or \
                   accessibility_manager.settings.enable_audio_descriptions, \
                   "Vision impaired mode should enable audio features"
        
        elif accessibility_mode == AccessibilityMode.MOTOR_IMPAIRED:
            # Should allow more time for interactions
            assert accessibility_manager.settings.listening_timeout >= 30.0 or \
                   accessibility_manager.settings.max_recognition_attempts >= 3, \
                   "Motor impaired mode should allow extended interaction time"
        
        elif accessibility_mode == AccessibilityMode.ELDERLY_FRIENDLY:
            # Should use slower, clearer settings
            assert accessibility_manager.settings.speech_rate.value <= SpeechRate.NORMAL.value or \
                   accessibility_manager.settings.volume_level.value >= VolumeLevel.MEDIUM.value, \
                   "Elderly friendly mode should use appropriate audio settings"
        
        # Property 4: Accessibility report should be comprehensive
        report = accessibility_manager.get_accessibility_report()
        
        assert isinstance(report, dict), "Accessibility report should be dictionary"
        assert "settings" in report, "Report should include settings"
        assert "current_state" in report, "Report should include current state"
        assert "features_enabled" in report, "Report should include enabled features"
        
        # Verify report accuracy
        assert report["settings"]["accessibility_mode"] == accessibility_mode.value, \
            "Report should reflect current accessibility mode"
        
        assert report["settings"]["preferred_language"] == accessibility_manager.settings.preferred_language.value, \
            "Report should reflect current language"
        
        # Property 5: Status message should be informative
        status = accessibility_manager.get_status_message()
        assert isinstance(status, str), "Status should be string"
        assert len(status) > 0, "Status should not be empty"
        
        # Status should reflect current state
        if accessibility_manager.is_listening:
            assert "listening" in status.lower(), "Status should indicate listening state"
        
        if accessibility_manager.settings.volume_level == VolumeLevel.MUTE:
            assert "mute" in status.lower(), "Status should indicate muted state"


# Run stateful tests
TestAccessibilityStateMachine = AccessibilityTestStateMachine.TestCase


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
