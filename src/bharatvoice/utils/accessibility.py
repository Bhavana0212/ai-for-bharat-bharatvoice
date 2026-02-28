"""
Accessibility features for BharatVoice Assistant.

This module provides comprehensive accessibility support including adjustable
volume levels, extended listening time, seamless mode switching, visual indicators,
and voice-guided help system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import structlog

from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)


class AccessibilityMode(Enum):
    """Accessibility modes for different user needs."""
    STANDARD = "standard"
    HEARING_IMPAIRED = "hearing_impaired"
    VISION_IMPAIRED = "vision_impaired"
    MOTOR_IMPAIRED = "motor_impaired"
    COGNITIVE_SUPPORT = "cognitive_support"
    ELDERLY_FRIENDLY = "elderly_friendly"


class VolumeLevel(Enum):
    """Volume levels for audio output."""
    MUTE = 0
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    MAXIMUM = 6


class SpeechRate(Enum):
    """Speech rate options for TTS."""
    VERY_SLOW = 0.5
    SLOW = 0.7
    NORMAL = 1.0
    FAST = 1.3
    VERY_FAST = 1.6


class InteractionMode(Enum):
    """Interaction modes for the assistant."""
    VOICE_ONLY = "voice_only"
    TEXT_ONLY = "text_only"
    MIXED = "mixed"
    VISUAL_INDICATORS = "visual_indicators"


@dataclass
class AccessibilitySettings:
    """User accessibility settings and preferences."""
    
    # Audio settings
    volume_level: VolumeLevel = VolumeLevel.MEDIUM
    speech_rate: SpeechRate = SpeechRate.NORMAL
    enable_audio_descriptions: bool = False
    enable_sound_effects: bool = True
    
    # Timing settings
    listening_timeout: float = 30.0  # seconds
    response_delay: float = 0.5  # seconds before speaking
    max_recognition_attempts: int = 3
    pause_between_words: float = 0.1  # seconds
    
    # Interaction settings
    interaction_mode: InteractionMode = InteractionMode.MIXED
    enable_confirmation_prompts: bool = False
    enable_detailed_feedback: bool = False
    enable_progress_updates: bool = True
    
    # Visual settings
    enable_visual_indicators: bool = True
    high_contrast_mode: bool = False
    large_text_mode: bool = False
    
    # Language and cultural settings
    preferred_language: LanguageCode = LanguageCode.ENGLISH_INDIA
    enable_cultural_context: bool = True
    use_formal_language: bool = False
    
    # Assistance settings
    enable_voice_guided_help: bool = True
    enable_tutorial_mode: bool = False
    enable_step_by_step_guidance: bool = False
    
    # Accessibility mode
    accessibility_mode: AccessibilityMode = AccessibilityMode.STANDARD
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class VoiceGuidedHelp:
    """Voice-guided help and tutorial system."""
    
    def __init__(self, language: LanguageCode = LanguageCode.ENGLISH_INDIA):
        self.language = language
        self.help_topics = self._initialize_help_topics()
        self.tutorial_steps = self._initialize_tutorial_steps()
    
    def _initialize_help_topics(self) -> Dict[str, Dict[str, str]]:
        """Initialize help topics in multiple languages."""
        return {
            LanguageCode.ENGLISH_INDIA.value: {
                "basic_commands": "You can ask me to recognize speech, translate text, or help with various tasks. Just speak naturally and I'll understand.",
                "voice_commands": "Say 'Hello BharatVoice' to start, 'Stop' to pause, 'Help' for assistance, or 'Settings' to change preferences.",
                "language_switching": "You can switch languages by saying 'Switch to Hindi' or 'Change language to Tamil'. I support 11 Indian languages.",
                "volume_control": "Say 'Volume up', 'Volume down', 'Louder', 'Softer', or 'Mute' to adjust audio levels.",
                "accessibility": "I have special features for different needs. Say 'Accessibility options' to learn about hearing, vision, and motor assistance.",
                "offline_mode": "I can work offline for basic tasks. Say 'Go offline' or 'Work without internet' to enable offline mode.",
                "troubleshooting": "If I don't understand, speak clearly and closer to the microphone. Check your internet connection for online features.",
            },
            LanguageCode.HINDI.value: {
                "basic_commands": "आप मुझसे भाषण पहचानने, पाठ का अनुवाद करने, या विभिन्न कार्यों में मदद करने के लिए कह सकते हैं। बस स्वाभाविक रूप से बोलें और मैं समझ जाऊंगा।",
                "voice_commands": "शुरू करने के लिए 'नमस्ते भारतवॉयस' कहें, रोकने के लिए 'रुको', सहायता के लिए 'मदद', या प्राथमिकताएं बदलने के लिए 'सेटिंग्स' कहें।",
                "language_switching": "आप 'हिंदी में बदलें' या 'भाषा तमिल में बदलें' कहकर भाषा बदल सकते हैं। मैं 11 भारतीय भाषाओं का समर्थन करता हूं।",
                "volume_control": "ऑडियो स्तर समायोजित करने के लिए 'आवाज़ बढ़ाओ', 'आवाज़ कम करो', 'तेज़', 'धीमे', या 'मूक' कहें।",
                "accessibility": "मेरे पास विभिन्न आवश्यकताओं के लिए विशेष सुविधाएं हैं। सुनने, देखने और मोटर सहायता के बारे में जानने के लिए 'पहुंच विकल्प' कहें।",
                "offline_mode": "मैं बुनियादी कार्यों के लिए ऑफ़लाइन काम कर सकता हूं। ऑफ़लाइन मोड सक्षम करने के लिए 'ऑफ़लाइन जाओ' या 'इंटरनेट के बिना काम करो' कहें।",
                "troubleshooting": "यदि मैं नहीं समझता, तो स्पष्ट रूप से और माइक्रोफ़ोन के करीब बोलें। ऑनलाइन सुविधाओं के लिए अपना इंटरनेट कनेक्शन जांचें।",
            }
        }
    
    def _initialize_tutorial_steps(self) -> Dict[str, List[str]]:
        """Initialize tutorial steps for different languages."""
        return {
            LanguageCode.ENGLISH_INDIA.value: [
                "Welcome to BharatVoice! I'm your AI assistant that understands multiple Indian languages.",
                "Let's start with a simple test. Please say 'Hello' in any language you prefer.",
                "Great! Now try asking me something like 'What's the weather today?' or 'Translate hello to Hindi'.",
                "You can adjust my volume by saying 'Volume up' or 'Volume down'. Try it now.",
                "I can work in different modes. Say 'Switch to text mode' if you prefer typing instead of speaking.",
                "For help anytime, just say 'Help' or 'What can you do?'. I'm here to assist you!",
                "Tutorial complete! You're ready to use BharatVoice. Say 'Help' if you need assistance."
            ],
            LanguageCode.HINDI.value: [
                "भारतवॉयस में आपका स्वागत है! मैं आपका AI सहायक हूं जो कई भारतीय भाषाओं को समझता हूं।",
                "आइए एक सरल परीक्षण से शुरू करते हैं। कृपया अपनी पसंदीदा भाषा में 'नमस्ते' कहें।",
                "बहुत बढ़िया! अब मुझसे कुछ पूछने की कोशिश करें जैसे 'आज मौसम कैसा है?' या 'हैलो का हिंदी में अनुवाद करें'।",
                "आप 'आवाज़ बढ़ाओ' या 'आवाज़ कम करो' कहकर मेरी आवाज़ समायोजित कर सकते हैं। अभी कोशिश करें।",
                "मैं विभिन्न मोड में काम कर सकता हूं। यदि आप बोलने के बजाय टाइप करना पसंद करते हैं तो 'टेक्स्ट मोड में बदलें' कहें।",
                "किसी भी समय मदद के लिए, बस 'मदद' या 'आप क्या कर सकते हैं?' कहें। मैं आपकी सहायता के लिए यहां हूं!",
                "ट्यूटोरियल पूरा! आप भारतवॉयस का उपयोग करने के लिए तैयार हैं। यदि आपको सहायता चाहिए तो 'मदद' कहें।"
            ]
        }
    
    def get_help_text(self, topic: str) -> str:
        """
        Get help text for a specific topic.
        
        Args:
            topic: Help topic
            
        Returns:
            Help text in the user's language
        """
        lang_key = self.language.value
        if lang_key in self.help_topics and topic in self.help_topics[lang_key]:
            return self.help_topics[lang_key][topic]
        
        # Fallback to English
        if topic in self.help_topics.get(LanguageCode.ENGLISH_INDIA.value, {}):
            return self.help_topics[LanguageCode.ENGLISH_INDIA.value][topic]
        
        return f"Help topic '{topic}' not found. Say 'Help topics' to see available help."
    
    def get_tutorial_step(self, step: int) -> Optional[str]:
        """
        Get tutorial step text.
        
        Args:
            step: Tutorial step number (0-based)
            
        Returns:
            Tutorial step text or None if step doesn't exist
        """
        lang_key = self.language.value
        steps = self.tutorial_steps.get(lang_key, self.tutorial_steps.get(LanguageCode.ENGLISH_INDIA.value, []))
        
        if 0 <= step < len(steps):
            return steps[step]
        
        return None
    
    def get_total_tutorial_steps(self) -> int:
        """Get total number of tutorial steps."""
        lang_key = self.language.value
        steps = self.tutorial_steps.get(lang_key, self.tutorial_steps.get(LanguageCode.ENGLISH_INDIA.value, []))
        return len(steps)


class VisualIndicator:
    """Visual indicator system for accessibility."""
    
    def __init__(self):
        self.indicators = {
            "listening": {"color": "blue", "animation": "pulse", "text": "Listening..."},
            "processing": {"color": "yellow", "animation": "spin", "text": "Processing..."},
            "speaking": {"color": "green", "animation": "wave", "text": "Speaking..."},
            "error": {"color": "red", "animation": "flash", "text": "Error occurred"},
            "success": {"color": "green", "animation": "check", "text": "Success"},
            "waiting": {"color": "gray", "animation": "dots", "text": "Waiting..."},
            "offline": {"color": "orange", "animation": "static", "text": "Offline mode"},
        }
    
    def get_indicator(self, state: str) -> Dict[str, str]:
        """
        Get visual indicator for a state.
        
        Args:
            state: Current system state
            
        Returns:
            Visual indicator configuration
        """
        return self.indicators.get(state, self.indicators["waiting"])


class AccessibilityManager:
    """
    Main accessibility manager for BharatVoice Assistant.
    """
    
    def __init__(self):
        self.settings = AccessibilitySettings()
        self.voice_help = VoiceGuidedHelp()
        self.visual_indicators = VisualIndicator()
        self.current_tutorial_step = 0
        self.recognition_attempts = 0
        self.last_interaction_time = time.time()
        
        # State tracking
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.current_mode = InteractionMode.MIXED
        
        # Callbacks for different events
        self.callbacks: Dict[str, List[Callable]] = {
            "volume_changed": [],
            "mode_switched": [],
            "help_requested": [],
            "tutorial_step": [],
            "accessibility_enabled": [],
        }
    
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update accessibility settings.
        
        Args:
            new_settings: Dictionary of settings to update
        """
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.info("Accessibility setting updated", setting=key, value=value)
        
        # Update voice help language if changed
        if "preferred_language" in new_settings:
            self.voice_help.language = self.settings.preferred_language
        
        # Trigger callbacks
        self._trigger_callbacks("accessibility_enabled", new_settings)
    
    def adjust_volume(self, direction: str) -> VolumeLevel:
        """
        Adjust volume level.
        
        Args:
            direction: "up", "down", "mute", or specific level
            
        Returns:
            New volume level
        """
        current_level = self.settings.volume_level.value
        
        if direction == "up" and current_level < VolumeLevel.MAXIMUM.value:
            new_level = VolumeLevel(current_level + 1)
        elif direction == "down" and current_level > VolumeLevel.MUTE.value:
            new_level = VolumeLevel(current_level - 1)
        elif direction == "mute":
            new_level = VolumeLevel.MUTE
        elif direction == "max":
            new_level = VolumeLevel.MAXIMUM
        else:
            # Try to parse as specific level
            try:
                level_value = int(direction)
                if VolumeLevel.MUTE.value <= level_value <= VolumeLevel.MAXIMUM.value:
                    new_level = VolumeLevel(level_value)
                else:
                    return self.settings.volume_level
            except ValueError:
                return self.settings.volume_level
        
        self.settings.volume_level = new_level
        logger.info("Volume adjusted", old_level=current_level, new_level=new_level.value)
        
        # Trigger callbacks
        self._trigger_callbacks("volume_changed", new_level)
        
        return new_level
    
    def switch_interaction_mode(self, mode: Union[str, InteractionMode]) -> InteractionMode:
        """
        Switch interaction mode.
        
        Args:
            mode: New interaction mode
            
        Returns:
            New interaction mode
        """
        if isinstance(mode, str):
            try:
                mode = InteractionMode(mode)
            except ValueError:
                logger.warning("Invalid interaction mode", mode=mode)
                return self.current_mode
        
        old_mode = self.current_mode
        self.current_mode = mode
        self.settings.interaction_mode = mode
        
        logger.info("Interaction mode switched", old_mode=old_mode.value, new_mode=mode.value)
        
        # Trigger callbacks
        self._trigger_callbacks("mode_switched", {"old_mode": old_mode, "new_mode": mode})
        
        return mode
    
    def start_listening(self, timeout: Optional[float] = None) -> None:
        """
        Start listening with accessibility considerations.
        
        Args:
            timeout: Custom timeout or use settings default
        """
        self.is_listening = True
        self.recognition_attempts = 0
        
        # Use custom timeout or setting
        listen_timeout = timeout or self.settings.listening_timeout
        
        logger.info("Started listening", timeout=listen_timeout, attempts_allowed=self.settings.max_recognition_attempts)
        
        # Visual indicator
        if self.settings.enable_visual_indicators:
            indicator = self.visual_indicators.get_indicator("listening")
            logger.info("Visual indicator", **indicator)
    
    def stop_listening(self) -> None:
        """Stop listening and reset state."""
        self.is_listening = False
        self.last_interaction_time = time.time()
        
        logger.info("Stopped listening")
    
    def handle_recognition_failure(self) -> bool:
        """
        Handle speech recognition failure with accessibility support.
        
        Returns:
            True if should retry, False if max attempts reached
        """
        self.recognition_attempts += 1
        
        if self.recognition_attempts < self.settings.max_recognition_attempts:
            logger.info("Recognition failed, retrying", attempt=self.recognition_attempts, max_attempts=self.settings.max_recognition_attempts)
            return True
        else:
            logger.warning("Max recognition attempts reached", attempts=self.recognition_attempts)
            return False
    
    def start_speaking(self, text: str) -> Dict[str, Any]:
        """
        Start speaking with accessibility adjustments.
        
        Args:
            text: Text to speak
            
        Returns:
            Speech configuration
        """
        self.is_speaking = True
        
        # Apply accessibility settings
        speech_config = {
            "text": text,
            "rate": self.settings.speech_rate.value,
            "volume": self.settings.volume_level.value / VolumeLevel.MAXIMUM.value,
            "pause_between_words": self.settings.pause_between_words,
            "language": self.settings.preferred_language.value,
            "enable_audio_descriptions": self.settings.enable_audio_descriptions,
        }
        
        # Add delay before speaking if configured
        if self.settings.response_delay > 0:
            speech_config["delay"] = self.settings.response_delay
        
        logger.info("Started speaking", **speech_config)
        
        # Visual indicator
        if self.settings.enable_visual_indicators:
            indicator = self.visual_indicators.get_indicator("speaking")
            logger.info("Visual indicator", **indicator)
        
        return speech_config
    
    def stop_speaking(self) -> None:
        """Stop speaking and reset state."""
        self.is_speaking = False
        logger.info("Stopped speaking")
    
    def provide_help(self, topic: Optional[str] = None) -> str:
        """
        Provide voice-guided help.
        
        Args:
            topic: Specific help topic or None for general help
            
        Returns:
            Help text
        """
        if topic:
            help_text = self.voice_help.get_help_text(topic)
        else:
            # Provide general help based on current context
            if self.settings.tutorial_mode:
                help_text = self.get_next_tutorial_step()
            else:
                help_text = self.voice_help.get_help_text("basic_commands")
        
        logger.info("Help provided", topic=topic or "general")
        
        # Trigger callbacks
        self._trigger_callbacks("help_requested", {"topic": topic, "text": help_text})
        
        return help_text
    
    def start_tutorial(self) -> str:
        """
        Start the voice-guided tutorial.
        
        Returns:
            First tutorial step
        """
        self.settings.enable_tutorial_mode = True
        self.current_tutorial_step = 0
        
        first_step = self.voice_help.get_tutorial_step(0)
        logger.info("Tutorial started", step=0, total_steps=self.voice_help.get_total_tutorial_steps())
        
        # Trigger callbacks
        self._trigger_callbacks("tutorial_step", {"step": 0, "text": first_step, "total": self.voice_help.get_total_tutorial_steps()})
        
        return first_step or "Tutorial not available in your language."
    
    def get_next_tutorial_step(self) -> str:
        """
        Get the next tutorial step.
        
        Returns:
            Next tutorial step or completion message
        """
        self.current_tutorial_step += 1
        total_steps = self.voice_help.get_total_tutorial_steps()
        
        if self.current_tutorial_step >= total_steps:
            self.settings.enable_tutorial_mode = False
            return "Tutorial completed! You're ready to use BharatVoice."
        
        step_text = self.voice_help.get_tutorial_step(self.current_tutorial_step)
        
        logger.info("Tutorial step", step=self.current_tutorial_step, total_steps=total_steps)
        
        # Trigger callbacks
        self._trigger_callbacks("tutorial_step", {
            "step": self.current_tutorial_step,
            "text": step_text,
            "total": total_steps
        })
        
        return step_text or "Tutorial step not available."
    
    def get_status_message(self) -> str:
        """
        Get current status message for accessibility.
        
        Returns:
            Status message describing current state
        """
        status_parts = []
        
        if self.is_listening:
            status_parts.append("Listening for your voice")
        elif self.is_speaking:
            status_parts.append("Speaking")
        elif self.is_processing:
            status_parts.append("Processing your request")
        else:
            status_parts.append("Ready")
        
        # Add mode information
        if self.current_mode != InteractionMode.MIXED:
            status_parts.append(f"in {self.current_mode.value.replace('_', ' ')} mode")
        
        # Add volume information if not standard
        if self.settings.volume_level != VolumeLevel.MEDIUM:
            if self.settings.volume_level == VolumeLevel.MUTE:
                status_parts.append("(muted)")
            else:
                status_parts.append(f"(volume {self.settings.volume_level.value})")
        
        return " ".join(status_parts)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for accessibility events.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
            logger.debug("Callback registered", event=event)
    
    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """
        Trigger callbacks for an event.
        
        Args:
            event: Event name
            data: Event data
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error("Callback error", event=event, exc_info=e)
    
    def get_accessibility_report(self) -> Dict[str, Any]:
        """
        Get comprehensive accessibility status report.
        
        Returns:
            Accessibility status and settings
        """
        return {
            "settings": {
                "volume_level": self.settings.volume_level.value,
                "speech_rate": self.settings.speech_rate.value,
                "listening_timeout": self.settings.listening_timeout,
                "max_recognition_attempts": self.settings.max_recognition_attempts,
                "interaction_mode": self.settings.interaction_mode.value,
                "accessibility_mode": self.settings.accessibility_mode.value,
                "preferred_language": self.settings.preferred_language.value,
            },
            "current_state": {
                "is_listening": self.is_listening,
                "is_speaking": self.is_speaking,
                "is_processing": self.is_processing,
                "current_mode": self.current_mode.value,
                "recognition_attempts": self.recognition_attempts,
                "tutorial_active": self.settings.enable_tutorial_mode,
                "tutorial_step": self.current_tutorial_step if self.settings.enable_tutorial_mode else None,
            },
            "features_enabled": {
                "visual_indicators": self.settings.enable_visual_indicators,
                "voice_guided_help": self.settings.enable_voice_guided_help,
                "confirmation_prompts": self.settings.enable_confirmation_prompts,
                "detailed_feedback": self.settings.enable_detailed_feedback,
                "progress_updates": self.settings.enable_progress_updates,
                "audio_descriptions": self.settings.enable_audio_descriptions,
            }
        }


# Global accessibility manager instance
_accessibility_manager: Optional[AccessibilityManager] = None


def get_accessibility_manager() -> AccessibilityManager:
    """Get the global accessibility manager instance."""
    global _accessibility_manager
    if _accessibility_manager is None:
        _accessibility_manager = AccessibilityManager()
    return _accessibility_manager