"""
Accent Adaptation Module for BharatVoice Assistant.

This module implements regional accent and dialect adaptation capabilities,
learning from user speech patterns and adapting recognition and synthesis accordingly.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import numpy as np

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    AccentType,
    AudioBuffer,
    RecognitionResult
)

logger = logging.getLogger(__name__)


class AccentProfile:
    """Represents a user's accent profile for a specific language."""
    
    def __init__(
        self,
        language: LanguageCode,
        region: str,
        accent_type: AccentType = AccentType.STANDARD
    ):
        self.language = language
        self.region = region
        self.accent_type = accent_type
        self.phoneme_variations: Dict[str, List[str]] = {}
        self.pronunciation_patterns: Dict[str, float] = {}
        self.confidence_score = 0.5
        self.sample_count = 0
        self.last_updated = datetime.utcnow()
        
        # Acoustic features for accent adaptation
        self.pitch_range = {"min": 80.0, "max": 300.0, "mean": 150.0}
        self.formant_patterns: Dict[str, List[float]] = {}
        self.speaking_rate = 1.0  # Relative to standard rate
        self.voice_quality_features: Dict[str, float] = {}
    
    def update_from_audio(self, audio_features: Dict[str, Any]) -> None:
        """Update accent profile from audio analysis."""
        self.sample_count += 1
        self.last_updated = datetime.utcnow()
        
        # Update pitch characteristics
        if "pitch" in audio_features:
            pitch_data = audio_features["pitch"]
            self.pitch_range["min"] = min(self.pitch_range["min"], pitch_data.get("min", 80))
            self.pitch_range["max"] = max(self.pitch_range["max"], pitch_data.get("max", 300))
            
            # Running average for mean pitch
            alpha = 0.1  # Learning rate
            self.pitch_range["mean"] = (
                (1 - alpha) * self.pitch_range["mean"] + 
                alpha * pitch_data.get("mean", 150)
            )
        
        # Update formant patterns
        if "formants" in audio_features:
            formant_data = audio_features["formants"]
            for phoneme, formants in formant_data.items():
                if phoneme not in self.formant_patterns:
                    self.formant_patterns[phoneme] = formants
                else:
                    # Update with exponential moving average
                    current = np.array(self.formant_patterns[phoneme])
                    new_data = np.array(formants)
                    self.formant_patterns[phoneme] = (
                        0.9 * current + 0.1 * new_data
                    ).tolist()
        
        # Update speaking rate
        if "speaking_rate" in audio_features:
            rate = audio_features["speaking_rate"]
            self.speaking_rate = 0.9 * self.speaking_rate + 0.1 * rate
        
        # Update confidence based on sample count
        self.confidence_score = min(0.95, 0.5 + (self.sample_count * 0.05))


class AccentAdapter:
    """
    Adapts speech recognition and synthesis to user's regional accent and dialect.
    """
    
    def __init__(
        self,
        adaptation_threshold: int = 10,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize accent adapter.
        
        Args:
            adaptation_threshold: Minimum samples needed for adaptation
            confidence_threshold: Minimum confidence for accent detection
        """
        self.adaptation_threshold = adaptation_threshold
        self.confidence_threshold = confidence_threshold
        
        # User accent profiles
        self._user_accents: Dict[UUID, Dict[LanguageCode, AccentProfile]] = defaultdict(dict)
        
        # Regional accent mappings
        self._regional_mappings = {
            "Maharashtra": AccentType.MUMBAI,
            "Delhi": AccentType.DELHI,
            "Karnataka": AccentType.BANGALORE,
            "Tamil Nadu": AccentType.CHENNAI,
            "West Bengal": AccentType.KOLKATA,
            "North India": AccentType.NORTH_INDIAN,
            "South India": AccentType.SOUTH_INDIAN,
            "West India": AccentType.WEST_INDIAN,
            "East India": AccentType.EAST_INDIAN
        }
        
        # Language-specific accent characteristics
        self._accent_characteristics = {
            LanguageCode.ENGLISH_IN: {
                AccentType.MUMBAI: {
                    "r_pronunciation": "retroflex",
                    "vowel_shifts": {"a": "aa", "o": "oo"},
                    "consonant_patterns": {"th": "d", "v": "w"}
                },
                AccentType.SOUTH_INDIAN: {
                    "r_pronunciation": "rolled",
                    "vowel_shifts": {"i": "ee", "u": "oo"},
                    "consonant_patterns": {"p": "b", "t": "d"}
                },
                AccentType.NORTH_INDIAN: {
                    "r_pronunciation": "soft",
                    "vowel_shifts": {"e": "ay", "o": "aw"},
                    "consonant_patterns": {"z": "j", "f": "ph"}
                }
            }
        }
        
        logger.info("Accent Adapter initialized")
    
    async def learn_accent_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        audio_data: Optional[AudioBuffer] = None,
        recognition_result: Optional[RecognitionResult] = None
    ) -> Dict[str, Any]:
        """
        Learn user's accent from interaction data.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            audio_data: Audio data if available
            recognition_result: Speech recognition result
            
        Returns:
            Accent learning results
        """
        learning_result = {
            "accent_detected": False,
            "accent_type": None,
            "confidence": 0.0,
            "adaptations_made": [],
            "samples_collected": 0
        }
        
        language = interaction.input_language
        
        # Initialize accent profile if not exists
        if language not in self._user_accents[user_id]:
            region = await self._detect_region_from_interaction(interaction)
            accent_type = self._regional_mappings.get(region, AccentType.STANDARD)
            
            self._user_accents[user_id][language] = AccentProfile(
                language=language,
                region=region,
                accent_type=accent_type
            )
        
        accent_profile = self._user_accents[user_id][language]
        
        # Analyze audio features if available
        if audio_data:
            audio_features = await self._extract_accent_features(audio_data, language)
            accent_profile.update_from_audio(audio_features)
            learning_result["samples_collected"] = accent_profile.sample_count
        
        # Analyze text patterns for accent indicators
        if recognition_result:
            text_patterns = await self._analyze_text_patterns(
                recognition_result.transcribed_text,
                language
            )
            await self._update_pronunciation_patterns(accent_profile, text_patterns)
        
        # Detect accent type if enough samples
        if accent_profile.sample_count >= self.adaptation_threshold:
            detected_accent = await self._detect_accent_type(accent_profile)
            if detected_accent != accent_profile.accent_type:
                accent_profile.accent_type = detected_accent
                learning_result["adaptations_made"].append(f"accent_type_updated_to_{detected_accent.value}")
            
            learning_result["accent_detected"] = True
            learning_result["accent_type"] = accent_profile.accent_type.value
            learning_result["confidence"] = accent_profile.confidence_score
        
        logger.info(f"Learned accent patterns for user {user_id}, language {language.value}")
        return learning_result
    
    async def adapt_recognition_model(
        self,
        user_id: UUID,
        language: LanguageCode,
        base_model_id: str
    ) -> Optional[str]:
        """
        Adapt speech recognition model for user's accent.
        
        Args:
            user_id: User identifier
            language: Target language
            base_model_id: Base recognition model identifier
            
        Returns:
            Adapted model identifier if successful
        """
        if language not in self._user_accents[user_id]:
            return None
        
        accent_profile = self._user_accents[user_id][language]
        
        if accent_profile.confidence_score < self.confidence_threshold:
            return None
        
        # Generate adapted model configuration
        adaptation_config = {
            "base_model": base_model_id,
            "accent_type": accent_profile.accent_type.value,
            "phoneme_variations": accent_profile.phoneme_variations,
            "pronunciation_patterns": accent_profile.pronunciation_patterns,
            "acoustic_features": {
                "pitch_range": accent_profile.pitch_range,
                "formant_patterns": accent_profile.formant_patterns,
                "speaking_rate": accent_profile.speaking_rate
            }
        }
        
        # In production, this would interface with the actual model adaptation system
        adapted_model_id = f"{base_model_id}_accent_{accent_profile.accent_type.value}_{user_id}"
        
        logger.info(f"Adapted recognition model for user {user_id}: {adapted_model_id}")
        return adapted_model_id
    
    async def adapt_synthesis_parameters(
        self,
        user_id: UUID,
        language: LanguageCode,
        text: str
    ) -> Dict[str, Any]:
        """
        Adapt text-to-speech parameters for user's accent preference.
        
        Args:
            user_id: User identifier
            language: Target language
            text: Text to synthesize
            
        Returns:
            Adapted synthesis parameters
        """
        synthesis_params = {
            "accent_type": AccentType.STANDARD,
            "pitch_adjustment": 1.0,
            "rate_adjustment": 1.0,
            "pronunciation_rules": {}
        }
        
        if language not in self._user_accents[user_id]:
            return synthesis_params
        
        accent_profile = self._user_accents[user_id][language]
        
        if accent_profile.confidence_score < self.confidence_threshold:
            return synthesis_params
        
        # Apply accent-specific adaptations
        synthesis_params["accent_type"] = accent_profile.accent_type
        
        # Adjust pitch based on user's natural range
        user_mean_pitch = accent_profile.pitch_range["mean"]
        standard_pitch = 150.0  # Standard reference pitch
        synthesis_params["pitch_adjustment"] = user_mean_pitch / standard_pitch
        
        # Adjust speaking rate
        synthesis_params["rate_adjustment"] = accent_profile.speaking_rate
        
        # Apply pronunciation rules
        accent_chars = self._accent_characteristics.get(language, {}).get(
            accent_profile.accent_type, {}
        )
        synthesis_params["pronunciation_rules"] = accent_chars
        
        logger.info(f"Adapted synthesis parameters for user {user_id}, accent {accent_profile.accent_type.value}")
        return synthesis_params
    
    async def get_accent_profile(
        self,
        user_id: UUID,
        language: LanguageCode
    ) -> Optional[Dict[str, Any]]:
        """
        Get user's accent profile for a language.
        
        Args:
            user_id: User identifier
            language: Target language
            
        Returns:
            Accent profile data if available
        """
        if language not in self._user_accents[user_id]:
            return None
        
        accent_profile = self._user_accents[user_id][language]
        
        return {
            "language": language.value,
            "region": accent_profile.region,
            "accent_type": accent_profile.accent_type.value,
            "confidence": accent_profile.confidence_score,
            "sample_count": accent_profile.sample_count,
            "last_updated": accent_profile.last_updated.isoformat(),
            "characteristics": {
                "pitch_range": accent_profile.pitch_range,
                "speaking_rate": accent_profile.speaking_rate,
                "phoneme_variations_count": len(accent_profile.phoneme_variations)
            }
        }
    
    async def suggest_accent_improvements(
        self,
        user_id: UUID,
        language: LanguageCode
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements for accent adaptation.
        
        Args:
            user_id: User identifier
            language: Target language
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if language not in self._user_accents[user_id]:
            suggestions.append({
                "type": "data_collection",
                "message": "Speak more to help us learn your accent",
                "priority": "high"
            })
            return suggestions
        
        accent_profile = self._user_accents[user_id][language]
        
        # Check if more samples are needed
        if accent_profile.sample_count < self.adaptation_threshold:
            remaining = self.adaptation_threshold - accent_profile.sample_count
            suggestions.append({
                "type": "more_samples",
                "message": f"Speak {remaining} more times to improve accent adaptation",
                "priority": "medium"
            })
        
        # Check confidence level
        if accent_profile.confidence_score < self.confidence_threshold:
            suggestions.append({
                "type": "confidence_improvement",
                "message": "Try speaking more clearly to improve accent detection",
                "priority": "medium"
            })
        
        # Check for missing acoustic features
        if not accent_profile.formant_patterns:
            suggestions.append({
                "type": "audio_quality",
                "message": "Use better audio quality for improved accent learning",
                "priority": "low"
            })
        
        return suggestions
    
    async def _extract_accent_features(
        self,
        audio_data: AudioBuffer,
        language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Extract accent-relevant features from audio.
        
        Args:
            audio_data: Audio buffer
            language: Audio language
            
        Returns:
            Extracted acoustic features
        """
        # This is a simplified implementation
        # In production, this would use advanced audio analysis
        
        audio_array = audio_data.numpy_array
        sample_rate = audio_data.sample_rate
        
        features = {}
        
        # Basic pitch analysis
        # In production, use libraries like librosa or praat-parselmouth
        features["pitch"] = {
            "min": 80.0,
            "max": 300.0,
            "mean": 150.0
        }
        
        # Speaking rate estimation (words per minute)
        duration = audio_data.duration
        estimated_words = max(1, int(duration * 2))  # Rough estimate
        features["speaking_rate"] = estimated_words / (duration / 60.0) / 150.0  # Normalized
        
        # Placeholder for formant analysis
        features["formants"] = {
            "a": [700, 1200, 2500],
            "i": [300, 2300, 3000],
            "u": [300, 800, 2200]
        }
        
        return features
    
    async def _analyze_text_patterns(
        self,
        text: str,
        language: LanguageCode
    ) -> Dict[str, Any]:
        """
        Analyze text for accent-indicative patterns.
        
        Args:
            text: Transcribed text
            language: Text language
            
        Returns:
            Text pattern analysis
        """
        patterns = {
            "pronunciation_indicators": [],
            "dialect_markers": [],
            "code_switching_points": []
        }
        
        # Look for common accent indicators in English
        if language == LanguageCode.ENGLISH_IN:
            # Check for common Indian English patterns
            if "itself" in text.lower():
                patterns["pronunciation_indicators"].append("itself_usage")
            
            if "only" in text.lower() and text.lower().endswith("only"):
                patterns["pronunciation_indicators"].append("only_emphasis")
            
            # Check for retroflex pronunciations
            if any(word in text.lower() for word in ["very", "party", "thirty"]):
                patterns["pronunciation_indicators"].append("retroflex_r")
        
        return patterns
    
    async def _update_pronunciation_patterns(
        self,
        accent_profile: AccentProfile,
        text_patterns: Dict[str, Any]
    ) -> None:
        """Update pronunciation patterns in accent profile."""
        for indicator in text_patterns.get("pronunciation_indicators", []):
            if indicator not in accent_profile.pronunciation_patterns:
                accent_profile.pronunciation_patterns[indicator] = 0.0
            
            accent_profile.pronunciation_patterns[indicator] += 0.1
            # Cap at 1.0
            accent_profile.pronunciation_patterns[indicator] = min(
                1.0, accent_profile.pronunciation_patterns[indicator]
            )
    
    async def _detect_accent_type(self, accent_profile: AccentProfile) -> AccentType:
        """
        Detect accent type based on accumulated patterns.
        
        Args:
            accent_profile: User's accent profile
            
        Returns:
            Detected accent type
        """
        # Use region mapping as primary indicator
        region_accent = self._regional_mappings.get(accent_profile.region, AccentType.STANDARD)
        
        # Refine based on pronunciation patterns
        if accent_profile.pronunciation_patterns:
            # Check for specific accent markers
            if accent_profile.pronunciation_patterns.get("retroflex_r", 0) > 0.5:
                if region_accent == AccentType.STANDARD:
                    return AccentType.MUMBAI
            
            if accent_profile.pronunciation_patterns.get("only_emphasis", 0) > 0.3:
                return AccentType.SOUTH_INDIAN
        
        return region_accent
    
    async def _detect_region_from_interaction(
        self,
        interaction: UserInteraction
    ) -> str:
        """
        Detect user's region from interaction context.
        
        Args:
            interaction: User interaction
            
        Returns:
            Detected region
        """
        # This would typically use location data or other context
        # For now, return a default region
        return "India"  # Placeholder
    
    async def cleanup_old_profiles(
        self,
        user_id: UUID,
        days_threshold: int = 180
    ) -> int:
        """
        Clean up old accent profiles.
        
        Args:
            user_id: User identifier
            days_threshold: Days threshold for cleanup
            
        Returns:
            Number of profiles cleaned up
        """
        user_accents = self._user_accents.get(user_id, {})
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        profiles_to_remove = []
        for language, profile in user_accents.items():
            if profile.last_updated < cutoff_date and profile.sample_count < 5:
                profiles_to_remove.append(language)
        
        for language in profiles_to_remove:
            del user_accents[language]
        
        logger.info(f"Cleaned up {len(profiles_to_remove)} old accent profiles for user {user_id}")
        return len(profiles_to_remove)