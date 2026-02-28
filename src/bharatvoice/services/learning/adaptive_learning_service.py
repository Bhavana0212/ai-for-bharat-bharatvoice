"""
Adaptive Learning Service for BharatVoice Assistant.

This service orchestrates all learning and adaptation mechanisms, providing
a unified interface for vocabulary learning, accent adaptation, preference learning,
feedback processing, and response style adaptation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    UserProfile,
    AudioBuffer,
    RecognitionResult
)

from .vocabulary_learner import VocabularyLearner
from .accent_adapter import AccentAdapter
from .preference_learner import PreferenceLearner
from .feedback_processor import FeedbackProcessor, FeedbackType
from .response_style_adapter import ResponseStyleAdapter

logger = logging.getLogger(__name__)


class AdaptiveLearningService:
    """
    Main service that orchestrates all adaptive learning mechanisms.
    """
    
    def __init__(
        self,
        enable_vocabulary_learning: bool = True,
        enable_accent_adaptation: bool = True,
        enable_preference_learning: bool = True,
        enable_feedback_processing: bool = True,
        enable_style_adaptation: bool = True,
        learning_rate: float = 0.1
    ):
        """
        Initialize adaptive learning service.
        
        Args:
            enable_vocabulary_learning: Enable vocabulary learning
            enable_accent_adaptation: Enable accent adaptation
            enable_preference_learning: Enable preference learning
            enable_feedback_processing: Enable feedback processing
            enable_style_adaptation: Enable response style adaptation
            learning_rate: Global learning rate
        """
        self.enable_vocabulary_learning = enable_vocabulary_learning
        self.enable_accent_adaptation = enable_accent_adaptation
        self.enable_preference_learning = enable_preference_learning
        self.enable_feedback_processing = enable_feedback_processing
        self.enable_style_adaptation = enable_style_adaptation
        self.learning_rate = learning_rate
        
        # Initialize learning components
        self.vocabulary_learner = VocabularyLearner() if enable_vocabulary_learning else None
        self.accent_adapter = AccentAdapter() if enable_accent_adaptation else None
        self.preference_learner = PreferenceLearner(learning_rate=learning_rate) if enable_preference_learning else None
        self.feedback_processor = FeedbackProcessor() if enable_feedback_processing else None
        self.response_style_adapter = ResponseStyleAdapter() if enable_style_adaptation else None
        
        # Learning statistics
        self._learning_stats = {
            "total_interactions_processed": 0,
            "vocabulary_words_learned": 0,
            "accent_profiles_created": 0,
            "preferences_updated": 0,
            "feedback_entries_processed": 0,
            "style_adaptations_made": 0,
            "last_learning_session": None
        }
        
        logger.info("Adaptive Learning Service initialized")
    
    async def process_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        audio_data: Optional[AudioBuffer] = None,
        recognition_result: Optional[RecognitionResult] = None,
        user_feedback: Optional[Dict[str, Any]] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process user interaction through all learning mechanisms.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            audio_data: Audio data if available
            recognition_result: Speech recognition result
            user_feedback: Explicit user feedback
            response_time: Response time in seconds
            
        Returns:
            Comprehensive learning results
        """
        learning_results = {
            "vocabulary_learning": {},
            "accent_adaptation": {},
            "preference_learning": {},
            "style_adaptation": {},
            "overall_improvements": [],
            "learning_confidence": 0.0
        }
        
        # Process vocabulary learning
        if self.vocabulary_learner and self.enable_vocabulary_learning:
            try:
                vocab_result = await self.vocabulary_learner.learn_from_interaction(
                    user_id, interaction
                )
                learning_results["vocabulary_learning"] = vocab_result
                self._learning_stats["vocabulary_words_learned"] += len(vocab_result.get("new_words", []))
            except Exception as e:
                logger.error(f"Vocabulary learning error: {e}")
                learning_results["vocabulary_learning"] = {"error": str(e)}
        
        # Process accent adaptation
        if self.accent_adapter and self.enable_accent_adaptation:
            try:
                accent_result = await self.accent_adapter.learn_accent_from_interaction(
                    user_id, interaction, audio_data, recognition_result
                )
                learning_results["accent_adaptation"] = accent_result
                if accent_result.get("accent_detected"):
                    self._learning_stats["accent_profiles_created"] += 1
            except Exception as e:
                logger.error(f"Accent adaptation error: {e}")
                learning_results["accent_adaptation"] = {"error": str(e)}
        
        # Process preference learning
        if self.preference_learner and self.enable_preference_learning:
            try:
                pref_result = await self.preference_learner.learn_from_interaction(
                    user_id, interaction, user_feedback, response_time
                )
                learning_results["preference_learning"] = pref_result
                self._learning_stats["preferences_updated"] += len(pref_result.get("preferences_updated", []))
            except Exception as e:
                logger.error(f"Preference learning error: {e}")
                learning_results["preference_learning"] = {"error": str(e)}
        
        # Process response style adaptation
        if self.response_style_adapter and self.enable_style_adaptation:
            try:
                # Calculate response effectiveness (simplified)
                response_effectiveness = await self._calculate_response_effectiveness(
                    interaction, user_feedback
                )
                
                style_result = await self.response_style_adapter.learn_style_from_interaction(
                    user_id, interaction, user_feedback, response_effectiveness
                )
                learning_results["style_adaptation"] = style_result
                self._learning_stats["style_adaptations_made"] += len(style_result.get("style_updates", []))
            except Exception as e:
                logger.error(f"Style adaptation error: {e}")
                learning_results["style_adaptation"] = {"error": str(e)}
        
        # Update overall statistics
        self._learning_stats["total_interactions_processed"] += 1
        self._learning_stats["last_learning_session"] = datetime.utcnow().isoformat()
        
        # Calculate overall learning confidence
        learning_results["learning_confidence"] = await self._calculate_overall_confidence(
            user_id, learning_results
        )
        
        # Generate overall improvement suggestions
        learning_results["overall_improvements"] = await self._generate_overall_improvements(
            user_id, learning_results
        )
        
        logger.info(f"Processed interaction for user {user_id} through all learning mechanisms")
        return learning_results
    
    async def collect_user_feedback(
        self,
        user_id: UUID,
        interaction_id: UUID,
        feedback_type: str,
        feedback_content: Dict[str, Any]
    ) -> UUID:
        """
        Collect user feedback for learning improvement.
        
        Args:
            user_id: User identifier
            interaction_id: Interaction identifier
            feedback_type: Type of feedback
            feedback_content: Feedback content
            
        Returns:
            Feedback entry ID
        """
        if not self.feedback_processor or not self.enable_feedback_processing:
            raise ValueError("Feedback processing is not enabled")
        
        # Convert string feedback type to enum
        feedback_type_enum = FeedbackType(feedback_type)
        
        feedback_id = await self.feedback_processor.collect_feedback(
            user_id, interaction_id, feedback_type_enum, feedback_content
        )
        
        self._learning_stats["feedback_entries_processed"] += 1
        
        logger.info(f"Collected {feedback_type} feedback from user {user_id}")
        return feedback_id
    
    async def adapt_response_for_user(
        self,
        user_id: UUID,
        base_response: str,
        language: LanguageCode,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Adapt response using all learned user preferences.
        
        Args:
            user_id: User identifier
            base_response: Base response text
            language: Response language
            context: Additional context
            
        Returns:
            Fully adapted response
        """
        adapted_response = base_response
        
        # Apply preference-based adaptations
        if self.preference_learner and self.enable_preference_learning:
            try:
                adapted_response = await self.preference_learner.adapt_response_for_user(
                    user_id, adapted_response, context or {}
                )
            except Exception as e:
                logger.error(f"Preference adaptation error: {e}")
        
        # Apply style adaptations
        if self.response_style_adapter and self.enable_style_adaptation:
            try:
                adapted_response = await self.response_style_adapter.adapt_response_style(
                    user_id, adapted_response, language, context
                )
            except Exception as e:
                logger.error(f"Style adaptation error: {e}")
        
        return adapted_response
    
    async def get_user_learning_profile(
        self,
        user_id: UUID,
        include_detailed_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive user learning profile.
        
        Args:
            user_id: User identifier
            include_detailed_stats: Include detailed statistics
            
        Returns:
            User learning profile
        """
        profile = {
            "user_id": str(user_id),
            "learning_enabled": {
                "vocabulary": self.enable_vocabulary_learning,
                "accent": self.enable_accent_adaptation,
                "preferences": self.enable_preference_learning,
                "feedback": self.enable_feedback_processing,
                "style": self.enable_style_adaptation
            },
            "vocabulary_profile": {},
            "accent_profile": {},
            "preference_profile": {},
            "style_profile": {},
            "overall_confidence": 0.0
        }
        
        # Get vocabulary profile
        if self.vocabulary_learner and self.enable_vocabulary_learning:
            try:
                vocab_profile = await self.vocabulary_learner.get_user_vocabulary(user_id)
                profile["vocabulary_profile"] = vocab_profile
            except Exception as e:
                logger.error(f"Error getting vocabulary profile: {e}")
        
        # Get accent profile
        if self.accent_adapter and self.enable_accent_adaptation:
            try:
                # Get accent profiles for all languages
                accent_profiles = {}
                for lang in LanguageCode:
                    accent_profile = await self.accent_adapter.get_accent_profile(user_id, lang)
                    if accent_profile:
                        accent_profiles[lang.value] = accent_profile
                profile["accent_profile"] = accent_profiles
            except Exception as e:
                logger.error(f"Error getting accent profile: {e}")
        
        # Get preference profile
        if self.preference_learner and self.enable_preference_learning:
            try:
                pref_profile = await self.preference_learner.get_user_preferences(user_id)
                profile["preference_profile"] = pref_profile
            except Exception as e:
                logger.error(f"Error getting preference profile: {e}")
        
        # Get style profile
        if self.response_style_adapter and self.enable_style_adaptation:
            try:
                style_profile = await self.response_style_adapter.get_user_style_profile(user_id)
                profile["style_profile"] = style_profile
            except Exception as e:
                logger.error(f"Error getting style profile: {e}")
        
        # Calculate overall confidence
        profile["overall_confidence"] = await self._calculate_user_overall_confidence(user_id)
        
        if include_detailed_stats:
            profile["detailed_stats"] = await self._get_detailed_user_stats(user_id)
        
        return profile
    
    async def get_personalization_suggestions(
        self,
        user_id: UUID,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get personalization suggestions for user.
        
        Args:
            user_id: User identifier
            context: Current context
            
        Returns:
            List of personalization suggestions
        """
        all_suggestions = []
        
        # Get preference-based suggestions
        if self.preference_learner and self.enable_preference_learning:
            try:
                pref_suggestions = await self.preference_learner.suggest_personalization(
                    user_id, context
                )
                all_suggestions.extend(pref_suggestions)
            except Exception as e:
                logger.error(f"Error getting preference suggestions: {e}")
        
        # Get style improvement suggestions
        if self.response_style_adapter and self.enable_style_adaptation:
            try:
                for lang in [LanguageCode.ENGLISH_IN, LanguageCode.HINDI]:
                    style_suggestions = await self.response_style_adapter.suggest_style_improvements(
                        user_id, lang
                    )
                    all_suggestions.extend(style_suggestions)
            except Exception as e:
                logger.error(f"Error getting style suggestions: {e}")
        
        # Get accent improvement suggestions
        if self.accent_adapter and self.enable_accent_adaptation:
            try:
                for lang in [LanguageCode.ENGLISH_IN, LanguageCode.HINDI]:
                    accent_suggestions = await self.accent_adapter.suggest_accent_improvements(
                        user_id, lang
                    )
                    all_suggestions.extend(accent_suggestions)
            except Exception as e:
                logger.error(f"Error getting accent suggestions: {e}")
        
        # Get vocabulary expansion suggestions
        if self.vocabulary_learner and self.enable_vocabulary_learning:
            try:
                for lang in [LanguageCode.ENGLISH_IN, LanguageCode.HINDI]:
                    vocab_suggestions = await self.vocabulary_learner.suggest_vocabulary_expansion(
                        user_id, lang
                    )
                    # Convert to standard suggestion format
                    for suggestion in vocab_suggestions:
                        all_suggestions.append({
                            "type": "vocabulary_expansion",
                            "suggestion": f"Learn word: {suggestion['suggested_word']}",
                            "confidence": suggestion["confidence"],
                            "reason": f"Related to {suggestion['related_to']}"
                        })
            except Exception as e:
                logger.error(f"Error getting vocabulary suggestions: {e}")
        
        # Sort suggestions by priority and confidence
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        all_suggestions.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 1),
                x.get("confidence", 0.0)
            ),
            reverse=True
        )
        
        return all_suggestions[:20]  # Return top 20 suggestions
    
    async def process_feedback_batch(self, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process a batch of collected feedback.
        
        Args:
            batch_size: Number of feedback entries to process
            
        Returns:
            Batch processing results
        """
        if not self.feedback_processor or not self.enable_feedback_processing:
            return {"error": "Feedback processing is not enabled"}
        
        try:
            results = await self.feedback_processor.process_feedback_batch(batch_size)
            self._learning_stats["feedback_entries_processed"] += results.get("processed_count", 0)
            return results
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
            return {"error": str(e)}
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get overall learning system statistics.
        
        Returns:
            Learning system statistics
        """
        stats = self._learning_stats.copy()
        
        # Add component-specific statistics
        if self.feedback_processor and self.enable_feedback_processing:
            try:
                feedback_stats = await self.feedback_processor.get_feedback_statistics()
                stats["feedback_statistics"] = feedback_stats
            except Exception as e:
                logger.error(f"Error getting feedback statistics: {e}")
        
        return stats
    
    async def cleanup_old_data(
        self,
        days_threshold: int = 90
    ) -> Dict[str, int]:
        """
        Clean up old learning data across all components.
        
        Args:
            days_threshold: Days threshold for cleanup
            
        Returns:
            Cleanup results by component
        """
        cleanup_results = {}
        
        # Cleanup feedback data
        if self.feedback_processor and self.enable_feedback_processing:
            try:
                cleaned_feedback = await self.feedback_processor.cleanup_old_feedback()
                cleanup_results["feedback_entries"] = cleaned_feedback
            except Exception as e:
                logger.error(f"Error cleaning up feedback data: {e}")
                cleanup_results["feedback_entries"] = 0
        
        return cleanup_results
    
    async def _calculate_response_effectiveness(
        self,
        interaction: UserInteraction,
        user_feedback: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate response effectiveness score."""
        effectiveness = 0.5  # Default neutral score
        
        if user_feedback:
            # Use explicit feedback
            if "rating" in user_feedback:
                effectiveness = user_feedback["rating"] / 5.0
            elif "positive" in user_feedback:
                effectiveness = 0.8 if user_feedback["positive"] else 0.2
        else:
            # Use implicit indicators
            response_length = len(interaction.response_text)
            if 50 <= response_length <= 200:  # Optimal length range
                effectiveness += 0.1
            
            # Check for successful task completion indicators
            if any(word in interaction.response_text.lower() for word in 
                   ["here", "found", "completed", "done", "success"]):
                effectiveness += 0.2
        
        return min(1.0, max(0.0, effectiveness))
    
    async def _calculate_overall_confidence(
        self,
        user_id: UUID,
        learning_results: Dict[str, Any]
    ) -> float:
        """Calculate overall learning confidence."""
        confidences = []
        
        # Collect confidence scores from different components
        if "vocabulary_learning" in learning_results:
            vocab_result = learning_results["vocabulary_learning"]
            if "total_vocabulary_size" in vocab_result and vocab_result["total_vocabulary_size"] > 0:
                confidences.append(min(1.0, vocab_result["total_vocabulary_size"] / 100.0))
        
        if "accent_adaptation" in learning_results:
            accent_result = learning_results["accent_adaptation"]
            if "confidence" in accent_result:
                confidences.append(accent_result["confidence"])
        
        if "preference_learning" in learning_results:
            pref_result = learning_results["preference_learning"]
            if "confidence_changes" in pref_result:
                pref_confidences = list(pref_result["confidence_changes"].values())
                if pref_confidences:
                    confidences.append(sum(pref_confidences) / len(pref_confidences))
        
        if "style_adaptation" in learning_results:
            style_result = learning_results["style_adaptation"]
            if "confidence_changes" in style_result:
                style_confidences = list(style_result["confidence_changes"].values())
                if style_confidences:
                    confidences.append(sum(style_confidences) / len(style_confidences))
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    async def _generate_overall_improvements(
        self,
        user_id: UUID,
        learning_results: Dict[str, Any]
    ) -> List[str]:
        """Generate overall improvement suggestions."""
        improvements = []
        
        # Check vocabulary learning
        vocab_result = learning_results.get("vocabulary_learning", {})
        if vocab_result.get("new_words"):
            improvements.append(f"Learned {len(vocab_result['new_words'])} new vocabulary words")
        
        # Check accent adaptation
        accent_result = learning_results.get("accent_adaptation", {})
        if accent_result.get("accent_detected"):
            improvements.append(f"Detected accent: {accent_result.get('accent_type', 'unknown')}")
        
        # Check preference updates
        pref_result = learning_results.get("preference_learning", {})
        if pref_result.get("preferences_updated"):
            improvements.append(f"Updated {len(pref_result['preferences_updated'])} preferences")
        
        # Check style adaptations
        style_result = learning_results.get("style_adaptation", {})
        if style_result.get("style_updates"):
            improvements.append(f"Applied {len(style_result['style_updates'])} style adaptations")
        
        return improvements
    
    async def _calculate_user_overall_confidence(self, user_id: UUID) -> float:
        """Calculate user's overall learning confidence across all components."""
        confidences = []
        
        # This would aggregate confidence scores from all learning components
        # For now, return a placeholder
        return 0.7  # Placeholder confidence score
    
    async def _get_detailed_user_stats(self, user_id: UUID) -> Dict[str, Any]:
        """Get detailed user statistics across all learning components."""
        return {
            "interactions_processed": self._learning_stats["total_interactions_processed"],
            "learning_components_active": sum([
                self.enable_vocabulary_learning,
                self.enable_accent_adaptation,
                self.enable_preference_learning,
                self.enable_feedback_processing,
                self.enable_style_adaptation
            ]),
            "last_learning_session": self._learning_stats["last_learning_session"]
        }