"""
Preference Learning Module for BharatVoice Assistant.

This module learns user preferences from usage patterns, including response styles,
interaction preferences, and service usage patterns.
"""

import asyncio
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import statistics

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    ServiceType,
    UserProfile
)

logger = logging.getLogger(__name__)


class PreferenceCategory:
    """Represents a category of user preferences."""
    
    def __init__(self, name: str):
        self.name = name
        self.preferences: Dict[str, float] = {}
        self.confidence = 0.0
        self.sample_count = 0
        self.last_updated = datetime.utcnow()
    
    def update_preference(self, preference_key: str, value: float, weight: float = 1.0) -> None:
        """Update a specific preference with weighted learning."""
        if preference_key not in self.preferences:
            self.preferences[preference_key] = value
        else:
            # Exponential moving average with weight
            alpha = min(0.3, weight * 0.1)  # Learning rate
            self.preferences[preference_key] = (
                (1 - alpha) * self.preferences[preference_key] + alpha * value
            )
        
        self.sample_count += 1
        self.last_updated = datetime.utcnow()
        self.confidence = min(0.95, 0.3 + (self.sample_count * 0.02))
    
    def get_top_preferences(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K preferences sorted by value."""
        sorted_prefs = sorted(self.preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_prefs[:top_k]


class PreferenceLearner:
    """
    Learns user preferences from interaction patterns and usage behavior.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_interactions: int = 5,
        preference_decay_days: int = 30
    ):
        """
        Initialize preference learner.
        
        Args:
            learning_rate: Rate of preference adaptation
            min_interactions: Minimum interactions before making adaptations
            preference_decay_days: Days after which preferences start decaying
        """
        self.learning_rate = learning_rate
        self.min_interactions = min_interactions
        self.preference_decay_days = preference_decay_days
        
        # User preference storage
        self._user_preferences: Dict[UUID, Dict[str, PreferenceCategory]] = defaultdict(
            lambda: defaultdict(lambda: PreferenceCategory(""))
        )
        
        # Preference categories
        self._preference_categories = {
            "response_style": [
                "formal", "casual", "friendly", "professional", "humorous",
                "detailed", "concise", "explanatory", "direct"
            ],
            "interaction_style": [
                "quick_responses", "detailed_explanations", "step_by_step",
                "examples_preferred", "voice_only", "text_preferred"
            ],
            "service_preferences": [
                service_type.value for service_type in ServiceType
            ],
            "language_mixing": [
                "code_switching_preferred", "single_language_preferred",
                "english_technical_terms", "local_language_preferred"
            ],
            "time_preferences": [
                "morning_user", "afternoon_user", "evening_user", "night_user",
                "weekend_user", "weekday_user"
            ],
            "content_preferences": [
                "news_interested", "weather_frequent", "travel_queries",
                "food_recommendations", "entertainment_queries", "work_related"
            ]
        }
        
        logger.info("Preference Learner initialized")
    
    async def learn_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        user_feedback: Optional[Dict[str, Any]] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Learn preferences from user interaction.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            user_feedback: Explicit user feedback if available
            response_time: Time taken to respond
            
        Returns:
            Learning results with preference updates
        """
        learning_result = {
            "preferences_updated": [],
            "new_patterns_detected": [],
            "confidence_changes": {}
        }
        
        # Learn response style preferences
        response_style_updates = await self._learn_response_style(
            user_id, interaction, user_feedback
        )
        learning_result["preferences_updated"].extend(response_style_updates)
        
        # Learn interaction style preferences
        interaction_style_updates = await self._learn_interaction_style(
            user_id, interaction, response_time
        )
        learning_result["preferences_updated"].extend(interaction_style_updates)
        
        # Learn service preferences
        service_updates = await self._learn_service_preferences(user_id, interaction)
        learning_result["preferences_updated"].extend(service_updates)
        
        # Learn language mixing preferences
        language_updates = await self._learn_language_preferences(user_id, interaction)
        learning_result["preferences_updated"].extend(language_updates)
        
        # Learn time-based preferences
        time_updates = await self._learn_time_preferences(user_id, interaction)
        learning_result["preferences_updated"].extend(time_updates)
        
        # Learn content preferences
        content_updates = await self._learn_content_preferences(user_id, interaction)
        learning_result["preferences_updated"].extend(content_updates)
        
        # Update confidence scores
        for category_name, category in self._user_preferences[user_id].items():
            learning_result["confidence_changes"][category_name] = category.confidence
        
        logger.info(f"Learned preferences for user {user_id}: {len(learning_result['preferences_updated'])} updates")
        return learning_result
    
    async def get_user_preferences(
        self,
        user_id: UUID,
        category: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Get user preferences for a category or all categories.
        
        Args:
            user_id: User identifier
            category: Specific category to retrieve (optional)
            min_confidence: Minimum confidence threshold
            
        Returns:
            User preferences data
        """
        user_prefs = self._user_preferences.get(user_id, {})
        
        if category:
            if category in user_prefs and user_prefs[category].confidence >= min_confidence:
                pref_category = user_prefs[category]
                return {
                    "category": category,
                    "preferences": dict(pref_category.preferences),
                    "confidence": pref_category.confidence,
                    "sample_count": pref_category.sample_count,
                    "last_updated": pref_category.last_updated.isoformat(),
                    "top_preferences": pref_category.get_top_preferences()
                }
            else:
                return {"category": category, "preferences": {}, "confidence": 0.0}
        
        # Return all categories
        all_preferences = {}
        for cat_name, pref_category in user_prefs.items():
            if pref_category.confidence >= min_confidence:
                all_preferences[cat_name] = {
                    "preferences": dict(pref_category.preferences),
                    "confidence": pref_category.confidence,
                    "top_preferences": pref_category.get_top_preferences(3)
                }
        
        return all_preferences
    
    async def suggest_personalization(
        self,
        user_id: UUID,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest personalization options based on learned preferences.
        
        Args:
            user_id: User identifier
            context: Current context for suggestions
            
        Returns:
            List of personalization suggestions
        """
        suggestions = []
        user_prefs = self._user_preferences.get(user_id, {})
        
        # Response style suggestions
        if "response_style" in user_prefs:
            style_prefs = user_prefs["response_style"]
            if style_prefs.confidence > 0.7:
                top_style = style_prefs.get_top_preferences(1)
                if top_style:
                    suggestions.append({
                        "type": "response_style",
                        "suggestion": f"Use {top_style[0][0]} response style",
                        "confidence": style_prefs.confidence,
                        "reason": "Based on your interaction patterns"
                    })
        
        # Language mixing suggestions
        if "language_mixing" in user_prefs:
            lang_prefs = user_prefs["language_mixing"]
            if lang_prefs.confidence > 0.6:
                if lang_prefs.preferences.get("code_switching_preferred", 0) > 0.7:
                    suggestions.append({
                        "type": "language_mixing",
                        "suggestion": "Enable natural code-switching in responses",
                        "confidence": lang_prefs.confidence,
                        "reason": "You seem to prefer mixed language responses"
                    })
        
        # Service preferences suggestions
        if "service_preferences" in user_prefs:
            service_prefs = user_prefs["service_preferences"]
            top_services = service_prefs.get_top_preferences(3)
            if top_services:
                suggestions.append({
                    "type": "service_shortcuts",
                    "suggestion": f"Create shortcuts for {', '.join([s[0] for s in top_services])}",
                    "confidence": service_prefs.confidence,
                    "reason": "These are your most used services"
                })
        
        # Time-based suggestions
        if "time_preferences" in user_prefs:
            time_prefs = user_prefs["time_preferences"]
            current_hour = datetime.now().hour
            
            if current_hour < 12 and time_prefs.preferences.get("morning_user", 0) > 0.8:
                suggestions.append({
                    "type": "time_optimization",
                    "suggestion": "Enable morning briefing mode",
                    "confidence": time_prefs.confidence,
                    "reason": "You're most active in the morning"
                })
        
        return suggestions
    
    async def adapt_response_for_user(
        self,
        user_id: UUID,
        base_response: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Adapt response based on user preferences.
        
        Args:
            user_id: User identifier
            base_response: Base response text
            context: Response context
            
        Returns:
            Adapted response
        """
        user_prefs = self._user_preferences.get(user_id, {})
        adapted_response = base_response
        
        # Apply response style adaptations
        if "response_style" in user_prefs:
            style_prefs = user_prefs["response_style"]
            
            if style_prefs.confidence > 0.6:
                # Make response more formal if preferred
                if style_prefs.preferences.get("formal", 0) > 0.7:
                    adapted_response = await self._make_response_formal(adapted_response)
                
                # Make response more concise if preferred
                elif style_prefs.preferences.get("concise", 0) > 0.7:
                    adapted_response = await self._make_response_concise(adapted_response)
                
                # Add friendliness if preferred
                elif style_prefs.preferences.get("friendly", 0) > 0.7:
                    adapted_response = await self._make_response_friendly(adapted_response)
        
        # Apply language mixing preferences
        if "language_mixing" in user_prefs:
            lang_prefs = user_prefs["language_mixing"]
            
            if (lang_prefs.confidence > 0.6 and 
                lang_prefs.preferences.get("code_switching_preferred", 0) > 0.6):
                adapted_response = await self._add_code_switching(adapted_response, context)
        
        return adapted_response
    
    async def _learn_response_style(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        feedback: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Learn response style preferences from interaction."""
        updates = []
        style_category = self._user_preferences[user_id]["response_style"]
        style_category.name = "response_style"
        
        # Analyze response length preference
        response_length = len(interaction.response_text.split())
        if response_length < 10:
            style_category.update_preference("concise", 1.0, 0.5)
            updates.append("concise_preference")
        elif response_length > 30:
            style_category.update_preference("detailed", 1.0, 0.5)
            updates.append("detailed_preference")
        
        # Learn from explicit feedback
        if feedback:
            if feedback.get("response_too_long"):
                style_category.update_preference("concise", 1.0, 1.0)
                updates.append("concise_from_feedback")
            elif feedback.get("response_too_short"):
                style_category.update_preference("detailed", 1.0, 1.0)
                updates.append("detailed_from_feedback")
            
            if feedback.get("tone_rating", 0) > 4:  # 5-point scale
                # Determine which style was used and reinforce it
                if "please" in interaction.response_text.lower():
                    style_category.update_preference("formal", 1.0, 1.0)
                    updates.append("formal_from_positive_feedback")
        
        return updates
    
    async def _learn_interaction_style(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        response_time: Optional[float]
    ) -> List[str]:
        """Learn interaction style preferences."""
        updates = []
        interaction_category = self._user_preferences[user_id]["interaction_style"]
        interaction_category.name = "interaction_style"
        
        # Learn from response time preferences
        if response_time:
            if response_time < 2.0:  # Quick response
                interaction_category.update_preference("quick_responses", 1.0, 0.7)
                updates.append("quick_response_preference")
        
        # Analyze query complexity preference
        query_words = len(interaction.input_text.split())
        if query_words > 15:  # Complex query
            interaction_category.update_preference("detailed_explanations", 1.0, 0.6)
            updates.append("detailed_explanation_preference")
        
        # Check for step-by-step request patterns
        if any(phrase in interaction.input_text.lower() for phrase in 
               ["step by step", "how to", "explain", "guide me"]):
            interaction_category.update_preference("step_by_step", 1.0, 0.8)
            updates.append("step_by_step_preference")
        
        return updates
    
    async def _learn_service_preferences(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> List[str]:
        """Learn service usage preferences."""
        updates = []
        service_category = self._user_preferences[user_id]["service_preferences"]
        service_category.name = "service_preferences"
        
        # Detect service usage from intent or entities
        if interaction.intent:
            # Map intents to services
            intent_service_mapping = {
                "weather": ServiceType.WEATHER.value,
                "train": ServiceType.INDIAN_RAILWAYS.value,
                "food": ServiceType.FOOD_DELIVERY.value,
                "ride": ServiceType.RIDE_SHARING.value,
                "payment": ServiceType.UPI_PAYMENT.value,
                "government": ServiceType.GOVERNMENT_SERVICE.value,
                "cricket": ServiceType.CRICKET_SCORES.value,
                "bollywood": ServiceType.BOLLYWOOD_NEWS.value
            }
            
            for intent_key, service_type in intent_service_mapping.items():
                if intent_key in interaction.intent.lower():
                    service_category.update_preference(service_type, 1.0, 1.0)
                    updates.append(f"service_preference_{service_type}")
                    break
        
        return updates
    
    async def _learn_language_preferences(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> List[str]:
        """Learn language mixing preferences."""
        updates = []
        lang_category = self._user_preferences[user_id]["language_mixing"]
        lang_category.name = "language_mixing"
        
        # Detect code-switching in input
        input_words = interaction.input_text.split()
        english_words = sum(1 for word in input_words if word.isascii())
        total_words = len(input_words)
        
        if total_words > 0:
            english_ratio = english_words / total_words
            
            if 0.2 < english_ratio < 0.8:  # Mixed language usage
                lang_category.update_preference("code_switching_preferred", 1.0, 0.8)
                updates.append("code_switching_detected")
            elif english_ratio < 0.2:  # Mostly local language
                lang_category.update_preference("local_language_preferred", 1.0, 0.6)
                updates.append("local_language_preference")
        
        return updates
    
    async def _learn_time_preferences(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> List[str]:
        """Learn time-based usage preferences."""
        updates = []
        time_category = self._user_preferences[user_id]["time_preferences"]
        time_category.name = "time_preferences"
        
        interaction_hour = interaction.timestamp.hour
        interaction_day = interaction.timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Learn time of day preferences
        if 6 <= interaction_hour < 12:
            time_category.update_preference("morning_user", 1.0, 0.5)
            updates.append("morning_usage")
        elif 12 <= interaction_hour < 17:
            time_category.update_preference("afternoon_user", 1.0, 0.5)
            updates.append("afternoon_usage")
        elif 17 <= interaction_hour < 21:
            time_category.update_preference("evening_user", 1.0, 0.5)
            updates.append("evening_usage")
        else:
            time_category.update_preference("night_user", 1.0, 0.5)
            updates.append("night_usage")
        
        # Learn day of week preferences
        if interaction_day < 5:  # Weekday
            time_category.update_preference("weekday_user", 1.0, 0.3)
            updates.append("weekday_usage")
        else:  # Weekend
            time_category.update_preference("weekend_user", 1.0, 0.3)
            updates.append("weekend_usage")
        
        return updates
    
    async def _learn_content_preferences(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> List[str]:
        """Learn content type preferences."""
        updates = []
        content_category = self._user_preferences[user_id]["content_preferences"]
        content_category.name = "content_preferences"
        
        # Analyze query content for preferences
        query_lower = interaction.input_text.lower()
        
        content_keywords = {
            "news_interested": ["news", "latest", "current", "today"],
            "weather_frequent": ["weather", "temperature", "rain", "climate"],
            "travel_queries": ["travel", "trip", "journey", "train", "flight"],
            "food_recommendations": ["food", "restaurant", "eat", "hungry", "order"],
            "entertainment_queries": ["movie", "music", "song", "entertainment", "fun"],
            "work_related": ["work", "office", "meeting", "schedule", "business"]
        }
        
        for preference, keywords in content_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                content_category.update_preference(preference, 1.0, 0.7)
                updates.append(f"content_preference_{preference}")
        
        return updates
    
    async def _make_response_formal(self, response: str) -> str:
        """Make response more formal."""
        # Simple formalization rules
        response = response.replace("can't", "cannot")
        response = response.replace("won't", "will not")
        response = response.replace("don't", "do not")
        
        if not response.startswith(("Please", "I would", "May I")):
            response = "I would be happy to help. " + response
        
        return response
    
    async def _make_response_concise(self, response: str) -> str:
        """Make response more concise."""
        # Remove filler words and phrases
        filler_phrases = [
            "I would be happy to help. ",
            "Let me help you with that. ",
            "Here's what I found: ",
            "I hope this helps. "
        ]
        
        for phrase in filler_phrases:
            response = response.replace(phrase, "")
        
        # Limit to first two sentences
        sentences = response.split('. ')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'
        
        return response.strip()
    
    async def _make_response_friendly(self, response: str) -> str:
        """Make response more friendly."""
        friendly_starters = [
            "Great question! ",
            "I'd be happy to help! ",
            "Sure thing! ",
            "Absolutely! "
        ]
        
        if not any(response.startswith(starter) for starter in friendly_starters):
            import random
            starter = random.choice(friendly_starters)
            response = starter + response
        
        return response
    
    async def _add_code_switching(self, response: str, context: Dict[str, Any]) -> str:
        """Add natural code-switching to response."""
        # Simple code-switching additions
        # In production, this would be more sophisticated
        
        if "weather" in response.lower():
            response = response.replace("weather", "weather/मौसम")
        
        if "food" in response.lower():
            response = response.replace("food", "food/खाना")
        
        return response
    
    async def decay_old_preferences(
        self,
        user_id: UUID,
        decay_factor: float = 0.95
    ) -> int:
        """
        Apply decay to old preferences to adapt to changing user behavior.
        
        Args:
            user_id: User identifier
            decay_factor: Factor by which to decay old preferences
            
        Returns:
            Number of preferences decayed
        """
        user_prefs = self._user_preferences.get(user_id, {})
        decayed_count = 0
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.preference_decay_days)
        
        for category in user_prefs.values():
            if category.last_updated < cutoff_date:
                # Apply decay to all preferences in this category
                for pref_key in category.preferences:
                    category.preferences[pref_key] *= decay_factor
                    decayed_count += 1
                
                # Reduce confidence
                category.confidence *= decay_factor
        
        logger.info(f"Applied decay to {decayed_count} preferences for user {user_id}")
        return decayed_count