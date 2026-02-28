"""
Response Style Adaptation Module for BharatVoice Assistant.

This module adapts response styles based on user preferences, cultural context,
and interaction patterns to provide personalized communication.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import re

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    Response
)

logger = logging.getLogger(__name__)


class ResponseStyle:
    """Represents a response style configuration."""
    
    def __init__(self, name: str):
        self.name = name
        self.formality_level = 0.5  # 0.0 = very casual, 1.0 = very formal
        self.friendliness_level = 0.5  # 0.0 = neutral, 1.0 = very friendly
        self.verbosity_level = 0.5  # 0.0 = very concise, 1.0 = very detailed
        self.cultural_adaptation = 0.5  # 0.0 = generic, 1.0 = highly localized
        self.code_switching_preference = 0.0  # 0.0 = single language, 1.0 = mixed
        self.humor_level = 0.0  # 0.0 = serious, 1.0 = humorous
        self.empathy_level = 0.5  # 0.0 = factual, 1.0 = empathetic
        
        # Style-specific patterns
        self.greeting_patterns: List[str] = []
        self.transition_phrases: List[str] = []
        self.closing_patterns: List[str] = []
        self.emphasis_markers: List[str] = []
        
        # Cultural elements
        self.cultural_references: List[str] = []
        self.local_expressions: List[str] = []
        
        self.confidence = 0.5
        self.usage_count = 0
        self.last_updated = datetime.utcnow()
    
    def update_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update style based on user feedback."""
        self.usage_count += 1
        self.last_updated = datetime.utcnow()
        
        # Adjust parameters based on feedback
        if feedback.get("too_formal"):
            self.formality_level = max(0.0, self.formality_level - 0.1)
        elif feedback.get("too_casual"):
            self.formality_level = min(1.0, self.formality_level + 0.1)
        
        if feedback.get("too_long"):
            self.verbosity_level = max(0.0, self.verbosity_level - 0.1)
        elif feedback.get("too_short"):
            self.verbosity_level = min(1.0, self.verbosity_level + 0.1)
        
        if feedback.get("not_friendly_enough"):
            self.friendliness_level = min(1.0, self.friendliness_level + 0.1)
        elif feedback.get("too_friendly"):
            self.friendliness_level = max(0.0, self.friendliness_level - 0.1)
        
        # Update confidence based on positive feedback
        if feedback.get("rating", 3) >= 4:
            self.confidence = min(0.95, self.confidence + 0.05)
        elif feedback.get("rating", 3) <= 2:
            self.confidence = max(0.1, self.confidence - 0.05)


class ResponseStyleAdapter:
    """
    Adapts response styles based on user preferences and cultural context.
    """
    
    def __init__(
        self,
        adaptation_threshold: int = 5,
        style_confidence_threshold: float = 0.7
    ):
        """
        Initialize response style adapter.
        
        Args:
            adaptation_threshold: Minimum interactions before style adaptation
            style_confidence_threshold: Minimum confidence for style application
        """
        self.adaptation_threshold = adaptation_threshold
        self.style_confidence_threshold = style_confidence_threshold
        
        # User style profiles
        self._user_styles: Dict[UUID, Dict[str, ResponseStyle]] = defaultdict(dict)
        
        # Predefined style templates
        self._style_templates = {
            "formal_professional": ResponseStyle("formal_professional"),
            "casual_friendly": ResponseStyle("casual_friendly"),
            "helpful_detailed": ResponseStyle("helpful_detailed"),
            "concise_direct": ResponseStyle("concise_direct"),
            "cultural_local": ResponseStyle("cultural_local"),
            "mixed_language": ResponseStyle("mixed_language")
        }
        
        # Initialize style templates
        self._initialize_style_templates()
        
        # Cultural adaptation patterns
        self._cultural_patterns = {
            LanguageCode.HINDI: {
                "respectful_address": ["आप", "जी", "साहब", "मैडम"],
                "polite_expressions": ["कृपया", "धन्यवाद", "माफ करें"],
                "cultural_greetings": ["नमस्ते", "नमस्कार", "आदाब"],
                "local_expressions": ["अच्छा", "बहुत बढ़िया", "शाबाश"]
            },
            LanguageCode.ENGLISH_IN: {
                "respectful_address": ["Sir", "Madam", "ji"],
                "polite_expressions": ["please", "thank you", "sorry"],
                "cultural_greetings": ["Namaste", "Good morning", "How are you"],
                "local_expressions": ["very good", "excellent", "perfect"]
            }
        }
        
        logger.info("Response Style Adapter initialized")
    
    async def learn_style_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        user_feedback: Optional[Dict[str, Any]] = None,
        response_effectiveness: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Learn user's preferred response style from interaction.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            user_feedback: Explicit user feedback
            response_effectiveness: Measured response effectiveness (0.0-1.0)
            
        Returns:
            Style learning results
        """
        learning_result = {
            "style_updates": [],
            "new_patterns_detected": [],
            "confidence_changes": {},
            "style_recommendations": []
        }
        
        # Determine current interaction style characteristics
        interaction_style = await self._analyze_interaction_style(interaction)
        
        # Get or create user style profile
        style_key = f"{interaction.input_language.value}_general"
        if style_key not in self._user_styles[user_id]:
            self._user_styles[user_id][style_key] = ResponseStyle(style_key)
        
        user_style = self._user_styles[user_id][style_key]
        
        # Update style based on interaction patterns
        await self._update_style_from_patterns(user_style, interaction_style)
        learning_result["style_updates"].append(f"updated_{style_key}")
        
        # Apply feedback if available
        if user_feedback:
            user_style.update_from_feedback(user_feedback)
            learning_result["style_updates"].append("feedback_applied")
        
        # Update based on response effectiveness
        if response_effectiveness is not None:
            await self._update_style_from_effectiveness(user_style, response_effectiveness)
            learning_result["style_updates"].append("effectiveness_applied")
        
        # Detect new patterns
        new_patterns = await self._detect_new_patterns(user_id, interaction)
        learning_result["new_patterns_detected"] = new_patterns
        
        # Update confidence
        learning_result["confidence_changes"][style_key] = user_style.confidence
        
        # Generate style recommendations
        if user_style.usage_count >= self.adaptation_threshold:
            recommendations = await self._generate_style_recommendations(user_style)
            learning_result["style_recommendations"] = recommendations
        
        logger.info(f"Learned response style for user {user_id}: {len(learning_result['style_updates'])} updates")
        return learning_result
    
    async def adapt_response_style(
        self,
        user_id: UUID,
        base_response: str,
        language: LanguageCode,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Adapt response style for user preferences.
        
        Args:
            user_id: User identifier
            base_response: Base response text
            language: Response language
            context: Additional context
            
        Returns:
            Style-adapted response
        """
        style_key = f"{language.value}_general"
        
        # Get user style or use default
        if (user_id in self._user_styles and 
            style_key in self._user_styles[user_id] and
            self._user_styles[user_id][style_key].confidence >= self.style_confidence_threshold):
            
            user_style = self._user_styles[user_id][style_key]
        else:
            # Use default style template
            user_style = self._style_templates["helpful_detailed"]
        
        adapted_response = base_response
        
        # Apply formality adjustments
        adapted_response = await self._adjust_formality(
            adapted_response, user_style.formality_level, language
        )
        
        # Apply friendliness adjustments
        adapted_response = await self._adjust_friendliness(
            adapted_response, user_style.friendliness_level, language
        )
        
        # Apply verbosity adjustments
        adapted_response = await self._adjust_verbosity(
            adapted_response, user_style.verbosity_level
        )
        
        # Apply cultural adaptations
        adapted_response = await self._apply_cultural_adaptations(
            adapted_response, user_style.cultural_adaptation, language
        )
        
        # Apply code-switching if preferred
        if user_style.code_switching_preference > 0.5:
            adapted_response = await self._apply_code_switching(
                adapted_response, language, user_style.code_switching_preference
            )
        
        # Apply empathy adjustments
        adapted_response = await self._adjust_empathy(
            adapted_response, user_style.empathy_level, context
        )
        
        return adapted_response
    
    async def get_user_style_profile(
        self,
        user_id: UUID,
        language: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """
        Get user's response style profile.
        
        Args:
            user_id: User identifier
            language: Specific language (optional)
            
        Returns:
            User style profile data
        """
        user_styles = self._user_styles.get(user_id, {})
        
        if language:
            style_key = f"{language.value}_general"
            if style_key in user_styles:
                style = user_styles[style_key]
                return {
                    "language": language.value,
                    "formality_level": style.formality_level,
                    "friendliness_level": style.friendliness_level,
                    "verbosity_level": style.verbosity_level,
                    "cultural_adaptation": style.cultural_adaptation,
                    "code_switching_preference": style.code_switching_preference,
                    "confidence": style.confidence,
                    "usage_count": style.usage_count,
                    "last_updated": style.last_updated.isoformat()
                }
            else:
                return {"language": language.value, "style": "default"}
        
        # Return all language styles
        all_styles = {}
        for style_key, style in user_styles.items():
            language_code = style_key.split('_')[0]
            all_styles[language_code] = {
                "formality_level": style.formality_level,
                "friendliness_level": style.friendliness_level,
                "verbosity_level": style.verbosity_level,
                "confidence": style.confidence,
                "usage_count": style.usage_count
            }
        
        return all_styles
    
    async def suggest_style_improvements(
        self,
        user_id: UUID,
        language: LanguageCode
    ) -> List[Dict[str, Any]]:
        """
        Suggest style improvements based on usage patterns.
        
        Args:
            user_id: User identifier
            language: Target language
            
        Returns:
            List of style improvement suggestions
        """
        suggestions = []
        style_key = f"{language.value}_general"
        
        if (user_id not in self._user_styles or 
            style_key not in self._user_styles[user_id]):
            suggestions.append({
                "type": "data_collection",
                "message": "Interact more to learn your preferred response style",
                "priority": "medium"
            })
            return suggestions
        
        user_style = self._user_styles[user_id][style_key]
        
        # Check confidence level
        if user_style.confidence < self.style_confidence_threshold:
            suggestions.append({
                "type": "confidence_improvement",
                "message": f"Need {self.adaptation_threshold - user_style.usage_count} more interactions to improve style adaptation",
                "priority": "medium"
            })
        
        # Suggest style optimizations
        if user_style.formality_level > 0.8:
            suggestions.append({
                "type": "style_balance",
                "message": "Consider using more casual language for better engagement",
                "priority": "low"
            })
        elif user_style.formality_level < 0.2:
            suggestions.append({
                "type": "style_balance",
                "message": "Consider more formal responses for professional contexts",
                "priority": "low"
            })
        
        if user_style.verbosity_level > 0.8:
            suggestions.append({
                "type": "conciseness",
                "message": "Responses might be too detailed - consider more concise communication",
                "priority": "low"
            })
        
        return suggestions
    
    async def _initialize_style_templates(self) -> None:
        """Initialize predefined style templates."""
        # Formal Professional
        formal = self._style_templates["formal_professional"]
        formal.formality_level = 0.9
        formal.friendliness_level = 0.3
        formal.verbosity_level = 0.7
        formal.greeting_patterns = ["Good morning", "Good afternoon", "I hope this message finds you well"]
        formal.transition_phrases = ["Furthermore", "Additionally", "Please note that"]
        formal.closing_patterns = ["Thank you for your attention", "Best regards", "Sincerely"]
        
        # Casual Friendly
        casual = self._style_templates["casual_friendly"]
        casual.formality_level = 0.2
        casual.friendliness_level = 0.9
        casual.verbosity_level = 0.4
        casual.greeting_patterns = ["Hey!", "Hi there!", "What's up?"]
        casual.transition_phrases = ["Also", "By the way", "Oh, and"]
        casual.closing_patterns = ["Take care!", "Catch you later!", "Have a great day!"]
        
        # Helpful Detailed
        helpful = self._style_templates["helpful_detailed"]
        helpful.formality_level = 0.5
        helpful.friendliness_level = 0.8
        helpful.verbosity_level = 0.9
        helpful.empathy_level = 0.7
        helpful.greeting_patterns = ["I'd be happy to help!", "Let me assist you with that"]
        helpful.transition_phrases = ["Here's what you need to know", "Let me explain", "Step by step"]
        
        # Concise Direct
        concise = self._style_templates["concise_direct"]
        concise.formality_level = 0.6
        concise.friendliness_level = 0.4
        concise.verbosity_level = 0.2
        concise.greeting_patterns = ["Sure", "Okay", "Right"]
        concise.transition_phrases = ["Next", "Then", "Finally"]
        
        # Cultural Local
        cultural = self._style_templates["cultural_local"]
        cultural.formality_level = 0.6
        cultural.friendliness_level = 0.7
        cultural.cultural_adaptation = 0.9
        cultural.greeting_patterns = ["Namaste", "Sat Sri Akal", "Vanakkam"]
        cultural.local_expressions = ["Bahut accha", "Shabash", "Bilkul sahi"]
        
        # Mixed Language
        mixed = self._style_templates["mixed_language"]
        mixed.code_switching_preference = 0.8
        mixed.cultural_adaptation = 0.7
        mixed.friendliness_level = 0.7
    
    async def _analyze_interaction_style(
        self,
        interaction: UserInteraction
    ) -> Dict[str, Any]:
        """Analyze style characteristics from interaction."""
        style_analysis = {
            "formality_indicators": 0.0,
            "friendliness_indicators": 0.0,
            "verbosity_preference": 0.0,
            "cultural_elements": 0.0,
            "code_switching_detected": 0.0
        }
        
        input_text = interaction.input_text.lower()
        response_text = interaction.response_text.lower()
        
        # Analyze formality
        formal_words = ["please", "thank you", "kindly", "would you", "could you"]
        casual_words = ["hey", "hi", "yeah", "ok", "cool", "awesome"]
        
        formal_count = sum(1 for word in formal_words if word in input_text)
        casual_count = sum(1 for word in casual_words if word in input_text)
        
        if formal_count + casual_count > 0:
            style_analysis["formality_indicators"] = formal_count / (formal_count + casual_count)
        
        # Analyze verbosity preference from response length satisfaction
        response_length = len(interaction.response_text.split())
        if response_length > 50:
            style_analysis["verbosity_preference"] = 0.8
        elif response_length < 15:
            style_analysis["verbosity_preference"] = 0.2
        else:
            style_analysis["verbosity_preference"] = 0.5
        
        # Detect code-switching
        if interaction.input_language == LanguageCode.ENGLISH_IN:
            # Check for Hindi words in English text
            hindi_pattern = re.compile(r'[\u0900-\u097F]+')
            if hindi_pattern.search(input_text):
                style_analysis["code_switching_detected"] = 1.0
        
        return style_analysis
    
    async def _update_style_from_patterns(
        self,
        user_style: ResponseStyle,
        interaction_style: Dict[str, Any]
    ) -> None:
        """Update user style based on interaction patterns."""
        alpha = 0.1  # Learning rate
        
        # Update formality
        if "formality_indicators" in interaction_style:
            user_style.formality_level = (
                (1 - alpha) * user_style.formality_level + 
                alpha * interaction_style["formality_indicators"]
            )
        
        # Update verbosity preference
        if "verbosity_preference" in interaction_style:
            user_style.verbosity_level = (
                (1 - alpha) * user_style.verbosity_level + 
                alpha * interaction_style["verbosity_preference"]
            )
        
        # Update code-switching preference
        if "code_switching_detected" in interaction_style:
            user_style.code_switching_preference = (
                (1 - alpha) * user_style.code_switching_preference + 
                alpha * interaction_style["code_switching_detected"]
            )
        
        user_style.usage_count += 1
        user_style.last_updated = datetime.utcnow()
        user_style.confidence = min(0.95, 0.3 + (user_style.usage_count * 0.05))
    
    async def _update_style_from_effectiveness(
        self,
        user_style: ResponseStyle,
        effectiveness: float
    ) -> None:
        """Update style based on response effectiveness."""
        if effectiveness > 0.8:
            # Reinforce current style settings
            user_style.confidence = min(0.95, user_style.confidence + 0.02)
        elif effectiveness < 0.4:
            # Slightly adjust style towards more helpful/detailed
            user_style.verbosity_level = min(1.0, user_style.verbosity_level + 0.05)
            user_style.empathy_level = min(1.0, user_style.empathy_level + 0.05)
    
    async def _detect_new_patterns(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> List[str]:
        """Detect new style patterns from interaction."""
        patterns = []
        
        # Check for time-based patterns
        hour = interaction.timestamp.hour
        if 6 <= hour < 12:
            patterns.append("morning_interaction")
        elif 12 <= hour < 17:
            patterns.append("afternoon_interaction")
        elif 17 <= hour < 21:
            patterns.append("evening_interaction")
        else:
            patterns.append("night_interaction")
        
        # Check for context-based patterns
        if any(word in interaction.input_text.lower() for word in ["work", "office", "meeting"]):
            patterns.append("professional_context")
        elif any(word in interaction.input_text.lower() for word in ["family", "home", "personal"]):
            patterns.append("personal_context")
        
        return patterns
    
    async def _generate_style_recommendations(
        self,
        user_style: ResponseStyle
    ) -> List[Dict[str, Any]]:
        """Generate style recommendations based on learned preferences."""
        recommendations = []
        
        if user_style.formality_level > 0.7:
            recommendations.append({
                "type": "formality",
                "suggestion": "Use formal, professional language",
                "confidence": user_style.confidence
            })
        elif user_style.formality_level < 0.3:
            recommendations.append({
                "type": "formality",
                "suggestion": "Use casual, friendly language",
                "confidence": user_style.confidence
            })
        
        if user_style.verbosity_level > 0.7:
            recommendations.append({
                "type": "verbosity",
                "suggestion": "Provide detailed, comprehensive responses",
                "confidence": user_style.confidence
            })
        elif user_style.verbosity_level < 0.3:
            recommendations.append({
                "type": "verbosity",
                "suggestion": "Keep responses concise and direct",
                "confidence": user_style.confidence
            })
        
        if user_style.code_switching_preference > 0.6:
            recommendations.append({
                "type": "language_mixing",
                "suggestion": "Use natural code-switching in responses",
                "confidence": user_style.confidence
            })
        
        return recommendations
    
    async def _adjust_formality(
        self,
        response: str,
        formality_level: float,
        language: LanguageCode
    ) -> str:
        """Adjust response formality level."""
        if formality_level > 0.7:
            # Make more formal
            response = response.replace("can't", "cannot")
            response = response.replace("won't", "will not")
            response = response.replace("don't", "do not")
            
            # Add formal address if not present
            if language == LanguageCode.HINDI:
                if not any(addr in response for addr in ["आप", "जी"]):
                    response = "जी, " + response
            elif language == LanguageCode.ENGLISH_IN:
                if not response.startswith(("Please", "I would", "May I")):
                    response = "I would be pleased to help. " + response
        
        elif formality_level < 0.3:
            # Make more casual
            response = response.replace("I would be pleased", "I'd be happy")
            response = response.replace("I shall", "I'll")
            response = response.replace("cannot", "can't")
        
        return response
    
    async def _adjust_friendliness(
        self,
        response: str,
        friendliness_level: float,
        language: LanguageCode
    ) -> str:
        """Adjust response friendliness level."""
        if friendliness_level > 0.7:
            # Add friendly elements
            friendly_starters = {
                LanguageCode.ENGLISH_IN: ["Great question! ", "I'd love to help! ", "Absolutely! "],
                LanguageCode.HINDI: ["बहुत अच्छा सवाल! ", "मैं खुशी से मदद करूंगा! ", "बिल्कुल! "]
            }
            
            starters = friendly_starters.get(language, friendly_starters[LanguageCode.ENGLISH_IN])
            if not any(response.startswith(starter.strip()) for starter in starters):
                import random
                response = random.choice(starters) + response
        
        return response
    
    async def _adjust_verbosity(
        self,
        response: str,
        verbosity_level: float
    ) -> str:
        """Adjust response verbosity level."""
        if verbosity_level < 0.3:
            # Make more concise
            sentences = response.split('. ')
            if len(sentences) > 2:
                response = '. '.join(sentences[:2]) + '.'
            
            # Remove filler phrases
            filler_phrases = [
                "I hope this helps. ",
                "Let me know if you need more information. ",
                "Please feel free to ask if you have any questions. "
            ]
            
            for phrase in filler_phrases:
                response = response.replace(phrase, "")
        
        elif verbosity_level > 0.7:
            # Add more detail
            if not response.endswith(("?", ".")):
                response += "."
            
            response += " Would you like me to provide more details about this?"
        
        return response.strip()
    
    async def _apply_cultural_adaptations(
        self,
        response: str,
        cultural_level: float,
        language: LanguageCode
    ) -> str:
        """Apply cultural adaptations to response."""
        if cultural_level > 0.6:
            cultural_patterns = self._cultural_patterns.get(language, {})
            
            # Add respectful address
            respectful_addresses = cultural_patterns.get("respectful_address", [])
            if respectful_addresses and not any(addr in response for addr in respectful_addresses):
                if language == LanguageCode.HINDI:
                    response = "जी, " + response
                elif language == LanguageCode.ENGLISH_IN:
                    response = response  # Already handled in formality
            
            # Add cultural expressions
            local_expressions = cultural_patterns.get("local_expressions", [])
            if local_expressions and "good" in response.lower():
                import random
                expression = random.choice(local_expressions)
                response = response.replace("good", f"good ({expression})", 1)
        
        return response
    
    async def _apply_code_switching(
        self,
        response: str,
        language: LanguageCode,
        switching_preference: float
    ) -> str:
        """Apply code-switching to response."""
        if switching_preference > 0.5 and language == LanguageCode.ENGLISH_IN:
            # Add Hindi equivalents for common words
            code_switch_mappings = {
                "good": "good/अच्छा",
                "yes": "yes/हाँ",
                "no": "no/नहीं",
                "thank you": "thank you/धन्यवाद",
                "please": "please/कृपया",
                "help": "help/मदद",
                "food": "food/खाना",
                "water": "water/पानी"
            }
            
            for english, mixed in code_switch_mappings.items():
                if english in response.lower():
                    response = response.replace(english, mixed, 1)
                    break  # Only apply one code-switch per response
        
        return response
    
    async def _adjust_empathy(
        self,
        response: str,
        empathy_level: float,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Adjust empathy level in response."""
        if empathy_level > 0.6 and context:
            # Check for emotional context
            if context.get("user_emotion") in ["frustrated", "sad", "worried"]:
                empathetic_starters = [
                    "I understand this can be frustrating. ",
                    "I can see why you might be concerned. ",
                    "That sounds challenging. "
                ]
                
                if not any(response.startswith(starter.strip()) for starter in empathetic_starters):
                    import random
                    response = random.choice(empathetic_starters) + response
        
        return response
    
    async def cleanup_old_styles(
        self,
        user_id: UUID,
        days_threshold: int = 90
    ) -> int:
        """Clean up old, unused style profiles."""
        user_styles = self._user_styles.get(user_id, {})
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        styles_to_remove = []
        for style_key, style in user_styles.items():
            if style.last_updated < cutoff_date and style.usage_count < 5:
                styles_to_remove.append(style_key)
        
        for style_key in styles_to_remove:
            del user_styles[style_key]
        
        logger.info(f"Cleaned up {len(styles_to_remove)} old style profiles for user {user_id}")
        return len(styles_to_remove)