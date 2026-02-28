<<<<<<< HEAD
"""
Response Generator for BharatVoice Assistant.

This module implements comprehensive response generation with multilingual output,
Indian localization for currency/measurements/time formats, grammatically correct
responses in all supported languages, and natural code-switching capabilities.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass

from bharatvoice.core.models import (
    LanguageCode,
    Response,
    Intent,
    Entity,
    ConversationState,
    RegionalContextData,
    CulturalContext,
    UserProfile,
    ServiceParameters,
    ServiceResult,
    ServiceType
)
from bharatvoice.core.interfaces import ResponseGenerator as ResponseGeneratorInterface
from bharatvoice.services.response_generation.nlu_service import IntentCategory, EntityType
from bharatvoice.services.language_engine.translation_engine import TranslationEngine


class ResponseStyle(str, Enum):
    """Response style options."""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    RESPECTFUL = "respectful"
    ENTHUSIASTIC = "enthusiastic"
    HELPFUL = "helpful"
    PROFESSIONAL = "professional"


class LocalizationFormat(str, Enum):
    """Localization format types."""
    CURRENCY = "currency"
    MEASUREMENT = "measurement"
    TIME = "time"
    DATE = "date"
    NUMBER = "number"
    TEMPERATURE = "temperature"


@dataclass
class LocalizedValue:
    """Represents a localized value with formatting."""
    original_value: str
    localized_value: str
    format_type: LocalizationFormat
    locale: str
    explanation: Optional[str] = None


@dataclass
class CodeSwitchingPoint:
    """Represents a point where code-switching occurs in response."""
    position: int
    from_language: LanguageCode
    to_language: LanguageCode
    reason: str
    confidence: float


class IndianLocalizationEngine:
    """Handles Indian-specific localization for various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._currency_mappings = self._initialize_currency_mappings()
        self._measurement_mappings = self._initialize_measurement_mappings()
        self._time_formats = self._initialize_time_formats()
        self._number_formats = self._initialize_number_formats()
    
    def _initialize_currency_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize currency conversion mappings."""
        return {
            "USD": {
                "symbol": "$",
                "indian_equivalent": "₹",
                "conversion_rate": 83.0,  # Approximate rate
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "EUR": {
                "symbol": "€",
                "indian_equivalent": "₹",
                "conversion_rate": 90.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "GBP": {
                "symbol": "£",
                "indian_equivalent": "₹",
                "conversion_rate": 105.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "INR": {
                "symbol": "₹",
                "indian_equivalent": "₹",
                "conversion_rate": 1.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            }
        }
    
    def _initialize_measurement_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize measurement conversion mappings."""
        return {
            "fahrenheit": {
                "unit": "°F",
                "indian_equivalent": "°C",
                "conversion": lambda f: (f - 32) * 5/9,
                "format": "{value:.1f}°C",
                "spoken_format": "{value} degrees celsius"
            },
            "miles": {
                "unit": "mi",
                "indian_equivalent": "km",
                "conversion": lambda mi: mi * 1.60934,
                "format": "{value:.1f} km",
                "spoken_format": "{value} kilometers"
            },
            "feet": {
                "unit": "ft",
                "indian_equivalent": "m",
                "conversion": lambda ft: ft * 0.3048,
                "format": "{value:.1f} meters",
                "spoken_format": "{value} meters"
            },
            "pounds": {
                "unit": "lbs",
                "indian_equivalent": "kg",
                "conversion": lambda lbs: lbs * 0.453592,
                "format": "{value:.1f} kg",
                "spoken_format": "{value} kilograms"
            },
            "gallons": {
                "unit": "gal",
                "indian_equivalent": "L",
                "conversion": lambda gal: gal * 3.78541,
                "format": "{value:.1f} liters",
                "spoken_format": "{value} liters"
            }
        }
    
    def _initialize_time_formats(self) -> Dict[str, Dict[str, str]]:
        """Initialize time format mappings."""
        return {
            "12_hour": {
                "format": "%I:%M %p",
                "indian_format": "%H:%M",
                "spoken_format": "{hour} baje {minute} minute"
            },
            "24_hour": {
                "format": "%H:%M",
                "indian_format": "%H:%M",
                "spoken_format": "{hour} baje {minute} minute"
            },
            "date": {
                "format": "%m/%d/%Y",
                "indian_format": "%d/%m/%Y",
                "spoken_format": "{day} {month} {year}"
            }
        }
    
    def _initialize_number_formats(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Indian number formatting (lakhs, crores)."""
        return {
            "indian_system": {
                "lakh": 100000,
                "crore": 10000000,
                "format_rules": {
                    "use_lakhs_crores": True,
                    "decimal_separator": ".",
                    "thousands_separator": ","
                }
            }
        }
    
    async def localize_currency(self, amount: float, currency: str = "USD") -> LocalizedValue:
        """Localize currency to Indian format."""
        try:
            currency_info = self._currency_mappings.get(currency.upper(), self._currency_mappings["USD"])
            
            if currency.upper() != "INR":
                # Convert to INR
                inr_amount = amount * currency_info["conversion_rate"]
            else:
                inr_amount = amount
            
            # Format in Indian numbering system
            formatted_amount = self._format_indian_number(inr_amount)
            localized_text = f"₹{formatted_amount}"
            
            return LocalizedValue(
                original_value=f"{currency_info['symbol']}{amount:,.2f}",
                localized_value=localized_text,
                format_type=LocalizationFormat.CURRENCY,
                locale="en-IN",
                explanation=f"Converted from {currency} to INR" if currency.upper() != "INR" else None
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing currency: {e}")
            return LocalizedValue(
                original_value=f"{amount}",
                localized_value=f"₹{amount:,.2f}",
                format_type=LocalizationFormat.CURRENCY,
                locale="en-IN"
            )
    
    async def localize_measurement(self, value: float, unit: str) -> LocalizedValue:
        """Localize measurements to Indian standards."""
        try:
            unit_lower = unit.lower()
            measurement_info = self._measurement_mappings.get(unit_lower)
            
            if not measurement_info:
                # Return original if no conversion available
                return LocalizedValue(
                    original_value=f"{value} {unit}",
                    localized_value=f"{value} {unit}",
                    format_type=LocalizationFormat.MEASUREMENT,
                    locale="en-IN"
                )
            
            # Convert to Indian equivalent
            converted_value = measurement_info["conversion"](value)
            localized_text = measurement_info["format"].format(value=converted_value)
            
            return LocalizedValue(
                original_value=f"{value} {unit}",
                localized_value=localized_text,
                format_type=LocalizationFormat.MEASUREMENT,
                locale="en-IN",
                explanation=f"Converted from {unit} to {measurement_info['indian_equivalent']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing measurement: {e}")
            return LocalizedValue(
                original_value=f"{value} {unit}",
                localized_value=f"{value} {unit}",
                format_type=LocalizationFormat.MEASUREMENT,
                locale="en-IN"
            )
    
    async def localize_time(self, time_str: str, format_type: str = "12_hour") -> LocalizedValue:
        """Localize time format to Indian standards."""
        try:
            time_info = self._time_formats.get(format_type, self._time_formats["12_hour"])
            
            # Parse time string and convert to Indian format
            try:
                if ":" in time_str:
                    if "AM" in time_str.upper() or "PM" in time_str.upper():
                        # 12-hour format
                        time_obj = datetime.strptime(time_str.strip(), "%I:%M %p")
                    else:
                        # 24-hour format
                        time_obj = datetime.strptime(time_str.strip(), "%H:%M")
                    
                    # Format in Indian style (24-hour preferred)
                    indian_time = time_obj.strftime("%H:%M")
                    
                    # Create spoken format
                    hour = time_obj.hour
                    minute = time_obj.minute
                    
                    if minute == 0:
                        spoken = f"{hour} baje"
                    else:
                        spoken = f"{hour} baje {minute} minute"
                    
                    return LocalizedValue(
                        original_value=time_str,
                        localized_value=indian_time,
                        format_type=LocalizationFormat.TIME,
                        locale="en-IN",
                        explanation=f"Spoken as: {spoken}"
                    )
                    
            except ValueError:
                # If parsing fails, return original
                pass
            
            return LocalizedValue(
                original_value=time_str,
                localized_value=time_str,
                format_type=LocalizationFormat.TIME,
                locale="en-IN"
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing time: {e}")
            return LocalizedValue(
                original_value=time_str,
                localized_value=time_str,
                format_type=LocalizationFormat.TIME,
                locale="en-IN"
            )
    
    async def localize_temperature(self, temp: float, unit: str = "F") -> LocalizedValue:
        """Localize temperature to Celsius."""
        try:
            if unit.upper() == "F":
                celsius = (temp - 32) * 5/9
                localized_text = f"{celsius:.1f}°C"
                explanation = f"Converted from {temp}°F to Celsius"
            elif unit.upper() == "C":
                celsius = temp
                localized_text = f"{celsius:.1f}°C"
                explanation = None
            else:
                # Unknown unit, assume Celsius
                celsius = temp
                localized_text = f"{celsius:.1f}°C"
                explanation = None
            
            return LocalizedValue(
                original_value=f"{temp}°{unit}",
                localized_value=localized_text,
                format_type=LocalizationFormat.TEMPERATURE,
                locale="en-IN",
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing temperature: {e}")
            return LocalizedValue(
                original_value=f"{temp}°{unit}",
                localized_value=f"{temp}°{unit}",
                format_type=LocalizationFormat.TEMPERATURE,
                locale="en-IN"
            )
    
    def _format_indian_number(self, number: float) -> str:
        """Format number in Indian numbering system (lakhs, crores)."""
        try:
            if number >= 10000000:  # 1 crore
                crores = number / 10000000
                if crores >= 1:
                    return f"{crores:.2f} crore"
            elif number >= 100000:  # 1 lakh
                lakhs = number / 100000
                if lakhs >= 1:
                    return f"{lakhs:.2f} lakh"
            elif number >= 1000:  # 1 thousand
                thousands = number / 1000
                return f"{thousands:.2f} thousand"
            else:
                return f"{number:,.2f}"
                
        except Exception as e:
            self.logger.error(f"Error formatting Indian number: {e}")
            return f"{number:,.2f}"


class CodeSwitchingEngine:
    """Handles natural code-switching in multilingual responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._switching_patterns = self._initialize_switching_patterns()
        self._cultural_triggers = self._initialize_cultural_triggers()
    
    def _initialize_switching_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize code-switching patterns."""
        return {
            "greeting_patterns": [
                {
                    "trigger": "formal_greeting",
                    "pattern": "Namaste! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.9
                },
                {
                    "trigger": "casual_greeting",
                    "pattern": "Hey yaar, {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                }
            ],
            "emphasis_patterns": [
                {
                    "trigger": "strong_agreement",
                    "pattern": "Bilkul! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                },
                {
                    "trigger": "excitement",
                    "pattern": "Arre waah! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.7
                }
            ],
            "cultural_patterns": [
                {
                    "trigger": "festival_mention",
                    "pattern": "{hindi_festival_name} ke liye {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.9
                },
                {
                    "trigger": "family_context",
                    "pattern": "Ghar mein {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                }
            ],
            "explanation_patterns": [
                {
                    "trigger": "clarification",
                    "pattern": "{english_content}, matlab {hindi_explanation}",
                    "languages": [LanguageCode.ENGLISH_IN, LanguageCode.HINDI],
                    "confidence": 0.8
                }
            ]
        }
    
    def _initialize_cultural_triggers(self) -> Dict[str, List[str]]:
        """Initialize cultural triggers for code-switching."""
        return {
            "festivals": ["diwali", "holi", "eid", "dussehra", "ganpati", "navratri"],
            "family_terms": ["mummy", "papa", "bhai", "didi", "family", "ghar"],
            "food_terms": ["khana", "chai", "roti", "dal", "sabzi"],
            "emotions": ["khushi", "dukh", "gussa", "pyaar", "masti"],
            "respect_terms": ["ji", "sahab", "madam", "sir", "namaste"],
            "casual_terms": ["yaar", "boss", "dude", "bro", "arre"]
        }
    
    async def apply_code_switching(
        self,
        text: str,
        primary_language: LanguageCode,
        secondary_language: LanguageCode,
        cultural_context: Dict[str, Any],
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply natural code-switching to response text."""
        try:
            switching_points = []
            modified_text = text
            
            # Determine switching probability based on user profile
            switching_probability = self._calculate_switching_probability(
                cultural_context, user_profile
            )
            
            if switching_probability < 0.3:
                return text, []
            
            # Apply greeting patterns
            modified_text, greeting_points = await self._apply_greeting_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(greeting_points)
            
            # Apply emphasis patterns
            modified_text, emphasis_points = await self._apply_emphasis_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(emphasis_points)
            
            # Apply cultural patterns
            modified_text, cultural_points = await self._apply_cultural_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(cultural_points)
            
            # Apply explanation patterns
            modified_text, explanation_points = await self._apply_explanation_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(explanation_points)
            
            self.logger.info(f"Applied {len(switching_points)} code-switching points")
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying code-switching: {e}")
            return text, []
    
    def _calculate_switching_probability(
        self,
        cultural_context: Dict[str, Any],
        user_profile: Optional[UserProfile]
    ) -> float:
        """Calculate probability of code-switching based on context."""
        try:
            base_probability = 0.5
            
            # Increase probability for casual communication
            if cultural_context.get("formality_level") == "low":
                base_probability += 0.2
            elif cultural_context.get("formality_level") == "high":
                base_probability -= 0.2
            
            # Increase probability for family/friend context
            if cultural_context.get("communication_style") == "casual_friendly":
                base_probability += 0.3
            elif cultural_context.get("communication_style") == "formal_respectful":
                base_probability -= 0.1
            
            # Consider user profile language preferences
            if user_profile and len(user_profile.preferred_languages) > 1:
                base_probability += 0.2
            
            # Consider regional influence
            if cultural_context.get("regional_influence") in ["north_india", "west_india"]:
                base_probability += 0.1  # More common in these regions
            
            return min(max(base_probability, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating switching probability: {e}")
            return 0.5
    
    async def _apply_greeting_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply greeting-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for greeting context
            greeting_indicators = ["hello", "hi", "hey", "good morning", "good evening"]
            
            if any(indicator in text.lower() for indicator in greeting_indicators):
                formality = cultural_context.get("formality_level", "medium")
                
                if formality == "high":
                    # Use formal Hindi greeting
                    if text.lower().startswith(("hello", "hi")):
                        modified_text = re.sub(
                            r'^(hello|hi)\b',
                            'Namaste',
                            text,
                            flags=re.IGNORECASE
                        )
                        switching_points.append(CodeSwitchingPoint(
                            position=0,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="formal_greeting",
                            confidence=0.9
                        ))
                elif formality == "low":
                    # Use casual mixed greeting
                    if text.lower().startswith(("hello", "hi")):
                        modified_text = re.sub(
                            r'^(hello|hi)\b',
                            'Hey yaar',
                            text,
                            flags=re.IGNORECASE
                        )
                        switching_points.append(CodeSwitchingPoint(
                            position=0,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="casual_greeting",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying greeting patterns: {e}")
            return text, []
    
    async def _apply_emphasis_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply emphasis-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for emphasis opportunities
            emphasis_words = ["yes", "absolutely", "definitely", "exactly", "perfect"]
            
            for word in emphasis_words:
                if word in text.lower():
                    # Replace with Hindi emphasis
                    if word == "yes":
                        modified_text = re.sub(
                            r'\byes\b',
                            'Haan bilkul',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif word in ["absolutely", "definitely"]:
                        modified_text = re.sub(
                            rf'\b{word}\b',
                            'Bilkul',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif word == "perfect":
                        modified_text = re.sub(
                            r'\bperfect\b',
                            'Ekdum perfect',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    
                    # Find position and add switching point
                    position = modified_text.lower().find(word)
                    if position != -1:
                        switching_points.append(CodeSwitchingPoint(
                            position=position,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="emphasis",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying emphasis patterns: {e}")
            return text, []
    
    async def _apply_cultural_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply culture-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for cultural triggers
            for category, triggers in self._cultural_triggers.items():
                for trigger in triggers:
                    if trigger in text.lower():
                        if category == "festivals":
                            # Keep festival names in original language
                            continue
                        elif category == "family_terms":
                            # Use Hindi family terms
                            if "family" in text.lower():
                                modified_text = re.sub(
                                    r'\bfamily\b',
                                    'parivaar',
                                    modified_text,
                                    flags=re.IGNORECASE
                                )
                        elif category == "food_terms":
                            # Keep food terms in Hindi
                            continue
                        elif category == "respect_terms":
                            # Add respectful suffixes
                            if "sir" not in text.lower() and "madam" not in text.lower():
                                modified_text += " ji"
                                switching_points.append(CodeSwitchingPoint(
                                    position=len(modified_text) - 3,
                                    from_language=primary_lang,
                                    to_language=LanguageCode.HINDI,
                                    reason="respect_suffix",
                                    confidence=0.7
                                ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying cultural patterns: {e}")
            return text, []
    
    async def _apply_explanation_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply explanation-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Look for explanation opportunities
            explanation_markers = ["that is", "which means", "in other words"]
            
            for marker in explanation_markers:
                if marker in text.lower():
                    # Replace with Hindi equivalent
                    if marker == "that is":
                        modified_text = re.sub(
                            r'\bthat is\b',
                            'matlab',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif marker == "which means":
                        modified_text = re.sub(
                            r'\bwhich means\b',
                            'matlab',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    
                    position = modified_text.lower().find("matlab")
                    if position != -1:
                        switching_points.append(CodeSwitchingPoint(
                            position=position,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="explanation",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying explanation patterns: {e}")
            return text, []


class MultilingualResponseGenerator(ResponseGeneratorInterface):
    """
    Comprehensive multilingual response generator with Indian localization.
    
    Provides grammatically correct responses in all supported languages,
    natural code-switching, and culturally appropriate formatting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translation_engine = TranslationEngine()
        self.localization_engine = IndianLocalizationEngine()
        self.code_switching_engine = CodeSwitchingEngine()
        
        # Response templates by language and intent
        self._response_templates = self._initialize_response_templates()
        self._grammar_rules = self._initialize_grammar_rules()
        self._cultural_adaptations = self._initialize_cultural_adaptations()
    
    def _initialize_response_templates(self) -> Dict[LanguageCode, Dict[str, Dict[str, str]]]:
        """Initialize response templates for different languages and intents."""
        return {
            LanguageCode.HINDI: {
                IntentCategory.GREETING.value: {
                    "formal": "नमस्ते! आज मैं आपकी कैसे सहायता कर सकता हूँ?",
                    "casual": "हैलो! क्या चाहिए आपको?",
                    "respectful": "नमस्कार जी! आपका स्वागत है।",
                    "default": "नमस्ते! मैं आपकी सहायता के लिए यहाँ हूँ।"
                },
                IntentCategory.WEATHER_INQUIRY.value: {
                    "formal": "मैं आपके लिए मौसम की जानकारी देख रहा हूँ।",
                    "casual": "मौसम का हाल देखता हूँ।",
                    "monsoon": "बारिश की जानकारी देख रहा हूँ।",
                    "default": "मौसम की जानकारी प्राप्त कर रहा हूँ।"
                },
                IntentCategory.TRAIN_INQUIRY.value: {
                    "formal": "मैं आपके लिए ट्रेन की जानकारी खोज रहा हूँ।",
                    "casual": "ट्रेन का टाइम देखता हूँ।",
                    "booking": "ट्रेन बुकिंग में मदद कर सकता हूँ।",
                    "default": "रेल की जानकारी प्राप्त कर रहा हूँ।"
                },
                IntentCategory.FOOD_ORDER.value: {
                    "formal": "मैं आपके लिए खाने का ऑर्डर करने में सहायता कर सकता हूँ।",
                    "casual": "खाना ऑर्डर करना है? बताइए क्या चाहिए।",
                    "local": "स्थानीय खाने के विकल्प देखता हूँ।",
                    "default": "खाने के विकल्प खोज रहा हूँ।"
                }
            },
            LanguageCode.ENGLISH_IN: {
                IntentCategory.GREETING.value: {
                    "formal": "Good day! How may I assist you today?",
                    "casual": "Hey there! What can I do for you?",
                    "respectful": "Namaste! Welcome, how can I help?",
                    "default": "Hello! I'm here to help you."
                },
                IntentCategory.WEATHER_INQUIRY.value: {
                    "formal": "I shall check the weather information for you.",
                    "casual": "Let me get the weather for you!",
                    "monsoon": "Checking monsoon updates for you.",
                    "default": "Getting weather information..."
                },
                IntentCategory.TRAIN_INQUIRY.value: {
                    "formal": "I will help you with train information.",
                    "casual": "Sure, let me check train details!",
                    "booking": "I can help with train booking information.",
                    "default": "Looking up train information..."
                },
                IntentCategory.FOOD_ORDER.value: {
                    "formal": "I can assist you with food ordering.",
                    "casual": "Hungry? Let me help you order food!",
                    "local": "Finding good local food options for you.",
                    "default": "Looking for food options..."
                }
            },
            LanguageCode.TAMIL: {
                IntentCategory.GREETING.value: {
                    "formal": "வணக்கம்! இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                    "casual": "ஹலோ! என்ன வேண்டும்?",
                    "respectful": "வணக்கம் ஐயா! உங்களை வரவேற்கிறேன்.",
                    "default": "வணக்கம்! நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன்."
                }
            }
        }
    
    def _initialize_grammar_rules(self) -> Dict[LanguageCode, Dict[str, Any]]:
        """Initialize grammar rules for different languages."""
        return {
            LanguageCode.HINDI: {
                "verb_conjugation": {
                    "present": {"1st_person": "हूँ", "2nd_person_formal": "हैं", "3rd_person": "है"},
                    "future": {"1st_person": "होऊंगा", "2nd_person_formal": "होंगे", "3rd_person": "होगा"}
                },
                "honorifics": {
                    "formal_you": "आप",
                    "informal_you": "तुम",
                    "respect_suffix": "जी"
                },
                "sentence_structure": "SOV"  # Subject-Object-Verb
            },
            LanguageCode.ENGLISH_IN: {
                "verb_conjugation": {
                    "present": {"1st_person": "am", "2nd_person": "are", "3rd_person": "is"},
                    "future": {"1st_person": "will", "2nd_person": "will", "3rd_person": "will"}
                },
                "honorifics": {
                    "formal_address": "Sir/Madam",
                    "casual_address": "you"
                },
                "sentence_structure": "SVO",  # Subject-Verb-Object
                "indian_english_features": {
                    "use_present_continuous": True,
                    "question_tags": ["isn't it?", "no?", "right?"],
                    "emphasis_words": ["only", "itself", "same"]
                }
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation rules."""
        return {
            "time_references": {
                "morning": {"hindi": "सुबह", "tamil": "காலை", "formal_time": "प्रातःकाल"},
                "evening": {"hindi": "शाम", "tamil": "மாலை", "formal_time": "सायंकाल"},
                "night": {"hindi": "रात", "tamil": "இரவு", "formal_time": "रात्रि"}
            },
            "relationship_terms": {
                "mother": {"hindi": "माँ", "formal": "माता जी", "casual": "मम्मी"},
                "father": {"hindi": "पिता", "formal": "पिता जी", "casual": "पापा"},
                "brother": {"hindi": "भाई", "elder": "भैया", "younger": "छोटा भाई"},
                "sister": {"hindi": "बहन", "elder": "दीदी", "younger": "छोटी बहन"}
            },
            "festival_greetings": {
                "diwali": {"hindi": "दीपावली की शुभकामनाएं", "english": "Happy Diwali"},
                "holi": {"hindi": "होली की शुभकामनाएं", "english": "Happy Holi"},
                "eid": {"hindi": "ईद मुबारक", "english": "Eid Mubarak"}
            }
        }
    
    async def process_query(
        self,
        query: str,
        context: ConversationState
    ) -> Dict[str, Any]:
        """Process user query - delegates to NLU interface."""
        # This method is implemented in the NLU interface
        # We'll import and use it here
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
        return await nlu_interface.process_query(query, context)
    
    async def generate_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: RegionalContextData
    ) -> Response:
        """Generate comprehensive multilingual response with localization."""
        try:
            self.logger.info(f"Generating response for intent: {intent.name}")
            
            # Determine response language and style
            response_language = context.local_language if context else LanguageCode.HINDI
            response_style = await self._determine_response_style(intent, entities, context)
            
            # Generate base response text
            base_response = await self._generate_base_response(
                intent, entities, context, response_language, response_style
            )
            
            # Apply localization
            localized_response = await self._apply_localization(
                base_response, entities, context
            )
            
            # Apply code-switching if appropriate
            final_response, switching_points = await self._apply_code_switching(
                localized_response, intent, entities, context, response_language
            )
            
            # Ensure grammatical correctness
            grammatical_response = await self._ensure_grammar(
                final_response, response_language, response_style
            )
            
            # Create response object
            response = Response(
                text=grammatical_response,
                language=response_language,
                intent=intent,
                entities=[Entity(**entity) if isinstance(entity, dict) else entity for entity in entities],
                confidence=intent.confidence,
                requires_followup=await self._requires_followup(intent, entities),
                suggested_actions=await self._generate_suggested_actions(intent, entities, context),
                external_service_used=await self._determine_external_service(intent),
                processing_time=0.5  # Would be measured in real implementation
            )
            
            self.logger.info(f"Response generated successfully: {len(grammatical_response)} characters")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return await self._create_error_response(intent, entities, context)
    
    async def _determine_response_style(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> ResponseStyle:
        """Determine appropriate response style based on context."""
        try:
            # Default to helpful style
            style = ResponseStyle.HELPFUL
            
            # Analyze intent for style cues
            if intent.name in [IntentCategory.GREETING.value, IntentCategory.FAREWELL.value]:
                style = ResponseStyle.FRIENDLY
            elif intent.name in [IntentCategory.GOVERNMENT_SERVICE.value, IntentCategory.HOSPITAL_INQUIRY.value]:
                style = ResponseStyle.FORMAL
            elif intent.name in [IntentCategory.FOOD_ORDER.value, IntentCategory.CRICKET_SCORES.value]:
                style = ResponseStyle.CASUAL
            elif intent.name in [IntentCategory.FESTIVAL_INQUIRY.value, IntentCategory.CULTURAL_EVENT.value]:
                style = ResponseStyle.ENTHUSIASTIC
            
            # Consider regional context
            if context and context.cultural_events:
                # During festivals, be more enthusiastic
                style = ResponseStyle.ENTHUSIASTIC
            
            return style
            
        except Exception as e:
            self.logger.error(f"Error determining response style: {e}")
            return ResponseStyle.HELPFUL
    
    async def _generate_base_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        language: LanguageCode,
        style: ResponseStyle
    ) -> str:
        """Generate base response text in specified language."""
        try:
            # Get templates for the language
            language_templates = self._response_templates.get(
                language, self._response_templates[LanguageCode.ENGLISH_IN]
            )
            
            # Get templates for the intent
            intent_templates = language_templates.get(
                intent.name, language_templates.get(IntentCategory.HELP.value, {})
            )
            
            # Select template based on style
            style_key = style.value if style.value in intent_templates else "default"
            base_template = intent_templates.get(style_key, intent_templates.get("default", "I can help you with that."))
            
            # Customize based on entities
            customized_response = await self._customize_with_entities(
                base_template, intent, entities, context, language
            )
            
            return customized_response
            
        except Exception as e:
            self.logger.error(f"Error generating base response: {e}")
            return "I'm here to help you."
    
    async def _customize_with_entities(
        self,
        template: str,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        language: LanguageCode
    ) -> str:
        """Customize response template with entity information."""
        try:
            customized = template
            
            # Extract entity values
            entity_dict = {}
            for entity in entities:
                if isinstance(entity, dict):
                    entity_dict[entity.get('type', 'unknown')] = entity.get('value', '')
                else:
                    entity_dict[entity.type] = entity.value
            
            # Customize based on intent and entities
            if intent.name == IntentCategory.WEATHER_INQUIRY.value:
                city = entity_dict.get(EntityType.CITY.value)
                if city:
                    if language == LanguageCode.HINDI:
                        customized = f"{city} के लिए मौसम की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking weather for {city}."
                elif context and context.location:
                    if language == LanguageCode.HINDI:
                        customized = f"{context.location.city} के लिए मौसम की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking weather for {context.location.city}."
            
            elif intent.name == IntentCategory.TRAIN_INQUIRY.value:
                stations = [v for k, v in entity_dict.items() if k == EntityType.STATION.value]
                if len(stations) >= 2:
                    if language == LanguageCode.HINDI:
                        customized = f"{stations[0]} से {stations[1]} के लिए ट्रेन की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking train information from {stations[0]} to {stations[1]}."
                elif len(stations) == 1:
                    if language == LanguageCode.HINDI:
                        customized = f"{stations[0]} के लिए ट्रेन की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking train information for {stations[0]}."
            
            elif intent.name == IntentCategory.FOOD_ORDER.value:
                dishes = [v for k, v in entity_dict.items() if k == EntityType.DISH.value]
                if dishes:
                    dish_list = ", ".join(dishes)
                    if language == LanguageCode.HINDI:
                        customized = f"{dish_list} का ऑर्डर करने में मदद कर सकता हूँ।"
                    else:
                        customized = f"I can help you order {dish_list}."
            
            elif intent.name == IntentCategory.FESTIVAL_INQUIRY.value:
                festival = entity_dict.get(EntityType.FESTIVAL.value)
                if festival:
                    if language == LanguageCode.HINDI:
                        customized = f"{festival} के बारे में जानकारी दे रहा हूँ।"
                    else:
                        customized = f"Getting information about {festival}."
            
            return customized
            
        except Exception as e:
            self.logger.error(f"Error customizing with entities: {e}")
            return template
    
    async def _apply_localization(
        self,
        response_text: str,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> str:
        """Apply Indian localization to response text."""
        try:
            localized_text = response_text
            
            # Find and localize currency mentions
            currency_pattern = r'\$(\d+(?:\.\d{2})?)'
            currency_matches = re.finditer(currency_pattern, localized_text)
            
            for match in currency_matches:
                amount = float(match.group(1))
                localized_currency = await self.localization_engine.localize_currency(amount, "USD")
                localized_text = localized_text.replace(match.group(0), localized_currency.localized_value)
            
            # Find and localize temperature mentions
            temp_pattern = r'(\d+(?:\.\d+)?)\s*°?F'
            temp_matches = re.finditer(temp_pattern, localized_text)
            
            for match in temp_matches:
                temp = float(match.group(1))
                localized_temp = await self.localization_engine.localize_temperature(temp, "F")
                localized_text = localized_text.replace(match.group(0), localized_temp.localized_value)
            
            # Find and localize distance mentions
            distance_pattern = r'(\d+(?:\.\d+)?)\s*(miles?|mi)'
            distance_matches = re.finditer(distance_pattern, localized_text)
            
            for match in distance_matches:
                distance = float(match.group(1))
                unit = match.group(2)
                localized_distance = await self.localization_engine.localize_measurement(distance, unit)
                localized_text = localized_text.replace(match.group(0), localized_distance.localized_value)
            
            # Find and localize time mentions
            time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))'
            time_matches = re.finditer(time_pattern, localized_text, re.IGNORECASE)
            
            for match in time_matches:
                time_str = match.group(1)
                localized_time = await self.localization_engine.localize_time(time_str)
                localized_text = localized_text.replace(match.group(0), localized_time.localized_value)
            
            return localized_text
            
        except Exception as e:
            self.logger.error(f"Error applying localization: {e}")
            return response_text
    
    async def _apply_code_switching(
        self,
        response_text: str,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        primary_language: LanguageCode
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply natural code-switching to response."""
        try:
            # Determine if code-switching is appropriate
            if primary_language == LanguageCode.ENGLISH_IN:
                secondary_language = LanguageCode.HINDI
            elif primary_language == LanguageCode.HINDI:
                secondary_language = LanguageCode.ENGLISH_IN
            else:
                # For other languages, minimal code-switching with English
                secondary_language = LanguageCode.ENGLISH_IN
            
            # Create cultural context for code-switching
            cultural_context = {
                "formality_level": "medium",
                "communication_style": "helpful",
                "regional_influence": context.location.state.lower() if context and context.location else None
            }
            
            # Determine formality based on intent
            if intent.name in [IntentCategory.GOVERNMENT_SERVICE.value, IntentCategory.HOSPITAL_INQUIRY.value]:
                cultural_context["formality_level"] = "high"
            elif intent.name in [IntentCategory.FOOD_ORDER.value, IntentCategory.CRICKET_SCORES.value]:
                cultural_context["formality_level"] = "low"
                cultural_context["communication_style"] = "casual_friendly"
            
            # Apply code-switching
            switched_text, switching_points = await self.code_switching_engine.apply_code_switching(
                response_text,
                primary_language,
                secondary_language,
                cultural_context
            )
            
            return switched_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying code-switching: {e}")
            return response_text, []
    
    async def _ensure_grammar(
        self,
        response_text: str,
        language: LanguageCode,
        style: ResponseStyle
    ) -> str:
        """Ensure grammatical correctness in the specified language."""
        try:
            grammar_rules = self._grammar_rules.get(language, {})
            corrected_text = response_text
            
            if language == LanguageCode.HINDI:
                # Apply Hindi grammar rules
                corrected_text = await self._apply_hindi_grammar(corrected_text, style, grammar_rules)
            elif language == LanguageCode.ENGLISH_IN:
                # Apply Indian English grammar features
                corrected_text = await self._apply_indian_english_grammar(corrected_text, style, grammar_rules)
            
            return corrected_text
            
        except Exception as e:
            self.logger.error(f"Error ensuring grammar: {e}")
            return response_text
    
    async def _apply_hindi_grammar(
        self,
        text: str,
        style: ResponseStyle,
        grammar_rules: Dict[str, Any]
    ) -> str:
        """Apply Hindi grammar rules."""
        try:
            corrected = text
            
            # Ensure proper verb conjugation
            if "हूँ" in corrected and style == ResponseStyle.FORMAL:
                # Use more formal verb forms
                corrected = corrected.replace("हूँ", "हूं")
            
            # Add respectful suffixes for formal style
            if style == ResponseStyle.FORMAL and not corrected.endswith(("जी", "जी।")):
                if corrected.endswith("।"):
                    corrected = corrected[:-1] + " जी।"
                else:
                    corrected += " जी"
            
            # Ensure proper sentence structure (SOV)
            # This is a simplified implementation
            # In a real system, you'd use a proper grammar parser
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying Hindi grammar: {e}")
            return text
    
    async def _apply_indian_english_grammar(
        self,
        text: str,
        style: ResponseStyle,
        grammar_rules: Dict[str, Any]
    ) -> str:
        """Apply Indian English grammar features."""
        try:
            corrected = text
            
            # Add Indian English features
            indian_features = grammar_rules.get("indian_english_features", {})
            
            # Use present continuous for ongoing actions
            if indian_features.get("use_present_continuous"):
                corrected = re.sub(r'\bI check\b', 'I am checking', corrected)
                corrected = re.sub(r'\bI get\b', 'I am getting', corrected)
                corrected = re.sub(r'\bI find\b', 'I am finding', corrected)
            
            # Add question tags for confirmation
            if "?" not in corrected and style in [ResponseStyle.CASUAL, ResponseStyle.FRIENDLY]:
                if corrected.endswith("."):
                    corrected = corrected[:-1] + ", right?"
            
            # Add emphasis words
            if style == ResponseStyle.ENTHUSIASTIC:
                corrected = re.sub(r'\bgood\b', 'very good', corrected)
                corrected = re.sub(r'\bnice\b', 'very nice', corrected)
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying Indian English grammar: {e}")
            return text
    
    async def _requires_followup(self, intent: Intent, entities: List[Dict[str, Any]]) -> bool:
        """Determine if response requires followup."""
        followup_intents = [
            IntentCategory.TRAIN_INQUIRY.value,
            IntentCategory.FOOD_ORDER.value,
            IntentCategory.RIDE_BOOKING.value,
            IntentCategory.GOVERNMENT_SERVICE.value,
            IntentCategory.HOSPITAL_INQUIRY.value
        ]
        
        return intent.name in followup_intents or intent.confidence < 0.7
    
    async def _generate_suggested_actions(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> List[str]:
        """Generate suggested actions for the user."""
        suggestions = []
        
        try:
            if intent.name == IntentCategory.WEATHER_INQUIRY.value:
                suggestions = ["Check 7-day forecast", "Get weather alerts", "Check air quality index"]
            elif intent.name == IntentCategory.TRAIN_INQUIRY.value:
                suggestions = ["Book train ticket", "Check PNR status", "Find nearby stations"]
            elif intent.name == IntentCategory.FOOD_ORDER.value:
                suggestions = ["Browse restaurants", "Check delivery options", "See local specialties"]
            elif intent.name == IntentCategory.FESTIVAL_INQUIRY.value:
                suggestions = ["Get festival calendar", "Learn about traditions", "Find local celebrations"]
            elif intent.name == IntentCategory.HELP.value:
                suggestions = ["Ask about weather", "Check train schedules", "Order food", "Get local info"]
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggested actions: {e}")
            return ["Ask me anything else", "Get help", "Try another query"]
    
    async def _determine_external_service(self, intent: Intent) -> Optional[ServiceType]:
        """Determine which external service to use."""
        service_mapping = {
            IntentCategory.WEATHER_INQUIRY.value: ServiceType.WEATHER,
            IntentCategory.TRAIN_INQUIRY.value: ServiceType.INDIAN_RAILWAYS,
            IntentCategory.FOOD_ORDER.value: ServiceType.FOOD_DELIVERY,
            IntentCategory.RIDE_BOOKING.value: ServiceType.RIDE_SHARING,
            IntentCategory.PAYMENT_UPI.value: ServiceType.UPI_PAYMENT,
            IntentCategory.CRICKET_SCORES.value: ServiceType.CRICKET_SCORES,
            IntentCategory.BOLLYWOOD_NEWS.value: ServiceType.BOLLYWOOD_NEWS,
            IntentCategory.GOVERNMENT_SERVICE.value: ServiceType.GOVERNMENT_SERVICE
        }
        
        return service_mapping.get(intent.name)
    
    async def _create_error_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> Response:
        """Create error response."""
        error_text = "I apologize, but I'm having trouble processing your request. Could you please try again?"
        
        if context and context.local_language == LanguageCode.HINDI:
            error_text = "माफ करें, मुझे आपके अनुरोध को समझने में कुछ समस्या हो रही है। कृपया फिर से कोशिश करें।"
        
        return Response(
            text=error_text,
            language=context.local_language if context else LanguageCode.ENGLISH_IN,
            intent=intent,
            entities=[],
            confidence=0.1,
            requires_followup=True,
            suggested_actions=["Try rephrasing your question", "Ask for help", "Start over"],
            external_service_used=None,
            processing_time=0.1
        )
    
    async def integrate_external_service(
        self,
        service_params: ServiceParameters
    ) -> ServiceResult:
        """Integrate with external service - delegates to NLU interface."""
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
        return await nlu_interface.integrate_external_service(service_params)
    
    async def format_cultural_response(
        self,
        response: Response,
        cultural_context: CulturalContext
    ) -> Response:
        """Format response with cultural appropriateness - delegates to NLU interface."""
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
=======
"""
Response Generator for BharatVoice Assistant.

This module implements comprehensive response generation with multilingual output,
Indian localization for currency/measurements/time formats, grammatically correct
responses in all supported languages, and natural code-switching capabilities.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass

from bharatvoice.core.models import (
    LanguageCode,
    Response,
    Intent,
    Entity,
    ConversationState,
    RegionalContextData,
    CulturalContext,
    UserProfile,
    ServiceParameters,
    ServiceResult,
    ServiceType
)
from bharatvoice.core.interfaces import ResponseGenerator as ResponseGeneratorInterface
from bharatvoice.services.response_generation.nlu_service import IntentCategory, EntityType
from bharatvoice.services.language_engine.translation_engine import TranslationEngine


class ResponseStyle(str, Enum):
    """Response style options."""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    RESPECTFUL = "respectful"
    ENTHUSIASTIC = "enthusiastic"
    HELPFUL = "helpful"
    PROFESSIONAL = "professional"


class LocalizationFormat(str, Enum):
    """Localization format types."""
    CURRENCY = "currency"
    MEASUREMENT = "measurement"
    TIME = "time"
    DATE = "date"
    NUMBER = "number"
    TEMPERATURE = "temperature"


@dataclass
class LocalizedValue:
    """Represents a localized value with formatting."""
    original_value: str
    localized_value: str
    format_type: LocalizationFormat
    locale: str
    explanation: Optional[str] = None


@dataclass
class CodeSwitchingPoint:
    """Represents a point where code-switching occurs in response."""
    position: int
    from_language: LanguageCode
    to_language: LanguageCode
    reason: str
    confidence: float


class IndianLocalizationEngine:
    """Handles Indian-specific localization for various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._currency_mappings = self._initialize_currency_mappings()
        self._measurement_mappings = self._initialize_measurement_mappings()
        self._time_formats = self._initialize_time_formats()
        self._number_formats = self._initialize_number_formats()
    
    def _initialize_currency_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize currency conversion mappings."""
        return {
            "USD": {
                "symbol": "$",
                "indian_equivalent": "₹",
                "conversion_rate": 83.0,  # Approximate rate
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "EUR": {
                "symbol": "€",
                "indian_equivalent": "₹",
                "conversion_rate": 90.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "GBP": {
                "symbol": "£",
                "indian_equivalent": "₹",
                "conversion_rate": 105.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            },
            "INR": {
                "symbol": "₹",
                "indian_equivalent": "₹",
                "conversion_rate": 1.0,
                "format": "₹{amount:,.2f}",
                "spoken_format": "{amount} rupees"
            }
        }
    
    def _initialize_measurement_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize measurement conversion mappings."""
        return {
            "fahrenheit": {
                "unit": "°F",
                "indian_equivalent": "°C",
                "conversion": lambda f: (f - 32) * 5/9,
                "format": "{value:.1f}°C",
                "spoken_format": "{value} degrees celsius"
            },
            "miles": {
                "unit": "mi",
                "indian_equivalent": "km",
                "conversion": lambda mi: mi * 1.60934,
                "format": "{value:.1f} km",
                "spoken_format": "{value} kilometers"
            },
            "feet": {
                "unit": "ft",
                "indian_equivalent": "m",
                "conversion": lambda ft: ft * 0.3048,
                "format": "{value:.1f} meters",
                "spoken_format": "{value} meters"
            },
            "pounds": {
                "unit": "lbs",
                "indian_equivalent": "kg",
                "conversion": lambda lbs: lbs * 0.453592,
                "format": "{value:.1f} kg",
                "spoken_format": "{value} kilograms"
            },
            "gallons": {
                "unit": "gal",
                "indian_equivalent": "L",
                "conversion": lambda gal: gal * 3.78541,
                "format": "{value:.1f} liters",
                "spoken_format": "{value} liters"
            }
        }
    
    def _initialize_time_formats(self) -> Dict[str, Dict[str, str]]:
        """Initialize time format mappings."""
        return {
            "12_hour": {
                "format": "%I:%M %p",
                "indian_format": "%H:%M",
                "spoken_format": "{hour} baje {minute} minute"
            },
            "24_hour": {
                "format": "%H:%M",
                "indian_format": "%H:%M",
                "spoken_format": "{hour} baje {minute} minute"
            },
            "date": {
                "format": "%m/%d/%Y",
                "indian_format": "%d/%m/%Y",
                "spoken_format": "{day} {month} {year}"
            }
        }
    
    def _initialize_number_formats(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Indian number formatting (lakhs, crores)."""
        return {
            "indian_system": {
                "lakh": 100000,
                "crore": 10000000,
                "format_rules": {
                    "use_lakhs_crores": True,
                    "decimal_separator": ".",
                    "thousands_separator": ","
                }
            }
        }
    
    async def localize_currency(self, amount: float, currency: str = "USD") -> LocalizedValue:
        """Localize currency to Indian format."""
        try:
            currency_info = self._currency_mappings.get(currency.upper(), self._currency_mappings["USD"])
            
            if currency.upper() != "INR":
                # Convert to INR
                inr_amount = amount * currency_info["conversion_rate"]
            else:
                inr_amount = amount
            
            # Format in Indian numbering system
            formatted_amount = self._format_indian_number(inr_amount)
            localized_text = f"₹{formatted_amount}"
            
            return LocalizedValue(
                original_value=f"{currency_info['symbol']}{amount:,.2f}",
                localized_value=localized_text,
                format_type=LocalizationFormat.CURRENCY,
                locale="en-IN",
                explanation=f"Converted from {currency} to INR" if currency.upper() != "INR" else None
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing currency: {e}")
            return LocalizedValue(
                original_value=f"{amount}",
                localized_value=f"₹{amount:,.2f}",
                format_type=LocalizationFormat.CURRENCY,
                locale="en-IN"
            )
    
    async def localize_measurement(self, value: float, unit: str) -> LocalizedValue:
        """Localize measurements to Indian standards."""
        try:
            unit_lower = unit.lower()
            measurement_info = self._measurement_mappings.get(unit_lower)
            
            if not measurement_info:
                # Return original if no conversion available
                return LocalizedValue(
                    original_value=f"{value} {unit}",
                    localized_value=f"{value} {unit}",
                    format_type=LocalizationFormat.MEASUREMENT,
                    locale="en-IN"
                )
            
            # Convert to Indian equivalent
            converted_value = measurement_info["conversion"](value)
            localized_text = measurement_info["format"].format(value=converted_value)
            
            return LocalizedValue(
                original_value=f"{value} {unit}",
                localized_value=localized_text,
                format_type=LocalizationFormat.MEASUREMENT,
                locale="en-IN",
                explanation=f"Converted from {unit} to {measurement_info['indian_equivalent']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing measurement: {e}")
            return LocalizedValue(
                original_value=f"{value} {unit}",
                localized_value=f"{value} {unit}",
                format_type=LocalizationFormat.MEASUREMENT,
                locale="en-IN"
            )
    
    async def localize_time(self, time_str: str, format_type: str = "12_hour") -> LocalizedValue:
        """Localize time format to Indian standards."""
        try:
            time_info = self._time_formats.get(format_type, self._time_formats["12_hour"])
            
            # Parse time string and convert to Indian format
            try:
                if ":" in time_str:
                    if "AM" in time_str.upper() or "PM" in time_str.upper():
                        # 12-hour format
                        time_obj = datetime.strptime(time_str.strip(), "%I:%M %p")
                    else:
                        # 24-hour format
                        time_obj = datetime.strptime(time_str.strip(), "%H:%M")
                    
                    # Format in Indian style (24-hour preferred)
                    indian_time = time_obj.strftime("%H:%M")
                    
                    # Create spoken format
                    hour = time_obj.hour
                    minute = time_obj.minute
                    
                    if minute == 0:
                        spoken = f"{hour} baje"
                    else:
                        spoken = f"{hour} baje {minute} minute"
                    
                    return LocalizedValue(
                        original_value=time_str,
                        localized_value=indian_time,
                        format_type=LocalizationFormat.TIME,
                        locale="en-IN",
                        explanation=f"Spoken as: {spoken}"
                    )
                    
            except ValueError:
                # If parsing fails, return original
                pass
            
            return LocalizedValue(
                original_value=time_str,
                localized_value=time_str,
                format_type=LocalizationFormat.TIME,
                locale="en-IN"
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing time: {e}")
            return LocalizedValue(
                original_value=time_str,
                localized_value=time_str,
                format_type=LocalizationFormat.TIME,
                locale="en-IN"
            )
    
    async def localize_temperature(self, temp: float, unit: str = "F") -> LocalizedValue:
        """Localize temperature to Celsius."""
        try:
            if unit.upper() == "F":
                celsius = (temp - 32) * 5/9
                localized_text = f"{celsius:.1f}°C"
                explanation = f"Converted from {temp}°F to Celsius"
            elif unit.upper() == "C":
                celsius = temp
                localized_text = f"{celsius:.1f}°C"
                explanation = None
            else:
                # Unknown unit, assume Celsius
                celsius = temp
                localized_text = f"{celsius:.1f}°C"
                explanation = None
            
            return LocalizedValue(
                original_value=f"{temp}°{unit}",
                localized_value=localized_text,
                format_type=LocalizationFormat.TEMPERATURE,
                locale="en-IN",
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Error localizing temperature: {e}")
            return LocalizedValue(
                original_value=f"{temp}°{unit}",
                localized_value=f"{temp}°{unit}",
                format_type=LocalizationFormat.TEMPERATURE,
                locale="en-IN"
            )
    
    def _format_indian_number(self, number: float) -> str:
        """Format number in Indian numbering system (lakhs, crores)."""
        try:
            if number >= 10000000:  # 1 crore
                crores = number / 10000000
                if crores >= 1:
                    return f"{crores:.2f} crore"
            elif number >= 100000:  # 1 lakh
                lakhs = number / 100000
                if lakhs >= 1:
                    return f"{lakhs:.2f} lakh"
            elif number >= 1000:  # 1 thousand
                thousands = number / 1000
                return f"{thousands:.2f} thousand"
            else:
                return f"{number:,.2f}"
                
        except Exception as e:
            self.logger.error(f"Error formatting Indian number: {e}")
            return f"{number:,.2f}"


class CodeSwitchingEngine:
    """Handles natural code-switching in multilingual responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._switching_patterns = self._initialize_switching_patterns()
        self._cultural_triggers = self._initialize_cultural_triggers()
    
    def _initialize_switching_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize code-switching patterns."""
        return {
            "greeting_patterns": [
                {
                    "trigger": "formal_greeting",
                    "pattern": "Namaste! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.9
                },
                {
                    "trigger": "casual_greeting",
                    "pattern": "Hey yaar, {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                }
            ],
            "emphasis_patterns": [
                {
                    "trigger": "strong_agreement",
                    "pattern": "Bilkul! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                },
                {
                    "trigger": "excitement",
                    "pattern": "Arre waah! {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.7
                }
            ],
            "cultural_patterns": [
                {
                    "trigger": "festival_mention",
                    "pattern": "{hindi_festival_name} ke liye {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.9
                },
                {
                    "trigger": "family_context",
                    "pattern": "Ghar mein {english_content}",
                    "languages": [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
                    "confidence": 0.8
                }
            ],
            "explanation_patterns": [
                {
                    "trigger": "clarification",
                    "pattern": "{english_content}, matlab {hindi_explanation}",
                    "languages": [LanguageCode.ENGLISH_IN, LanguageCode.HINDI],
                    "confidence": 0.8
                }
            ]
        }
    
    def _initialize_cultural_triggers(self) -> Dict[str, List[str]]:
        """Initialize cultural triggers for code-switching."""
        return {
            "festivals": ["diwali", "holi", "eid", "dussehra", "ganpati", "navratri"],
            "family_terms": ["mummy", "papa", "bhai", "didi", "family", "ghar"],
            "food_terms": ["khana", "chai", "roti", "dal", "sabzi"],
            "emotions": ["khushi", "dukh", "gussa", "pyaar", "masti"],
            "respect_terms": ["ji", "sahab", "madam", "sir", "namaste"],
            "casual_terms": ["yaar", "boss", "dude", "bro", "arre"]
        }
    
    async def apply_code_switching(
        self,
        text: str,
        primary_language: LanguageCode,
        secondary_language: LanguageCode,
        cultural_context: Dict[str, Any],
        user_profile: Optional[UserProfile] = None
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply natural code-switching to response text."""
        try:
            switching_points = []
            modified_text = text
            
            # Determine switching probability based on user profile
            switching_probability = self._calculate_switching_probability(
                cultural_context, user_profile
            )
            
            if switching_probability < 0.3:
                return text, []
            
            # Apply greeting patterns
            modified_text, greeting_points = await self._apply_greeting_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(greeting_points)
            
            # Apply emphasis patterns
            modified_text, emphasis_points = await self._apply_emphasis_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(emphasis_points)
            
            # Apply cultural patterns
            modified_text, cultural_points = await self._apply_cultural_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(cultural_points)
            
            # Apply explanation patterns
            modified_text, explanation_points = await self._apply_explanation_patterns(
                modified_text, primary_language, secondary_language, cultural_context
            )
            switching_points.extend(explanation_points)
            
            self.logger.info(f"Applied {len(switching_points)} code-switching points")
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying code-switching: {e}")
            return text, []
    
    def _calculate_switching_probability(
        self,
        cultural_context: Dict[str, Any],
        user_profile: Optional[UserProfile]
    ) -> float:
        """Calculate probability of code-switching based on context."""
        try:
            base_probability = 0.5
            
            # Increase probability for casual communication
            if cultural_context.get("formality_level") == "low":
                base_probability += 0.2
            elif cultural_context.get("formality_level") == "high":
                base_probability -= 0.2
            
            # Increase probability for family/friend context
            if cultural_context.get("communication_style") == "casual_friendly":
                base_probability += 0.3
            elif cultural_context.get("communication_style") == "formal_respectful":
                base_probability -= 0.1
            
            # Consider user profile language preferences
            if user_profile and len(user_profile.preferred_languages) > 1:
                base_probability += 0.2
            
            # Consider regional influence
            if cultural_context.get("regional_influence") in ["north_india", "west_india"]:
                base_probability += 0.1  # More common in these regions
            
            return min(max(base_probability, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating switching probability: {e}")
            return 0.5
    
    async def _apply_greeting_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply greeting-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for greeting context
            greeting_indicators = ["hello", "hi", "hey", "good morning", "good evening"]
            
            if any(indicator in text.lower() for indicator in greeting_indicators):
                formality = cultural_context.get("formality_level", "medium")
                
                if formality == "high":
                    # Use formal Hindi greeting
                    if text.lower().startswith(("hello", "hi")):
                        modified_text = re.sub(
                            r'^(hello|hi)\b',
                            'Namaste',
                            text,
                            flags=re.IGNORECASE
                        )
                        switching_points.append(CodeSwitchingPoint(
                            position=0,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="formal_greeting",
                            confidence=0.9
                        ))
                elif formality == "low":
                    # Use casual mixed greeting
                    if text.lower().startswith(("hello", "hi")):
                        modified_text = re.sub(
                            r'^(hello|hi)\b',
                            'Hey yaar',
                            text,
                            flags=re.IGNORECASE
                        )
                        switching_points.append(CodeSwitchingPoint(
                            position=0,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="casual_greeting",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying greeting patterns: {e}")
            return text, []
    
    async def _apply_emphasis_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply emphasis-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for emphasis opportunities
            emphasis_words = ["yes", "absolutely", "definitely", "exactly", "perfect"]
            
            for word in emphasis_words:
                if word in text.lower():
                    # Replace with Hindi emphasis
                    if word == "yes":
                        modified_text = re.sub(
                            r'\byes\b',
                            'Haan bilkul',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif word in ["absolutely", "definitely"]:
                        modified_text = re.sub(
                            rf'\b{word}\b',
                            'Bilkul',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif word == "perfect":
                        modified_text = re.sub(
                            r'\bperfect\b',
                            'Ekdum perfect',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    
                    # Find position and add switching point
                    position = modified_text.lower().find(word)
                    if position != -1:
                        switching_points.append(CodeSwitchingPoint(
                            position=position,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="emphasis",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying emphasis patterns: {e}")
            return text, []
    
    async def _apply_cultural_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply culture-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Check for cultural triggers
            for category, triggers in self._cultural_triggers.items():
                for trigger in triggers:
                    if trigger in text.lower():
                        if category == "festivals":
                            # Keep festival names in original language
                            continue
                        elif category == "family_terms":
                            # Use Hindi family terms
                            if "family" in text.lower():
                                modified_text = re.sub(
                                    r'\bfamily\b',
                                    'parivaar',
                                    modified_text,
                                    flags=re.IGNORECASE
                                )
                        elif category == "food_terms":
                            # Keep food terms in Hindi
                            continue
                        elif category == "respect_terms":
                            # Add respectful suffixes
                            if "sir" not in text.lower() and "madam" not in text.lower():
                                modified_text += " ji"
                                switching_points.append(CodeSwitchingPoint(
                                    position=len(modified_text) - 3,
                                    from_language=primary_lang,
                                    to_language=LanguageCode.HINDI,
                                    reason="respect_suffix",
                                    confidence=0.7
                                ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying cultural patterns: {e}")
            return text, []
    
    async def _apply_explanation_patterns(
        self,
        text: str,
        primary_lang: LanguageCode,
        secondary_lang: LanguageCode,
        cultural_context: Dict[str, Any]
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply explanation-based code-switching patterns."""
        switching_points = []
        modified_text = text
        
        try:
            # Look for explanation opportunities
            explanation_markers = ["that is", "which means", "in other words"]
            
            for marker in explanation_markers:
                if marker in text.lower():
                    # Replace with Hindi equivalent
                    if marker == "that is":
                        modified_text = re.sub(
                            r'\bthat is\b',
                            'matlab',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    elif marker == "which means":
                        modified_text = re.sub(
                            r'\bwhich means\b',
                            'matlab',
                            modified_text,
                            flags=re.IGNORECASE
                        )
                    
                    position = modified_text.lower().find("matlab")
                    if position != -1:
                        switching_points.append(CodeSwitchingPoint(
                            position=position,
                            from_language=primary_lang,
                            to_language=LanguageCode.HINDI,
                            reason="explanation",
                            confidence=0.8
                        ))
            
            return modified_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying explanation patterns: {e}")
            return text, []


class MultilingualResponseGenerator(ResponseGeneratorInterface):
    """
    Comprehensive multilingual response generator with Indian localization.
    
    Provides grammatically correct responses in all supported languages,
    natural code-switching, and culturally appropriate formatting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translation_engine = TranslationEngine()
        self.localization_engine = IndianLocalizationEngine()
        self.code_switching_engine = CodeSwitchingEngine()
        
        # Response templates by language and intent
        self._response_templates = self._initialize_response_templates()
        self._grammar_rules = self._initialize_grammar_rules()
        self._cultural_adaptations = self._initialize_cultural_adaptations()
    
    def _initialize_response_templates(self) -> Dict[LanguageCode, Dict[str, Dict[str, str]]]:
        """Initialize response templates for different languages and intents."""
        return {
            LanguageCode.HINDI: {
                IntentCategory.GREETING.value: {
                    "formal": "नमस्ते! आज मैं आपकी कैसे सहायता कर सकता हूँ?",
                    "casual": "हैलो! क्या चाहिए आपको?",
                    "respectful": "नमस्कार जी! आपका स्वागत है।",
                    "default": "नमस्ते! मैं आपकी सहायता के लिए यहाँ हूँ।"
                },
                IntentCategory.WEATHER_INQUIRY.value: {
                    "formal": "मैं आपके लिए मौसम की जानकारी देख रहा हूँ।",
                    "casual": "मौसम का हाल देखता हूँ।",
                    "monsoon": "बारिश की जानकारी देख रहा हूँ।",
                    "default": "मौसम की जानकारी प्राप्त कर रहा हूँ।"
                },
                IntentCategory.TRAIN_INQUIRY.value: {
                    "formal": "मैं आपके लिए ट्रेन की जानकारी खोज रहा हूँ।",
                    "casual": "ट्रेन का टाइम देखता हूँ।",
                    "booking": "ट्रेन बुकिंग में मदद कर सकता हूँ।",
                    "default": "रेल की जानकारी प्राप्त कर रहा हूँ।"
                },
                IntentCategory.FOOD_ORDER.value: {
                    "formal": "मैं आपके लिए खाने का ऑर्डर करने में सहायता कर सकता हूँ।",
                    "casual": "खाना ऑर्डर करना है? बताइए क्या चाहिए।",
                    "local": "स्थानीय खाने के विकल्प देखता हूँ।",
                    "default": "खाने के विकल्प खोज रहा हूँ।"
                }
            },
            LanguageCode.ENGLISH_IN: {
                IntentCategory.GREETING.value: {
                    "formal": "Good day! How may I assist you today?",
                    "casual": "Hey there! What can I do for you?",
                    "respectful": "Namaste! Welcome, how can I help?",
                    "default": "Hello! I'm here to help you."
                },
                IntentCategory.WEATHER_INQUIRY.value: {
                    "formal": "I shall check the weather information for you.",
                    "casual": "Let me get the weather for you!",
                    "monsoon": "Checking monsoon updates for you.",
                    "default": "Getting weather information..."
                },
                IntentCategory.TRAIN_INQUIRY.value: {
                    "formal": "I will help you with train information.",
                    "casual": "Sure, let me check train details!",
                    "booking": "I can help with train booking information.",
                    "default": "Looking up train information..."
                },
                IntentCategory.FOOD_ORDER.value: {
                    "formal": "I can assist you with food ordering.",
                    "casual": "Hungry? Let me help you order food!",
                    "local": "Finding good local food options for you.",
                    "default": "Looking for food options..."
                }
            },
            LanguageCode.TAMIL: {
                IntentCategory.GREETING.value: {
                    "formal": "வணக்கம்! இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                    "casual": "ஹலோ! என்ன வேண்டும்?",
                    "respectful": "வணக்கம் ஐயா! உங்களை வரவேற்கிறேன்.",
                    "default": "வணக்கம்! நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன்."
                }
            }
        }
    
    def _initialize_grammar_rules(self) -> Dict[LanguageCode, Dict[str, Any]]:
        """Initialize grammar rules for different languages."""
        return {
            LanguageCode.HINDI: {
                "verb_conjugation": {
                    "present": {"1st_person": "हूँ", "2nd_person_formal": "हैं", "3rd_person": "है"},
                    "future": {"1st_person": "होऊंगा", "2nd_person_formal": "होंगे", "3rd_person": "होगा"}
                },
                "honorifics": {
                    "formal_you": "आप",
                    "informal_you": "तुम",
                    "respect_suffix": "जी"
                },
                "sentence_structure": "SOV"  # Subject-Object-Verb
            },
            LanguageCode.ENGLISH_IN: {
                "verb_conjugation": {
                    "present": {"1st_person": "am", "2nd_person": "are", "3rd_person": "is"},
                    "future": {"1st_person": "will", "2nd_person": "will", "3rd_person": "will"}
                },
                "honorifics": {
                    "formal_address": "Sir/Madam",
                    "casual_address": "you"
                },
                "sentence_structure": "SVO",  # Subject-Verb-Object
                "indian_english_features": {
                    "use_present_continuous": True,
                    "question_tags": ["isn't it?", "no?", "right?"],
                    "emphasis_words": ["only", "itself", "same"]
                }
            }
        }
    
    def _initialize_cultural_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation rules."""
        return {
            "time_references": {
                "morning": {"hindi": "सुबह", "tamil": "காலை", "formal_time": "प्रातःकाल"},
                "evening": {"hindi": "शाम", "tamil": "மாலை", "formal_time": "सायंकाल"},
                "night": {"hindi": "रात", "tamil": "இரவு", "formal_time": "रात्रि"}
            },
            "relationship_terms": {
                "mother": {"hindi": "माँ", "formal": "माता जी", "casual": "मम्मी"},
                "father": {"hindi": "पिता", "formal": "पिता जी", "casual": "पापा"},
                "brother": {"hindi": "भाई", "elder": "भैया", "younger": "छोटा भाई"},
                "sister": {"hindi": "बहन", "elder": "दीदी", "younger": "छोटी बहन"}
            },
            "festival_greetings": {
                "diwali": {"hindi": "दीपावली की शुभकामनाएं", "english": "Happy Diwali"},
                "holi": {"hindi": "होली की शुभकामनाएं", "english": "Happy Holi"},
                "eid": {"hindi": "ईद मुबारक", "english": "Eid Mubarak"}
            }
        }
    
    async def process_query(
        self,
        query: str,
        context: ConversationState
    ) -> Dict[str, Any]:
        """Process user query - delegates to NLU interface."""
        # This method is implemented in the NLU interface
        # We'll import and use it here
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
        return await nlu_interface.process_query(query, context)
    
    async def generate_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: RegionalContextData
    ) -> Response:
        """Generate comprehensive multilingual response with localization."""
        try:
            self.logger.info(f"Generating response for intent: {intent.name}")
            
            # Determine response language and style
            response_language = context.local_language if context else LanguageCode.HINDI
            response_style = await self._determine_response_style(intent, entities, context)
            
            # Generate base response text
            base_response = await self._generate_base_response(
                intent, entities, context, response_language, response_style
            )
            
            # Apply localization
            localized_response = await self._apply_localization(
                base_response, entities, context
            )
            
            # Apply code-switching if appropriate
            final_response, switching_points = await self._apply_code_switching(
                localized_response, intent, entities, context, response_language
            )
            
            # Ensure grammatical correctness
            grammatical_response = await self._ensure_grammar(
                final_response, response_language, response_style
            )
            
            # Create response object
            response = Response(
                text=grammatical_response,
                language=response_language,
                intent=intent,
                entities=[Entity(**entity) if isinstance(entity, dict) else entity for entity in entities],
                confidence=intent.confidence,
                requires_followup=await self._requires_followup(intent, entities),
                suggested_actions=await self._generate_suggested_actions(intent, entities, context),
                external_service_used=await self._determine_external_service(intent),
                processing_time=0.5  # Would be measured in real implementation
            )
            
            self.logger.info(f"Response generated successfully: {len(grammatical_response)} characters")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return await self._create_error_response(intent, entities, context)
    
    async def _determine_response_style(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> ResponseStyle:
        """Determine appropriate response style based on context."""
        try:
            # Default to helpful style
            style = ResponseStyle.HELPFUL
            
            # Analyze intent for style cues
            if intent.name in [IntentCategory.GREETING.value, IntentCategory.FAREWELL.value]:
                style = ResponseStyle.FRIENDLY
            elif intent.name in [IntentCategory.GOVERNMENT_SERVICE.value, IntentCategory.HOSPITAL_INQUIRY.value]:
                style = ResponseStyle.FORMAL
            elif intent.name in [IntentCategory.FOOD_ORDER.value, IntentCategory.CRICKET_SCORES.value]:
                style = ResponseStyle.CASUAL
            elif intent.name in [IntentCategory.FESTIVAL_INQUIRY.value, IntentCategory.CULTURAL_EVENT.value]:
                style = ResponseStyle.ENTHUSIASTIC
            
            # Consider regional context
            if context and context.cultural_events:
                # During festivals, be more enthusiastic
                style = ResponseStyle.ENTHUSIASTIC
            
            return style
            
        except Exception as e:
            self.logger.error(f"Error determining response style: {e}")
            return ResponseStyle.HELPFUL
    
    async def _generate_base_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        language: LanguageCode,
        style: ResponseStyle
    ) -> str:
        """Generate base response text in specified language."""
        try:
            # Get templates for the language
            language_templates = self._response_templates.get(
                language, self._response_templates[LanguageCode.ENGLISH_IN]
            )
            
            # Get templates for the intent
            intent_templates = language_templates.get(
                intent.name, language_templates.get(IntentCategory.HELP.value, {})
            )
            
            # Select template based on style
            style_key = style.value if style.value in intent_templates else "default"
            base_template = intent_templates.get(style_key, intent_templates.get("default", "I can help you with that."))
            
            # Customize based on entities
            customized_response = await self._customize_with_entities(
                base_template, intent, entities, context, language
            )
            
            return customized_response
            
        except Exception as e:
            self.logger.error(f"Error generating base response: {e}")
            return "I'm here to help you."
    
    async def _customize_with_entities(
        self,
        template: str,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        language: LanguageCode
    ) -> str:
        """Customize response template with entity information."""
        try:
            customized = template
            
            # Extract entity values
            entity_dict = {}
            for entity in entities:
                if isinstance(entity, dict):
                    entity_dict[entity.get('type', 'unknown')] = entity.get('value', '')
                else:
                    entity_dict[entity.type] = entity.value
            
            # Customize based on intent and entities
            if intent.name == IntentCategory.WEATHER_INQUIRY.value:
                city = entity_dict.get(EntityType.CITY.value)
                if city:
                    if language == LanguageCode.HINDI:
                        customized = f"{city} के लिए मौसम की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking weather for {city}."
                elif context and context.location:
                    if language == LanguageCode.HINDI:
                        customized = f"{context.location.city} के लिए मौसम की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking weather for {context.location.city}."
            
            elif intent.name == IntentCategory.TRAIN_INQUIRY.value:
                stations = [v for k, v in entity_dict.items() if k == EntityType.STATION.value]
                if len(stations) >= 2:
                    if language == LanguageCode.HINDI:
                        customized = f"{stations[0]} से {stations[1]} के लिए ट्रेन की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking train information from {stations[0]} to {stations[1]}."
                elif len(stations) == 1:
                    if language == LanguageCode.HINDI:
                        customized = f"{stations[0]} के लिए ट्रेन की जानकारी देख रहा हूँ।"
                    else:
                        customized = f"Checking train information for {stations[0]}."
            
            elif intent.name == IntentCategory.FOOD_ORDER.value:
                dishes = [v for k, v in entity_dict.items() if k == EntityType.DISH.value]
                if dishes:
                    dish_list = ", ".join(dishes)
                    if language == LanguageCode.HINDI:
                        customized = f"{dish_list} का ऑर्डर करने में मदद कर सकता हूँ।"
                    else:
                        customized = f"I can help you order {dish_list}."
            
            elif intent.name == IntentCategory.FESTIVAL_INQUIRY.value:
                festival = entity_dict.get(EntityType.FESTIVAL.value)
                if festival:
                    if language == LanguageCode.HINDI:
                        customized = f"{festival} के बारे में जानकारी दे रहा हूँ।"
                    else:
                        customized = f"Getting information about {festival}."
            
            return customized
            
        except Exception as e:
            self.logger.error(f"Error customizing with entities: {e}")
            return template
    
    async def _apply_localization(
        self,
        response_text: str,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> str:
        """Apply Indian localization to response text."""
        try:
            localized_text = response_text
            
            # Find and localize currency mentions
            currency_pattern = r'\$(\d+(?:\.\d{2})?)'
            currency_matches = re.finditer(currency_pattern, localized_text)
            
            for match in currency_matches:
                amount = float(match.group(1))
                localized_currency = await self.localization_engine.localize_currency(amount, "USD")
                localized_text = localized_text.replace(match.group(0), localized_currency.localized_value)
            
            # Find and localize temperature mentions
            temp_pattern = r'(\d+(?:\.\d+)?)\s*°?F'
            temp_matches = re.finditer(temp_pattern, localized_text)
            
            for match in temp_matches:
                temp = float(match.group(1))
                localized_temp = await self.localization_engine.localize_temperature(temp, "F")
                localized_text = localized_text.replace(match.group(0), localized_temp.localized_value)
            
            # Find and localize distance mentions
            distance_pattern = r'(\d+(?:\.\d+)?)\s*(miles?|mi)'
            distance_matches = re.finditer(distance_pattern, localized_text)
            
            for match in distance_matches:
                distance = float(match.group(1))
                unit = match.group(2)
                localized_distance = await self.localization_engine.localize_measurement(distance, unit)
                localized_text = localized_text.replace(match.group(0), localized_distance.localized_value)
            
            # Find and localize time mentions
            time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM))'
            time_matches = re.finditer(time_pattern, localized_text, re.IGNORECASE)
            
            for match in time_matches:
                time_str = match.group(1)
                localized_time = await self.localization_engine.localize_time(time_str)
                localized_text = localized_text.replace(match.group(0), localized_time.localized_value)
            
            return localized_text
            
        except Exception as e:
            self.logger.error(f"Error applying localization: {e}")
            return response_text
    
    async def _apply_code_switching(
        self,
        response_text: str,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData],
        primary_language: LanguageCode
    ) -> Tuple[str, List[CodeSwitchingPoint]]:
        """Apply natural code-switching to response."""
        try:
            # Determine if code-switching is appropriate
            if primary_language == LanguageCode.ENGLISH_IN:
                secondary_language = LanguageCode.HINDI
            elif primary_language == LanguageCode.HINDI:
                secondary_language = LanguageCode.ENGLISH_IN
            else:
                # For other languages, minimal code-switching with English
                secondary_language = LanguageCode.ENGLISH_IN
            
            # Create cultural context for code-switching
            cultural_context = {
                "formality_level": "medium",
                "communication_style": "helpful",
                "regional_influence": context.location.state.lower() if context and context.location else None
            }
            
            # Determine formality based on intent
            if intent.name in [IntentCategory.GOVERNMENT_SERVICE.value, IntentCategory.HOSPITAL_INQUIRY.value]:
                cultural_context["formality_level"] = "high"
            elif intent.name in [IntentCategory.FOOD_ORDER.value, IntentCategory.CRICKET_SCORES.value]:
                cultural_context["formality_level"] = "low"
                cultural_context["communication_style"] = "casual_friendly"
            
            # Apply code-switching
            switched_text, switching_points = await self.code_switching_engine.apply_code_switching(
                response_text,
                primary_language,
                secondary_language,
                cultural_context
            )
            
            return switched_text, switching_points
            
        except Exception as e:
            self.logger.error(f"Error applying code-switching: {e}")
            return response_text, []
    
    async def _ensure_grammar(
        self,
        response_text: str,
        language: LanguageCode,
        style: ResponseStyle
    ) -> str:
        """Ensure grammatical correctness in the specified language."""
        try:
            grammar_rules = self._grammar_rules.get(language, {})
            corrected_text = response_text
            
            if language == LanguageCode.HINDI:
                # Apply Hindi grammar rules
                corrected_text = await self._apply_hindi_grammar(corrected_text, style, grammar_rules)
            elif language == LanguageCode.ENGLISH_IN:
                # Apply Indian English grammar features
                corrected_text = await self._apply_indian_english_grammar(corrected_text, style, grammar_rules)
            
            return corrected_text
            
        except Exception as e:
            self.logger.error(f"Error ensuring grammar: {e}")
            return response_text
    
    async def _apply_hindi_grammar(
        self,
        text: str,
        style: ResponseStyle,
        grammar_rules: Dict[str, Any]
    ) -> str:
        """Apply Hindi grammar rules."""
        try:
            corrected = text
            
            # Ensure proper verb conjugation
            if "हूँ" in corrected and style == ResponseStyle.FORMAL:
                # Use more formal verb forms
                corrected = corrected.replace("हूँ", "हूं")
            
            # Add respectful suffixes for formal style
            if style == ResponseStyle.FORMAL and not corrected.endswith(("जी", "जी।")):
                if corrected.endswith("।"):
                    corrected = corrected[:-1] + " जी।"
                else:
                    corrected += " जी"
            
            # Ensure proper sentence structure (SOV)
            # This is a simplified implementation
            # In a real system, you'd use a proper grammar parser
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying Hindi grammar: {e}")
            return text
    
    async def _apply_indian_english_grammar(
        self,
        text: str,
        style: ResponseStyle,
        grammar_rules: Dict[str, Any]
    ) -> str:
        """Apply Indian English grammar features."""
        try:
            corrected = text
            
            # Add Indian English features
            indian_features = grammar_rules.get("indian_english_features", {})
            
            # Use present continuous for ongoing actions
            if indian_features.get("use_present_continuous"):
                corrected = re.sub(r'\bI check\b', 'I am checking', corrected)
                corrected = re.sub(r'\bI get\b', 'I am getting', corrected)
                corrected = re.sub(r'\bI find\b', 'I am finding', corrected)
            
            # Add question tags for confirmation
            if "?" not in corrected and style in [ResponseStyle.CASUAL, ResponseStyle.FRIENDLY]:
                if corrected.endswith("."):
                    corrected = corrected[:-1] + ", right?"
            
            # Add emphasis words
            if style == ResponseStyle.ENTHUSIASTIC:
                corrected = re.sub(r'\bgood\b', 'very good', corrected)
                corrected = re.sub(r'\bnice\b', 'very nice', corrected)
            
            return corrected
            
        except Exception as e:
            self.logger.error(f"Error applying Indian English grammar: {e}")
            return text
    
    async def _requires_followup(self, intent: Intent, entities: List[Dict[str, Any]]) -> bool:
        """Determine if response requires followup."""
        followup_intents = [
            IntentCategory.TRAIN_INQUIRY.value,
            IntentCategory.FOOD_ORDER.value,
            IntentCategory.RIDE_BOOKING.value,
            IntentCategory.GOVERNMENT_SERVICE.value,
            IntentCategory.HOSPITAL_INQUIRY.value
        ]
        
        return intent.name in followup_intents or intent.confidence < 0.7
    
    async def _generate_suggested_actions(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> List[str]:
        """Generate suggested actions for the user."""
        suggestions = []
        
        try:
            if intent.name == IntentCategory.WEATHER_INQUIRY.value:
                suggestions = ["Check 7-day forecast", "Get weather alerts", "Check air quality index"]
            elif intent.name == IntentCategory.TRAIN_INQUIRY.value:
                suggestions = ["Book train ticket", "Check PNR status", "Find nearby stations"]
            elif intent.name == IntentCategory.FOOD_ORDER.value:
                suggestions = ["Browse restaurants", "Check delivery options", "See local specialties"]
            elif intent.name == IntentCategory.FESTIVAL_INQUIRY.value:
                suggestions = ["Get festival calendar", "Learn about traditions", "Find local celebrations"]
            elif intent.name == IntentCategory.HELP.value:
                suggestions = ["Ask about weather", "Check train schedules", "Order food", "Get local info"]
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggested actions: {e}")
            return ["Ask me anything else", "Get help", "Try another query"]
    
    async def _determine_external_service(self, intent: Intent) -> Optional[ServiceType]:
        """Determine which external service to use."""
        service_mapping = {
            IntentCategory.WEATHER_INQUIRY.value: ServiceType.WEATHER,
            IntentCategory.TRAIN_INQUIRY.value: ServiceType.INDIAN_RAILWAYS,
            IntentCategory.FOOD_ORDER.value: ServiceType.FOOD_DELIVERY,
            IntentCategory.RIDE_BOOKING.value: ServiceType.RIDE_SHARING,
            IntentCategory.PAYMENT_UPI.value: ServiceType.UPI_PAYMENT,
            IntentCategory.CRICKET_SCORES.value: ServiceType.CRICKET_SCORES,
            IntentCategory.BOLLYWOOD_NEWS.value: ServiceType.BOLLYWOOD_NEWS,
            IntentCategory.GOVERNMENT_SERVICE.value: ServiceType.GOVERNMENT_SERVICE
        }
        
        return service_mapping.get(intent.name)
    
    async def _create_error_response(
        self,
        intent: Intent,
        entities: List[Dict[str, Any]],
        context: Optional[RegionalContextData]
    ) -> Response:
        """Create error response."""
        error_text = "I apologize, but I'm having trouble processing your request. Could you please try again?"
        
        if context and context.local_language == LanguageCode.HINDI:
            error_text = "माफ करें, मुझे आपके अनुरोध को समझने में कुछ समस्या हो रही है। कृपया फिर से कोशिश करें।"
        
        return Response(
            text=error_text,
            language=context.local_language if context else LanguageCode.ENGLISH_IN,
            intent=intent,
            entities=[],
            confidence=0.1,
            requires_followup=True,
            suggested_actions=["Try rephrasing your question", "Ask for help", "Start over"],
            external_service_used=None,
            processing_time=0.1
        )
    
    async def integrate_external_service(
        self,
        service_params: ServiceParameters
    ) -> ServiceResult:
        """Integrate with external service - delegates to NLU interface."""
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
        return await nlu_interface.integrate_external_service(service_params)
    
    async def format_cultural_response(
        self,
        response: Response,
        cultural_context: CulturalContext
    ) -> Response:
        """Format response with cultural appropriateness - delegates to NLU interface."""
        from bharatvoice.services.response_generation.nlu_interface import NLUInterface
        
        nlu_interface = NLUInterface()
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        return await nlu_interface.format_cultural_response(response, cultural_context)