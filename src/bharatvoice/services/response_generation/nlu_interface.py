"""
NLU Interface Implementation for BharatVoice Assistant.

This module provides the main interface for Natural Language Understanding
services, implementing the ResponseGenerator interface for query processing
and cultural context integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from bharatvoice.core.interfaces import ResponseGenerator
from bharatvoice.core.models import (
    Intent,
    Entity,
    Response,
    ConversationState,
    RegionalContextData,
    CulturalContext,
    ServiceParameters,
    ServiceResult,
    LanguageCode,
    UserProfile
)
from bharatvoice.services.response_generation.nlu_service import (
    NLUService,
    IntentCategory,
    EntityType
)


class NLUInterface(ResponseGenerator):
    """
    Main NLU interface implementing ResponseGenerator for query processing.
    
    This class serves as the primary entry point for NLU operations,
    integrating intent recognition, entity extraction, and cultural context
    interpretation for Indian users.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlu_service = NLUService()
        self._response_templates = self._initialize_response_templates()
        self._cultural_response_patterns = self._initialize_cultural_patterns()
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize response templates for different intents."""
        return {
            IntentCategory.GREETING.value: {
                "formal": "Namaste! How may I assist you today?",
                "casual": "Hey there! What's up?",
                "regional_north": "Namaste ji! Kaise hain aap?",
                "regional_south": "Vanakkam! How can I help you?",
                "default": "Hello! How can I help you today?"
            },
            IntentCategory.WEATHER_INQUIRY.value: {
                "formal": "I'll check the weather information for you.",
                "casual": "Let me get the weather for you!",
                "monsoon": "Let me check the monsoon updates for you.",
                "default": "Checking weather information..."
            },
            IntentCategory.TRAIN_INQUIRY.value: {
                "formal": "I'll help you with train information.",
                "casual": "Sure, let me check train details for you!",
                "booking": "I can help you with train booking information.",
                "default": "Looking up train information..."
            },
            IntentCategory.FESTIVAL_INQUIRY.value: {
                "formal": "I'll provide information about upcoming festivals.",
                "casual": "Let me tell you about the festivals coming up!",
                "cultural": "Here are the auspicious occasions ahead:",
                "default": "Checking festival information..."
            },
            IntentCategory.FOOD_ORDER.value: {
                "formal": "I can assist you with food ordering.",
                "casual": "Hungry? Let me help you order some food!",
                "local": "Let me find some good local food options for you.",
                "default": "Looking for food options..."
            },
            IntentCategory.HELP.value: {
                "formal": "I'm here to assist you with various services.",
                "casual": "I'm here to help! What do you need?",
                "detailed": "I can help with weather, trains, food orders, and much more!",
                "default": "How can I help you?"
            }
        }
    
    def _initialize_cultural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural response patterns."""
        return {
            "respectful_formal": {
                "prefixes": ["Ji haan", "Bilkul", "Zaroor"],
                "suffixes": ["ji", "sahab", "madam"],
                "tone": "respectful"
            },
            "casual_friendly": {
                "prefixes": ["Yaar", "Boss", "Arre"],
                "suffixes": ["yaar", "bro", "dude"],
                "tone": "friendly"
            },
            "family_oriented": {
                "prefixes": ["Beta", "Bhai", "Didi"],
                "suffixes": ["beta", "bhai", "didi"],
                "tone": "warm"
            },
            "regional_north": {
                "prefixes": ["Acha", "Theek hai", "Bilkul"],
                "suffixes": ["ji", "sahab"],
                "tone": "regional_north"
            },
            "regional_south": {
                "prefixes": ["Sure", "Of course", "Definitely"],
                "suffixes": ["sir", "madam"],
                "tone": "regional_south"
            }
        }
    
    async def process_query(
        self, 
        query: str, 
        context: ConversationState
    ) -> Dict[str, Any]:
        """
        Process user query and extract intent and entities.
        
        Args:
            query: User query text
            context: Current conversation context
            
        Returns:
            Query processing result with intent and entities
        """
        try:
            self.logger.info(f"Processing query: '{query[:50]}...' for session {context.session_id}")
            
            # Get user profile and regional context (would be injected in real implementation)
            user_profile = None  # Would be retrieved from context manager
            regional_context = None  # Would be retrieved from context manager
            
            # Process through NLU pipeline
            nlu_result = await self.nlu_service.process_user_input(
                text=query,
                language=context.current_language,
                conversation_state=context,
                user_profile=user_profile,
                regional_context=regional_context
            )
            
            # Extract key components
            intent = nlu_result["intent"]
            entities = nlu_result["entities"]
            cultural_context = nlu_result["cultural_context"]
            
            # Validate cultural appropriateness
            validation = await self.nlu_service.validate_cultural_appropriateness(
                query, intent, cultural_context
            )
            
            processing_result = {
                "intent": intent,
                "entities": entities,
                "cultural_context": cultural_context,
                "confidence": nlu_result["confidence"],
                "language": context.current_language,
                "processing_metadata": nlu_result["processing_metadata"],
                "cultural_validation": validation,
                "suggestions": await self._get_response_suggestions(intent, entities, cultural_context)
            }
            
            self.logger.info(f"Query processed successfully. Intent: {intent.name}, Confidence: {intent.confidence:.2f}")
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "intent": Intent(
                    name=IntentCategory.UNKNOWN.value,
                    confidence=0.1,
                    category=IntentCategory.UNKNOWN.value,
                    parameters={}
                ),
                "entities": [],
                "cultural_context": {},
                "confidence": 0.1,
                "error": str(e)
            }
    
    async def generate_response(
        self, 
        intent: Intent, 
        entities: List[Dict[str, Any]], 
        context: RegionalContextData
    ) -> Response:
        """
        Generate appropriate response based on intent and context.
        
        Args:
            intent: Detected user intent
            entities: Extracted entities from query
            context: Regional context data
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Generating response for intent: {intent.name}")
            
            # Convert entities to Entity objects if they're dicts
            entity_objects = []
            for entity in entities:
                if isinstance(entity, dict):
                    entity_obj = Entity(
                        name=entity.get("name", "unknown"),
                        value=entity.get("value", ""),
                        type=entity.get("type", "unknown"),
                        confidence=entity.get("confidence", 0.5),
                        start_pos=entity.get("start_pos", 0),
                        end_pos=entity.get("end_pos", 0)
                    )
                    entity_objects.append(entity_obj)
                else:
                    entity_objects.append(entity)
            
            # Determine response language based on context
            response_language = context.local_language if context else LanguageCode.HINDI
            
            # Generate base response text
            response_text = await self._generate_response_text(intent, entity_objects, context)
            
            # Determine if followup is needed
            requires_followup = await self._requires_followup(intent, entity_objects)
            
            # Generate suggested actions
            suggested_actions = await self._generate_suggested_actions(intent, entity_objects, context)
            
            # Determine external service usage
            external_service = await self._determine_external_service(intent, entity_objects)
            
            response = Response(
                text=response_text,
                language=response_language,
                intent=intent,
                entities=entity_objects,
                confidence=intent.confidence,
                requires_followup=requires_followup,
                suggested_actions=suggested_actions,
                external_service_used=external_service,
                processing_time=0.5  # Would be measured in real implementation
            )
            
            self.logger.info(f"Response generated successfully for intent: {intent.name}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return Response(
                text="I apologize, but I'm having trouble understanding your request. Could you please rephrase?",
                language=LanguageCode.ENGLISH_IN,
                intent=intent,
                entities=[],
                confidence=0.1,
                requires_followup=True,
                suggested_actions=["Try rephrasing your question", "Ask for help"],
                external_service_used=None,
                processing_time=0.1
            )
    
    async def integrate_external_service(
        self, 
        service_params: ServiceParameters
    ) -> ServiceResult:
        """
        Integrate with external service to fulfill user request.
        
        Args:
            service_params: Service integration parameters
            
        Returns:
            Service integration result
        """
        try:
            self.logger.info(f"Integrating with external service: {service_params.service_type}")
            
            # This would integrate with actual external services
            # For now, return a mock successful result
            
            service_result = ServiceResult(
                service_type=service_params.service_type,
                success=True,
                data={
                    "status": "success",
                    "message": f"Successfully integrated with {service_params.service_type.value}",
                    "parameters": service_params.parameters
                },
                error_message=None,
                response_time=1.0
            )
            
            self.logger.info(f"External service integration completed: {service_params.service_type}")
            return service_result
            
        except Exception as e:
            self.logger.error(f"Error integrating external service: {e}")
            return ServiceResult(
                service_type=service_params.service_type,
                success=False,
                data={},
                error_message=str(e),
                response_time=0.1
            )
    
    async def format_cultural_response(
        self, 
        response: Response, 
        cultural_context: CulturalContext
    ) -> Response:
        """
        Format response with cultural appropriateness.
        
        Args:
            response: Base response to format
            cultural_context: Cultural context for formatting
            
        Returns:
            Culturally formatted response
        """
        try:
            self.logger.info("Formatting response with cultural context")
            
            formatted_response = response.copy()
            
            # Apply cultural formatting based on context
            cultural_pattern = self._get_cultural_pattern(cultural_context)
            
            if cultural_pattern:
                # Add cultural prefixes/suffixes
                formatted_text = self._apply_cultural_formatting(
                    response.text, 
                    cultural_pattern,
                    cultural_context
                )
                formatted_response.text = formatted_text
            
            # Adjust formality based on cultural context
            if hasattr(cultural_context, 'preferred_greetings') and cultural_context.preferred_greetings:
                formatted_response.text = self._adjust_formality(
                    formatted_response.text,
                    cultural_context.preferred_greetings[0]
                )
            
            self.logger.info("Cultural formatting applied successfully")
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting cultural response: {e}")
            return response
    
    async def _generate_response_text(
        self, 
        intent: Intent, 
        entities: List[Entity], 
        context: Optional[RegionalContextData]
    ) -> str:
        """Generate response text based on intent and entities."""
        try:
            intent_name = intent.name
            
            # Get base template
            templates = self._response_templates.get(intent_name, {})
            base_template = templates.get("default", "I understand your request.")
            
            # Customize based on entities and context
            if intent_name == IntentCategory.WEATHER_INQUIRY.value:
                city_entity = next((e for e in entities if e.type == EntityType.CITY.value), None)
                if city_entity:
                    return f"Let me check the weather for {city_entity.value}."
                elif context and context.weather_info:
                    if context.weather_info.is_monsoon_season:
                        return templates.get("monsoon", base_template)
                
            elif intent_name == IntentCategory.TRAIN_INQUIRY.value:
                station_entities = [e for e in entities if e.type == EntityType.STATION.value]
                if len(station_entities) >= 2:
                    return f"Let me check train information from {station_entities[0].value} to {station_entities[1].value}."
                elif station_entities:
                    return f"Checking train information for {station_entities[0].value}."
                
            elif intent_name == IntentCategory.FESTIVAL_INQUIRY.value:
                festival_entity = next((e for e in entities if e.type == EntityType.FESTIVAL.value), None)
                if festival_entity:
                    return f"Let me tell you about {festival_entity.value}."
                elif context and context.cultural_events:
                    upcoming_festival = context.cultural_events[0].name if context.cultural_events else None
                    if upcoming_festival:
                        return f"The next major festival is {upcoming_festival}."
                
            elif intent_name == IntentCategory.FOOD_ORDER.value:
                dish_entities = [e for e in entities if e.type == EntityType.DISH.value]
                if dish_entities:
                    dishes = ", ".join([e.value for e in dish_entities])
                    return f"I can help you order {dishes}."
                
            return base_template
            
        except Exception as e:
            self.logger.error(f"Error generating response text: {e}")
            return "I'm here to help you with your request."
    
    async def _requires_followup(self, intent: Intent, entities: List[Entity]) -> bool:
        """Determine if the intent requires followup questions."""
        followup_intents = [
            IntentCategory.TRAIN_INQUIRY.value,
            IntentCategory.FOOD_ORDER.value,
            IntentCategory.RIDE_BOOKING.value,
            IntentCategory.GOVERNMENT_SERVICE.value
        ]
        
        # Check if intent typically requires followup
        if intent.name in followup_intents:
            return True
        
        # Check if we have insufficient entities for complete response
        if intent.name == IntentCategory.TRAIN_INQUIRY.value:
            station_entities = [e for e in entities if e.type == EntityType.STATION.value]
            return len(station_entities) < 2
        
        return False
    
    async def _generate_suggested_actions(
        self, 
        intent: Intent, 
        entities: List[Entity], 
        context: Optional[RegionalContextData]
    ) -> List[str]:
        """Generate suggested actions based on intent and context."""
        suggestions = []
        
        if intent.name == IntentCategory.WEATHER_INQUIRY.value:
            suggestions = ["Check 7-day forecast", "Get weather alerts", "Check air quality"]
            
        elif intent.name == IntentCategory.TRAIN_INQUIRY.value:
            suggestions = ["Book train ticket", "Check PNR status", "Find nearby stations"]
            
        elif intent.name == IntentCategory.FESTIVAL_INQUIRY.value:
            suggestions = ["Get festival calendar", "Learn about traditions", "Find local celebrations"]
            
        elif intent.name == IntentCategory.FOOD_ORDER.value:
            suggestions = ["Browse restaurants", "Check food delivery", "See local specialties"]
            
        elif intent.name == IntentCategory.HELP.value:
            suggestions = ["Ask about weather", "Check train schedules", "Order food", "Get local information"]
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _determine_external_service(self, intent: Intent, entities: List[Entity]) -> Optional[str]:
        """Determine which external service to use based on intent."""
        service_mapping = {
            IntentCategory.WEATHER_INQUIRY.value: "weather",
            IntentCategory.TRAIN_INQUIRY.value: "indian_railways",
            IntentCategory.FOOD_ORDER.value: "food_delivery",
            IntentCategory.RIDE_BOOKING.value: "ride_sharing",
            IntentCategory.PAYMENT_UPI.value: "upi_payment",
            IntentCategory.CRICKET_SCORES.value: "cricket_scores",
            IntentCategory.BOLLYWOOD_NEWS.value: "bollywood_news",
            IntentCategory.GOVERNMENT_SERVICE.value: "government_service"
        }
        
        return service_mapping.get(intent.name)
    
    async def _get_response_suggestions(
        self, 
        intent: Intent, 
        entities: List[Entity], 
        cultural_context: Dict[str, Any]
    ) -> List[str]:
        """Get response suggestions based on processing results."""
        suggestions = []
        
        # Add suggestions based on cultural context
        if cultural_context.get("formality_level") == "high":
            suggestions.append("Use formal language in response")
        elif cultural_context.get("formality_level") == "low":
            suggestions.append("Use casual, friendly tone")
        
        # Add suggestions based on regional influence
        regional_influence = cultural_context.get("regional_influence")
        if regional_influence:
            suggestions.append(f"Consider {regional_influence} cultural context")
        
        # Add suggestions based on intent
        if intent.confidence < 0.7:
            suggestions.append("Ask for clarification due to low confidence")
        
        return suggestions
    
    def _get_cultural_pattern(self, cultural_context: CulturalContext) -> Optional[Dict[str, Any]]:
        """Get appropriate cultural pattern based on context."""
        # This would analyze the cultural context and return appropriate pattern
        # For now, return a default pattern
        return self._cultural_response_patterns.get("respectful_formal")
    
    def _apply_cultural_formatting(
        self, 
        text: str, 
        pattern: Dict[str, Any], 
        cultural_context: CulturalContext
    ) -> str:
        """Apply cultural formatting to response text."""
        try:
            formatted_text = text
            
            # Add cultural prefixes if appropriate
            if pattern.get("prefixes") and not any(prefix.lower() in text.lower() for prefix in pattern["prefixes"]):
                prefix = pattern["prefixes"][0]
                formatted_text = f"{prefix}, {formatted_text}"
            
            # Add cultural suffixes if appropriate
            if pattern.get("suffixes") and pattern["tone"] == "respectful":
                suffix = pattern["suffixes"][0]
                if not formatted_text.endswith(suffix):
                    formatted_text = f"{formatted_text} {suffix}"
            
            return formatted_text
            
        except Exception as e:
            self.logger.error(f"Error applying cultural formatting: {e}")
            return text
    
    def _adjust_formality(self, text: str, greeting_style: str) -> str:
        """Adjust formality level based on greeting style."""
        try:
            if greeting_style in ["namaste", "namaskar"]:
                # More formal approach
                text = text.replace("Hey", "Hello").replace("What's up", "How may I help you")
            elif greeting_style in ["hi", "hello"]:
                # Casual approach
                text = text.replace("How may I assist", "How can I help")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error adjusting formality: {e}")
            return text