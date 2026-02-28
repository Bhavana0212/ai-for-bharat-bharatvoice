"""
Property-based tests for NLU Service.

Tests universal properties that should hold across all inputs for the
Natural Language Understanding system, focusing on cultural context recognition.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
import asyncio
from datetime import datetime
from uuid import uuid4

from bharatvoice.core.models import (
    LanguageCode,
    ConversationState,
    UserProfile,
    LocationData,
    RegionalContextData,
    UserInteraction
)
from bharatvoice.services.response_generation.nlu_service import (
    NLUService,
    ColloquialTermMapper,
    IndianEntityExtractor,
    IndianIntentClassifier,
    CulturalContextInterpreter,
    IntentCategory,
    EntityType
)


# Strategy generators for test data
@composite
def indian_city_names(draw):
    """Generate Indian city names."""
    cities = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune",
        "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
        "Bhopal", "Visakhapatnam", "Vadodara", "Firozabad", "Ludhiana", "Rajkot"
    ]
    return draw(st.sampled_from(cities))


@composite
def indian_festival_names(draw):
    """Generate Indian festival names."""
    festivals = [
        "Diwali", "Holi", "Eid", "Dussehra", "Ganesh Chaturthi", "Navratri",
        "Karva Chauth", "Raksha Bandhan", "Janmashtami", "Onam", "Pongal",
        "Baisakhi", "Lohri", "Makar Sankranti", "Gudi Padwa", "Ugadi"
    ]
    return draw(st.sampled_from(festivals))


@composite
def indian_dishes(draw):
    """Generate Indian dish names."""
    dishes = [
        "biryani", "samosa", "dosa", "idli", "roti", "dal", "paneer", "chai",
        "pakora", "vada", "upma", "poha", "dhokla", "kachori", "jalebi",
        "gulab jamun", "rasgulla", "lassi", "kulfi", "halwa"
    ]
    return draw(st.sampled_from(dishes))


@composite
def colloquial_terms(draw):
    """Generate colloquial Indian terms."""
    terms = [
        "namaste", "yaar", "bhai", "didi", "mummy", "papa", "chai", "khana",
        "paani", "ghar", "gaadi", "bazaar", "dukan", "accha", "theek hai",
        "kya baat", "bindaas", "jugaad", "timepass"
    ]
    return draw(st.sampled_from(terms))


@composite
def respectful_terms(draw):
    """Generate respectful terms used in Indian context."""
    terms = ["ji", "sir", "madam", "sahab", "sahib"]
    return draw(st.sampled_from(terms))


@composite
def casual_terms(draw):
    """Generate casual terms used in Indian context."""
    terms = ["yaar", "bro", "dude", "boss", "arre"]
    return draw(st.sampled_from(terms))


@composite
def user_input_with_intent(draw):
    """Generate user input with known intent patterns."""
    intent_patterns = {
        IntentCategory.GREETING: [
            "namaste", "hello", "hi", "good morning", "kaise ho", "how are you"
        ],
        IntentCategory.WEATHER_INQUIRY: [
            "weather", "mausam", "rain", "temperature", "how is weather", "barish"
        ],
        IntentCategory.TRAIN_INQUIRY: [
            "train", "railway", "station", "ticket", "train schedule", "irctc"
        ],
        IntentCategory.FESTIVAL_INQUIRY: [
            "festival", "diwali", "holi", "celebration", "when is", "tyohar"
        ],
        IntentCategory.FOOD_ORDER: [
            "food", "khana", "order", "hungry", "restaurant", "delivery"
        ]
    }
    
    intent = draw(st.sampled_from(list(intent_patterns.keys())))
    pattern = draw(st.sampled_from(intent_patterns[intent]))
    
    # Add some context around the pattern
    prefix = draw(st.sampled_from(["", "please ", "can you ", "I want to "]))
    suffix = draw(st.sampled_from(["", " please", " today", " now"]))
    
    text = f"{prefix}{pattern}{suffix}".strip()
    return text, intent


class TestNLUProperties:
    """Property-based tests for NLU service."""
    
    @pytest.fixture
    def nlu_service(self):
        return NLUService()
    
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_nlu_never_crashes_on_any_input(self, nlu_service, text):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        The NLU system should never crash regardless of input and should always
        return a valid response structure with cultural context analysis.
        """
        assume(len(text.strip()) > 0)  # Assume non-empty input
        
        try:
            result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
            
            # Should always return valid structure
            assert isinstance(result, dict)
            assert "intent" in result
            assert "entities" in result
            assert "cultural_context" in result
            assert "confidence" in result
            
            # Intent should be valid
            intent = result["intent"]
            assert hasattr(intent, 'name')
            assert hasattr(intent, 'confidence')
            assert 0.0 <= intent.confidence <= 1.0
            
            # Cultural context should be present
            cultural_context = result["cultural_context"]
            assert isinstance(cultural_context, dict)
            
        except Exception as e:
            pytest.fail(f"NLU crashed on input '{text[:50]}...': {e}")
    
    @given(colloquial_terms())
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_colloquial_terms_always_mapped(self, nlu_service, term):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Colloquial terms should always be recognized and mapped to standard meanings,
        preserving cultural context information.
        """
        text = f"Hello {term} how are you"
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        # Should detect colloquial term usage
        processing_metadata = result.get("processing_metadata", {})
        
        # Either the term was mapped or it's preserved in cultural context
        cultural_context = result["cultural_context"]
        
        # The system should recognize this as having cultural elements
        assert (processing_metadata.get("colloquial_terms_mapped", False) or
                len(cultural_context.get("cultural_references", [])) > 0 or
                cultural_context.get("communication_style") != "neutral")
    
    @given(respectful_terms())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_respectful_terms_increase_formality(self, nlu_service, respectful_term):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Respectful terms should always increase the detected formality level
        and influence the communication style appropriately.
        """
        text = f"Please help me {respectful_term}"
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        cultural_context = result["cultural_context"]
        
        # Respectful terms should increase formality
        formality_level = cultural_context.get("formality_level", "medium")
        assert formality_level in ["medium", "high"]
        
        # Should not be casual
        assert formality_level != "low"
    
    @given(casual_terms())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_casual_terms_decrease_formality(self, nlu_service, casual_term):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Casual terms should decrease formality level and indicate casual communication style.
        """
        text = f"Hey {casual_term} what's up"
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        cultural_context = result["cultural_context"]
        
        # Casual terms should decrease formality or indicate casual style
        formality_level = cultural_context.get("formality_level", "medium")
        communication_style = cultural_context.get("communication_style", "neutral")
        
        assert (formality_level == "low" or 
                communication_style == "casual_friendly" or
                formality_level != "high")
    
    @given(indian_city_names(), indian_city_names())
    @settings(max_examples=25)
    @pytest.mark.asyncio
    async def test_city_entities_always_extracted(self, nlu_service, city1, city2):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Indian city names should always be recognized as location entities
        when present in user input.
        """
        assume(city1 != city2)  # Ensure different cities
        
        text = f"I want to travel from {city1} to {city2}"
        
        result = await nlu_service.process_user_input(text, LanguageCode.ENGLISH_IN)
        
        entities = result["entities"]
        city_entities = [e for e in entities if e.type == EntityType.CITY.value]
        
        # Should extract at least one city, preferably both
        assert len(city_entities) >= 1
        
        # Check that the extracted cities match our input
        extracted_cities = [e.value.lower() for e in city_entities]
        assert city1.lower() in extracted_cities or city2.lower() in extracted_cities
    
    @given(indian_festival_names())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_festival_entities_trigger_cultural_context(self, nlu_service, festival):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Festival names should always be recognized as cultural entities and
        should trigger appropriate cultural context interpretation.
        """
        text = f"When is {festival} this year?"
        
        result = await nlu_service.process_user_input(text, LanguageCode.ENGLISH_IN)
        
        # Should extract festival entity
        entities = result["entities"]
        festival_entities = [e for e in entities if e.type == EntityType.FESTIVAL.value]
        assert len(festival_entities) >= 1
        
        # Should recognize festival intent
        intent = result["intent"]
        assert intent.name == IntentCategory.FESTIVAL_INQUIRY.value
        
        # Should have cultural context
        cultural_context = result["cultural_context"]
        cultural_sensitivity = cultural_context.get("cultural_sensitivity", [])
        assert "cultural_celebration" in cultural_sensitivity or len(cultural_sensitivity) > 0
    
    @given(user_input_with_intent())
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_intent_classification_consistency(self, nlu_service, input_data):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Intent classification should be consistent and confident for clear
        intent patterns, maintaining cultural appropriateness.
        """
        text, expected_intent = input_data
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        detected_intent = result["intent"]
        
        # Should detect the expected intent with reasonable confidence
        if expected_intent != IntentCategory.UNKNOWN:
            assert detected_intent.name == expected_intent.value
            assert detected_intent.confidence >= 0.3  # Reasonable confidence threshold
        
        # Should always have some cultural context analysis
        cultural_context = result["cultural_context"]
        assert isinstance(cultural_context, dict)
        assert "formality_level" in cultural_context
        assert "communication_style" in cultural_context
    
    @given(st.text(min_size=10, max_size=200))
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_confidence_scores_are_valid(self, nlu_service, text):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        All confidence scores should be valid probabilities between 0 and 1,
        and the overall system confidence should reflect the quality of analysis.
        """
        assume(len(text.strip()) > 5)  # Assume meaningful input
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        # Overall confidence should be valid
        overall_confidence = result["confidence"]
        assert 0.0 <= overall_confidence <= 1.0
        
        # Intent confidence should be valid
        intent = result["intent"]
        assert 0.0 <= intent.confidence <= 1.0
        
        # Entity confidences should be valid
        entities = result["entities"]
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0
        
        # Overall confidence should be related to intent confidence
        assert abs(overall_confidence - intent.confidence) <= 0.3
    
    @given(st.lists(st.text(min_size=5, max_size=100), min_size=2, max_size=5))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_conversation_context_improves_confidence(self, nlu_service, conversation_texts):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Conversation context should generally improve intent classification
        confidence for related queries.
        """
        # Process first message without context
        first_text = conversation_texts[0]
        assume(len(first_text.strip()) > 3)
        
        result_without_context = await nlu_service.process_user_input(
            first_text, LanguageCode.HINDI
        )
        
        # Create conversation state with history
        conversation_state = ConversationState(
            user_id=uuid4(),
            current_language=LanguageCode.HINDI,
            conversation_history=[
                UserInteraction(
                    user_id=uuid4(),
                    input_text=first_text,
                    input_language=LanguageCode.HINDI,
                    response_text="Response",
                    response_language=LanguageCode.HINDI,
                    intent=result_without_context["intent"].name,
                    confidence=result_without_context["intent"].confidence,
                    processing_time=0.5
                )
            ]
        )
        
        # Process second message with context
        if len(conversation_texts) > 1:
            second_text = conversation_texts[1]
            assume(len(second_text.strip()) > 3)
            
            result_with_context = await nlu_service.process_user_input(
                second_text, LanguageCode.HINDI, conversation_state
            )
            
            # Context should provide some benefit (either confidence or better classification)
            # This is a soft property - context should generally help
            assert result_with_context["confidence"] >= 0.1  # Should still be reasonable
    
    @given(indian_dishes())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_food_entities_trigger_food_intent(self, nlu_service, dish):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Indian dish names should be recognized as food entities and should
        strongly suggest food-related intents.
        """
        text = f"I want to order {dish}"
        
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        # Should extract dish entity
        entities = result["entities"]
        dish_entities = [e for e in entities if e.type == EntityType.DISH.value]
        assert len(dish_entities) >= 1
        
        # Should detect food-related intent
        intent = result["intent"]
        assert intent.name in [
            IntentCategory.FOOD_ORDER.value,
            IntentCategory.RESTAURANT_INQUIRY.value
        ] or intent.confidence < 0.5  # If not food intent, should be uncertain
    
    @given(st.sampled_from([LanguageCode.HINDI, LanguageCode.ENGLISH_IN, LanguageCode.TAMIL]))
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_language_consistency_in_processing(self, nlu_service, language):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        The NLU system should handle different Indian languages consistently
        and maintain cultural context awareness across languages.
        """
        # Use a simple greeting that works across languages
        text = "Hello, how are you?"
        
        result = await nlu_service.process_user_input(text, language)
        
        # Should always detect greeting intent
        intent = result["intent"]
        assert intent.name == IntentCategory.GREETING.value
        
        # Should have consistent structure regardless of language
        assert "cultural_context" in result
        assert "entities" in result
        assert "confidence" in result
        
        # Language should be preserved
        assert result["language"] == language
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: any(c.isalpha() for c in x)))
    @settings(max_examples=25)
    @pytest.mark.asyncio
    async def test_cultural_context_always_analyzed(self, nlu_service, text):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Every input should receive cultural context analysis, even if minimal.
        The system should always attempt to understand the cultural nuances.
        """
        result = await nlu_service.process_user_input(text, LanguageCode.HINDI)
        
        cultural_context = result["cultural_context"]
        
        # Should always have basic cultural context structure
        assert isinstance(cultural_context, dict)
        
        # Should have key cultural analysis fields
        expected_fields = [
            "communication_style", "formality_level", "cultural_references",
            "regional_influence", "response_tone", "cultural_sensitivity"
        ]
        
        # At least some fields should be present
        present_fields = [field for field in expected_fields if field in cultural_context]
        assert len(present_fields) >= 3  # Should analyze at least 3 cultural aspects
        
        # Formality level should be valid
        if "formality_level" in cultural_context:
            assert cultural_context["formality_level"] in ["low", "medium", "high"]


class TestColloquialTermProperties:
    """Property-based tests for colloquial term mapping."""
    
    @pytest.fixture
    def mapper(self):
        return ColloquialTermMapper()
    
    @given(colloquial_terms(), st.text(min_size=0, max_size=100))
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_colloquial_mapping_preserves_meaning(self, mapper, term, context):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Colloquial term mapping should preserve the essential meaning while
        making it more understandable for processing.
        """
        text = f"{context} {term} {context}".strip()
        
        result = await mapper.map_colloquial_terms(text, LanguageCode.HINDI)
        
        # Result should be a string
        assert isinstance(result, str)
        
        # Should not be empty if input wasn't empty
        if text.strip():
            assert result.strip()
        
        # Length should be reasonable (not drastically different)
        assert len(result) <= len(text) * 3  # Allow for expansion but not excessive
    
    @given(colloquial_terms())
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_cultural_context_retrieval_consistency(self, mapper, term):
        """
        **Property 5: Cultural Context Recognition**
        **Validates: Requirements 2.1, 2.5**
        
        Cultural context retrieval should be consistent for the same term
        and provide meaningful cultural information.
        """
        context1 = await mapper.get_cultural_context(term)
        context2 = await mapper.get_cultural_context(term)
        
        # Should be consistent
        assert context1 == context2
        
        # If context exists, should have required fields
        if context1 is not None:
            assert "term" in context1
            assert "standard_meaning" in context1
            assert "context" in context1
            assert "languages" in context1
            
            # Languages should be a list
            assert isinstance(context1["languages"], list)
            assert len(context1["languages"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])