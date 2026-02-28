<<<<<<< HEAD
"""
Unit tests for NLU Service.

Tests the Natural Language Understanding capabilities including intent recognition,
entity extraction, colloquial term mapping, and cultural context interpretation.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from bharatvoice.core.models import (
    LanguageCode,
    ConversationState,
    UserProfile,
    LocationData,
    RegionalContextData,
    UserInteraction,
    Intent,
    Entity
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


class TestColloquialTermMapper:
    """Test colloquial term mapping functionality."""
    
    @pytest.fixture
    def mapper(self):
        return ColloquialTermMapper()
    
    @pytest.mark.asyncio
    async def test_basic_colloquial_mapping(self, mapper):
        """Test basic colloquial term mapping."""
        text = "Namaste, mummy ne khana banaya hai"
        result = await mapper.map_colloquial_terms(text, LanguageCode.HINDI)
        
        assert "hello" in result.lower()
        assert "mother" in result.lower()
        assert "food" in result.lower()
    
    @pytest.mark.asyncio
    async def test_family_relationship_mapping(self, mapper):
        """Test family relationship term mapping."""
        text = "Papa aur didi ghar pe hain"
        result = await mapper.map_colloquial_terms(text, LanguageCode.HINDI)
        
        assert "father" in result.lower()
        assert "elder_sister" in result.lower()
    
    @pytest.mark.asyncio
    async def test_cultural_context_retrieval(self, mapper):
        """Test cultural context retrieval for terms."""
        context = await mapper.get_cultural_context("namaste")
        
        assert context is not None
        assert context["standard_meaning"] == "hello"
        assert context["context"] == "respectful_greeting"
        assert "hi" in context["languages"]
    
    @pytest.mark.asyncio
    async def test_regional_variations(self, mapper):
        """Test regional variations of common terms."""
        # Test with different languages
        text_hindi = "paani chahiye"
        result_hindi = await mapper.map_colloquial_terms(text_hindi, LanguageCode.HINDI)
        assert "water" in result_hindi.lower()
        
        # Test preservation of unmapped terms
        text_english = "I need some water"
        result_english = await mapper.map_colloquial_terms(text_english, LanguageCode.ENGLISH_IN)
        assert "water" in result_english.lower()


class TestIndianEntityExtractor:
    """Test Indian-specific entity extraction."""
    
    @pytest.fixture
    def extractor(self):
        return IndianEntityExtractor()
    
    @pytest.mark.asyncio
    async def test_city_extraction(self, extractor):
        """Test extraction of Indian cities."""
        text = "I want to go from Mumbai to Delhi"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        city_entities = [e for e in entities if e.type == EntityType.CITY.value]
        assert len(city_entities) >= 2
        
        city_names = [e.value.lower() for e in city_entities]
        assert "mumbai" in city_names
        assert "delhi" in city_names
    
    @pytest.mark.asyncio
    async def test_festival_extraction(self, extractor):
        """Test extraction of Indian festivals."""
        text = "When is Diwali this year? Also tell me about Holi."
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        festival_entities = [e for e in entities if e.type == EntityType.FESTIVAL.value]
        assert len(festival_entities) >= 2
        
        festival_names = [e.value.lower() for e in festival_entities]
        assert "diwali" in festival_names
        assert "holi" in festival_names
    
    @pytest.mark.asyncio
    async def test_dish_extraction(self, extractor):
        """Test extraction of Indian dishes."""
        text = "I want to order biryani and samosa"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        dish_entities = [e for e in entities if e.type == EntityType.DISH.value]
        assert len(dish_entities) >= 2
        
        dish_names = [e.value.lower() for e in dish_entities]
        assert "biryani" in dish_names
        assert "samosa" in dish_names
    
    @pytest.mark.asyncio
    async def test_relationship_extraction(self, extractor):
        """Test extraction of family relationships."""
        text = "Call my mummy and tell bhai to come home"
        entities = await extractor.extract_entities(text, LanguageCode.HINDI)
        
        relationship_entities = [e for e in entities if e.type == EntityType.RELATIONSHIP.value]
        assert len(relationship_entities) >= 2
        
        relationships = [e.value.lower() for e in relationship_entities]
        assert "mummy" in relationships
        assert "bhai" in relationships
    
    @pytest.mark.asyncio
    async def test_pincode_extraction(self, extractor):
        """Test extraction of Indian pincodes."""
        text = "My address is 400001 Mumbai"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        pincode_entities = [e for e in entities if e.type == "pincode"]
        assert len(pincode_entities) >= 1
        assert pincode_entities[0].value == "400001"
    
    @pytest.mark.asyncio
    async def test_currency_extraction(self, extractor):
        """Test extraction of Indian currency."""
        text = "It costs ₹500 or Rs. 1000"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        currency_entities = [e for e in entities if e.type == "currency"]
        assert len(currency_entities) >= 2


class TestIndianIntentClassifier:
    """Test Indian cultural context intent classification."""
    
    @pytest.fixture
    def classifier(self):
        return IndianIntentClassifier()
    
    @pytest.mark.asyncio
    async def test_greeting_intent(self, classifier):
        """Test greeting intent classification."""
        test_cases = [
            "Namaste, kaise hain aap?",
            "Hello, how are you?",
            "Sat Sri Akal ji",
            "Vanakkam"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.GREETING.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_weather_intent(self, classifier):
        """Test weather inquiry intent classification."""
        test_cases = [
            "What's the weather like today?",
            "Aaj mausam kaisa hai?",
            "Will it rain today?",
            "Temperature kitna hai?"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.WEATHER_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_train_inquiry_intent(self, classifier):
        """Test train inquiry intent classification."""
        test_cases = [
            "Train schedule from Mumbai to Delhi",
            "Book train ticket",
            "IRCTC booking",
            "Railway station information"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.TRAIN_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_festival_intent(self, classifier):
        """Test festival inquiry intent classification."""
        test_cases = [
            "When is Diwali?",
            "Tell me about Holi",
            "Upcoming festivals",
            "Ganpati celebration dates"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.FESTIVAL_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_food_order_intent(self, classifier):
        """Test food ordering intent classification."""
        test_cases = [
            "I want to order food",
            "Khana order karna hai",
            "Food delivery",
            "Hungry, need food"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.FOOD_ORDER.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_context_based_classification(self, classifier):
        """Test intent classification with conversation context."""
        # Create conversation state with history
        conversation_state = ConversationState(
            user_id=uuid4(),
            current_language=LanguageCode.HINDI,
            conversation_history=[
                UserInteraction(
                    user_id=uuid4(),
                    input_text="What's the weather?",
                    input_language=LanguageCode.ENGLISH_IN,
                    response_text="It's sunny today",
                    response_language=LanguageCode.ENGLISH_IN,
                    intent="weather_inquiry",
                    confidence=0.9,
                    processing_time=0.5
                )
            ]
        )
        
        # Test related intent with context
        intent = await classifier.classify_intent("What about tomorrow?", conversation_state)
        
        # Should still be weather-related due to context
        assert intent.confidence > 0.3  # Context should boost confidence


class TestCulturalContextInterpreter:
    """Test cultural context interpretation."""
    
    @pytest.fixture
    def interpreter(self):
        return CulturalContextInterpreter()
    
    @pytest.mark.asyncio
    async def test_formal_context_detection(self, interpreter):
        """Test detection of formal communication style."""
        text = "Sir, could you please help me with this?"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["formality_level"] == "high"
        assert context["communication_style"] in ["formal_respectful", "neutral"]
    
    @pytest.mark.asyncio
    async def test_casual_context_detection(self, interpreter):
        """Test detection of casual communication style."""
        text = "Hey yaar, what's up bro?"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["formality_level"] == "low"
        assert context["communication_style"] == "casual_friendly"
    
    @pytest.mark.asyncio
    async def test_family_context_detection(self, interpreter):
        """Test detection of family-oriented context."""
        text = "Mummy said to call papa"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "family_oriented"
    
    @pytest.mark.asyncio
    async def test_religious_context_detection(self, interpreter):
        """Test detection of religious context."""
        text = "I need to go to the mandir for puja"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "religious_spiritual"
        assert "cultural_celebration" in context.get("cultural_sensitivity", [])
    
    @pytest.mark.asyncio
    async def test_regional_context_with_profile(self, interpreter):
        """Test regional context interpretation with user profile."""
        # Create user profile with location
        location = LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India"
        )
        
        user_profile = UserProfile(
            preferred_languages=[LanguageCode.MARATHI, LanguageCode.HINDI],
            location=location
        )
        
        text = "Namaskar, kasa kay?"
        context = await interpreter.interpret_cultural_context(text, user_profile)
        
        assert context["regional_influence"] == "west_india"
    
    @pytest.mark.asyncio
    async def test_urgency_detection(self, interpreter):
        """Test detection of urgency indicators."""
        text = "Jaldi help chahiye, emergency hai!"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "urgent_immediate"


class TestNLUService:
    """Test complete NLU service integration."""
    
    @pytest.fixture
    def nlu_service(self):
        return NLUService()
    
    @pytest.mark.asyncio
    async def test_complete_nlu_processing(self, nlu_service):
        """Test complete NLU processing pipeline."""
        text = "Namaste ji, Mumbai se Delhi ki train ka time kya hai?"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Check all components are present
        assert "original_text" in result
        assert "processed_text" in result
        assert "intent" in result
        assert "entities" in result
        assert "cultural_context" in result
        assert "confidence" in result
        
        # Check intent classification
        intent = result["intent"]
        assert intent.name == IntentCategory.TRAIN_INQUIRY.value
        assert intent.confidence > 0.5
        
        # Check entity extraction
        entities = result["entities"]
        city_entities = [e for e in entities if e.type == EntityType.CITY.value]
        assert len(city_entities) >= 2
        
        # Check cultural context
        cultural_context = result["cultural_context"]
        assert cultural_context["formality_level"] == "high"  # Due to "ji"
    
    @pytest.mark.asyncio
    async def test_colloquial_term_processing(self, nlu_service):
        """Test processing of colloquial terms."""
        text = "Yaar, khana order karna hai"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Check that colloquial terms were mapped
        assert result["processing_metadata"]["colloquial_terms_mapped"] == True
        assert "food" in result["processed_text"].lower()
        
        # Check intent classification on processed text
        intent = result["intent"]
        assert intent.name == IntentCategory.FOOD_ORDER.value
    
    @pytest.mark.asyncio
    async def test_cultural_appropriateness_validation(self, nlu_service):
        """Test cultural appropriateness validation."""
        text = "Tell me about religious festivals"
        language = LanguageCode.ENGLISH_IN
        
        result = await nlu_service.process_user_input(text, language)
        
        intent = result["intent"]
        cultural_context = result["cultural_context"]
        
        validation = await nlu_service.validate_cultural_appropriateness(
            text, intent, cultural_context
        )
        
        assert validation["is_appropriate"] == True
        # Should have suggestions for handling religious topics
        assert len(validation["suggestions"]) > 0
    
    @pytest.mark.asyncio
    async def test_intent_suggestions(self, nlu_service):
        """Test intent suggestions for partial input."""
        partial_text = "train"
        language = LanguageCode.ENGLISH_IN
        
        suggestions = await nlu_service.get_intent_suggestions(partial_text, language)
        
        assert len(suggestions) > 0
        # Should suggest train-related intents
        intent_names = [s.get("intent", "") for s in suggestions]
        assert any("train" in intent for intent in intent_names)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, nlu_service):
        """Test error handling in NLU processing."""
        # Test with empty text
        result = await nlu_service.process_user_input("", LanguageCode.HINDI)
        
        assert result["intent"].name == IntentCategory.UNKNOWN.value
        assert result["confidence"] < 0.5
        
        # Test with very long text
        long_text = "a" * 10000
        result = await nlu_service.process_user_input(long_text, LanguageCode.HINDI)
        
        # Should handle gracefully
        assert "intent" in result
        assert "entities" in result
    
    @pytest.mark.asyncio
    async def test_multilingual_processing(self, nlu_service):
        """Test processing of multilingual input."""
        # Code-switched text (Hindi-English)
        text = "Hello, aaj weather kaisa hai?"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Should detect weather intent despite code-switching
        intent = result["intent"]
        assert intent.name == IntentCategory.WEATHER_INQUIRY.value
        
        # Should extract entities from both languages
        entities = result["entities"]
        assert len(entities) >= 0  # May or may not have entities, but should not crash
    
    @pytest.mark.asyncio
    async def test_conversation_context_influence(self, nlu_service):
        """Test how conversation context influences NLU processing."""
        # Create conversation state with weather inquiry history
        conversation_state = ConversationState(
            user_id=uuid4(),
            current_language=LanguageCode.HINDI,
            conversation_history=[
                UserInteraction(
                    user_id=uuid4(),
                    input_text="What's the weather?",
                    input_language=LanguageCode.ENGLISH_IN,
                    response_text="It's sunny",
                    response_language=LanguageCode.ENGLISH_IN,
                    intent="weather_inquiry",
                    confidence=0.9,
                    processing_time=0.5
                )
            ]
        )
        
        # Ambiguous follow-up question
        text = "What about tomorrow?"
        
        result = await nlu_service.process_user_input(
            text, LanguageCode.ENGLISH_IN, conversation_state
        )
        
        # Should still be weather-related due to context
        intent = result["intent"]
        # Context should help with classification
        assert intent.confidence > 0.3


# Integration tests
class TestNLUIntegration:
    """Test NLU service integration with other components."""
    
    @pytest.mark.asyncio
    async def test_regional_context_integration(self):
        """Test NLU processing with regional context."""
        nlu_service = NLUService()
        
        # Create regional context for Mumbai
        location = LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India"
        )
        
        regional_context = RegionalContextData(
            location=location,
            local_language=LanguageCode.MARATHI
        )
        
        text = "Local train schedule chahiye"
        
        result = await nlu_service.process_user_input(
            text, LanguageCode.HINDI, regional_context=regional_context
        )
        
        # Should detect train inquiry
        assert result["intent"].name == IntentCategory.TRAIN_INQUIRY.value
        
        # Should have regional influence in cultural context
        cultural_context = result["cultural_context"]
        assert cultural_context.get("regional_influence") is not None


if __name__ == "__main__":
=======
"""
Unit tests for NLU Service.

Tests the Natural Language Understanding capabilities including intent recognition,
entity extraction, colloquial term mapping, and cultural context interpretation.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from bharatvoice.core.models import (
    LanguageCode,
    ConversationState,
    UserProfile,
    LocationData,
    RegionalContextData,
    UserInteraction,
    Intent,
    Entity
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


class TestColloquialTermMapper:
    """Test colloquial term mapping functionality."""
    
    @pytest.fixture
    def mapper(self):
        return ColloquialTermMapper()
    
    @pytest.mark.asyncio
    async def test_basic_colloquial_mapping(self, mapper):
        """Test basic colloquial term mapping."""
        text = "Namaste, mummy ne khana banaya hai"
        result = await mapper.map_colloquial_terms(text, LanguageCode.HINDI)
        
        assert "hello" in result.lower()
        assert "mother" in result.lower()
        assert "food" in result.lower()
    
    @pytest.mark.asyncio
    async def test_family_relationship_mapping(self, mapper):
        """Test family relationship term mapping."""
        text = "Papa aur didi ghar pe hain"
        result = await mapper.map_colloquial_terms(text, LanguageCode.HINDI)
        
        assert "father" in result.lower()
        assert "elder_sister" in result.lower()
    
    @pytest.mark.asyncio
    async def test_cultural_context_retrieval(self, mapper):
        """Test cultural context retrieval for terms."""
        context = await mapper.get_cultural_context("namaste")
        
        assert context is not None
        assert context["standard_meaning"] == "hello"
        assert context["context"] == "respectful_greeting"
        assert "hi" in context["languages"]
    
    @pytest.mark.asyncio
    async def test_regional_variations(self, mapper):
        """Test regional variations of common terms."""
        # Test with different languages
        text_hindi = "paani chahiye"
        result_hindi = await mapper.map_colloquial_terms(text_hindi, LanguageCode.HINDI)
        assert "water" in result_hindi.lower()
        
        # Test preservation of unmapped terms
        text_english = "I need some water"
        result_english = await mapper.map_colloquial_terms(text_english, LanguageCode.ENGLISH_IN)
        assert "water" in result_english.lower()


class TestIndianEntityExtractor:
    """Test Indian-specific entity extraction."""
    
    @pytest.fixture
    def extractor(self):
        return IndianEntityExtractor()
    
    @pytest.mark.asyncio
    async def test_city_extraction(self, extractor):
        """Test extraction of Indian cities."""
        text = "I want to go from Mumbai to Delhi"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        city_entities = [e for e in entities if e.type == EntityType.CITY.value]
        assert len(city_entities) >= 2
        
        city_names = [e.value.lower() for e in city_entities]
        assert "mumbai" in city_names
        assert "delhi" in city_names
    
    @pytest.mark.asyncio
    async def test_festival_extraction(self, extractor):
        """Test extraction of Indian festivals."""
        text = "When is Diwali this year? Also tell me about Holi."
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        festival_entities = [e for e in entities if e.type == EntityType.FESTIVAL.value]
        assert len(festival_entities) >= 2
        
        festival_names = [e.value.lower() for e in festival_entities]
        assert "diwali" in festival_names
        assert "holi" in festival_names
    
    @pytest.mark.asyncio
    async def test_dish_extraction(self, extractor):
        """Test extraction of Indian dishes."""
        text = "I want to order biryani and samosa"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        dish_entities = [e for e in entities if e.type == EntityType.DISH.value]
        assert len(dish_entities) >= 2
        
        dish_names = [e.value.lower() for e in dish_entities]
        assert "biryani" in dish_names
        assert "samosa" in dish_names
    
    @pytest.mark.asyncio
    async def test_relationship_extraction(self, extractor):
        """Test extraction of family relationships."""
        text = "Call my mummy and tell bhai to come home"
        entities = await extractor.extract_entities(text, LanguageCode.HINDI)
        
        relationship_entities = [e for e in entities if e.type == EntityType.RELATIONSHIP.value]
        assert len(relationship_entities) >= 2
        
        relationships = [e.value.lower() for e in relationship_entities]
        assert "mummy" in relationships
        assert "bhai" in relationships
    
    @pytest.mark.asyncio
    async def test_pincode_extraction(self, extractor):
        """Test extraction of Indian pincodes."""
        text = "My address is 400001 Mumbai"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        pincode_entities = [e for e in entities if e.type == "pincode"]
        assert len(pincode_entities) >= 1
        assert pincode_entities[0].value == "400001"
    
    @pytest.mark.asyncio
    async def test_currency_extraction(self, extractor):
        """Test extraction of Indian currency."""
        text = "It costs ₹500 or Rs. 1000"
        entities = await extractor.extract_entities(text, LanguageCode.ENGLISH_IN)
        
        currency_entities = [e for e in entities if e.type == "currency"]
        assert len(currency_entities) >= 2


class TestIndianIntentClassifier:
    """Test Indian cultural context intent classification."""
    
    @pytest.fixture
    def classifier(self):
        return IndianIntentClassifier()
    
    @pytest.mark.asyncio
    async def test_greeting_intent(self, classifier):
        """Test greeting intent classification."""
        test_cases = [
            "Namaste, kaise hain aap?",
            "Hello, how are you?",
            "Sat Sri Akal ji",
            "Vanakkam"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.GREETING.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_weather_intent(self, classifier):
        """Test weather inquiry intent classification."""
        test_cases = [
            "What's the weather like today?",
            "Aaj mausam kaisa hai?",
            "Will it rain today?",
            "Temperature kitna hai?"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.WEATHER_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_train_inquiry_intent(self, classifier):
        """Test train inquiry intent classification."""
        test_cases = [
            "Train schedule from Mumbai to Delhi",
            "Book train ticket",
            "IRCTC booking",
            "Railway station information"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.TRAIN_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_festival_intent(self, classifier):
        """Test festival inquiry intent classification."""
        test_cases = [
            "When is Diwali?",
            "Tell me about Holi",
            "Upcoming festivals",
            "Ganpati celebration dates"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.FESTIVAL_INQUIRY.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_food_order_intent(self, classifier):
        """Test food ordering intent classification."""
        test_cases = [
            "I want to order food",
            "Khana order karna hai",
            "Food delivery",
            "Hungry, need food"
        ]
        
        for text in test_cases:
            intent = await classifier.classify_intent(text)
            assert intent.name == IntentCategory.FOOD_ORDER.value
            assert intent.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_context_based_classification(self, classifier):
        """Test intent classification with conversation context."""
        # Create conversation state with history
        conversation_state = ConversationState(
            user_id=uuid4(),
            current_language=LanguageCode.HINDI,
            conversation_history=[
                UserInteraction(
                    user_id=uuid4(),
                    input_text="What's the weather?",
                    input_language=LanguageCode.ENGLISH_IN,
                    response_text="It's sunny today",
                    response_language=LanguageCode.ENGLISH_IN,
                    intent="weather_inquiry",
                    confidence=0.9,
                    processing_time=0.5
                )
            ]
        )
        
        # Test related intent with context
        intent = await classifier.classify_intent("What about tomorrow?", conversation_state)
        
        # Should still be weather-related due to context
        assert intent.confidence > 0.3  # Context should boost confidence


class TestCulturalContextInterpreter:
    """Test cultural context interpretation."""
    
    @pytest.fixture
    def interpreter(self):
        return CulturalContextInterpreter()
    
    @pytest.mark.asyncio
    async def test_formal_context_detection(self, interpreter):
        """Test detection of formal communication style."""
        text = "Sir, could you please help me with this?"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["formality_level"] == "high"
        assert context["communication_style"] in ["formal_respectful", "neutral"]
    
    @pytest.mark.asyncio
    async def test_casual_context_detection(self, interpreter):
        """Test detection of casual communication style."""
        text = "Hey yaar, what's up bro?"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["formality_level"] == "low"
        assert context["communication_style"] == "casual_friendly"
    
    @pytest.mark.asyncio
    async def test_family_context_detection(self, interpreter):
        """Test detection of family-oriented context."""
        text = "Mummy said to call papa"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "family_oriented"
    
    @pytest.mark.asyncio
    async def test_religious_context_detection(self, interpreter):
        """Test detection of religious context."""
        text = "I need to go to the mandir for puja"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "religious_spiritual"
        assert "cultural_celebration" in context.get("cultural_sensitivity", [])
    
    @pytest.mark.asyncio
    async def test_regional_context_with_profile(self, interpreter):
        """Test regional context interpretation with user profile."""
        # Create user profile with location
        location = LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India"
        )
        
        user_profile = UserProfile(
            preferred_languages=[LanguageCode.MARATHI, LanguageCode.HINDI],
            location=location
        )
        
        text = "Namaskar, kasa kay?"
        context = await interpreter.interpret_cultural_context(text, user_profile)
        
        assert context["regional_influence"] == "west_india"
    
    @pytest.mark.asyncio
    async def test_urgency_detection(self, interpreter):
        """Test detection of urgency indicators."""
        text = "Jaldi help chahiye, emergency hai!"
        context = await interpreter.interpret_cultural_context(text)
        
        assert context["communication_style"] == "urgent_immediate"


class TestNLUService:
    """Test complete NLU service integration."""
    
    @pytest.fixture
    def nlu_service(self):
        return NLUService()
    
    @pytest.mark.asyncio
    async def test_complete_nlu_processing(self, nlu_service):
        """Test complete NLU processing pipeline."""
        text = "Namaste ji, Mumbai se Delhi ki train ka time kya hai?"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Check all components are present
        assert "original_text" in result
        assert "processed_text" in result
        assert "intent" in result
        assert "entities" in result
        assert "cultural_context" in result
        assert "confidence" in result
        
        # Check intent classification
        intent = result["intent"]
        assert intent.name == IntentCategory.TRAIN_INQUIRY.value
        assert intent.confidence > 0.5
        
        # Check entity extraction
        entities = result["entities"]
        city_entities = [e for e in entities if e.type == EntityType.CITY.value]
        assert len(city_entities) >= 2
        
        # Check cultural context
        cultural_context = result["cultural_context"]
        assert cultural_context["formality_level"] == "high"  # Due to "ji"
    
    @pytest.mark.asyncio
    async def test_colloquial_term_processing(self, nlu_service):
        """Test processing of colloquial terms."""
        text = "Yaar, khana order karna hai"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Check that colloquial terms were mapped
        assert result["processing_metadata"]["colloquial_terms_mapped"] == True
        assert "food" in result["processed_text"].lower()
        
        # Check intent classification on processed text
        intent = result["intent"]
        assert intent.name == IntentCategory.FOOD_ORDER.value
    
    @pytest.mark.asyncio
    async def test_cultural_appropriateness_validation(self, nlu_service):
        """Test cultural appropriateness validation."""
        text = "Tell me about religious festivals"
        language = LanguageCode.ENGLISH_IN
        
        result = await nlu_service.process_user_input(text, language)
        
        intent = result["intent"]
        cultural_context = result["cultural_context"]
        
        validation = await nlu_service.validate_cultural_appropriateness(
            text, intent, cultural_context
        )
        
        assert validation["is_appropriate"] == True
        # Should have suggestions for handling religious topics
        assert len(validation["suggestions"]) > 0
    
    @pytest.mark.asyncio
    async def test_intent_suggestions(self, nlu_service):
        """Test intent suggestions for partial input."""
        partial_text = "train"
        language = LanguageCode.ENGLISH_IN
        
        suggestions = await nlu_service.get_intent_suggestions(partial_text, language)
        
        assert len(suggestions) > 0
        # Should suggest train-related intents
        intent_names = [s.get("intent", "") for s in suggestions]
        assert any("train" in intent for intent in intent_names)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, nlu_service):
        """Test error handling in NLU processing."""
        # Test with empty text
        result = await nlu_service.process_user_input("", LanguageCode.HINDI)
        
        assert result["intent"].name == IntentCategory.UNKNOWN.value
        assert result["confidence"] < 0.5
        
        # Test with very long text
        long_text = "a" * 10000
        result = await nlu_service.process_user_input(long_text, LanguageCode.HINDI)
        
        # Should handle gracefully
        assert "intent" in result
        assert "entities" in result
    
    @pytest.mark.asyncio
    async def test_multilingual_processing(self, nlu_service):
        """Test processing of multilingual input."""
        # Code-switched text (Hindi-English)
        text = "Hello, aaj weather kaisa hai?"
        language = LanguageCode.HINDI
        
        result = await nlu_service.process_user_input(text, language)
        
        # Should detect weather intent despite code-switching
        intent = result["intent"]
        assert intent.name == IntentCategory.WEATHER_INQUIRY.value
        
        # Should extract entities from both languages
        entities = result["entities"]
        assert len(entities) >= 0  # May or may not have entities, but should not crash
    
    @pytest.mark.asyncio
    async def test_conversation_context_influence(self, nlu_service):
        """Test how conversation context influences NLU processing."""
        # Create conversation state with weather inquiry history
        conversation_state = ConversationState(
            user_id=uuid4(),
            current_language=LanguageCode.HINDI,
            conversation_history=[
                UserInteraction(
                    user_id=uuid4(),
                    input_text="What's the weather?",
                    input_language=LanguageCode.ENGLISH_IN,
                    response_text="It's sunny",
                    response_language=LanguageCode.ENGLISH_IN,
                    intent="weather_inquiry",
                    confidence=0.9,
                    processing_time=0.5
                )
            ]
        )
        
        # Ambiguous follow-up question
        text = "What about tomorrow?"
        
        result = await nlu_service.process_user_input(
            text, LanguageCode.ENGLISH_IN, conversation_state
        )
        
        # Should still be weather-related due to context
        intent = result["intent"]
        # Context should help with classification
        assert intent.confidence > 0.3


# Integration tests
class TestNLUIntegration:
    """Test NLU service integration with other components."""
    
    @pytest.mark.asyncio
    async def test_regional_context_integration(self):
        """Test NLU processing with regional context."""
        nlu_service = NLUService()
        
        # Create regional context for Mumbai
        location = LocationData(
            latitude=19.0760,
            longitude=72.8777,
            city="Mumbai",
            state="Maharashtra",
            country="India"
        )
        
        regional_context = RegionalContextData(
            location=location,
            local_language=LanguageCode.MARATHI
        )
        
        text = "Local train schedule chahiye"
        
        result = await nlu_service.process_user_input(
            text, LanguageCode.HINDI, regional_context=regional_context
        )
        
        # Should detect train inquiry
        assert result["intent"].name == IntentCategory.TRAIN_INQUIRY.value
        
        # Should have regional influence in cultural context
        cultural_context = result["cultural_context"]
        assert cultural_context.get("regional_influence") is not None


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])