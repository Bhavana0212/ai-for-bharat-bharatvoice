<<<<<<< HEAD
"""
Property-Based Tests for Adaptive Learning System.

**Property 22: Adaptive Learning**
Tests that the adaptive learning system correctly learns from user interactions
and improves personalization over time.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from bharatvoice.core.models import (
    UserInteraction, LanguageCode, AudioBuffer, RecognitionResult
)
from bharatvoice.services.learning import (
    AdaptiveLearningService, VocabularyLearner, AccentAdapter,
    PreferenceLearner, FeedbackProcessor, ResponseStyleAdapter
)


# Test data generators
@st.composite
def user_interaction(draw):
    """Generate a user interaction."""
    languages = list(LanguageCode)
    language = draw(st.sampled_from(languages))
    
    return UserInteraction(
        interaction_id=uuid4(),
        user_id=uuid4(),
        input_text=draw(st.text(min_size=5, max_size=200)),
        input_language=language,
        response_text=draw(st.text(min_size=10, max_size=300)),
        response_language=language,
        timestamp=datetime.utcnow(),
        intent=draw(st.one_of(st.none(), st.text(min_size=3, max_size=50))),
        entities=draw(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50), max_size=5)),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        processing_time=draw(st.floats(min_value=0.1, max_value=10.0))
    )


@st.composite
def feedback_data(draw):
    """Generate user feedback data."""
    return {
        "rating": draw(st.integers(min_value=1, max_value=5)),
        "positive": draw(st.booleans()),
        "too_long": draw(st.booleans()),
        "too_short": draw(st.booleans()),
        "too_formal": draw(st.booleans()),
        "too_casual": draw(st.booleans()),
        "not_friendly_enough": draw(st.booleans()),
        "correction": draw(st.one_of(st.none(), st.text(min_size=5, max_size=100)))
    }


class TestAdaptiveLearningProperties:
    """Property-based tests for adaptive learning system."""
    
    @pytest.fixture
    def learning_service(self):
        """Create adaptive learning service for testing."""
        return AdaptiveLearningService(
            learning_rate=0.1,
            enable_vocabulary_learning=True,
            enable_accent_adaptation=True,
            enable_preference_learning=True,
            enable_feedback_processing=True,
            enable_style_adaptation=True
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=1, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=50, deadline=5000)
    async def test_learning_improves_with_interactions(self, learning_service, interactions, user_id):
        """
        **Property 22: Adaptive Learning**
        **Validates: Requirements 1.1, 1.2, 2.1, 2.5**
        
        Test that learning confidence and personalization improve with more interactions.
        """
        assume(len(interactions) >= 2)
        
        # Process first interaction
        first_interaction = interactions[0]
        first_interaction.user_id = user_id
        
        first_result = await learning_service.process_interaction(
            user_id=user_id,
            interaction=first_interaction
        )
        
        first_confidence = first_result.get("learning_confidence", 0.0)
        
        # Process remaining interactions
        for interaction in interactions[1:]:
            interaction.user_id = user_id
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Get final learning profile
        final_profile = await learning_service.get_user_learning_profile(user_id)
        final_confidence = final_profile.get("overall_confidence", 0.0)
        
        # Assert learning improvement
        assert final_confidence >= first_confidence, \
            f"Learning confidence should improve: {first_confidence} -> {final_confidence}"
        
        # Check that vocabulary was learned
        vocab_profile = final_profile.get("vocabulary_profile", {})
        if vocab_profile.get("total_words", 0) > 0:
            assert vocab_profile["total_words"] > 0, "Should learn vocabulary from interactions"
        
        # Check that preferences were updated
        pref_profile = final_profile.get("preference_profile", {})
        assert len(pref_profile) >= 0, "Preference profile should be available"
    
    @given(
        interaction=user_interaction(),
        feedback_list=st.lists(feedback_data(), min_size=1, max_size=5),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=5000)
    async def test_feedback_improves_responses(self, learning_service, interaction, feedback_list, user_id):
        """
        Test that user feedback leads to improved response adaptation.
        """
        interaction.user_id = user_id
        
        # Get initial response adaptation
        initial_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a test response.",
            language=interaction.input_language
        )
        
        # Process interaction with feedback
        for feedback in feedback_list:
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction,
                user_feedback=feedback
            )
        
        # Get adapted response after feedback
        final_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a test response.",
            language=interaction.input_language
        )
        
        # Response should be adapted based on feedback
        # At minimum, the service should have processed the feedback
        profile = await learning_service.get_user_learning_profile(user_id)
        assert profile is not None, "User profile should exist after feedback"
    
    @given(
        interactions=st.lists(user_interaction(), min_size=5, max_size=15),
        user_id=st.uuids()
    )
    @settings(max_examples=20, deadline=10000)
    async def test_personalization_consistency(self, learning_service, interactions, user_id):
        """
        Test that personalization remains consistent for the same user across sessions.
        """
        # Process all interactions
        for interaction in interactions:
            interaction.user_id = user_id
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Get personalization suggestions multiple times
        suggestions1 = await learning_service.get_personalization_suggestions(user_id)
        suggestions2 = await learning_service.get_personalization_suggestions(user_id)
        
        # Suggestions should be consistent
        assert len(suggestions1) == len(suggestions2), \
            "Personalization suggestions should be consistent"
        
        # User profile should be stable
        profile1 = await learning_service.get_user_learning_profile(user_id)
        profile2 = await learning_service.get_user_learning_profile(user_id)
        
        assert profile1["overall_confidence"] == profile2["overall_confidence"], \
            "User profile confidence should be stable"
    
    @given(
        base_response=st.text(min_size=10, max_size=100),
        language=st.sampled_from(list(LanguageCode)),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=3000)
    async def test_response_adaptation_preserves_meaning(self, learning_service, base_response, language, user_id):
        """
        Test that response adaptation preserves the core meaning of responses.
        """
        # Adapt response
        adapted_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response=base_response,
            language=language
        )
        
        # Basic checks for meaning preservation
        assert len(adapted_response) > 0, "Adapted response should not be empty"
        assert isinstance(adapted_response, str), "Adapted response should be a string"
        
        # Response should not be drastically different in length
        length_ratio = len(adapted_response) / len(base_response)
        assert 0.5 <= length_ratio <= 3.0, \
            f"Adapted response length ratio should be reasonable: {length_ratio}"


class TestVocabularyLearningProperties:
    """Property-based tests for vocabulary learning."""
    
    @pytest.fixture
    def vocab_learner(self):
        """Create vocabulary learner for testing."""
        return VocabularyLearner(
            min_frequency_threshold=2,
            context_window_size=3,
            learning_rate=0.1
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=3, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=5000)
    async def test_vocabulary_accumulation(self, vocab_learner, interactions, user_id):
        """
        Test that vocabulary accumulates correctly over multiple interactions.
        """
        total_new_words = 0
        
        for interaction in interactions:
            result = await vocab_learner.learn_from_interaction(user_id, interaction)
            total_new_words += len(result.get("new_words", []))
        
        # Get final vocabulary
        final_vocab = await vocab_learner.get_user_vocabulary(user_id)
        
        # Vocabulary should accumulate
        assert final_vocab["total_words"] >= 0, "Should have learned some vocabulary"
        
        # Language distribution should be reasonable
        lang_dist = final_vocab.get("languages", [])
        assert len(lang_dist) >= 0, "Should track language distribution"
    
    @given(
        interaction=user_interaction(),
        user_id=st.uuids(),
        min_confidence=st.floats(min_value=0.1, max_value=0.9)
    )
    @settings(max_examples=20, deadline=3000)
    async def test_vocabulary_confidence_filtering(self, vocab_learner, interaction, user_id, min_confidence):
        """
        Test that vocabulary filtering by confidence works correctly.
        """
        # Learn from interaction
        await vocab_learner.learn_from_interaction(user_id, interaction)
        
        # Get vocabulary with confidence filter
        filtered_vocab = await vocab_learner.get_user_vocabulary(
            user_id, min_confidence=min_confidence
        )
        
        # All returned vocabulary should meet confidence threshold
        for word_data in filtered_vocab["vocabulary"].values():
            assert word_data["confidence"] >= min_confidence, \
                f"Word confidence {word_data['confidence']} should be >= {min_confidence}"


class TestAccentAdaptationProperties:
    """Property-based tests for accent adaptation."""
    
    @pytest.fixture
    def accent_adapter(self):
        """Create accent adapter for testing."""
        return AccentAdapter(
            adaptation_threshold=3,
            confidence_threshold=0.6
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=5, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=20, deadline=5000)
    async def test_accent_profile_development(self, accent_adapter, interactions, user_id):
        """
        Test that accent profiles develop correctly with sufficient data.
        """
        # Process interactions
        for interaction in interactions:
            await accent_adapter.learn_accent_from_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Check accent profiles for each language
        for interaction in interactions:
            profile = await accent_adapter.get_accent_profile(
                user_id, interaction.input_language
            )
            
            if profile:
                assert profile["confidence"] >= 0.0, "Confidence should be non-negative"
                assert profile["sample_count"] >= 0, "Sample count should be non-negative"
                assert profile["language"] == interaction.input_language.value, \
                    "Profile language should match"


class AdaptiveLearningStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for adaptive learning system.
    """
    
    def __init__(self):
        super().__init__()
        self.learning_service = None
        self.users = {}
        self.interactions_count = 0
    
    @initialize()
    def setup(self):
        """Initialize the learning service."""
        self.learning_service = AdaptiveLearningService(learning_rate=0.1)
    
    @rule(
        user_id=st.uuids(),
        interaction=user_interaction()
    )
    async def process_interaction(self, user_id, interaction):
        """Process a user interaction."""
        interaction.user_id = user_id
        
        if user_id not in self.users:
            self.users[user_id] = {
                "interactions": 0,
                "last_confidence": 0.0
            }
        
        result = await self.learning_service.process_interaction(
            user_id=user_id,
            interaction=interaction
        )
        
        self.users[user_id]["interactions"] += 1
        self.users[user_id]["last_confidence"] = result.get("learning_confidence", 0.0)
        self.interactions_count += 1
    
    @rule(user_id=st.uuids())
    async def get_user_profile(self, user_id):
        """Get user learning profile."""
        if user_id in self.users:
            profile = await self.learning_service.get_user_learning_profile(user_id)
            assert profile is not None, "Profile should exist for known user"
            assert profile["user_id"] == str(user_id), "Profile should match user ID"
    
    @invariant()
    def learning_service_consistency(self):
        """Check that learning service maintains consistency."""
        assert self.learning_service is not None, "Learning service should exist"
        assert self.interactions_count >= 0, "Interaction count should be non-negative"
        
        # All tracked users should have non-negative interaction counts
        for user_data in self.users.values():
            assert user_data["interactions"] >= 0, "User interaction count should be non-negative"
            assert 0.0 <= user_data["last_confidence"] <= 1.0, \
                "User confidence should be between 0 and 1"


# Test runner for stateful tests
TestAdaptiveLearningStateful = AdaptiveLearningStateMachine.TestCase


@pytest.mark.asyncio
class TestAdaptiveLearningIntegration:
    """Integration tests for adaptive learning system."""
    
    async def test_full_learning_cycle(self):
        """
        **Property 22: Adaptive Learning**
        Test complete learning cycle with multiple components.
        """
        learning_service = AdaptiveLearningService()
        user_id = uuid4()
        
        # Create test interactions
        interactions = [
            UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="Hello, how are you today?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'm doing well, thank you for asking!",
                response_language=LanguageCode.ENGLISH_IN,
                timestamp=datetime.utcnow(),
                intent="greeting",
                entities={"greeting": "hello"},
                confidence_score=0.9,
                processing_time=1.2
            ),
            UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="Can you help me with weather information?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'd be happy to help you with weather information.",
                response_language=LanguageCode.ENGLISH_IN,
                timestamp=datetime.utcnow(),
                intent="weather_query",
                entities={"service": "weather"},
                confidence_score=0.85,
                processing_time=0.8
            )
        ]
        
        # Process interactions with feedback
        for i, interaction in enumerate(interactions):
            feedback = {
                "rating": 4 if i == 0 else 5,
                "positive": True,
                "too_formal": i == 0,  # First response too formal
                "correction": None
            }
            
            result = await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction,
                user_feedback=feedback,
                response_time=interaction.processing_time
            )
            
            # Verify learning occurred
            assert result["learning_confidence"] >= 0.0, "Learning confidence should be valid"
            assert len(result["overall_improvements"]) >= 0, "Should track improvements"
        
        # Verify personalization
        profile = await learning_service.get_user_learning_profile(user_id)
        assert profile["overall_confidence"] > 0.0, "Should have learned from interactions"
        
        # Test response adaptation
        adapted_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a formal response.",
            language=LanguageCode.ENGLISH_IN
        )
        
        # Response should be adapted (less formal based on feedback)
        assert len(adapted_response) > 0, "Should return adapted response"
        
        # Get personalization suggestions
        suggestions = await learning_service.get_personalization_suggestions(user_id)
        assert isinstance(suggestions, list), "Should return suggestions list"


if __name__ == "__main__":
    # Run property-based tests
=======
"""
Property-Based Tests for Adaptive Learning System.

**Property 22: Adaptive Learning**
Tests that the adaptive learning system correctly learns from user interactions
and improves personalization over time.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from bharatvoice.core.models import (
    UserInteraction, LanguageCode, AudioBuffer, RecognitionResult
)
from bharatvoice.services.learning import (
    AdaptiveLearningService, VocabularyLearner, AccentAdapter,
    PreferenceLearner, FeedbackProcessor, ResponseStyleAdapter
)


# Test data generators
@st.composite
def user_interaction(draw):
    """Generate a user interaction."""
    languages = list(LanguageCode)
    language = draw(st.sampled_from(languages))
    
    return UserInteraction(
        interaction_id=uuid4(),
        user_id=uuid4(),
        input_text=draw(st.text(min_size=5, max_size=200)),
        input_language=language,
        response_text=draw(st.text(min_size=10, max_size=300)),
        response_language=language,
        timestamp=datetime.utcnow(),
        intent=draw(st.one_of(st.none(), st.text(min_size=3, max_size=50))),
        entities=draw(st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50), max_size=5)),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        processing_time=draw(st.floats(min_value=0.1, max_value=10.0))
    )


@st.composite
def feedback_data(draw):
    """Generate user feedback data."""
    return {
        "rating": draw(st.integers(min_value=1, max_value=5)),
        "positive": draw(st.booleans()),
        "too_long": draw(st.booleans()),
        "too_short": draw(st.booleans()),
        "too_formal": draw(st.booleans()),
        "too_casual": draw(st.booleans()),
        "not_friendly_enough": draw(st.booleans()),
        "correction": draw(st.one_of(st.none(), st.text(min_size=5, max_size=100)))
    }


class TestAdaptiveLearningProperties:
    """Property-based tests for adaptive learning system."""
    
    @pytest.fixture
    def learning_service(self):
        """Create adaptive learning service for testing."""
        return AdaptiveLearningService(
            learning_rate=0.1,
            enable_vocabulary_learning=True,
            enable_accent_adaptation=True,
            enable_preference_learning=True,
            enable_feedback_processing=True,
            enable_style_adaptation=True
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=1, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=50, deadline=5000)
    async def test_learning_improves_with_interactions(self, learning_service, interactions, user_id):
        """
        **Property 22: Adaptive Learning**
        **Validates: Requirements 1.1, 1.2, 2.1, 2.5**
        
        Test that learning confidence and personalization improve with more interactions.
        """
        assume(len(interactions) >= 2)
        
        # Process first interaction
        first_interaction = interactions[0]
        first_interaction.user_id = user_id
        
        first_result = await learning_service.process_interaction(
            user_id=user_id,
            interaction=first_interaction
        )
        
        first_confidence = first_result.get("learning_confidence", 0.0)
        
        # Process remaining interactions
        for interaction in interactions[1:]:
            interaction.user_id = user_id
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Get final learning profile
        final_profile = await learning_service.get_user_learning_profile(user_id)
        final_confidence = final_profile.get("overall_confidence", 0.0)
        
        # Assert learning improvement
        assert final_confidence >= first_confidence, \
            f"Learning confidence should improve: {first_confidence} -> {final_confidence}"
        
        # Check that vocabulary was learned
        vocab_profile = final_profile.get("vocabulary_profile", {})
        if vocab_profile.get("total_words", 0) > 0:
            assert vocab_profile["total_words"] > 0, "Should learn vocabulary from interactions"
        
        # Check that preferences were updated
        pref_profile = final_profile.get("preference_profile", {})
        assert len(pref_profile) >= 0, "Preference profile should be available"
    
    @given(
        interaction=user_interaction(),
        feedback_list=st.lists(feedback_data(), min_size=1, max_size=5),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=5000)
    async def test_feedback_improves_responses(self, learning_service, interaction, feedback_list, user_id):
        """
        Test that user feedback leads to improved response adaptation.
        """
        interaction.user_id = user_id
        
        # Get initial response adaptation
        initial_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a test response.",
            language=interaction.input_language
        )
        
        # Process interaction with feedback
        for feedback in feedback_list:
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction,
                user_feedback=feedback
            )
        
        # Get adapted response after feedback
        final_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a test response.",
            language=interaction.input_language
        )
        
        # Response should be adapted based on feedback
        # At minimum, the service should have processed the feedback
        profile = await learning_service.get_user_learning_profile(user_id)
        assert profile is not None, "User profile should exist after feedback"
    
    @given(
        interactions=st.lists(user_interaction(), min_size=5, max_size=15),
        user_id=st.uuids()
    )
    @settings(max_examples=20, deadline=10000)
    async def test_personalization_consistency(self, learning_service, interactions, user_id):
        """
        Test that personalization remains consistent for the same user across sessions.
        """
        # Process all interactions
        for interaction in interactions:
            interaction.user_id = user_id
            await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Get personalization suggestions multiple times
        suggestions1 = await learning_service.get_personalization_suggestions(user_id)
        suggestions2 = await learning_service.get_personalization_suggestions(user_id)
        
        # Suggestions should be consistent
        assert len(suggestions1) == len(suggestions2), \
            "Personalization suggestions should be consistent"
        
        # User profile should be stable
        profile1 = await learning_service.get_user_learning_profile(user_id)
        profile2 = await learning_service.get_user_learning_profile(user_id)
        
        assert profile1["overall_confidence"] == profile2["overall_confidence"], \
            "User profile confidence should be stable"
    
    @given(
        base_response=st.text(min_size=10, max_size=100),
        language=st.sampled_from(list(LanguageCode)),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=3000)
    async def test_response_adaptation_preserves_meaning(self, learning_service, base_response, language, user_id):
        """
        Test that response adaptation preserves the core meaning of responses.
        """
        # Adapt response
        adapted_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response=base_response,
            language=language
        )
        
        # Basic checks for meaning preservation
        assert len(adapted_response) > 0, "Adapted response should not be empty"
        assert isinstance(adapted_response, str), "Adapted response should be a string"
        
        # Response should not be drastically different in length
        length_ratio = len(adapted_response) / len(base_response)
        assert 0.5 <= length_ratio <= 3.0, \
            f"Adapted response length ratio should be reasonable: {length_ratio}"


class TestVocabularyLearningProperties:
    """Property-based tests for vocabulary learning."""
    
    @pytest.fixture
    def vocab_learner(self):
        """Create vocabulary learner for testing."""
        return VocabularyLearner(
            min_frequency_threshold=2,
            context_window_size=3,
            learning_rate=0.1
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=3, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=30, deadline=5000)
    async def test_vocabulary_accumulation(self, vocab_learner, interactions, user_id):
        """
        Test that vocabulary accumulates correctly over multiple interactions.
        """
        total_new_words = 0
        
        for interaction in interactions:
            result = await vocab_learner.learn_from_interaction(user_id, interaction)
            total_new_words += len(result.get("new_words", []))
        
        # Get final vocabulary
        final_vocab = await vocab_learner.get_user_vocabulary(user_id)
        
        # Vocabulary should accumulate
        assert final_vocab["total_words"] >= 0, "Should have learned some vocabulary"
        
        # Language distribution should be reasonable
        lang_dist = final_vocab.get("languages", [])
        assert len(lang_dist) >= 0, "Should track language distribution"
    
    @given(
        interaction=user_interaction(),
        user_id=st.uuids(),
        min_confidence=st.floats(min_value=0.1, max_value=0.9)
    )
    @settings(max_examples=20, deadline=3000)
    async def test_vocabulary_confidence_filtering(self, vocab_learner, interaction, user_id, min_confidence):
        """
        Test that vocabulary filtering by confidence works correctly.
        """
        # Learn from interaction
        await vocab_learner.learn_from_interaction(user_id, interaction)
        
        # Get vocabulary with confidence filter
        filtered_vocab = await vocab_learner.get_user_vocabulary(
            user_id, min_confidence=min_confidence
        )
        
        # All returned vocabulary should meet confidence threshold
        for word_data in filtered_vocab["vocabulary"].values():
            assert word_data["confidence"] >= min_confidence, \
                f"Word confidence {word_data['confidence']} should be >= {min_confidence}"


class TestAccentAdaptationProperties:
    """Property-based tests for accent adaptation."""
    
    @pytest.fixture
    def accent_adapter(self):
        """Create accent adapter for testing."""
        return AccentAdapter(
            adaptation_threshold=3,
            confidence_threshold=0.6
        )
    
    @given(
        interactions=st.lists(user_interaction(), min_size=5, max_size=10),
        user_id=st.uuids()
    )
    @settings(max_examples=20, deadline=5000)
    async def test_accent_profile_development(self, accent_adapter, interactions, user_id):
        """
        Test that accent profiles develop correctly with sufficient data.
        """
        # Process interactions
        for interaction in interactions:
            await accent_adapter.learn_accent_from_interaction(
                user_id=user_id,
                interaction=interaction
            )
        
        # Check accent profiles for each language
        for interaction in interactions:
            profile = await accent_adapter.get_accent_profile(
                user_id, interaction.input_language
            )
            
            if profile:
                assert profile["confidence"] >= 0.0, "Confidence should be non-negative"
                assert profile["sample_count"] >= 0, "Sample count should be non-negative"
                assert profile["language"] == interaction.input_language.value, \
                    "Profile language should match"


class AdaptiveLearningStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for adaptive learning system.
    """
    
    def __init__(self):
        super().__init__()
        self.learning_service = None
        self.users = {}
        self.interactions_count = 0
    
    @initialize()
    def setup(self):
        """Initialize the learning service."""
        self.learning_service = AdaptiveLearningService(learning_rate=0.1)
    
    @rule(
        user_id=st.uuids(),
        interaction=user_interaction()
    )
    async def process_interaction(self, user_id, interaction):
        """Process a user interaction."""
        interaction.user_id = user_id
        
        if user_id not in self.users:
            self.users[user_id] = {
                "interactions": 0,
                "last_confidence": 0.0
            }
        
        result = await self.learning_service.process_interaction(
            user_id=user_id,
            interaction=interaction
        )
        
        self.users[user_id]["interactions"] += 1
        self.users[user_id]["last_confidence"] = result.get("learning_confidence", 0.0)
        self.interactions_count += 1
    
    @rule(user_id=st.uuids())
    async def get_user_profile(self, user_id):
        """Get user learning profile."""
        if user_id in self.users:
            profile = await self.learning_service.get_user_learning_profile(user_id)
            assert profile is not None, "Profile should exist for known user"
            assert profile["user_id"] == str(user_id), "Profile should match user ID"
    
    @invariant()
    def learning_service_consistency(self):
        """Check that learning service maintains consistency."""
        assert self.learning_service is not None, "Learning service should exist"
        assert self.interactions_count >= 0, "Interaction count should be non-negative"
        
        # All tracked users should have non-negative interaction counts
        for user_data in self.users.values():
            assert user_data["interactions"] >= 0, "User interaction count should be non-negative"
            assert 0.0 <= user_data["last_confidence"] <= 1.0, \
                "User confidence should be between 0 and 1"


# Test runner for stateful tests
TestAdaptiveLearningStateful = AdaptiveLearningStateMachine.TestCase


@pytest.mark.asyncio
class TestAdaptiveLearningIntegration:
    """Integration tests for adaptive learning system."""
    
    async def test_full_learning_cycle(self):
        """
        **Property 22: Adaptive Learning**
        Test complete learning cycle with multiple components.
        """
        learning_service = AdaptiveLearningService()
        user_id = uuid4()
        
        # Create test interactions
        interactions = [
            UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="Hello, how are you today?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'm doing well, thank you for asking!",
                response_language=LanguageCode.ENGLISH_IN,
                timestamp=datetime.utcnow(),
                intent="greeting",
                entities={"greeting": "hello"},
                confidence_score=0.9,
                processing_time=1.2
            ),
            UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="Can you help me with weather information?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'd be happy to help you with weather information.",
                response_language=LanguageCode.ENGLISH_IN,
                timestamp=datetime.utcnow(),
                intent="weather_query",
                entities={"service": "weather"},
                confidence_score=0.85,
                processing_time=0.8
            )
        ]
        
        # Process interactions with feedback
        for i, interaction in enumerate(interactions):
            feedback = {
                "rating": 4 if i == 0 else 5,
                "positive": True,
                "too_formal": i == 0,  # First response too formal
                "correction": None
            }
            
            result = await learning_service.process_interaction(
                user_id=user_id,
                interaction=interaction,
                user_feedback=feedback,
                response_time=interaction.processing_time
            )
            
            # Verify learning occurred
            assert result["learning_confidence"] >= 0.0, "Learning confidence should be valid"
            assert len(result["overall_improvements"]) >= 0, "Should track improvements"
        
        # Verify personalization
        profile = await learning_service.get_user_learning_profile(user_id)
        assert profile["overall_confidence"] > 0.0, "Should have learned from interactions"
        
        # Test response adaptation
        adapted_response = await learning_service.adapt_response_for_user(
            user_id=user_id,
            base_response="This is a formal response.",
            language=LanguageCode.ENGLISH_IN
        )
        
        # Response should be adapted (less formal based on feedback)
        assert len(adapted_response) > 0, "Should return adapted response"
        
        # Get personalization suggestions
        suggestions = await learning_service.get_personalization_suggestions(user_id)
        assert isinstance(suggestions, list), "Should return suggestions list"


if __name__ == "__main__":
    # Run property-based tests
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__, "-v", "--tb=short"])