"""
Unit tests for conversation state management.

Tests the ConversationManager, ConversationContextManager, and ContextManagementService
classes for proper session handling, multi-turn dialog support, context preservation,
timeout mechanisms, conversation history storage and retrieval, and enhanced user profile management.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from bharatvoice.core.models import (
    ConversationState,
    LanguageCode,
    UserInteraction,
    UserProfile,
    LocationData,
)
from bharatvoice.services.context_management import (
    ConversationManager,
    ConversationContextManager,
    ContextManagementService,
)


class TestConversationManager:
    """Test cases for ConversationManager."""
    
    @pytest.fixture
    async def conversation_manager(self):
        """Create a conversation manager for testing."""
        manager = ConversationManager(
            session_timeout_minutes=1,  # Short timeout for testing
            max_history_length=5,
            cleanup_interval_minutes=1
        )
        yield manager
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_session(self, conversation_manager):
        """Test creating a new conversation session."""
        user_id = uuid4()
        
        conversation_state = await conversation_manager.create_session(
            user_id=user_id,
            initial_language=LanguageCode.HINDI
        )
        
        assert conversation_state.user_id == user_id
        assert conversation_state.current_language == LanguageCode.HINDI
        assert conversation_state.is_active is True
        assert len(conversation_state.conversation_history) == 0
        assert len(conversation_state.context_variables) == 0
    
    @pytest.mark.asyncio
    async def test_get_session(self, conversation_manager):
        """Test retrieving an existing session."""
        user_id = uuid4()
        
        # Create session
        original_state = await conversation_manager.create_session(user_id)
        session_id = original_state.session_id
        
        # Retrieve session
        retrieved_state = await conversation_manager.get_session(session_id)
        
        assert retrieved_state is not None
        assert retrieved_state.session_id == session_id
        assert retrieved_state.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, conversation_manager):
        """Test retrieving a non-existent session."""
        fake_session_id = uuid4()
        
        retrieved_state = await conversation_manager.get_session(fake_session_id)
        
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_update_session_with_interaction(self, conversation_manager):
        """Test updating session with new interaction."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # Create interaction
        interaction = UserInteraction(
            user_id=user_id,
            input_text="Hello",
            input_language=LanguageCode.ENGLISH_IN,
            response_text="Hi there!",
            response_language=LanguageCode.ENGLISH_IN,
            intent="greeting",
            confidence=0.95,
            processing_time=0.1
        )
        
        # Update session
        updated_state = await conversation_manager.update_session(session_id, interaction)
        
        assert updated_state is not None
        assert len(updated_state.conversation_history) == 1
        assert updated_state.conversation_history[0] == interaction
        assert updated_state.current_language == LanguageCode.ENGLISH_IN  # Language switched
    
    @pytest.mark.asyncio
    async def test_context_variables(self, conversation_manager):
        """Test setting and getting context variables."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # Set context variable
        success = await conversation_manager.set_context_variable(
            session_id, "user_name", "Raj"
        )
        assert success is True
        
        # Get context variable
        value = await conversation_manager.get_context_variable(
            session_id, "user_name"
        )
        assert value == "Raj"
        
        # Get non-existent variable with default
        value = await conversation_manager.get_context_variable(
            session_id, "non_existent", "default_value"
        )
        assert value == "default_value"
    
    @pytest.mark.asyncio
    async def test_conversation_history_limit(self, conversation_manager):
        """Test conversation history length limiting."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # Add more interactions than the limit (5)
        for i in range(7):
            interaction = UserInteraction(
                user_id=user_id,
                input_text=f"Message {i}",
                input_language=LanguageCode.HINDI,
                response_text=f"Response {i}",
                response_language=LanguageCode.HINDI,
                confidence=0.9,
                processing_time=0.1
            )
            
            await conversation_manager.update_session(session_id, interaction)
        
        # Check that history is limited to max_history_length
        updated_state = await conversation_manager.get_session(session_id)
        assert len(updated_state.conversation_history) == 5
        
        # Check that the most recent interactions are kept
        assert updated_state.conversation_history[-1].input_text == "Message 6"
        assert updated_state.conversation_history[0].input_text == "Message 2"
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, conversation_manager):
        """Test retrieving conversation history."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # Add interactions
        interactions = []
        for i in range(3):
            interaction = UserInteraction(
                user_id=user_id,
                input_text=f"Message {i}",
                input_language=LanguageCode.HINDI,
                response_text=f"Response {i}",
                response_language=LanguageCode.HINDI,
                confidence=0.9,
                processing_time=0.1
            )
            interactions.append(interaction)
            await conversation_manager.update_session(session_id, interaction)
        
        # Get full history
        history = await conversation_manager.get_conversation_history(session_id)
        assert len(history) == 3
        
        # Get limited history
        limited_history = await conversation_manager.get_conversation_history(
            session_id, limit=2
        )
        assert len(limited_history) == 2
    
    @pytest.mark.asyncio
    async def test_session_timeout(self, conversation_manager):
        """Test session timeout and cleanup."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # Manually set last interaction time to past
        conversation_state.last_interaction_time = datetime.utcnow() - timedelta(minutes=2)
        
        # Try to get session - should return None due to timeout
        retrieved_state = await conversation_manager.get_session(session_id)
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_end_session(self, conversation_manager):
        """Test explicitly ending a session."""
        user_id = uuid4()
        
        # Create session
        conversation_state = await conversation_manager.create_session(user_id)
        session_id = conversation_state.session_id
        
        # End session
        success = await conversation_manager.end_session(session_id)
        assert success is True
        
        # Try to get session - should return None
        retrieved_state = await conversation_manager.get_session(session_id)
        assert retrieved_state is None
        
        # Try to end non-existent session
        success = await conversation_manager.end_session(uuid4())
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_active_sessions(self, conversation_manager):
        """Test getting active sessions."""
        user1_id = uuid4()
        user2_id = uuid4()
        
        # Create sessions for different users
        state1 = await conversation_manager.create_session(user1_id)
        state2 = await conversation_manager.create_session(user2_id)
        state3 = await conversation_manager.create_session(user1_id)
        
        # Get all active sessions
        all_sessions = await conversation_manager.get_active_sessions()
        assert len(all_sessions) == 3
        
        # Get sessions for specific user
        user1_sessions = await conversation_manager.get_active_sessions(user1_id)
        assert len(user1_sessions) == 2
        
        user2_sessions = await conversation_manager.get_active_sessions(user2_id)
        assert len(user2_sessions) == 1
    
    @pytest.mark.asyncio
    async def test_session_statistics(self, conversation_manager):
        """Test getting session statistics."""
        user_id = uuid4()
        
        # Initially no sessions
        stats = await conversation_manager.get_session_statistics()
        assert stats["total_sessions"] == 0
        assert stats["active_sessions"] == 0
        
        # Create sessions
        await conversation_manager.create_session(user_id)
        await conversation_manager.create_session(user_id)
        
        stats = await conversation_manager.get_session_statistics()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2


class TestConversationContextManager:
    """Test cases for ConversationContextManager."""
    
    @pytest.fixture
    async def context_manager(self):
        """Create a conversation context manager for testing."""
        conversation_manager = ConversationManager(
            session_timeout_minutes=30,
            max_history_length=50,
            cleanup_interval_minutes=5
        )
        manager = ConversationContextManager(conversation_manager)
        yield manager
        await conversation_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, context_manager):
        """Test starting a conversation with context initialization."""
        user_id = uuid4()
        
        conversation_state = await context_manager.start_conversation(
            user_id=user_id,
            initial_language=LanguageCode.TAMIL
        )
        
        assert conversation_state.user_id == user_id
        assert conversation_state.current_language == LanguageCode.TAMIL
        assert conversation_state.is_active is True
    
    @pytest.mark.asyncio
    async def test_process_interaction_with_context(self, context_manager):
        """Test processing interaction with context preservation."""
        user_id = uuid4()
        
        # Start conversation
        conversation_state = await context_manager.start_conversation(user_id)
        session_id = conversation_state.session_id
        
        # Create interaction with entities
        interaction = UserInteraction(
            user_id=user_id,
            input_text="Book a table for 4 people",
            input_language=LanguageCode.ENGLISH_IN,
            response_text="I'll help you book a table",
            response_language=LanguageCode.ENGLISH_IN,
            intent="restaurant.booking",
            entities={"party_size": 4, "service_type": "restaurant"},
            confidence=0.92,
            processing_time=0.15
        )
        
        # Process interaction
        updated_state = await context_manager.process_interaction(session_id, interaction)
        
        assert updated_state is not None
        assert len(updated_state.conversation_history) == 1
        
        # Check that context variables were updated
        last_lang = await context_manager.conversation_manager.get_context_variable(
            session_id, "last_input_language"
        )
        assert last_lang == LanguageCode.ENGLISH_IN.value
        
        entities = await context_manager.conversation_manager.get_context_variable(
            session_id, "mentioned_entities"
        )
        assert entities["party_size"] == 4
        assert entities["service_type"] == "restaurant"


class TestContextManagementService:
    """Test cases for ContextManagementService."""
    
    @pytest.fixture
    async def service(self):
        """Create a context management service for testing."""
        service = ContextManagementService(
            session_timeout_minutes=30,
            max_history_length=50,
            cleanup_interval_minutes=5
        )
        yield service
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_maintain_conversation_state(self, service):
        """Test maintaining conversation state."""
        user_id = uuid4()
        
        # Create interaction
        interaction = UserInteraction(
            user_id=user_id,
            input_text="Hello",
            input_language=LanguageCode.HINDI,
            response_text="Namaste",
            response_language=LanguageCode.HINDI,
            intent="greeting",
            confidence=0.95,
            processing_time=0.1
        )
        
        # Maintain conversation state (should create new session)
        conversation_state = await service.maintain_conversation_state(
            "non-existent-session", interaction
        )
        
        assert conversation_state is not None
        assert conversation_state.user_id == user_id
        assert len(conversation_state.conversation_history) == 1
    
    @pytest.mark.asyncio
    async def test_update_user_profile(self, service):
        """Test updating user profile."""
        user_id = str(uuid4())
        
        preferences = {
            "preferred_languages": [LanguageCode.TAMIL, LanguageCode.ENGLISH_IN],
            "primary_language": LanguageCode.TAMIL,
            "privacy_settings": {
                "data_retention_days": 60,
                "allow_analytics": False
            }
        }
        
        # Update profile
        user_profile = await service.update_user_profile(user_id, preferences)
        
        assert user_profile.preferred_languages == preferences["preferred_languages"]
        assert user_profile.primary_language == preferences["primary_language"]
        assert user_profile.privacy_settings.data_retention_days == 60
        assert user_profile.privacy_settings.allow_analytics is False
    
    @pytest.mark.asyncio
    async def test_get_regional_context(self, service):
        """Test getting regional context."""
        location = {
            "latitude": 13.0827,
            "longitude": 80.2707,
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India"
        }
        
        regional_context = await service.get_regional_context(location)
        
        assert regional_context.location.city == "Chennai"
        assert regional_context.location.state == "Tamil Nadu"
        assert regional_context.local_language == LanguageCode.TAMIL
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, service):
        """Test learning from user interaction."""
        user_id = uuid4()
        
        interaction = UserInteraction(
            user_id=user_id,
            input_text="What's the weather like?",
            input_language=LanguageCode.GUJARATI,
            response_text="The weather is sunny",
            response_language=LanguageCode.GUJARATI,
            intent="weather.query",
            confidence=0.88,
            processing_time=0.2
        )
        
        # Learn from interaction
        learning_result = await service.learn_from_interaction(interaction)
        
        assert learning_result["user_id"] == str(user_id)
        assert "created_new_profile" in learning_result["adaptations"]
        assert f"added_language_{LanguageCode.GUJARATI.value}" in learning_result["adaptations"]
    
    @pytest.mark.asyncio
    async def test_session_management_methods(self, service):
        """Test session management methods."""
        user_id = str(uuid4())
        
        # Create session
        conversation_state = await service.create_conversation_session(
            user_id, LanguageCode.BENGALI.value
        )
        session_id = str(conversation_state.session_id)
        
        assert conversation_state.current_language == LanguageCode.BENGALI
        
        # Get session
        retrieved_state = await service.get_conversation_state(session_id)
        assert retrieved_state is not None
        assert retrieved_state.session_id == conversation_state.session_id
        
        # Set context
        success = await service.set_session_context(session_id, "test_key", "test_value")
        assert success is True
        
        # Get context
        value = await service.get_session_context(session_id, "test_key")
        assert value == "test_value"
        
        # End session
        success = await service.end_conversation_session(session_id)
        assert success is True
        
        # Verify session is ended
        retrieved_state = await service.get_conversation_state(session_id)
        assert retrieved_state is None
    
    @pytest.mark.asyncio
    async def test_service_statistics(self, service):
        """Test getting service statistics."""
        stats = await service.get_service_statistics()
        
        assert "conversation_sessions" in stats
        assert "user_profiles" in stats
        assert "regional_contexts_cached" in stats
        assert isinstance(stats["user_profiles"], int)
    
    @pytest.mark.asyncio
    async def test_invalid_session_id_handling(self, service):
        """Test handling of invalid session IDs."""
        # Test with invalid UUID format
        result = await service.get_conversation_state("invalid-uuid")
        assert result is None
        
        success = await service.set_session_context("invalid-uuid", "key", "value")
        assert success is False
        
        value = await service.get_session_context("invalid-uuid", "key", "default")
        assert value == "default"
    
    @pytest.mark.asyncio
    async def test_invalid_user_id_handling(self, service):
        """Test handling of invalid user IDs."""
        # Test with invalid UUID format
        result = await service.get_user_profile("invalid-uuid")
        assert result is None
        
        # Should raise ValueError for invalid user ID in profile update
        with pytest.raises(ValueError):
            await service.update_user_profile("invalid-uuid", {})


# Property-based test for conversation state consistency
@pytest.mark.asyncio
async def test_conversation_state_consistency():
    """
    Property test: Conversation state should remain consistent across operations.
    
    **Validates: Requirements 2.4, 9.3**
    
    This test verifies that conversation state management maintains consistency
    when multiple operations are performed on the same session.
    """
    service = ContextManagementService()
    
    try:
        user_id = str(uuid4())
        
        # Create session
        conversation_state = await service.create_conversation_session(user_id)
        session_id = str(conversation_state.session_id)
        
        # Perform multiple operations
        interactions = []
        for i in range(10):
            interaction = UserInteraction(
                user_id=uuid4(),  # Use conversation_state.user_id instead
                input_text=f"Test message {i}",
                input_language=LanguageCode.HINDI,
                response_text=f"Test response {i}",
                response_language=LanguageCode.HINDI,
                intent=f"test.intent.{i % 3}",
                confidence=0.8 + (i % 3) * 0.05,
                processing_time=0.1 + i * 0.01
            )
            interaction.user_id = conversation_state.user_id  # Fix user_id
            interactions.append(interaction)
            
            # Update session
            updated_state = await service.maintain_conversation_state(session_id, interaction)
            
            # Verify consistency
            assert updated_state is not None
            assert updated_state.session_id == conversation_state.session_id
            assert updated_state.user_id == conversation_state.user_id
            assert len(updated_state.conversation_history) == i + 1
            
            # Set context variable
            await service.set_session_context(session_id, f"var_{i}", f"value_{i}")
        
        # Verify final state
        final_state = await service.get_conversation_state(session_id)
        assert final_state is not None
        assert len(final_state.conversation_history) == 10
        
        # Verify context variables
        for i in range(10):
            value = await service.get_session_context(session_id, f"var_{i}")
            assert value == f"value_{i}"
        
        # Verify conversation history
        history = await service.get_conversation_history(session_id)
        assert len(history) == 10
        
    finally:
        await service.shutdown()


class TestEnhancedProfileManagement:
    """Test cases for enhanced user profile management."""
    
    @pytest.fixture
    async def service(self):
        """Create a context management service with enhanced profile management."""
        service = ContextManagementService(
            session_timeout_minutes=30,
            learning_rate=0.2
        )
        yield service
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_enhanced_user_profile(self, service):
        """Test creating user profile with enhanced features."""
        user_id = str(uuid4())
        
        initial_preferences = {
            "preferred_languages": [LanguageCode.MARATHI, LanguageCode.ENGLISH_IN],
            "primary_language": LanguageCode.MARATHI,
            "privacy_settings": {
                "data_retention_days": 60,
                "allow_analytics": True,
                "location_sharing": True
            }
        }
        
        profile = await service.create_user_profile(user_id, initial_preferences)
        
        assert profile.user_id == UUID(user_id)
        assert LanguageCode.MARATHI in profile.preferred_languages
        assert profile.primary_language == LanguageCode.MARATHI
        assert profile.privacy_settings.data_retention_days == 60
    
    @pytest.mark.asyncio
    async def test_update_user_location_with_privacy(self, service):
        """Test updating user location with privacy compliance."""
        user_id = str(uuid4())
        
        # Create profile first
        await service.create_user_profile(user_id)
        
        location_data = {
            "latitude": 19.076090,
            "longitude": 72.877426,
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India",
            "postal_code": "400001",
            "timezone": "Asia/Kolkata"
        }
        
        updated_profile = await service.update_user_location(
            user_id, location_data, privacy_compliant=True
        )
        
        assert updated_profile is not None
        assert updated_profile.location.city == "Mumbai"
        # Check that precision was reduced for privacy
        assert updated_profile.location.latitude == 19.08
        assert updated_profile.location.longitude == 72.88
    
    @pytest.mark.asyncio
    async def test_get_user_regional_context(self, service):
        """Test getting regional context for user location."""
        user_id = str(uuid4())
        
        # Create profile with location
        await service.create_user_profile(user_id)
        
        location_data = {
            "latitude": 13.0827,
            "longitude": 80.2707,
            "city": "Chennai",
            "state": "Tamil Nadu",
            "country": "India"
        }
        
        await service.update_user_location(user_id, location_data)
        
        # Get regional context
        context = await service.get_user_regional_context(user_id)
        
        assert context is not None
        assert context.location.city == "Chennai"
        assert context.local_language == LanguageCode.TAMIL
    
    @pytest.mark.asyncio
    async def test_enhanced_learning_from_interaction(self, service):
        """Test enhanced learning from user interactions."""
        user_id = str(uuid4())
        
        # Create profile
        await service.create_user_profile(user_id)
        
        # Create multiple interactions to trigger learning
        interactions = [
            UserInteraction(
                user_id=UUID(user_id),
                input_text="Hello, how are you?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'm fine, thank you",
                response_language=LanguageCode.ENGLISH_IN,
                intent="greeting",
                confidence=0.9,
                processing_time=0.1
            ),
            UserInteraction(
                user_id=UUID(user_id),
                input_text="What's the weather like?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="It's sunny today",
                response_language=LanguageCode.ENGLISH_IN,
                intent="weather.query",
                confidence=0.92,
                processing_time=0.08
            ),
            UserInteraction(
                user_id=UUID(user_id),
                input_text="नमस्ते, आप कैसे हैं?",
                input_language=LanguageCode.HINDI,
                response_text="मैं ठीक हूं, धन्यवाद",
                response_language=LanguageCode.HINDI,
                intent="greeting",
                confidence=0.88,
                processing_time=0.12
            )
        ]
        
        # Process interactions
        for interaction in interactions:
            learning_result = await service.learn_from_interaction(interaction)
            assert "user_id" in learning_result
        
        # Get learning insights
        insights = await service.get_profile_learning_insights(user_id)
        
        assert insights["user_id"] == user_id
        assert insights["total_interactions"] == 3
        assert "english-in" in insights["language_usage_frequency"]
        assert "hindi" in insights["language_usage_frequency"]
    
    @pytest.mark.asyncio
    async def test_profile_deletion_with_compliance(self, service):
        """Test profile deletion with compliance logging."""
        user_id = str(uuid4())
        
        # Create profile
        await service.create_user_profile(user_id)
        
        # Verify profile exists
        profile = await service.get_user_profile(user_id)
        assert profile is not None
        
        # Delete profile
        deleted = await service.delete_user_profile(user_id, "user_request")
        assert deleted is True
        
        # Verify profile is deleted
        profile = await service.get_user_profile(user_id)
        assert profile is None
    
    @pytest.mark.asyncio
    async def test_privacy_compliant_operations(self, service):
        """Test privacy-compliant profile operations."""
        user_id = str(uuid4())
        
        # Create profile with strict privacy settings
        initial_preferences = {
            "privacy_settings": {
                "allow_analytics": False,
                "allow_personalization": False,
                "location_sharing": False,
                "data_retention_days": 30
            }
        }
        
        profile = await service.create_user_profile(user_id, initial_preferences)
        
        # Try to update location (should be blocked due to privacy settings)
        location_data = {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "city": "Bangalore",
            "state": "Karnataka"
        }
        
        updated_profile = await service.update_user_location(user_id, location_data)
        
        # Location should not be updated due to privacy settings
        assert updated_profile.location is None
    
    @pytest.mark.asyncio
    async def test_enhanced_service_statistics(self, service):
        """Test enhanced service statistics including profile management."""
        # Create some profiles
        for i in range(3):
            user_id = str(uuid4())
            await service.create_user_profile(user_id)
        
        stats = await service.get_service_statistics()
        
        assert "conversation_sessions" in stats
        assert "user_profiles" in stats
        assert "enhanced_profile_management" in stats
        assert stats["enhanced_profile_management"]["total_profiles"] >= 3
    
    @pytest.mark.asyncio
    async def test_adaptive_language_learning_integration(self, service):
        """Test adaptive language learning integration with context service."""
        user_id = str(uuid4())
        
        # Create profile with initial preferences
        initial_preferences = {
            "preferred_languages": [LanguageCode.HINDI],
            "primary_language": LanguageCode.HINDI
        }
        
        await service.create_user_profile(user_id, initial_preferences)
        
        # Create interactions that should trigger language adaptation
        for i in range(6):  # Enough interactions to trigger learning
            interaction = UserInteraction(
                user_id=UUID(user_id),
                input_text=f"English query {i}",
                input_language=LanguageCode.ENGLISH_IN,
                response_text=f"English response {i}",
                response_language=LanguageCode.ENGLISH_IN,
                intent="test.query",
                confidence=0.9,
                processing_time=0.1
            )
            
            await service.learn_from_interaction(interaction)
        
        # Check if English was added to preferences
        profile = await service.get_user_profile(user_id)
        assert profile is not None
        
        # English should be added due to frequent usage
        language_values = [lang.value for lang in profile.preferred_languages]
        assert "en-IN" in language_values or LanguageCode.ENGLISH_IN in profile.preferred_languages
    
    @pytest.mark.asyncio
    async def test_location_context_caching(self, service):
        """Test that location context is properly cached."""
        user_id1 = str(uuid4())
        user_id2 = str(uuid4())
        
        # Create two profiles with same location
        location_data = {
            "latitude": 28.6139,
            "longitude": 77.2090,
            "city": "Delhi",
            "state": "Delhi",
            "country": "India"
        }
        
        await service.create_user_profile(user_id1)
        await service.create_user_profile(user_id2)
        
        await service.update_user_location(user_id1, location_data)
        await service.update_user_location(user_id2, location_data)
        
        # Get regional context for both users
        context1 = await service.get_user_regional_context(user_id1)
        context2 = await service.get_user_regional_context(user_id2)
        
        # Both should have same context data
        assert context1 is not None
        assert context2 is not None
        assert context1.location.city == context2.location.city
        assert context1.local_language == context2.local_language


if __name__ == "__main__":
    pytest.main([__file__])