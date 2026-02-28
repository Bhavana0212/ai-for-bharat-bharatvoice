<<<<<<< HEAD
"""
Tests for User Profile Manager.

This module tests the enhanced user profile management functionality including
privacy compliance, encryption, adaptive learning, and location-based context.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch

from bharatvoice.core.models import (
    UserProfile,
    UserInteraction,
    LanguageCode,
    LocationData,
    UsageAnalytics,
    PrivacyConfiguration,
)
from bharatvoice.services.context_management.user_profile_manager import (
    UserProfileManager,
    ProfileEncryption,
    LanguageLearningEngine,
    LocationContextManager,
)


class TestProfileEncryption:
    """Test profile encryption functionality."""
    
    def test_encryption_initialization(self):
        """Test encryption initialization."""
        encryption = ProfileEncryption()
        assert encryption._cipher_suite is not None
        assert encryption.master_key is not None
    
    def test_encryption_with_custom_key(self):
        """Test encryption with custom master key."""
        custom_key = "test_master_key_123"
        encryption = ProfileEncryption(custom_key)
        assert encryption.master_key == custom_key.encode()
    
    def test_data_encryption_decryption(self):
        """Test data encryption and decryption."""
        encryption = ProfileEncryption()
        
        test_data = {
            "user_id": "test-user-123",
            "preferences": ["hindi", "english"],
            "location": {"city": "Mumbai", "state": "Maharashtra"}
        }
        
        # Encrypt data
        encrypted_data = encryption.encrypt_data(test_data)
        assert isinstance(encrypted_data, str)
        assert encrypted_data != str(test_data)
        
        # Decrypt data
        decrypted_data = encryption.decrypt_data(encrypted_data)
        assert decrypted_data == test_data
    
    def test_encryption_with_complex_data(self):
        """Test encryption with complex data structures."""
        encryption = ProfileEncryption()
        
        complex_data = {
            "nested": {
                "array": [1, 2, 3],
                "boolean": True,
                "null": None
            },
            "datetime": datetime.utcnow().isoformat(),
            "unicode": "हिंदी भाषा"
        }
        
        encrypted_data = encryption.encrypt_data(complex_data)
        decrypted_data = encryption.decrypt_data(encrypted_data)
        
        # Note: datetime objects become strings after JSON serialization
        assert decrypted_data["nested"]["array"] == [1, 2, 3]
        assert decrypted_data["nested"]["boolean"] is True
        assert decrypted_data["nested"]["null"] is None
        assert decrypted_data["unicode"] == "हिंदी भाषा"


class TestLanguageLearningEngine:
    """Test language learning and adaptation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.learning_engine = LanguageLearningEngine(learning_rate=0.1, min_interactions=3)
        self.user_id = uuid4()
    
    def create_interaction(self, language: LanguageCode, intent: str = "general.query") -> UserInteraction:
        """Create a test interaction."""
        return UserInteraction(
            user_id=self.user_id,
            input_text="Test query",
            input_language=language,
            response_text="Test response",
            response_language=language,
            intent=intent,
            confidence=0.9,
            processing_time=0.1
        )
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient interaction data."""
        interactions = [
            self.create_interaction(LanguageCode.HINDI),
            self.create_interaction(LanguageCode.ENGLISH_IN)
        ]
        
        current_preferences = [LanguageCode.HINDI]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        assert updated_preferences == current_preferences
        assert insights["reason"] == "insufficient_data"
    
    def test_new_language_detection(self):
        """Test detection and addition of new frequently used languages."""
        interactions = []
        
        # Create interactions with Tamil (not in current preferences)
        for _ in range(5):
            interactions.append(self.create_interaction(LanguageCode.TAMIL))
        
        # Add some Hindi interactions
        for _ in range(3):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        # Tamil should be added as it's used >20% of the time
        assert LanguageCode.TAMIL in updated_preferences
        assert len(insights["new_languages_detected"]) > 0
        assert "added_tamil" in insights["adaptations_made"]
    
    def test_preference_reordering(self):
        """Test reordering of preferences based on recent usage."""
        interactions = []
        
        # Create 15 interactions to trigger reordering
        # Recent interactions favor English
        for _ in range(8):
            interactions.append(self.create_interaction(LanguageCode.ENGLISH_IN))
        
        for _ in range(7):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        # English should be first due to higher recent usage
        assert updated_preferences[0] == LanguageCode.ENGLISH_IN
        assert "reordered_by_usage" in insights["adaptations_made"]
    
    def test_primary_language_shift_detection(self):
        """Test detection of primary language shifts."""
        interactions = []
        
        # Create interactions heavily favoring Tamil
        for _ in range(15):
            interactions.append(self.create_interaction(LanguageCode.TAMIL))
        
        for _ in range(3):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_primary = LanguageCode.HINDI
        
        new_primary, should_change = self.learning_engine.detect_primary_language_shift(
            interactions, current_primary
        )
        
        assert should_change is True
        assert new_primary == LanguageCode.TAMIL
    
    def test_no_primary_language_shift(self):
        """Test when primary language should not change."""
        interactions = []
        
        # Balanced usage, no clear preference
        for _ in range(5):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        for _ in range(4):
            interactions.append(self.create_interaction(LanguageCode.ENGLISH_IN))
        
        current_primary = LanguageCode.HINDI
        
        new_primary, should_change = self.learning_engine.detect_primary_language_shift(
            interactions, current_primary
        )
        
        assert should_change is False
        assert new_primary == current_primary


class TestLocationContextManager:
    """Test location-based context management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.location_manager = LocationContextManager()
        self.user_profile = UserProfile(user_id=uuid4())
    
    def test_location_update_with_privacy(self):
        """Test location update with privacy compliance."""
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data, privacy_compliant=True
        )
        
        # Check that precision was reduced for privacy
        assert updated_profile.location.latitude == 19.08  # Rounded to 2 decimal places
        assert updated_profile.location.longitude == 72.88
        assert updated_profile.location.city == "Mumbai"
    
    def test_location_update_without_privacy_filters(self):
        """Test location update without privacy filters."""
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data, privacy_compliant=False
        )
        
        # Check that original precision is maintained
        assert updated_profile.location.latitude == 19.076090
        assert updated_profile.location.longitude == 72.877426
        assert updated_profile.location.postal_code == "400001"
    
    def test_location_sharing_disabled(self):
        """Test behavior when location sharing is disabled."""
        self.user_profile.privacy_settings.location_sharing = False
        
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data
        )
        
        # Location should not be updated
        assert updated_profile.location is None
    
    def test_regional_context_generation(self):
        """Test regional context generation."""
        location_data = LocationData(
            latitude=13.0827,
            longitude=80.2707,
            city="Chennai",
            state="Tamil Nadu",
            country="India"
        )
        
        user_preferences = [LanguageCode.TAMIL, LanguageCode.ENGLISH_IN]
        
        context = self.location_manager.get_location_context(location_data, user_preferences)
        
        assert context is not None
        assert context.location.city == "Chennai"
        assert context.local_language == LanguageCode.TAMIL
        assert "Tamil" in context.dialect_info
    
    def test_context_caching(self):
        """Test that regional context is cached properly."""
        location_data = LocationData(
            latitude=28.6139,
            longitude=77.2090,
            city="Delhi",
            state="Delhi",
            country="India"
        )
        
        user_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        # First call
        context1 = self.location_manager.get_location_context(location_data, user_preferences)
        
        # Second call should return cached result
        context2 = self.location_manager.get_location_context(location_data, user_preferences)
        
        assert context1 is context2  # Same object reference due to caching


class TestUserProfileManager:
    """Test comprehensive user profile management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profile_manager = UserProfileManager(learning_rate=0.2)
        self.user_id = uuid4()
    
    @pytest.mark.asyncio
    async def test_profile_creation(self):
        """Test user profile creation."""
        initial_preferences = {
            "preferred_languages": [LanguageCode.HINDI, LanguageCode.GUJARATI],
            "primary_language": LanguageCode.GUJARATI
        }
        
        profile = await self.profile_manager.create_profile(
            self.user_id, initial_preferences
        )
        
        assert profile.user_id == self.user_id
        assert LanguageCode.GUJARATI in profile.preferred_languages
        assert profile.primary_language == LanguageCode.GUJARATI
    
    @pytest.mark.asyncio
    async def test_profile_retrieval(self):
        """Test profile retrieval and caching."""
        # Create profile
        profile = await self.profile_manager.create_profile(self.user_id)
        
        # Retrieve profile
        retrieved_profile = await self.profile_manager.get_profile(self.user_id)
        
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == self.user_id
    
    @pytest.mark.asyncio
    async def test_profile_update(self):
        """Test profile updates with privacy compliance."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Update preferences
        preferences = {
            "preferred_languages": [LanguageCode.TAMIL, LanguageCode.ENGLISH_IN],
            "privacy_settings": {
                "data_retention_days": 60,
                "allow_analytics": False
            }
        }
        
        updated_profile = await self.profile_manager.update_profile(
            self.user_id, preferences
        )
        
        assert updated_profile is not None
        assert LanguageCode.TAMIL in updated_profile.preferred_languages
        assert updated_profile.privacy_settings.data_retention_days == 60
        assert updated_profile.privacy_settings.allow_analytics is False
    
    @pytest.mark.asyncio
    async def test_learning_from_interaction(self):
        """Test adaptive learning from user interactions."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create interaction
        interaction = UserInteraction(
            user_id=self.user_id,
            input_text="मुंबई का मौसम कैसा है?",
            input_language=LanguageCode.HINDI,
            response_text="Mumbai weather is sunny",
            response_language=LanguageCode.ENGLISH_IN,
            intent="weather.query",
            confidence=0.92,
            processing_time=0.15
        )
        
        # Learn from interaction
        learning_result = await self.profile_manager.learn_from_interaction(
            self.user_id, interaction
        )
        
        assert learning_result["user_id"] == str(self.user_id)
        assert learning_result["total_interactions"] == 1
        
        # Check that profile was updated
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile.usage_patterns.total_interactions == 1
        assert LanguageCode.HINDI in profile.usage_patterns.language_usage_frequency
    
    @pytest.mark.asyncio
    async def test_location_update(self):
        """Test location updates with privacy compliance."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        location_data = LocationData(
            latitude=22.5726,
            longitude=88.3639,
            city="Kolkata",
            state="West Bengal",
            country="India"
        )
        
        updated_profile = await self.profile_manager.update_location(
            self.user_id, location_data
        )
        
        assert updated_profile is not None
        assert updated_profile.location.city == "Kolkata"
        assert updated_profile.location.state == "West Bengal"
    
    @pytest.mark.asyncio
    async def test_regional_context_retrieval(self):
        """Test regional context retrieval for user location."""
        # Create profile with location
        location_data = LocationData(
            latitude=17.3850,
            longitude=78.4867,
            city="Hyderabad",
            state="Telangana",
            country="India"
        )
        
        await self.profile_manager.create_profile(self.user_id)
        await self.profile_manager.update_location(self.user_id, location_data)
        
        # Get regional context
        context = await self.profile_manager.get_regional_context(self.user_id)
        
        assert context is not None
        assert context.location.city == "Hyderabad"
        assert context.local_language == LanguageCode.TELUGU
    
    @pytest.mark.asyncio
    async def test_profile_deletion(self):
        """Test profile deletion with compliance logging."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Verify profile exists
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile is not None
        
        # Delete profile
        deleted = await self.profile_manager.delete_profile(
            self.user_id, "user_request"
        )
        
        assert deleted is True
        
        # Verify profile is deleted
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile is None
    
    @pytest.mark.asyncio
    async def test_profile_statistics(self):
        """Test profile management statistics."""
        # Create multiple profiles
        user_ids = [uuid4() for _ in range(3)]
        
        for user_id in user_ids:
            await self.profile_manager.create_profile(user_id)
        
        # Get statistics
        stats = await self.profile_manager.get_profile_statistics()
        
        assert stats["total_profiles"] >= 3
        assert stats["active_profiles"] >= 3
        assert "average_interactions_per_user" in stats
    
    @pytest.mark.asyncio
    async def test_adaptive_language_learning(self):
        """Test adaptive language learning over multiple interactions."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create multiple interactions in different languages
        interactions = [
            UserInteraction(
                user_id=self.user_id,
                input_text="Hello, how are you?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'm fine, thank you",
                response_language=LanguageCode.ENGLISH_IN,
                intent="greeting",
                confidence=0.9,
                processing_time=0.1
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="வணக்கம், எப்படி இருக்கீங்க?",
                input_language=LanguageCode.TAMIL,
                response_text="நான் நல்லா இருக்கேன்",
                response_language=LanguageCode.TAMIL,
                intent="greeting",
                confidence=0.85,
                processing_time=0.12
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="What's the weather like?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="It's sunny today",
                response_language=LanguageCode.ENGLISH_IN,
                intent="weather.query",
                confidence=0.92,
                processing_time=0.08
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="இன்றைய வானிலை எப்படி?",
                input_language=LanguageCode.TAMIL,
                response_text="இன்று வெயில் அதிகம்",
                response_language=LanguageCode.TAMIL,
                intent="weather.query",
                confidence=0.88,
                processing_time=0.11
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="Good morning!",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="Good morning to you too!",
                response_language=LanguageCode.ENGLISH_IN,
                intent="greeting",
                confidence=0.95,
                processing_time=0.07
            )
        ]
        
        # Process interactions
        for interaction in interactions:
            await self.profile_manager.learn_from_interaction(self.user_id, interaction)
        
        # Check final profile state
        profile = await self.profile_manager.get_profile(self.user_id)
        
        assert profile.usage_patterns.total_interactions == 5
        assert LanguageCode.ENGLISH_IN in profile.usage_patterns.language_usage_frequency
        assert LanguageCode.TAMIL in profile.usage_patterns.language_usage_frequency
        
        # English should have higher frequency (3/5 = 0.6)
        english_freq = profile.usage_patterns.language_usage_frequency[LanguageCode.ENGLISH_IN]
        tamil_freq = profile.usage_patterns.language_usage_frequency[LanguageCode.TAMIL]
        
        assert english_freq > tamil_freq
        assert abs(english_freq - 0.6) < 0.1  # Allow for floating point precision
        assert abs(tamil_freq - 0.4) < 0.1
    
    @pytest.mark.asyncio
    async def test_privacy_compliant_operations(self):
        """Test privacy-compliant profile operations."""
        # Create profile with strict privacy settings
        initial_preferences = {
            "privacy_settings": {
                "allow_analytics": False,
                "allow_personalization": False,
                "location_sharing": False,
                "data_retention_days": 30
            }
        }
        
        profile = await self.profile_manager.create_profile(
            self.user_id, initial_preferences
        )
        
        # Try to update location (should be blocked)
        location_data = LocationData(
            latitude=12.9716,
            longitude=77.5946,
            city="Bangalore",
            state="Karnataka"
        )
        
        updated_profile = await self.profile_manager.update_location(
            self.user_id, location_data
        )
        
        # Location should not be updated due to privacy settings
        assert updated_profile.location is None
    
    @pytest.mark.asyncio
    async def test_concurrent_profile_operations(self):
        """Test concurrent profile operations with proper locking."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create multiple concurrent interactions
        async def create_interaction(lang: LanguageCode, intent: str):
            interaction = UserInteraction(
                user_id=self.user_id,
                input_text=f"Test query in {lang.value}",
                input_language=lang,
                response_text=f"Response in {lang.value}",
                response_language=lang,
                intent=intent,
                confidence=0.9,
                processing_time=0.1
            )
            return await self.profile_manager.learn_from_interaction(self.user_id, interaction)
        
        # Run concurrent operations
        tasks = [
            create_interaction(LanguageCode.HINDI, "test.query1"),
            create_interaction(LanguageCode.ENGLISH_IN, "test.query2"),
            create_interaction(LanguageCode.TAMIL, "test.query3"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 3
        for result in results:
            assert "user_id" in result
        
        # Final profile should have all interactions
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile.usage_patterns.total_interactions == 3
    
    @pytest.mark.asyncio
    async def test_profile_manager_shutdown(self):
        """Test proper shutdown of profile manager."""
        # Create some profiles
        for _ in range(2):
            await self.profile_manager.create_profile(uuid4())
        
        # Shutdown should complete without errors
        await self.profile_manager.shutdown()
        
        # Cleanup task should be cancelled
        assert self.profile_manager._cleanup_task.cancelled()


@pytest.mark.asyncio
async def test_integration_with_context_service():
    """Test integration between profile manager and context service."""
    from bharatvoice.services.context_management.service import ContextManagementService
    
    # Create service with enhanced profile management
    service = ContextManagementService(
        session_timeout_minutes=30,
        learning_rate=0.15
    )
    
    user_id = str(uuid4())
    
    try:
        # Create user profile
        profile = await service.create_user_profile(user_id, {
            "preferred_languages": [LanguageCode.MARATHI, LanguageCode.ENGLISH_IN],
            "primary_language": LanguageCode.MARATHI
        })
        
        assert profile.primary_language == LanguageCode.MARATHI
        
        # Update location
        location_data = {
            "latitude": 19.0760,
            "longitude": 72.8777,
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India"
        }
        
        updated_profile = await service.update_user_location(user_id, location_data)
        assert updated_profile.location.city == "Mumbai"
        
        # Get regional context
        context = await service.get_user_regional_context(user_id)
        assert context is not None
        assert context.local_language == LanguageCode.MARATHI
        
        # Get learning insights
        insights = await service.get_profile_learning_insights(user_id)
        assert insights["user_id"] == user_id
        assert insights["primary_language"] == "mr"
        
    finally:
=======
"""
Tests for User Profile Manager.

This module tests the enhanced user profile management functionality including
privacy compliance, encryption, adaptive learning, and location-based context.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch

from bharatvoice.core.models import (
    UserProfile,
    UserInteraction,
    LanguageCode,
    LocationData,
    UsageAnalytics,
    PrivacyConfiguration,
)
from bharatvoice.services.context_management.user_profile_manager import (
    UserProfileManager,
    ProfileEncryption,
    LanguageLearningEngine,
    LocationContextManager,
)


class TestProfileEncryption:
    """Test profile encryption functionality."""
    
    def test_encryption_initialization(self):
        """Test encryption initialization."""
        encryption = ProfileEncryption()
        assert encryption._cipher_suite is not None
        assert encryption.master_key is not None
    
    def test_encryption_with_custom_key(self):
        """Test encryption with custom master key."""
        custom_key = "test_master_key_123"
        encryption = ProfileEncryption(custom_key)
        assert encryption.master_key == custom_key.encode()
    
    def test_data_encryption_decryption(self):
        """Test data encryption and decryption."""
        encryption = ProfileEncryption()
        
        test_data = {
            "user_id": "test-user-123",
            "preferences": ["hindi", "english"],
            "location": {"city": "Mumbai", "state": "Maharashtra"}
        }
        
        # Encrypt data
        encrypted_data = encryption.encrypt_data(test_data)
        assert isinstance(encrypted_data, str)
        assert encrypted_data != str(test_data)
        
        # Decrypt data
        decrypted_data = encryption.decrypt_data(encrypted_data)
        assert decrypted_data == test_data
    
    def test_encryption_with_complex_data(self):
        """Test encryption with complex data structures."""
        encryption = ProfileEncryption()
        
        complex_data = {
            "nested": {
                "array": [1, 2, 3],
                "boolean": True,
                "null": None
            },
            "datetime": datetime.utcnow().isoformat(),
            "unicode": "हिंदी भाषा"
        }
        
        encrypted_data = encryption.encrypt_data(complex_data)
        decrypted_data = encryption.decrypt_data(encrypted_data)
        
        # Note: datetime objects become strings after JSON serialization
        assert decrypted_data["nested"]["array"] == [1, 2, 3]
        assert decrypted_data["nested"]["boolean"] is True
        assert decrypted_data["nested"]["null"] is None
        assert decrypted_data["unicode"] == "हिंदी भाषा"


class TestLanguageLearningEngine:
    """Test language learning and adaptation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.learning_engine = LanguageLearningEngine(learning_rate=0.1, min_interactions=3)
        self.user_id = uuid4()
    
    def create_interaction(self, language: LanguageCode, intent: str = "general.query") -> UserInteraction:
        """Create a test interaction."""
        return UserInteraction(
            user_id=self.user_id,
            input_text="Test query",
            input_language=language,
            response_text="Test response",
            response_language=language,
            intent=intent,
            confidence=0.9,
            processing_time=0.1
        )
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient interaction data."""
        interactions = [
            self.create_interaction(LanguageCode.HINDI),
            self.create_interaction(LanguageCode.ENGLISH_IN)
        ]
        
        current_preferences = [LanguageCode.HINDI]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        assert updated_preferences == current_preferences
        assert insights["reason"] == "insufficient_data"
    
    def test_new_language_detection(self):
        """Test detection and addition of new frequently used languages."""
        interactions = []
        
        # Create interactions with Tamil (not in current preferences)
        for _ in range(5):
            interactions.append(self.create_interaction(LanguageCode.TAMIL))
        
        # Add some Hindi interactions
        for _ in range(3):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        # Tamil should be added as it's used >20% of the time
        assert LanguageCode.TAMIL in updated_preferences
        assert len(insights["new_languages_detected"]) > 0
        assert "added_tamil" in insights["adaptations_made"]
    
    def test_preference_reordering(self):
        """Test reordering of preferences based on recent usage."""
        interactions = []
        
        # Create 15 interactions to trigger reordering
        # Recent interactions favor English
        for _ in range(8):
            interactions.append(self.create_interaction(LanguageCode.ENGLISH_IN))
        
        for _ in range(7):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        updated_preferences, insights = self.learning_engine.analyze_language_patterns(
            interactions, current_preferences
        )
        
        # English should be first due to higher recent usage
        assert updated_preferences[0] == LanguageCode.ENGLISH_IN
        assert "reordered_by_usage" in insights["adaptations_made"]
    
    def test_primary_language_shift_detection(self):
        """Test detection of primary language shifts."""
        interactions = []
        
        # Create interactions heavily favoring Tamil
        for _ in range(15):
            interactions.append(self.create_interaction(LanguageCode.TAMIL))
        
        for _ in range(3):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        current_primary = LanguageCode.HINDI
        
        new_primary, should_change = self.learning_engine.detect_primary_language_shift(
            interactions, current_primary
        )
        
        assert should_change is True
        assert new_primary == LanguageCode.TAMIL
    
    def test_no_primary_language_shift(self):
        """Test when primary language should not change."""
        interactions = []
        
        # Balanced usage, no clear preference
        for _ in range(5):
            interactions.append(self.create_interaction(LanguageCode.HINDI))
        
        for _ in range(4):
            interactions.append(self.create_interaction(LanguageCode.ENGLISH_IN))
        
        current_primary = LanguageCode.HINDI
        
        new_primary, should_change = self.learning_engine.detect_primary_language_shift(
            interactions, current_primary
        )
        
        assert should_change is False
        assert new_primary == current_primary


class TestLocationContextManager:
    """Test location-based context management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.location_manager = LocationContextManager()
        self.user_profile = UserProfile(user_id=uuid4())
    
    def test_location_update_with_privacy(self):
        """Test location update with privacy compliance."""
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data, privacy_compliant=True
        )
        
        # Check that precision was reduced for privacy
        assert updated_profile.location.latitude == 19.08  # Rounded to 2 decimal places
        assert updated_profile.location.longitude == 72.88
        assert updated_profile.location.city == "Mumbai"
    
    def test_location_update_without_privacy_filters(self):
        """Test location update without privacy filters."""
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra",
            country="India",
            postal_code="400001",
            timezone="Asia/Kolkata"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data, privacy_compliant=False
        )
        
        # Check that original precision is maintained
        assert updated_profile.location.latitude == 19.076090
        assert updated_profile.location.longitude == 72.877426
        assert updated_profile.location.postal_code == "400001"
    
    def test_location_sharing_disabled(self):
        """Test behavior when location sharing is disabled."""
        self.user_profile.privacy_settings.location_sharing = False
        
        location_data = LocationData(
            latitude=19.076090,
            longitude=72.877426,
            city="Mumbai",
            state="Maharashtra"
        )
        
        updated_profile = self.location_manager.update_user_location(
            self.user_profile, location_data
        )
        
        # Location should not be updated
        assert updated_profile.location is None
    
    def test_regional_context_generation(self):
        """Test regional context generation."""
        location_data = LocationData(
            latitude=13.0827,
            longitude=80.2707,
            city="Chennai",
            state="Tamil Nadu",
            country="India"
        )
        
        user_preferences = [LanguageCode.TAMIL, LanguageCode.ENGLISH_IN]
        
        context = self.location_manager.get_location_context(location_data, user_preferences)
        
        assert context is not None
        assert context.location.city == "Chennai"
        assert context.local_language == LanguageCode.TAMIL
        assert "Tamil" in context.dialect_info
    
    def test_context_caching(self):
        """Test that regional context is cached properly."""
        location_data = LocationData(
            latitude=28.6139,
            longitude=77.2090,
            city="Delhi",
            state="Delhi",
            country="India"
        )
        
        user_preferences = [LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
        
        # First call
        context1 = self.location_manager.get_location_context(location_data, user_preferences)
        
        # Second call should return cached result
        context2 = self.location_manager.get_location_context(location_data, user_preferences)
        
        assert context1 is context2  # Same object reference due to caching


class TestUserProfileManager:
    """Test comprehensive user profile management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profile_manager = UserProfileManager(learning_rate=0.2)
        self.user_id = uuid4()
    
    @pytest.mark.asyncio
    async def test_profile_creation(self):
        """Test user profile creation."""
        initial_preferences = {
            "preferred_languages": [LanguageCode.HINDI, LanguageCode.GUJARATI],
            "primary_language": LanguageCode.GUJARATI
        }
        
        profile = await self.profile_manager.create_profile(
            self.user_id, initial_preferences
        )
        
        assert profile.user_id == self.user_id
        assert LanguageCode.GUJARATI in profile.preferred_languages
        assert profile.primary_language == LanguageCode.GUJARATI
    
    @pytest.mark.asyncio
    async def test_profile_retrieval(self):
        """Test profile retrieval and caching."""
        # Create profile
        profile = await self.profile_manager.create_profile(self.user_id)
        
        # Retrieve profile
        retrieved_profile = await self.profile_manager.get_profile(self.user_id)
        
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == self.user_id
    
    @pytest.mark.asyncio
    async def test_profile_update(self):
        """Test profile updates with privacy compliance."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Update preferences
        preferences = {
            "preferred_languages": [LanguageCode.TAMIL, LanguageCode.ENGLISH_IN],
            "privacy_settings": {
                "data_retention_days": 60,
                "allow_analytics": False
            }
        }
        
        updated_profile = await self.profile_manager.update_profile(
            self.user_id, preferences
        )
        
        assert updated_profile is not None
        assert LanguageCode.TAMIL in updated_profile.preferred_languages
        assert updated_profile.privacy_settings.data_retention_days == 60
        assert updated_profile.privacy_settings.allow_analytics is False
    
    @pytest.mark.asyncio
    async def test_learning_from_interaction(self):
        """Test adaptive learning from user interactions."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create interaction
        interaction = UserInteraction(
            user_id=self.user_id,
            input_text="मुंबई का मौसम कैसा है?",
            input_language=LanguageCode.HINDI,
            response_text="Mumbai weather is sunny",
            response_language=LanguageCode.ENGLISH_IN,
            intent="weather.query",
            confidence=0.92,
            processing_time=0.15
        )
        
        # Learn from interaction
        learning_result = await self.profile_manager.learn_from_interaction(
            self.user_id, interaction
        )
        
        assert learning_result["user_id"] == str(self.user_id)
        assert learning_result["total_interactions"] == 1
        
        # Check that profile was updated
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile.usage_patterns.total_interactions == 1
        assert LanguageCode.HINDI in profile.usage_patterns.language_usage_frequency
    
    @pytest.mark.asyncio
    async def test_location_update(self):
        """Test location updates with privacy compliance."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        location_data = LocationData(
            latitude=22.5726,
            longitude=88.3639,
            city="Kolkata",
            state="West Bengal",
            country="India"
        )
        
        updated_profile = await self.profile_manager.update_location(
            self.user_id, location_data
        )
        
        assert updated_profile is not None
        assert updated_profile.location.city == "Kolkata"
        assert updated_profile.location.state == "West Bengal"
    
    @pytest.mark.asyncio
    async def test_regional_context_retrieval(self):
        """Test regional context retrieval for user location."""
        # Create profile with location
        location_data = LocationData(
            latitude=17.3850,
            longitude=78.4867,
            city="Hyderabad",
            state="Telangana",
            country="India"
        )
        
        await self.profile_manager.create_profile(self.user_id)
        await self.profile_manager.update_location(self.user_id, location_data)
        
        # Get regional context
        context = await self.profile_manager.get_regional_context(self.user_id)
        
        assert context is not None
        assert context.location.city == "Hyderabad"
        assert context.local_language == LanguageCode.TELUGU
    
    @pytest.mark.asyncio
    async def test_profile_deletion(self):
        """Test profile deletion with compliance logging."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Verify profile exists
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile is not None
        
        # Delete profile
        deleted = await self.profile_manager.delete_profile(
            self.user_id, "user_request"
        )
        
        assert deleted is True
        
        # Verify profile is deleted
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile is None
    
    @pytest.mark.asyncio
    async def test_profile_statistics(self):
        """Test profile management statistics."""
        # Create multiple profiles
        user_ids = [uuid4() for _ in range(3)]
        
        for user_id in user_ids:
            await self.profile_manager.create_profile(user_id)
        
        # Get statistics
        stats = await self.profile_manager.get_profile_statistics()
        
        assert stats["total_profiles"] >= 3
        assert stats["active_profiles"] >= 3
        assert "average_interactions_per_user" in stats
    
    @pytest.mark.asyncio
    async def test_adaptive_language_learning(self):
        """Test adaptive language learning over multiple interactions."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create multiple interactions in different languages
        interactions = [
            UserInteraction(
                user_id=self.user_id,
                input_text="Hello, how are you?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="I'm fine, thank you",
                response_language=LanguageCode.ENGLISH_IN,
                intent="greeting",
                confidence=0.9,
                processing_time=0.1
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="வணக்கம், எப்படி இருக்கீங்க?",
                input_language=LanguageCode.TAMIL,
                response_text="நான் நல்லா இருக்கேன்",
                response_language=LanguageCode.TAMIL,
                intent="greeting",
                confidence=0.85,
                processing_time=0.12
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="What's the weather like?",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="It's sunny today",
                response_language=LanguageCode.ENGLISH_IN,
                intent="weather.query",
                confidence=0.92,
                processing_time=0.08
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="இன்றைய வானிலை எப்படி?",
                input_language=LanguageCode.TAMIL,
                response_text="இன்று வெயில் அதிகம்",
                response_language=LanguageCode.TAMIL,
                intent="weather.query",
                confidence=0.88,
                processing_time=0.11
            ),
            UserInteraction(
                user_id=self.user_id,
                input_text="Good morning!",
                input_language=LanguageCode.ENGLISH_IN,
                response_text="Good morning to you too!",
                response_language=LanguageCode.ENGLISH_IN,
                intent="greeting",
                confidence=0.95,
                processing_time=0.07
            )
        ]
        
        # Process interactions
        for interaction in interactions:
            await self.profile_manager.learn_from_interaction(self.user_id, interaction)
        
        # Check final profile state
        profile = await self.profile_manager.get_profile(self.user_id)
        
        assert profile.usage_patterns.total_interactions == 5
        assert LanguageCode.ENGLISH_IN in profile.usage_patterns.language_usage_frequency
        assert LanguageCode.TAMIL in profile.usage_patterns.language_usage_frequency
        
        # English should have higher frequency (3/5 = 0.6)
        english_freq = profile.usage_patterns.language_usage_frequency[LanguageCode.ENGLISH_IN]
        tamil_freq = profile.usage_patterns.language_usage_frequency[LanguageCode.TAMIL]
        
        assert english_freq > tamil_freq
        assert abs(english_freq - 0.6) < 0.1  # Allow for floating point precision
        assert abs(tamil_freq - 0.4) < 0.1
    
    @pytest.mark.asyncio
    async def test_privacy_compliant_operations(self):
        """Test privacy-compliant profile operations."""
        # Create profile with strict privacy settings
        initial_preferences = {
            "privacy_settings": {
                "allow_analytics": False,
                "allow_personalization": False,
                "location_sharing": False,
                "data_retention_days": 30
            }
        }
        
        profile = await self.profile_manager.create_profile(
            self.user_id, initial_preferences
        )
        
        # Try to update location (should be blocked)
        location_data = LocationData(
            latitude=12.9716,
            longitude=77.5946,
            city="Bangalore",
            state="Karnataka"
        )
        
        updated_profile = await self.profile_manager.update_location(
            self.user_id, location_data
        )
        
        # Location should not be updated due to privacy settings
        assert updated_profile.location is None
    
    @pytest.mark.asyncio
    async def test_concurrent_profile_operations(self):
        """Test concurrent profile operations with proper locking."""
        # Create profile
        await self.profile_manager.create_profile(self.user_id)
        
        # Create multiple concurrent interactions
        async def create_interaction(lang: LanguageCode, intent: str):
            interaction = UserInteraction(
                user_id=self.user_id,
                input_text=f"Test query in {lang.value}",
                input_language=lang,
                response_text=f"Response in {lang.value}",
                response_language=lang,
                intent=intent,
                confidence=0.9,
                processing_time=0.1
            )
            return await self.profile_manager.learn_from_interaction(self.user_id, interaction)
        
        # Run concurrent operations
        tasks = [
            create_interaction(LanguageCode.HINDI, "test.query1"),
            create_interaction(LanguageCode.ENGLISH_IN, "test.query2"),
            create_interaction(LanguageCode.TAMIL, "test.query3"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        assert len(results) == 3
        for result in results:
            assert "user_id" in result
        
        # Final profile should have all interactions
        profile = await self.profile_manager.get_profile(self.user_id)
        assert profile.usage_patterns.total_interactions == 3
    
    @pytest.mark.asyncio
    async def test_profile_manager_shutdown(self):
        """Test proper shutdown of profile manager."""
        # Create some profiles
        for _ in range(2):
            await self.profile_manager.create_profile(uuid4())
        
        # Shutdown should complete without errors
        await self.profile_manager.shutdown()
        
        # Cleanup task should be cancelled
        assert self.profile_manager._cleanup_task.cancelled()


@pytest.mark.asyncio
async def test_integration_with_context_service():
    """Test integration between profile manager and context service."""
    from bharatvoice.services.context_management.service import ContextManagementService
    
    # Create service with enhanced profile management
    service = ContextManagementService(
        session_timeout_minutes=30,
        learning_rate=0.15
    )
    
    user_id = str(uuid4())
    
    try:
        # Create user profile
        profile = await service.create_user_profile(user_id, {
            "preferred_languages": [LanguageCode.MARATHI, LanguageCode.ENGLISH_IN],
            "primary_language": LanguageCode.MARATHI
        })
        
        assert profile.primary_language == LanguageCode.MARATHI
        
        # Update location
        location_data = {
            "latitude": 19.0760,
            "longitude": 72.8777,
            "city": "Mumbai",
            "state": "Maharashtra",
            "country": "India"
        }
        
        updated_profile = await service.update_user_location(user_id, location_data)
        assert updated_profile.location.city == "Mumbai"
        
        # Get regional context
        context = await service.get_user_regional_context(user_id)
        assert context is not None
        assert context.local_language == LanguageCode.MARATHI
        
        # Get learning insights
        insights = await service.get_profile_learning_insights(user_id)
        assert insights["user_id"] == user_id
        assert insights["primary_language"] == "mr"
        
    finally:
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        await service.shutdown()