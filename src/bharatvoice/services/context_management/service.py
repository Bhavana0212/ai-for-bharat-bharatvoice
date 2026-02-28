"""
Context Management Service for BharatVoice Assistant.

This module implements the ContextManager interface, providing conversation state
management, user profile management, and regional context services.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from bharatvoice.core.interfaces import ContextManager
from bharatvoice.core.models import (
    ConversationState,
    RegionalContextData,
    UserInteraction,
    UserProfile,
    CulturalEvent,
    LocalService,
    WeatherData,
    LocationData,
)
from bharatvoice.services.context_management.conversation_manager import (
    ConversationContextManager,
    ConversationManager,
)
from bharatvoice.services.context_management.user_profile_manager import (
    UserProfileManager,
)
from bharatvoice.services.context_management.regional_context_manager import (
    RegionalContextManager,
)


logger = logging.getLogger(__name__)


class ContextManagementService(ContextManager):
    """
    Context Management Service implementation.
    
    Provides conversation state management, user profile management,
    and regional context services for the BharatVoice Assistant.
    """
    
    def __init__(
        self,
        session_timeout_minutes: int = 30,
        max_history_length: int = 50,
        cleanup_interval_minutes: int = 5,
        encryption_key: Optional[str] = None,
        learning_rate: float = 0.1
    ):
        """
        Initialize context management service.
        
        Args:
            session_timeout_minutes: Minutes before inactive session expires
            max_history_length: Maximum number of interactions to keep in history
            cleanup_interval_minutes: Minutes between cleanup runs
            encryption_key: Master encryption key for profile data
            learning_rate: Rate for adaptive learning
        """
        self.conversation_manager = ConversationManager(
            session_timeout_minutes=session_timeout_minutes,
            max_history_length=max_history_length,
            cleanup_interval_minutes=cleanup_interval_minutes
        )
        
        self.context_manager = ConversationContextManager(self.conversation_manager)
        
        # Initialize enhanced user profile manager
        self.profile_manager = UserProfileManager(
            encryption_key=encryption_key,
            learning_rate=learning_rate
        )
        
        # Initialize regional context manager
        self.regional_context_manager = RegionalContextManager()
        
        # Legacy storage for backward compatibility
        self._user_profiles: Dict[UUID, UserProfile] = {}
        self._regional_contexts: Dict[str, RegionalContextData] = {}
        
        logger.info("Context Management Service initialized with enhanced profile management")
    
    async def maintain_conversation_state(
        self,
        session_id: str,
        interaction: UserInteraction
    ) -> ConversationState:
        """
        Maintain and update conversation state.
        
        Args:
            session_id: Unique session identifier
            interaction: New user interaction to process
            
        Returns:
            Updated conversation state
            
        Raises:
            ValueError: If session doesn't exist
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            raise ValueError(f"Invalid session ID format: {session_id}")
        
        # Process interaction through context manager
        conversation_state = await self.context_manager.process_interaction(
            session_uuid, interaction
        )
        
        if conversation_state is None:
            # Session doesn't exist or has expired, create a new one
            logger.info(f"Session {session_id} not found, creating new session")
            conversation_state = await self.context_manager.start_conversation(
                user_id=interaction.user_id
            )
            
            # Process the interaction again with the new session
            conversation_state = await self.context_manager.process_interaction(
                conversation_state.session_id, interaction
            )
        
        return conversation_state
    
    async def update_user_profile(
        self,
        user_id: str,
        preferences: Dict[str, any]
    ) -> UserProfile:
        """
        Update user profile with new preferences using enhanced profile manager.
        
        Args:
            user_id: Unique user identifier
            preferences: Updated user preferences
            
        Returns:
            Updated user profile
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user ID format: {user_id}")
        
        # Use enhanced profile manager
        user_profile = await self.profile_manager.update_profile(
            user_uuid, preferences, privacy_compliant=True
        )
        
        if user_profile is None:
            raise ValueError(f"Failed to update profile for user {user_id}")
        
        # Update legacy storage for backward compatibility
        self._user_profiles[user_uuid] = user_profile
        
        logger.info(f"Updated user profile for {user_id} using enhanced profile manager")
        return user_profile
    
    async def get_regional_context(
        self,
        location: Dict[str, any]
    ) -> RegionalContextData:
        """
        Get regional context information for a location using enhanced regional context manager.
        
        Args:
            location: Location data (latitude, longitude, etc.)
            
        Returns:
            Regional context data
        """
        # Create LocationData object
        from bharatvoice.core.models import LocationData
        
        location_data = LocationData(
            latitude=location.get("latitude", 28.6139),  # Default to Delhi
            longitude=location.get("longitude", 77.2090),
            city=location.get("city", "Delhi"),
            state=location.get("state", "Delhi"),
            country=location.get("country", "India"),
            postal_code=location.get("postal_code"),
            timezone=location.get("timezone", "Asia/Kolkata")
        )
        
        # Use enhanced regional context manager
        regional_context = await self.regional_context_manager.get_regional_context(location_data)
        
        # Cache the result for legacy compatibility
        location_key = f"{location_data.city}_{location_data.state}"
        self._regional_contexts[location_key] = regional_context
        
        logger.info(f"Retrieved enhanced regional context for {location_key}")
        return regional_context
    
    async def learn_from_interaction(self, interaction: UserInteraction) -> Dict[str, any]:
        """
        Learn and adapt from user interaction using enhanced profile manager.
        
        Args:
            interaction: User interaction to learn from
            
        Returns:
            Learning result with adaptation information
        """
        # Use enhanced profile manager for learning
        learning_result = await self.profile_manager.learn_from_interaction(
            interaction.user_id, interaction
        )
        
        # Update legacy storage for backward compatibility
        user_profile = await self.profile_manager.get_profile(interaction.user_id)
        if user_profile:
            self._user_profiles[interaction.user_id] = user_profile
        
        logger.info(f"Enhanced learning completed for user {interaction.user_id}")
        return learning_result
    
    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """
        Retrieve conversation state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation state if exists, None otherwise
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session ID format: {session_id}")
            return None
        
        return await self.conversation_manager.get_session(session_uuid)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve user profile using enhanced profile manager.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists, None otherwise
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            logger.warning(f"Invalid user ID format: {user_id}")
            return None
        
        # Try enhanced profile manager first
        user_profile = await self.profile_manager.get_profile(user_uuid)
        
        if user_profile is None:
            # Fallback to legacy storage
            user_profile = self._user_profiles.get(user_uuid)
        
        return user_profile
    
    # Additional methods for conversation management
    
    async def create_conversation_session(
        self,
        user_id: str,
        initial_language: Optional[str] = None
    ) -> ConversationState:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            initial_language: Initial conversation language code
            
        Returns:
            New conversation state
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user ID format: {user_id}")
        
        from bharatvoice.core.models import LanguageCode
        
        lang = None
        if initial_language:
            try:
                lang = LanguageCode(initial_language)
            except ValueError:
                logger.warning(f"Invalid language code: {initial_language}, using default")
        
        return await self.context_manager.start_conversation(
            user_id=user_uuid,
            initial_language=lang
        )
    
    async def end_conversation_session(self, session_id: str) -> bool:
        """
        End a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was ended successfully
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session ID format: {session_id}")
            return False
        
        return await self.conversation_manager.end_session(session_uuid)
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[UserInteraction]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of user interactions
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session ID format: {session_id}")
            return []
        
        return await self.conversation_manager.get_conversation_history(
            session_uuid, limit
        )
    
    async def set_session_context(
        self,
        session_id: str,
        key: str,
        value: any
    ) -> bool:
        """
        Set a context variable for a session.
        
        Args:
            session_id: Session identifier
            key: Context variable key
            value: Context variable value
            
        Returns:
            True if set successfully
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session ID format: {session_id}")
            return False
        
        return await self.conversation_manager.set_context_variable(
            session_uuid, key, value
        )
    
    async def get_session_context(
        self,
        session_id: str,
        key: str,
        default: any = None
    ) -> any:
        """
        Get a context variable from a session.
        
        Args:
            session_id: Session identifier
            key: Context variable key
            default: Default value if key doesn't exist
            
        Returns:
            Context variable value or default
        """
        try:
            session_uuid = UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session ID format: {session_id}")
            return default
        
        return await self.conversation_manager.get_context_variable(
            session_uuid, key, default
        )
    
    async def get_service_statistics(self) -> Dict[str, any]:
        """
        Get service statistics including enhanced profile management.
        
        Returns:
            Dictionary with service statistics
        """
        session_stats = await self.conversation_manager.get_session_statistics()
        profile_stats = await self.profile_manager.get_profile_statistics()
        
        return {
            "conversation_sessions": session_stats,
            "user_profiles": len(self._user_profiles),
            "regional_contexts_cached": len(self._regional_contexts),
            "enhanced_profile_management": profile_stats
        }
    
    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources."""
        await self.conversation_manager.shutdown()
        await self.profile_manager.shutdown()
        # Regional context manager doesn't need explicit shutdown as it's stateless
        logger.info("Context Management Service shutdown complete")
    
    def _get_local_language(self, state: str) -> "LanguageCode":
        """
        Get the primary local language for a state.
        
        Args:
            state: State name
            
        Returns:
            Primary local language code
        """
        from bharatvoice.core.models import LanguageCode
        
        # Mapping of states to primary local languages
        state_languages = {
            "Tamil Nadu": LanguageCode.TAMIL,
            "Andhra Pradesh": LanguageCode.TELUGU,
            "Telangana": LanguageCode.TELUGU,
            "West Bengal": LanguageCode.BENGALI,
            "Maharashtra": LanguageCode.MARATHI,
            "Gujarat": LanguageCode.GUJARATI,
            "Karnataka": LanguageCode.KANNADA,
            "Kerala": LanguageCode.MALAYALAM,
            "Punjab": LanguageCode.PUNJABI,
            "Odisha": LanguageCode.ODIA,
        }
        
        return state_languages.get(state, LanguageCode.HINDI)
    
    def _get_dialect_info(self, state: str) -> Optional[str]:
        """
        Get dialect information for a state.
        
        Args:
            state: State name
            
        Returns:
            Dialect information string
        """
        dialect_info = {
            "Maharashtra": "Marathi with Mumbai/Pune variations",
            "Tamil Nadu": "Tamil with Chennai/Madurai variations",
            "Karnataka": "Kannada with Bangalore/Mysore variations",
            "West Bengal": "Bengali with Kolkata variations",
            "Gujarat": "Gujarati with Ahmedabad/Surat variations",
            "Punjab": "Punjabi with Amritsar/Ludhiana variations",
        }
        
        return dialect_info.get(state)
    
    def _create_legacy_regional_context(self, location_data: LocationData) -> RegionalContextData:
        """Create regional context using legacy implementation."""
        local_language = self._get_local_language(location_data.state)
        
        return RegionalContextData(
            location=location_data,
            local_services=[],
            weather_info=None,
            cultural_events=[],
            transport_options=[],
            government_services=[],
            local_language=local_language,
            dialect_info=self._get_dialect_info(location_data.state)
        )
    
    # Enhanced Profile Management Methods
    
    async def create_user_profile(
        self,
        user_id: str,
        initial_preferences: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """
        Create a new user profile with enhanced features.
        
        Args:
            user_id: Unique user identifier
            initial_preferences: Initial user preferences
            
        Returns:
            New user profile
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user ID format: {user_id}")
        
        user_profile = await self.profile_manager.create_profile(
            user_uuid, initial_preferences
        )
        
        # Update legacy storage for backward compatibility
        self._user_profiles[user_uuid] = user_profile
        
        logger.info(f"Created enhanced user profile for {user_id}")
        return user_profile
    
    async def update_user_location(
        self,
        user_id: str,
        location_data: Dict[str, Any],
        privacy_compliant: bool = True
    ) -> Optional[UserProfile]:
        """
        Update user location with privacy compliance.
        
        Args:
            user_id: User identifier
            location_data: Location data dictionary
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user ID format: {user_id}")
        
        from bharatvoice.core.models import LocationData
        
        location = LocationData(**location_data)
        
        user_profile = await self.profile_manager.update_location(
            user_uuid, location, privacy_compliant
        )
        
        if user_profile:
            # Update legacy storage
            self._user_profiles[user_uuid] = user_profile
        
        return user_profile
    
    async def get_user_regional_context(self, user_id: str) -> Optional[RegionalContextData]:
        """
        Get regional context for user's current location.
        
        Args:
            user_id: User identifier
            
        Returns:
            Regional context data if available
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            logger.warning(f"Invalid user ID format: {user_id}")
            return None
        
        # Get user profile to extract location
        user_profile = await self.profile_manager.get_profile(user_uuid)
        if user_profile and user_profile.location:
            return await self.regional_context_manager.get_regional_context(user_profile.location)
        
        return None
    
    async def get_cultural_events_by_type(
        self,
        location: Dict[str, Any],
        event_type: str
    ) -> List[CulturalEvent]:
        """
        Get cultural events filtered by type for a location.
        
        Args:
            location: Location data dictionary
            event_type: Type of cultural event to filter by
            
        Returns:
            List of cultural events of the specified type
        """
        from bharatvoice.core.models import LocationData
        
        location_data = LocationData(**location)
        return await self.regional_context_manager.get_cultural_events_by_type(
            location_data, event_type
        )
    
    async def search_local_services(
        self,
        location: Dict[str, Any],
        query: str
    ) -> List[LocalService]:
        """
        Search for local services by query.
        
        Args:
            location: Location data dictionary
            query: Search query for services
            
        Returns:
            List of matching local services
        """
        from bharatvoice.core.models import LocationData
        
        location_data = LocationData(**location)
        return await self.regional_context_manager.search_local_services(
            location_data, query
        )
    
    async def get_weather_forecast(
        self,
        location: Dict[str, Any],
        days: int = 7
    ) -> List[WeatherData]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Location data dictionary
            days: Number of days for forecast
            
        Returns:
            List of weather data for the forecast period
        """
        from bharatvoice.core.models import LocationData
        
        location_data = LocationData(**location)
        return await self.regional_context_manager.get_weather_forecast(
            location_data, days
        )
    
    async def delete_user_profile(
        self,
        user_id: str,
        compliance_reason: str = "user_request"
    ) -> bool:
        """
        Delete user profile with compliance logging.
        
        Args:
            user_id: User identifier
            compliance_reason: Reason for deletion
            
        Returns:
            True if profile was deleted
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            logger.warning(f"Invalid user ID format: {user_id}")
            return False
        
        # Delete from enhanced profile manager
        deleted = await self.profile_manager.delete_profile(user_uuid, compliance_reason)
        
        # Delete from legacy storage
        if user_uuid in self._user_profiles:
            del self._user_profiles[user_uuid]
            deleted = True
        
        return deleted
    
    async def get_profile_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get learning insights for a user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Learning insights and statistics
        """
        try:
            user_uuid = UUID(user_id)
        except ValueError:
            raise ValueError(f"Invalid user ID format: {user_id}")
        
        user_profile = await self.profile_manager.get_profile(user_uuid)
        
        if user_profile is None:
            return {"error": "Profile not found"}
        
        return {
            "user_id": user_id,
            "total_interactions": user_profile.usage_patterns.total_interactions,
            "preferred_languages": [lang.value for lang in user_profile.preferred_languages],
            "primary_language": user_profile.primary_language.value,
            "language_usage_frequency": {
                lang.value: freq for lang, freq in user_profile.usage_patterns.language_usage_frequency.items()
            },
            "preferred_time_slots": user_profile.usage_patterns.preferred_time_slots,
            "common_query_types": user_profile.usage_patterns.common_query_types,
            "last_active": user_profile.usage_patterns.last_active.isoformat() if user_profile.usage_patterns.last_active else None,
            "profile_created": user_profile.created_at.isoformat(),
            "last_updated": user_profile.last_updated.isoformat(),
            "privacy_settings": {
                "data_retention_days": user_profile.privacy_settings.data_retention_days,
                "allow_analytics": user_profile.privacy_settings.allow_analytics,
                "allow_personalization": user_profile.privacy_settings.allow_personalization,
                "location_sharing": user_profile.privacy_settings.location_sharing
            }
        }