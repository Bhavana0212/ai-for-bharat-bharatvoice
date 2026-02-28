<<<<<<< HEAD
"""
User Profile Management for BharatVoice Assistant.

This module implements comprehensive user profile management with privacy compliance,
encryption, adaptive learning, and location-based context management.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID, uuid4
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from bharatvoice.core.models import (
    UserProfile,
    UserInteraction,
    LanguageCode,
    LocationData,
    UsageAnalytics,
    PrivacyConfiguration,
    RegionalContextData,
)


logger = logging.getLogger(__name__)


class ProfileEncryption:
    """Handles encryption and decryption of user profile data."""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption handler.
        
        Args:
            master_key: Master encryption key. If None, generates a new one.
        """
        if master_key is None:
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        self.master_key = master_key.encode()
        self._cipher_suite = None
        self._initialize_cipher()
    
    def _initialize_cipher(self) -> None:
        """Initialize the cipher suite."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'bharatvoice_salt',  # In production, use random salt per user
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self._cipher_suite = Fernet(key)
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt profile data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self._cipher_suite.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt profile data.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data dictionary
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise


class LanguageLearningEngine:
    """Learns and adapts language preferences from user interactions."""
    
    def __init__(self, learning_rate: float = 0.1, min_interactions: int = 5):
        """
        Initialize language learning engine.
        
        Args:
            learning_rate: Rate at which to adapt preferences
            min_interactions: Minimum interactions before making adaptations
        """
        self.learning_rate = learning_rate
        self.min_interactions = min_interactions
    
    def analyze_language_patterns(
        self,
        interactions: List[UserInteraction],
        current_preferences: List[LanguageCode]
    ) -> Tuple[List[LanguageCode], Dict[str, Any]]:
        """
        Analyze language usage patterns and suggest adaptations.
        
        Args:
            interactions: Recent user interactions
            current_preferences: Current language preferences
            
        Returns:
            Tuple of (updated_preferences, learning_insights)
        """
        if len(interactions) < self.min_interactions:
            return current_preferences, {"reason": "insufficient_data"}
        
        # Analyze language usage frequency
        language_counts = {}
        total_interactions = len(interactions)
        
        for interaction in interactions:
            lang = interaction.input_language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Calculate usage frequencies
        language_frequencies = {
            lang: count / total_interactions
            for lang, count in language_counts.items()
        }
        
        # Identify frequently used languages not in preferences
        new_languages = []
        for lang, frequency in language_frequencies.items():
            if lang not in current_preferences and frequency > 0.2:  # 20% threshold
                new_languages.append((lang, frequency))
        
        # Sort by frequency
        new_languages.sort(key=lambda x: x[1], reverse=True)
        
        # Update preferences
        updated_preferences = current_preferences.copy()
        insights = {
            "language_frequencies": language_frequencies,
            "new_languages_detected": [],
            "adaptations_made": []
        }
        
        for lang, frequency in new_languages:
            if len(updated_preferences) < 5:  # Max 5 preferred languages
                updated_preferences.append(lang)
                insights["new_languages_detected"].append({
                    "language": lang.value,
                    "frequency": frequency
                })
                insights["adaptations_made"].append(f"added_{lang.value}")
        
        # Reorder preferences by recent usage
        if len(interactions) >= 10:
            recent_interactions = interactions[-10:]
            recent_language_counts = {}
            
            for interaction in recent_interactions:
                lang = interaction.input_language
                recent_language_counts[lang] = recent_language_counts.get(lang, 0) + 1
            
            # Sort preferences by recent usage
            preference_scores = {}
            for lang in updated_preferences:
                recent_count = recent_language_counts.get(lang, 0)
                overall_frequency = language_frequencies.get(lang, 0)
                preference_scores[lang] = recent_count * 0.7 + overall_frequency * 0.3
            
            updated_preferences.sort(key=lambda x: preference_scores.get(x, 0), reverse=True)
            insights["adaptations_made"].append("reordered_by_usage")
        
        return updated_preferences, insights
    
    def detect_primary_language_shift(
        self,
        interactions: List[UserInteraction],
        current_primary: LanguageCode
    ) -> Tuple[LanguageCode, bool]:
        """
        Detect if primary language should be changed based on recent usage.
        
        Args:
            interactions: Recent user interactions
            current_primary: Current primary language
            
        Returns:
            Tuple of (suggested_primary, should_change)
        """
        if len(interactions) < self.min_interactions * 2:
            return current_primary, False
        
        # Analyze recent language usage (last 20 interactions)
        recent_interactions = interactions[-20:]
        language_counts = {}
        
        for interaction in recent_interactions:
            lang = interaction.input_language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Find most frequently used language
        if not language_counts:
            return current_primary, False
        
        most_used_lang = max(language_counts.items(), key=lambda x: x[1])
        most_used_frequency = most_used_lang[1] / len(recent_interactions)
        
        # Suggest change if a different language is used >60% of the time
        if (most_used_lang[0] != current_primary and 
            most_used_frequency > 0.6):
            return most_used_lang[0], True
        
        return current_primary, False


class LocationContextManager:
    """Manages location-based context and privacy-compliant location handling."""
    
    def __init__(self):
        """Initialize location context manager."""
        self._location_cache: Dict[str, Tuple[RegionalContextData, datetime]] = {}
        self._cache_duration = timedelta(hours=6)  # Cache location data for 6 hours
    
    def update_user_location(
        self,
        user_profile: UserProfile,
        location_data: LocationData,
        privacy_compliant: bool = True
    ) -> UserProfile:
        """
        Update user location with privacy compliance.
        
        Args:
            user_profile: User profile to update
            location_data: New location data
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        if not user_profile.privacy_settings.location_sharing:
            logger.info(f"Location sharing disabled for user {user_profile.user_id}")
            return user_profile
        
        # Apply privacy filters if requested
        if privacy_compliant:
            # Reduce precision for privacy (round to ~1km accuracy)
            location_data.latitude = round(location_data.latitude, 2)
            location_data.longitude = round(location_data.longitude, 2)
            
            # Remove postal code if privacy is high
            if not user_profile.privacy_settings.allow_analytics:
                location_data.postal_code = None
        
        user_profile.location = location_data
        user_profile.last_updated = datetime.utcnow()
        
        logger.info(f"Updated location for user {user_profile.user_id} to {location_data.city}")
        return user_profile
    
    def get_location_context(
        self,
        location: LocationData,
        user_preferences: List[LanguageCode]
    ) -> Optional[RegionalContextData]:
        """
        Get regional context for a location with caching.
        
        Args:
            location: Location to get context for
            user_preferences: User language preferences
            
        Returns:
            Regional context data if available
        """
        # Create cache key
        cache_key = f"{location.city}_{location.state}_{location.country}"
        
        # Check cache
        if cache_key in self._location_cache:
            cached_data, cached_time = self._location_cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_duration:
                return cached_data
        
        # Generate new context data
        context_data = self._generate_regional_context(location, user_preferences)
        
        # Cache the result
        self._location_cache[cache_key] = (context_data, datetime.utcnow())
        
        return context_data
    
    def _generate_regional_context(
        self,
        location: LocationData,
        user_preferences: List[LanguageCode]
    ) -> RegionalContextData:
        """
        Generate regional context data for a location.
        
        Args:
            location: Location data
            user_preferences: User language preferences
            
        Returns:
            Regional context data
        """
        # Determine local language based on state
        local_language = self._get_local_language(location.state)
        
        # Create regional context
        regional_context = RegionalContextData(
            location=location,
            local_services=[],  # Would be populated from external APIs
            weather_info=None,  # Would be fetched from weather service
            cultural_events=[],  # Would be fetched from cultural calendar
            transport_options=[],  # Would be fetched from transport APIs
            government_services=[],  # Would be fetched from government portals
            local_language=local_language,
            dialect_info=self._get_dialect_info(location.state)
        )
        
        return regional_context
    
    def _get_local_language(self, state: str) -> LanguageCode:
        """Get the primary local language for a state."""
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
        """Get dialect information for a state."""
        dialect_info = {
            "Maharashtra": "Marathi with Mumbai/Pune variations",
            "Tamil Nadu": "Tamil with Chennai/Madurai variations",
            "Karnataka": "Kannada with Bangalore/Mysore variations",
            "West Bengal": "Bengali with Kolkata variations",
            "Gujarat": "Gujarati with Ahmedabad/Surat variations",
            "Punjab": "Punjabi with Amritsar/Ludhiana variations",
        }
        return dialect_info.get(state)


class UserProfileManager:
    """
    Comprehensive user profile management with privacy compliance,
    encryption, adaptive learning, and location-based context.
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        learning_rate: float = 0.1,
        profile_cleanup_days: int = 90
    ):
        """
        Initialize user profile manager.
        
        Args:
            encryption_key: Master encryption key
            learning_rate: Rate for adaptive learning
            profile_cleanup_days: Days after which to cleanup inactive profiles
        """
        self.encryption = ProfileEncryption(encryption_key)
        self.language_learner = LanguageLearningEngine(learning_rate)
        self.location_manager = LocationContextManager()
        self.profile_cleanup_days = profile_cleanup_days
        
        # In-memory storage (in production, use database)
        self._profiles: Dict[UUID, UserProfile] = {}
        self._encrypted_profiles: Dict[UUID, str] = {}
        self._interaction_history: Dict[UUID, List[UserInteraction]] = {}
        
        # Profile locks for thread safety
        self._profile_locks: Dict[UUID, asyncio.Lock] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info("User Profile Manager initialized")
    
    async def create_profile(
        self,
        user_id: UUID,
        initial_preferences: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """
        Create a new user profile with privacy-compliant defaults.
        
        Args:
            user_id: Unique user identifier
            initial_preferences: Initial user preferences
            
        Returns:
            New user profile
        """
        # Create profile with defaults
        profile = UserProfile(user_id=user_id)
        
        # Apply initial preferences if provided
        if initial_preferences:
            profile = await self._apply_preferences(profile, initial_preferences)
        
        # Store profile
        await self._store_profile(profile)
        
        logger.info(f"Created new user profile for {user_id}")
        return profile
    
    async def get_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """
        Retrieve user profile with decryption.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists
        """
        if user_id in self._profiles:
            return self._profiles[user_id]
        
        # Try to load from encrypted storage
        if user_id in self._encrypted_profiles:
            try:
                encrypted_data = self._encrypted_profiles[user_id]
                profile_data = self.encryption.decrypt_data(encrypted_data)
                profile = UserProfile(**profile_data)
                
                # Cache in memory
                self._profiles[user_id] = profile
                return profile
            except Exception as e:
                logger.error(f"Failed to decrypt profile for {user_id}: {e}")
                return None
        
        return None
    
    async def update_profile(
        self,
        user_id: UUID,
        preferences: Dict[str, Any],
        privacy_compliant: bool = True
    ) -> Optional[UserProfile]:
        """
        Update user profile with privacy compliance.
        
        Args:
            user_id: User identifier
            preferences: Updated preferences
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                # Create new profile if it doesn't exist
                profile = await self.create_profile(user_id, preferences)
                return profile
            
            # Apply preferences
            profile = await self._apply_preferences(profile, preferences, privacy_compliant)
            
            # Store updated profile
            await self._store_profile(profile)
            
            logger.info(f"Updated profile for user {user_id}")
            return profile
    
    async def learn_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> Dict[str, Any]:
        """
        Learn and adapt user profile from interaction.
        
        Args:
            user_id: User identifier
            interaction: User interaction to learn from
            
        Returns:
            Learning results and adaptations made
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                profile = await self.create_profile(user_id)
            
            # Store interaction for learning
            if user_id not in self._interaction_history:
                self._interaction_history[user_id] = []
            
            self._interaction_history[user_id].append(interaction)
            
            # Keep only recent interactions for learning
            max_history = 100
            if len(self._interaction_history[user_id]) > max_history:
                self._interaction_history[user_id] = self._interaction_history[user_id][-max_history:]
            
            # Update usage analytics
            profile.usage_patterns.total_interactions += 1
            profile.usage_patterns.last_active = interaction.timestamp
            
            # Update language usage frequency
            input_lang = interaction.input_language
            current_freq = profile.usage_patterns.language_usage_frequency.get(input_lang, 0.0)
            total_interactions = profile.usage_patterns.total_interactions
            new_freq = (current_freq * (total_interactions - 1) + 1.0) / total_interactions
            profile.usage_patterns.language_usage_frequency[input_lang] = new_freq
            
            # Learn time preferences
            interaction_hour = interaction.timestamp.hour
            if interaction_hour not in profile.usage_patterns.preferred_time_slots:
                profile.usage_patterns.preferred_time_slots.append(interaction_hour)
            
            # Learn query types
            if interaction.intent:
                intent_category = interaction.intent.split('.')[0] if '.' in interaction.intent else interaction.intent
                current_count = profile.usage_patterns.common_query_types.get(intent_category, 0)
                profile.usage_patterns.common_query_types[intent_category] = current_count + 1
            
            # Adaptive language learning
            learning_result = {"adaptations": [], "insights": []}
            
            if len(self._interaction_history[user_id]) >= 5:
                # Analyze language patterns
                updated_preferences, language_insights = self.language_learner.analyze_language_patterns(
                    self._interaction_history[user_id],
                    profile.preferred_languages
                )
                
                if updated_preferences != profile.preferred_languages:
                    profile.preferred_languages = updated_preferences
                    learning_result["adaptations"].extend(language_insights.get("adaptations_made", []))
                    learning_result["insights"].append("language_preferences_updated")
                
                # Check for primary language shift
                new_primary, should_change = self.language_learner.detect_primary_language_shift(
                    self._interaction_history[user_id],
                    profile.primary_language
                )
                
                if should_change:
                    profile.primary_language = new_primary
                    learning_result["adaptations"].append(f"primary_language_changed_to_{new_primary.value}")
            
            # Update profile timestamp
            profile.last_updated = datetime.utcnow()
            
            # Store updated profile
            await self._store_profile(profile)
            
            learning_result["user_id"] = str(user_id)
            learning_result["total_interactions"] = profile.usage_patterns.total_interactions
            
            return learning_result
    
    async def update_location(
        self,
        user_id: UUID,
        location_data: LocationData,
        privacy_compliant: bool = True
    ) -> Optional[UserProfile]:
        """
        Update user location with privacy compliance.
        
        Args:
            user_id: User identifier
            location_data: New location data
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                profile = await self.create_profile(user_id)
            
            # Update location through location manager
            profile = self.location_manager.update_user_location(
                profile, location_data, privacy_compliant
            )
            
            # Store updated profile
            await self._store_profile(profile)
            
            return profile
    
    async def get_regional_context(
        self,
        user_id: UUID
    ) -> Optional[RegionalContextData]:
        """
        Get regional context for user's location.
        
        Args:
            user_id: User identifier
            
        Returns:
            Regional context data if available
        """
        profile = await self.get_profile(user_id)
        
        if profile is None or profile.location is None:
            return None
        
        return self.location_manager.get_location_context(
            profile.location,
            profile.preferred_languages
        )
    
    async def delete_profile(
        self,
        user_id: UUID,
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
        async with self._get_profile_lock(user_id):
            deleted = False
            
            if user_id in self._profiles:
                del self._profiles[user_id]
                deleted = True
            
            if user_id in self._encrypted_profiles:
                del self._encrypted_profiles[user_id]
                deleted = True
            
            if user_id in self._interaction_history:
                del self._interaction_history[user_id]
            
            if user_id in self._profile_locks:
                del self._profile_locks[user_id]
            
            if deleted:
                logger.info(f"Deleted profile for user {user_id}, reason: {compliance_reason}")
            
            return deleted
    
    async def get_profile_statistics(self) -> Dict[str, Any]:
        """
        Get profile management statistics.
        
        Returns:
            Statistics dictionary
        """
        total_profiles = len(self._profiles) + len(self._encrypted_profiles)
        active_profiles = len(self._profiles)
        encrypted_profiles = len(self._encrypted_profiles)
        
        # Calculate average interactions per user
        total_interactions = sum(
            len(interactions) for interactions in self._interaction_history.values()
        )
        avg_interactions = total_interactions / max(total_profiles, 1)
        
        return {
            "total_profiles": total_profiles,
            "active_profiles": active_profiles,
            "encrypted_profiles": encrypted_profiles,
            "total_interactions_stored": total_interactions,
            "average_interactions_per_user": avg_interactions
        }
    
    async def _apply_preferences(
        self,
        profile: UserProfile,
        preferences: Dict[str, Any],
        privacy_compliant: bool = True
    ) -> UserProfile:
        """Apply preferences to user profile."""
        if "preferred_languages" in preferences:
            profile.preferred_languages = preferences["preferred_languages"]
        
        if "primary_language" in preferences:
            profile.primary_language = preferences["primary_language"]
        
        if "location" in preferences:
            location_data = LocationData(**preferences["location"])
            profile = self.location_manager.update_user_location(
                profile, location_data, privacy_compliant
            )
        
        if "privacy_settings" in preferences:
            privacy_prefs = preferences["privacy_settings"]
            for key, value in privacy_prefs.items():
                if hasattr(profile.privacy_settings, key):
                    setattr(profile.privacy_settings, key, value)
        
        profile.last_updated = datetime.utcnow()
        return profile
    
    async def _store_profile(self, profile: UserProfile) -> None:
        """Store profile with encryption."""
        # Store in memory cache
        self._profiles[profile.user_id] = profile
        
        # Store encrypted version for persistence
        if profile.privacy_settings.allow_personalization:
            try:
                profile_data = profile.dict()
                encrypted_data = self.encryption.encrypt_data(profile_data)
                self._encrypted_profiles[profile.user_id] = encrypted_data
            except Exception as e:
                logger.error(f"Failed to encrypt profile for {profile.user_id}: {e}")
    
    def _get_profile_lock(self, user_id: UUID) -> asyncio.Lock:
        """Get or create a lock for a user profile."""
        if user_id not in self._profile_locks:
            self._profile_locks[user_id] = asyncio.Lock()
        return self._profile_locks[user_id]
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task for inactive profiles."""
        async def cleanup_inactive_profiles():
            while True:
                try:
                    await asyncio.sleep(24 * 3600)  # Run daily
                    
                    cutoff_date = datetime.utcnow() - timedelta(days=self.profile_cleanup_days)
                    inactive_profiles = []
                    
                    for user_id, profile in self._profiles.items():
                        if profile.last_updated < cutoff_date:
                            inactive_profiles.append(user_id)
                    
                    for user_id in inactive_profiles:
                        await self.delete_profile(user_id, "inactive_cleanup")
                    
                    if inactive_profiles:
                        logger.info(f"Cleaned up {len(inactive_profiles)} inactive profiles")
                
                except Exception as e:
                    logger.error(f"Error in profile cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_inactive_profiles())
    
    async def shutdown(self) -> None:
        """Shutdown the profile manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
=======
"""
User Profile Management for BharatVoice Assistant.

This module implements comprehensive user profile management with privacy compliance,
encryption, adaptive learning, and location-based context management.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID, uuid4
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from bharatvoice.core.models import (
    UserProfile,
    UserInteraction,
    LanguageCode,
    LocationData,
    UsageAnalytics,
    PrivacyConfiguration,
    RegionalContextData,
)


logger = logging.getLogger(__name__)


class ProfileEncryption:
    """Handles encryption and decryption of user profile data."""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption handler.
        
        Args:
            master_key: Master encryption key. If None, generates a new one.
        """
        if master_key is None:
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        self.master_key = master_key.encode()
        self._cipher_suite = None
        self._initialize_cipher()
    
    def _initialize_cipher(self) -> None:
        """Initialize the cipher suite."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'bharatvoice_salt',  # In production, use random salt per user
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self._cipher_suite = Fernet(key)
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt profile data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self._cipher_suite.encrypt(json_data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt profile data.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data dictionary
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher_suite.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise


class LanguageLearningEngine:
    """Learns and adapts language preferences from user interactions."""
    
    def __init__(self, learning_rate: float = 0.1, min_interactions: int = 5):
        """
        Initialize language learning engine.
        
        Args:
            learning_rate: Rate at which to adapt preferences
            min_interactions: Minimum interactions before making adaptations
        """
        self.learning_rate = learning_rate
        self.min_interactions = min_interactions
    
    def analyze_language_patterns(
        self,
        interactions: List[UserInteraction],
        current_preferences: List[LanguageCode]
    ) -> Tuple[List[LanguageCode], Dict[str, Any]]:
        """
        Analyze language usage patterns and suggest adaptations.
        
        Args:
            interactions: Recent user interactions
            current_preferences: Current language preferences
            
        Returns:
            Tuple of (updated_preferences, learning_insights)
        """
        if len(interactions) < self.min_interactions:
            return current_preferences, {"reason": "insufficient_data"}
        
        # Analyze language usage frequency
        language_counts = {}
        total_interactions = len(interactions)
        
        for interaction in interactions:
            lang = interaction.input_language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Calculate usage frequencies
        language_frequencies = {
            lang: count / total_interactions
            for lang, count in language_counts.items()
        }
        
        # Identify frequently used languages not in preferences
        new_languages = []
        for lang, frequency in language_frequencies.items():
            if lang not in current_preferences and frequency > 0.2:  # 20% threshold
                new_languages.append((lang, frequency))
        
        # Sort by frequency
        new_languages.sort(key=lambda x: x[1], reverse=True)
        
        # Update preferences
        updated_preferences = current_preferences.copy()
        insights = {
            "language_frequencies": language_frequencies,
            "new_languages_detected": [],
            "adaptations_made": []
        }
        
        for lang, frequency in new_languages:
            if len(updated_preferences) < 5:  # Max 5 preferred languages
                updated_preferences.append(lang)
                insights["new_languages_detected"].append({
                    "language": lang.value,
                    "frequency": frequency
                })
                insights["adaptations_made"].append(f"added_{lang.value}")
        
        # Reorder preferences by recent usage
        if len(interactions) >= 10:
            recent_interactions = interactions[-10:]
            recent_language_counts = {}
            
            for interaction in recent_interactions:
                lang = interaction.input_language
                recent_language_counts[lang] = recent_language_counts.get(lang, 0) + 1
            
            # Sort preferences by recent usage
            preference_scores = {}
            for lang in updated_preferences:
                recent_count = recent_language_counts.get(lang, 0)
                overall_frequency = language_frequencies.get(lang, 0)
                preference_scores[lang] = recent_count * 0.7 + overall_frequency * 0.3
            
            updated_preferences.sort(key=lambda x: preference_scores.get(x, 0), reverse=True)
            insights["adaptations_made"].append("reordered_by_usage")
        
        return updated_preferences, insights
    
    def detect_primary_language_shift(
        self,
        interactions: List[UserInteraction],
        current_primary: LanguageCode
    ) -> Tuple[LanguageCode, bool]:
        """
        Detect if primary language should be changed based on recent usage.
        
        Args:
            interactions: Recent user interactions
            current_primary: Current primary language
            
        Returns:
            Tuple of (suggested_primary, should_change)
        """
        if len(interactions) < self.min_interactions * 2:
            return current_primary, False
        
        # Analyze recent language usage (last 20 interactions)
        recent_interactions = interactions[-20:]
        language_counts = {}
        
        for interaction in recent_interactions:
            lang = interaction.input_language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Find most frequently used language
        if not language_counts:
            return current_primary, False
        
        most_used_lang = max(language_counts.items(), key=lambda x: x[1])
        most_used_frequency = most_used_lang[1] / len(recent_interactions)
        
        # Suggest change if a different language is used >60% of the time
        if (most_used_lang[0] != current_primary and 
            most_used_frequency > 0.6):
            return most_used_lang[0], True
        
        return current_primary, False


class LocationContextManager:
    """Manages location-based context and privacy-compliant location handling."""
    
    def __init__(self):
        """Initialize location context manager."""
        self._location_cache: Dict[str, Tuple[RegionalContextData, datetime]] = {}
        self._cache_duration = timedelta(hours=6)  # Cache location data for 6 hours
    
    def update_user_location(
        self,
        user_profile: UserProfile,
        location_data: LocationData,
        privacy_compliant: bool = True
    ) -> UserProfile:
        """
        Update user location with privacy compliance.
        
        Args:
            user_profile: User profile to update
            location_data: New location data
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        if not user_profile.privacy_settings.location_sharing:
            logger.info(f"Location sharing disabled for user {user_profile.user_id}")
            return user_profile
        
        # Apply privacy filters if requested
        if privacy_compliant:
            # Reduce precision for privacy (round to ~1km accuracy)
            location_data.latitude = round(location_data.latitude, 2)
            location_data.longitude = round(location_data.longitude, 2)
            
            # Remove postal code if privacy is high
            if not user_profile.privacy_settings.allow_analytics:
                location_data.postal_code = None
        
        user_profile.location = location_data
        user_profile.last_updated = datetime.utcnow()
        
        logger.info(f"Updated location for user {user_profile.user_id} to {location_data.city}")
        return user_profile
    
    def get_location_context(
        self,
        location: LocationData,
        user_preferences: List[LanguageCode]
    ) -> Optional[RegionalContextData]:
        """
        Get regional context for a location with caching.
        
        Args:
            location: Location to get context for
            user_preferences: User language preferences
            
        Returns:
            Regional context data if available
        """
        # Create cache key
        cache_key = f"{location.city}_{location.state}_{location.country}"
        
        # Check cache
        if cache_key in self._location_cache:
            cached_data, cached_time = self._location_cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_duration:
                return cached_data
        
        # Generate new context data
        context_data = self._generate_regional_context(location, user_preferences)
        
        # Cache the result
        self._location_cache[cache_key] = (context_data, datetime.utcnow())
        
        return context_data
    
    def _generate_regional_context(
        self,
        location: LocationData,
        user_preferences: List[LanguageCode]
    ) -> RegionalContextData:
        """
        Generate regional context data for a location.
        
        Args:
            location: Location data
            user_preferences: User language preferences
            
        Returns:
            Regional context data
        """
        # Determine local language based on state
        local_language = self._get_local_language(location.state)
        
        # Create regional context
        regional_context = RegionalContextData(
            location=location,
            local_services=[],  # Would be populated from external APIs
            weather_info=None,  # Would be fetched from weather service
            cultural_events=[],  # Would be fetched from cultural calendar
            transport_options=[],  # Would be fetched from transport APIs
            government_services=[],  # Would be fetched from government portals
            local_language=local_language,
            dialect_info=self._get_dialect_info(location.state)
        )
        
        return regional_context
    
    def _get_local_language(self, state: str) -> LanguageCode:
        """Get the primary local language for a state."""
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
        """Get dialect information for a state."""
        dialect_info = {
            "Maharashtra": "Marathi with Mumbai/Pune variations",
            "Tamil Nadu": "Tamil with Chennai/Madurai variations",
            "Karnataka": "Kannada with Bangalore/Mysore variations",
            "West Bengal": "Bengali with Kolkata variations",
            "Gujarat": "Gujarati with Ahmedabad/Surat variations",
            "Punjab": "Punjabi with Amritsar/Ludhiana variations",
        }
        return dialect_info.get(state)


class UserProfileManager:
    """
    Comprehensive user profile management with privacy compliance,
    encryption, adaptive learning, and location-based context.
    """
    
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        learning_rate: float = 0.1,
        profile_cleanup_days: int = 90
    ):
        """
        Initialize user profile manager.
        
        Args:
            encryption_key: Master encryption key
            learning_rate: Rate for adaptive learning
            profile_cleanup_days: Days after which to cleanup inactive profiles
        """
        self.encryption = ProfileEncryption(encryption_key)
        self.language_learner = LanguageLearningEngine(learning_rate)
        self.location_manager = LocationContextManager()
        self.profile_cleanup_days = profile_cleanup_days
        
        # In-memory storage (in production, use database)
        self._profiles: Dict[UUID, UserProfile] = {}
        self._encrypted_profiles: Dict[UUID, str] = {}
        self._interaction_history: Dict[UUID, List[UserInteraction]] = {}
        
        # Profile locks for thread safety
        self._profile_locks: Dict[UUID, asyncio.Lock] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info("User Profile Manager initialized")
    
    async def create_profile(
        self,
        user_id: UUID,
        initial_preferences: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """
        Create a new user profile with privacy-compliant defaults.
        
        Args:
            user_id: Unique user identifier
            initial_preferences: Initial user preferences
            
        Returns:
            New user profile
        """
        # Create profile with defaults
        profile = UserProfile(user_id=user_id)
        
        # Apply initial preferences if provided
        if initial_preferences:
            profile = await self._apply_preferences(profile, initial_preferences)
        
        # Store profile
        await self._store_profile(profile)
        
        logger.info(f"Created new user profile for {user_id}")
        return profile
    
    async def get_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """
        Retrieve user profile with decryption.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists
        """
        if user_id in self._profiles:
            return self._profiles[user_id]
        
        # Try to load from encrypted storage
        if user_id in self._encrypted_profiles:
            try:
                encrypted_data = self._encrypted_profiles[user_id]
                profile_data = self.encryption.decrypt_data(encrypted_data)
                profile = UserProfile(**profile_data)
                
                # Cache in memory
                self._profiles[user_id] = profile
                return profile
            except Exception as e:
                logger.error(f"Failed to decrypt profile for {user_id}: {e}")
                return None
        
        return None
    
    async def update_profile(
        self,
        user_id: UUID,
        preferences: Dict[str, Any],
        privacy_compliant: bool = True
    ) -> Optional[UserProfile]:
        """
        Update user profile with privacy compliance.
        
        Args:
            user_id: User identifier
            preferences: Updated preferences
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                # Create new profile if it doesn't exist
                profile = await self.create_profile(user_id, preferences)
                return profile
            
            # Apply preferences
            profile = await self._apply_preferences(profile, preferences, privacy_compliant)
            
            # Store updated profile
            await self._store_profile(profile)
            
            logger.info(f"Updated profile for user {user_id}")
            return profile
    
    async def learn_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> Dict[str, Any]:
        """
        Learn and adapt user profile from interaction.
        
        Args:
            user_id: User identifier
            interaction: User interaction to learn from
            
        Returns:
            Learning results and adaptations made
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                profile = await self.create_profile(user_id)
            
            # Store interaction for learning
            if user_id not in self._interaction_history:
                self._interaction_history[user_id] = []
            
            self._interaction_history[user_id].append(interaction)
            
            # Keep only recent interactions for learning
            max_history = 100
            if len(self._interaction_history[user_id]) > max_history:
                self._interaction_history[user_id] = self._interaction_history[user_id][-max_history:]
            
            # Update usage analytics
            profile.usage_patterns.total_interactions += 1
            profile.usage_patterns.last_active = interaction.timestamp
            
            # Update language usage frequency
            input_lang = interaction.input_language
            current_freq = profile.usage_patterns.language_usage_frequency.get(input_lang, 0.0)
            total_interactions = profile.usage_patterns.total_interactions
            new_freq = (current_freq * (total_interactions - 1) + 1.0) / total_interactions
            profile.usage_patterns.language_usage_frequency[input_lang] = new_freq
            
            # Learn time preferences
            interaction_hour = interaction.timestamp.hour
            if interaction_hour not in profile.usage_patterns.preferred_time_slots:
                profile.usage_patterns.preferred_time_slots.append(interaction_hour)
            
            # Learn query types
            if interaction.intent:
                intent_category = interaction.intent.split('.')[0] if '.' in interaction.intent else interaction.intent
                current_count = profile.usage_patterns.common_query_types.get(intent_category, 0)
                profile.usage_patterns.common_query_types[intent_category] = current_count + 1
            
            # Adaptive language learning
            learning_result = {"adaptations": [], "insights": []}
            
            if len(self._interaction_history[user_id]) >= 5:
                # Analyze language patterns
                updated_preferences, language_insights = self.language_learner.analyze_language_patterns(
                    self._interaction_history[user_id],
                    profile.preferred_languages
                )
                
                if updated_preferences != profile.preferred_languages:
                    profile.preferred_languages = updated_preferences
                    learning_result["adaptations"].extend(language_insights.get("adaptations_made", []))
                    learning_result["insights"].append("language_preferences_updated")
                
                # Check for primary language shift
                new_primary, should_change = self.language_learner.detect_primary_language_shift(
                    self._interaction_history[user_id],
                    profile.primary_language
                )
                
                if should_change:
                    profile.primary_language = new_primary
                    learning_result["adaptations"].append(f"primary_language_changed_to_{new_primary.value}")
            
            # Update profile timestamp
            profile.last_updated = datetime.utcnow()
            
            # Store updated profile
            await self._store_profile(profile)
            
            learning_result["user_id"] = str(user_id)
            learning_result["total_interactions"] = profile.usage_patterns.total_interactions
            
            return learning_result
    
    async def update_location(
        self,
        user_id: UUID,
        location_data: LocationData,
        privacy_compliant: bool = True
    ) -> Optional[UserProfile]:
        """
        Update user location with privacy compliance.
        
        Args:
            user_id: User identifier
            location_data: New location data
            privacy_compliant: Whether to apply privacy filters
            
        Returns:
            Updated user profile
        """
        async with self._get_profile_lock(user_id):
            profile = await self.get_profile(user_id)
            
            if profile is None:
                profile = await self.create_profile(user_id)
            
            # Update location through location manager
            profile = self.location_manager.update_user_location(
                profile, location_data, privacy_compliant
            )
            
            # Store updated profile
            await self._store_profile(profile)
            
            return profile
    
    async def get_regional_context(
        self,
        user_id: UUID
    ) -> Optional[RegionalContextData]:
        """
        Get regional context for user's location.
        
        Args:
            user_id: User identifier
            
        Returns:
            Regional context data if available
        """
        profile = await self.get_profile(user_id)
        
        if profile is None or profile.location is None:
            return None
        
        return self.location_manager.get_location_context(
            profile.location,
            profile.preferred_languages
        )
    
    async def delete_profile(
        self,
        user_id: UUID,
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
        async with self._get_profile_lock(user_id):
            deleted = False
            
            if user_id in self._profiles:
                del self._profiles[user_id]
                deleted = True
            
            if user_id in self._encrypted_profiles:
                del self._encrypted_profiles[user_id]
                deleted = True
            
            if user_id in self._interaction_history:
                del self._interaction_history[user_id]
            
            if user_id in self._profile_locks:
                del self._profile_locks[user_id]
            
            if deleted:
                logger.info(f"Deleted profile for user {user_id}, reason: {compliance_reason}")
            
            return deleted
    
    async def get_profile_statistics(self) -> Dict[str, Any]:
        """
        Get profile management statistics.
        
        Returns:
            Statistics dictionary
        """
        total_profiles = len(self._profiles) + len(self._encrypted_profiles)
        active_profiles = len(self._profiles)
        encrypted_profiles = len(self._encrypted_profiles)
        
        # Calculate average interactions per user
        total_interactions = sum(
            len(interactions) for interactions in self._interaction_history.values()
        )
        avg_interactions = total_interactions / max(total_profiles, 1)
        
        return {
            "total_profiles": total_profiles,
            "active_profiles": active_profiles,
            "encrypted_profiles": encrypted_profiles,
            "total_interactions_stored": total_interactions,
            "average_interactions_per_user": avg_interactions
        }
    
    async def _apply_preferences(
        self,
        profile: UserProfile,
        preferences: Dict[str, Any],
        privacy_compliant: bool = True
    ) -> UserProfile:
        """Apply preferences to user profile."""
        if "preferred_languages" in preferences:
            profile.preferred_languages = preferences["preferred_languages"]
        
        if "primary_language" in preferences:
            profile.primary_language = preferences["primary_language"]
        
        if "location" in preferences:
            location_data = LocationData(**preferences["location"])
            profile = self.location_manager.update_user_location(
                profile, location_data, privacy_compliant
            )
        
        if "privacy_settings" in preferences:
            privacy_prefs = preferences["privacy_settings"]
            for key, value in privacy_prefs.items():
                if hasattr(profile.privacy_settings, key):
                    setattr(profile.privacy_settings, key, value)
        
        profile.last_updated = datetime.utcnow()
        return profile
    
    async def _store_profile(self, profile: UserProfile) -> None:
        """Store profile with encryption."""
        # Store in memory cache
        self._profiles[profile.user_id] = profile
        
        # Store encrypted version for persistence
        if profile.privacy_settings.allow_personalization:
            try:
                profile_data = profile.dict()
                encrypted_data = self.encryption.encrypt_data(profile_data)
                self._encrypted_profiles[profile.user_id] = encrypted_data
            except Exception as e:
                logger.error(f"Failed to encrypt profile for {profile.user_id}: {e}")
    
    def _get_profile_lock(self, user_id: UUID) -> asyncio.Lock:
        """Get or create a lock for a user profile."""
        if user_id not in self._profile_locks:
            self._profile_locks[user_id] = asyncio.Lock()
        return self._profile_locks[user_id]
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task for inactive profiles."""
        async def cleanup_inactive_profiles():
            while True:
                try:
                    await asyncio.sleep(24 * 3600)  # Run daily
                    
                    cutoff_date = datetime.utcnow() - timedelta(days=self.profile_cleanup_days)
                    inactive_profiles = []
                    
                    for user_id, profile in self._profiles.items():
                        if profile.last_updated < cutoff_date:
                            inactive_profiles.append(user_id)
                    
                    for user_id in inactive_profiles:
                        await self.delete_profile(user_id, "inactive_cleanup")
                    
                    if inactive_profiles:
                        logger.info(f"Cleaned up {len(inactive_profiles)} inactive profiles")
                
                except Exception as e:
                    logger.error(f"Error in profile cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_inactive_profiles())
    
    async def shutdown(self) -> None:
        """Shutdown the profile manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        logger.info("User Profile Manager shutdown complete")