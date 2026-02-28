<<<<<<< HEAD
"""
Core interfaces for BharatVoice Assistant services.

This module defines the abstract interfaces that all service implementations
must follow, ensuring consistent contracts across the microservices architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    ConversationState,
    CulturalContext,
    Intent,
    LanguageCode,
    RecognitionResult,
    RegionalContextData,
    Response,
    ServiceParameters,
    ServiceResult,
    UserInteraction,
    UserProfile,
    VoiceActivityResult,
)


class AudioProcessor(ABC):
    """Interface for audio processing operations."""
    
    @abstractmethod
    async def process_audio_stream(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio stream with language-specific optimizations.
        
        Args:
            audio_data: Input audio buffer
            language: Target language for processing
            
        Returns:
            Processed audio buffer
        """
        pass
    
    @abstractmethod
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        pass
    
    @abstractmethod
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        pass
    
    @abstractmethod
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise from audio data.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        pass


class LanguageEngine(ABC):
    """Interface for language processing operations."""
    
    @abstractmethod
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        pass
    
    @abstractmethod
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        pass
    
    @abstractmethod
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        pass
    
    @abstractmethod
    async def adapt_to_regional_accent(
        self, 
        model_id: str, 
        accent_data: Dict[str, any]
    ) -> str:
        """
        Adapt language model to regional accent.
        
        Args:
            model_id: Base model identifier
            accent_data: Regional accent adaptation data
            
        Returns:
            Adapted model identifier
        """
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the primary language of input text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Detected language code
        """
        pass


class ContextManager(ABC):
    """Interface for conversation context and user profile management."""
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    async def update_user_profile(
        self, 
        user_id: str, 
        preferences: Dict[str, any]
    ) -> UserProfile:
        """
        Update user profile with new preferences.
        
        Args:
            user_id: Unique user identifier
            preferences: Updated user preferences
            
        Returns:
            Updated user profile
        """
        pass
    
    @abstractmethod
    async def get_regional_context(
        self, 
        location: Dict[str, any]
    ) -> RegionalContextData:
        """
        Get regional context information for a location.
        
        Args:
            location: Location data (latitude, longitude, etc.)
            
        Returns:
            Regional context data
        """
        pass
    
    @abstractmethod
    async def learn_from_interaction(self, interaction: UserInteraction) -> Dict[str, any]:
        """
        Learn and adapt from user interaction.
        
        Args:
            interaction: User interaction to learn from
            
        Returns:
            Learning result with adaptation information
        """
        pass
    
    @abstractmethod
    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """
        Retrieve conversation state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation state if exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists, None otherwise
        """
        pass


class ResponseGenerator(ABC):
    """Interface for query processing and response generation."""
    
    @abstractmethod
    async def process_query(
        self, 
        query: str, 
        context: ConversationState
    ) -> Dict[str, any]:
        """
        Process user query and extract intent and entities.
        
        Args:
            query: User query text
            context: Current conversation context
            
        Returns:
            Query processing result with intent and entities
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        intent: Intent, 
        entities: List[Dict[str, any]], 
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class AuthenticationService(ABC):
    """Interface for user authentication and authorization."""
    
    @abstractmethod
    async def authenticate_user(self, credentials: Dict[str, any]) -> Dict[str, any]:
        """
        Authenticate user with provided credentials.
        
        Args:
            credentials: User authentication credentials
            
        Returns:
            Authentication result with user information
        """
        pass
    
    @abstractmethod
    async def authorize_request(
        self, 
        user_id: str, 
        resource: str, 
        action: str
    ) -> bool:
        """
        Authorize user request for specific resource and action.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if authorized, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_session(self, user_id: str) -> str:
        """
        Create new user session.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session token
        """
        pass
    
    @abstractmethod
    async def validate_session(self, session_token: str) -> Optional[str]:
        """
        Validate session token and return user ID.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            User ID if valid, None otherwise
        """
        pass


class OfflineSyncService(ABC):
    """Interface for offline capability and data synchronization."""
    
    @abstractmethod
    async def cache_for_offline(self, data: Dict[str, any], cache_key: str) -> bool:
        """
        Cache data for offline access.
        
        Args:
            data: Data to cache
            cache_key: Unique cache key
            
        Returns:
            True if cached successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_offline_data(self, cache_key: str) -> Optional[Dict[str, any]]:
        """
        Retrieve cached data for offline use.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data if available, None otherwise
        """
        pass
    
    @abstractmethod
    async def sync_when_online(self, user_id: str) -> Dict[str, any]:
        """
        Synchronize offline data when connectivity is restored.
        
        Args:
            user_id: User identifier for data synchronization
            
        Returns:
            Synchronization result
        """
        pass
    
    @abstractmethod
    async def is_offline_mode(self) -> bool:
        """
        Check if system is currently in offline mode.
        
        Returns:
            True if offline, False if online
        """
        pass


class HealthCheckService(ABC):
    """Interface for service health monitoring."""
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> Dict[str, any]:
        """
        Check health status of a specific service.
        
        Args:
            service_name: Name of service to check
            
        Returns:
            Health check result with status and metrics
        """
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, any]:
        """
        Get overall system health metrics.
        
        Returns:
            System metrics and performance data
        """
        pass
    
    @abstractmethod
    async def register_service(self, service_name: str, endpoint: str) -> bool:
        """
        Register service for health monitoring.
        
        Args:
            service_name: Name of service to register
            endpoint: Health check endpoint URL
            
        Returns:
            True if registered successfully, False otherwise
        """
=======
"""
Core interfaces for BharatVoice Assistant services.

This module defines the abstract interfaces that all service implementations
must follow, ensuring consistent contracts across the microservices architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    ConversationState,
    CulturalContext,
    Intent,
    LanguageCode,
    RecognitionResult,
    RegionalContextData,
    Response,
    ServiceParameters,
    ServiceResult,
    UserInteraction,
    UserProfile,
    VoiceActivityResult,
)


class AudioProcessor(ABC):
    """Interface for audio processing operations."""
    
    @abstractmethod
    async def process_audio_stream(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio stream with language-specific optimizations.
        
        Args:
            audio_data: Input audio buffer
            language: Target language for processing
            
        Returns:
            Processed audio buffer
        """
        pass
    
    @abstractmethod
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        pass
    
    @abstractmethod
    async def synthesize_speech(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech from text with specified language and accent.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        pass
    
    @abstractmethod
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise from audio data.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        pass


class LanguageEngine(ABC):
    """Interface for language processing operations."""
    
    @abstractmethod
    async def recognize_speech(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech from audio input.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result with transcription and metadata
        """
        pass
    
    @abstractmethod
    async def detect_code_switching(self, text: str) -> List[Dict[str, any]]:
        """
        Detect code-switching points in multilingual text.
        
        Args:
            text: Input text potentially containing multiple languages
            
        Returns:
            List of code-switching detection results
        """
        pass
    
    @abstractmethod
    async def translate_text(
        self, 
        text: str, 
        source_lang: LanguageCode, 
        target_lang: LanguageCode
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        pass
    
    @abstractmethod
    async def adapt_to_regional_accent(
        self, 
        model_id: str, 
        accent_data: Dict[str, any]
    ) -> str:
        """
        Adapt language model to regional accent.
        
        Args:
            model_id: Base model identifier
            accent_data: Regional accent adaptation data
            
        Returns:
            Adapted model identifier
        """
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> LanguageCode:
        """
        Detect the primary language of input text.
        
        Args:
            text: Input text for language detection
            
        Returns:
            Detected language code
        """
        pass


class ContextManager(ABC):
    """Interface for conversation context and user profile management."""
    
    @abstractmethod
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
        """
        pass
    
    @abstractmethod
    async def update_user_profile(
        self, 
        user_id: str, 
        preferences: Dict[str, any]
    ) -> UserProfile:
        """
        Update user profile with new preferences.
        
        Args:
            user_id: Unique user identifier
            preferences: Updated user preferences
            
        Returns:
            Updated user profile
        """
        pass
    
    @abstractmethod
    async def get_regional_context(
        self, 
        location: Dict[str, any]
    ) -> RegionalContextData:
        """
        Get regional context information for a location.
        
        Args:
            location: Location data (latitude, longitude, etc.)
            
        Returns:
            Regional context data
        """
        pass
    
    @abstractmethod
    async def learn_from_interaction(self, interaction: UserInteraction) -> Dict[str, any]:
        """
        Learn and adapt from user interaction.
        
        Args:
            interaction: User interaction to learn from
            
        Returns:
            Learning result with adaptation information
        """
        pass
    
    @abstractmethod
    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """
        Retrieve conversation state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation state if exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists, None otherwise
        """
        pass


class ResponseGenerator(ABC):
    """Interface for query processing and response generation."""
    
    @abstractmethod
    async def process_query(
        self, 
        query: str, 
        context: ConversationState
    ) -> Dict[str, any]:
        """
        Process user query and extract intent and entities.
        
        Args:
            query: User query text
            context: Current conversation context
            
        Returns:
            Query processing result with intent and entities
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        intent: Intent, 
        entities: List[Dict[str, any]], 
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class AuthenticationService(ABC):
    """Interface for user authentication and authorization."""
    
    @abstractmethod
    async def authenticate_user(self, credentials: Dict[str, any]) -> Dict[str, any]:
        """
        Authenticate user with provided credentials.
        
        Args:
            credentials: User authentication credentials
            
        Returns:
            Authentication result with user information
        """
        pass
    
    @abstractmethod
    async def authorize_request(
        self, 
        user_id: str, 
        resource: str, 
        action: str
    ) -> bool:
        """
        Authorize user request for specific resource and action.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if authorized, False otherwise
        """
        pass
    
    @abstractmethod
    async def create_session(self, user_id: str) -> str:
        """
        Create new user session.
        
        Args:
            user_id: User identifier
            
        Returns:
            Session token
        """
        pass
    
    @abstractmethod
    async def validate_session(self, session_token: str) -> Optional[str]:
        """
        Validate session token and return user ID.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            User ID if valid, None otherwise
        """
        pass


class OfflineSyncService(ABC):
    """Interface for offline capability and data synchronization."""
    
    @abstractmethod
    async def cache_for_offline(self, data: Dict[str, any], cache_key: str) -> bool:
        """
        Cache data for offline access.
        
        Args:
            data: Data to cache
            cache_key: Unique cache key
            
        Returns:
            True if cached successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_offline_data(self, cache_key: str) -> Optional[Dict[str, any]]:
        """
        Retrieve cached data for offline use.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data if available, None otherwise
        """
        pass
    
    @abstractmethod
    async def sync_when_online(self, user_id: str) -> Dict[str, any]:
        """
        Synchronize offline data when connectivity is restored.
        
        Args:
            user_id: User identifier for data synchronization
            
        Returns:
            Synchronization result
        """
        pass
    
    @abstractmethod
    async def is_offline_mode(self) -> bool:
        """
        Check if system is currently in offline mode.
        
        Returns:
            True if offline, False if online
        """
        pass


class HealthCheckService(ABC):
    """Interface for service health monitoring."""
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> Dict[str, any]:
        """
        Check health status of a specific service.
        
        Args:
            service_name: Name of service to check
            
        Returns:
            Health check result with status and metrics
        """
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, any]:
        """
        Get overall system health metrics.
        
        Returns:
            System metrics and performance data
        """
        pass
    
    @abstractmethod
    async def register_service(self, service_name: str, endpoint: str) -> bool:
        """
        Register service for health monitoring.
        
        Args:
            service_name: Name of service to register
            endpoint: Health check endpoint URL
            
        Returns:
            True if registered successfully, False otherwise
        """
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        pass