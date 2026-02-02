"""
Core data models for BharatVoice Assistant.

This module defines the primary data structures used throughout the system,
including user profiles, conversation state, audio processing results, and
regional context information.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, validator


class LanguageCode(str, Enum):
    """Supported language codes for BharatVoice Assistant."""
    
    # Primary languages
    HINDI = "hi"
    ENGLISH_IN = "en-IN"  # Indian English
    
    # Regional Indian languages
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    ODIA = "or"


class AccentType(str, Enum):
    """Regional accent types for speech synthesis."""
    
    STANDARD = "standard"
    NORTH_INDIAN = "north_indian"
    SOUTH_INDIAN = "south_indian"
    WEST_INDIAN = "west_indian"
    EAST_INDIAN = "east_indian"
    MUMBAI = "mumbai"
    DELHI = "delhi"
    BANGALORE = "bangalore"
    CHENNAI = "chennai"
    KOLKATA = "kolkata"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class ServiceType(str, Enum):
    """Types of external services that can be integrated."""
    
    INDIAN_RAILWAYS = "indian_railways"
    WEATHER = "weather"
    UPI_PAYMENT = "upi_payment"
    FOOD_DELIVERY = "food_delivery"
    RIDE_SHARING = "ride_sharing"
    GOVERNMENT_SERVICE = "government_service"
    CRICKET_SCORES = "cricket_scores"
    BOLLYWOOD_NEWS = "bollywood_news"
    PLATFORM_INTEGRATION = "platform_integration"


class AudioBuffer(BaseModel):
    """Represents audio data with metadata."""
    
    data: List[float] = Field(..., description="Audio sample data")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    duration: float = Field(..., description="Duration in seconds")
    
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError('Sample rate must be one of: 8000, 16000, 22050, 44100, 48000')
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        if v not in [1, 2]:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v
    
    @property
    def numpy_array(self) -> np.ndarray:
        """Convert audio data to numpy array."""
        return np.array(self.data, dtype=np.float32)
    
    class Config:
        arbitrary_types_allowed = True


class VoiceActivityResult(BaseModel):
    """Result of voice activity detection."""
    
    is_speech: bool = Field(..., description="Whether speech was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_time: float = Field(..., description="Start time of speech segment")
    end_time: float = Field(..., description="End time of speech segment")
    energy_level: float = Field(..., description="Audio energy level")


class LanguageSwitchPoint(BaseModel):
    """Represents a point where language switching occurs in text."""
    
    position: int = Field(..., description="Character position of language switch")
    from_language: LanguageCode = Field(..., description="Source language")
    to_language: LanguageCode = Field(..., description="Target language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Switch detection confidence")


class AlternativeResult(BaseModel):
    """Alternative transcription result."""
    
    text: str = Field(..., description="Alternative transcribed text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    language: LanguageCode = Field(..., description="Detected language")


class RecognitionResult(BaseModel):
    """Result of speech recognition processing."""
    
    transcribed_text: str = Field(..., description="Primary transcribed text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    detected_language: LanguageCode = Field(..., description="Primary detected language")
    code_switching_points: List[LanguageSwitchPoint] = Field(
        default_factory=list, description="Points where language switching occurs"
    )
    alternative_transcriptions: List[AlternativeResult] = Field(
        default_factory=list, description="Alternative transcription results"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    
    @property
    def has_code_switching(self) -> bool:
        """Check if the result contains code-switching."""
        return len(self.code_switching_points) > 0


class LocationData(BaseModel):
    """Geographic location information."""
    
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State name")
    country: str = Field(default="India", description="Country name")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    timezone: str = Field(default="Asia/Kolkata", description="Timezone identifier")


class UsageAnalytics(BaseModel):
    """User usage pattern analytics."""
    
    total_interactions: int = Field(default=0, description="Total number of interactions")
    preferred_time_slots: List[int] = Field(
        default_factory=list, description="Preferred hours of day (0-23)"
    )
    common_query_types: Dict[str, int] = Field(
        default_factory=dict, description="Frequency of query types"
    )
    language_usage_frequency: Dict[LanguageCode, float] = Field(
        default_factory=dict, description="Language usage percentages"
    )
    average_session_duration: float = Field(
        default=0.0, description="Average session duration in minutes"
    )
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")


class PrivacyConfiguration(BaseModel):
    """User privacy settings and preferences."""
    
    data_retention_days: int = Field(default=90, description="Data retention period")
    allow_analytics: bool = Field(default=True, description="Allow usage analytics")
    allow_personalization: bool = Field(default=True, description="Allow personalization")
    voice_data_storage: bool = Field(default=False, description="Store voice recordings")
    location_sharing: bool = Field(default=True, description="Share location data")
    third_party_integrations: bool = Field(default=True, description="Allow third-party integrations")


class UserProfile(BaseModel):
    """User profile with preferences and settings."""
    
    user_id: UUID = Field(default_factory=uuid4, description="Unique user identifier")
    preferred_languages: List[LanguageCode] = Field(
        default_factory=lambda: [LanguageCode.HINDI, LanguageCode.ENGLISH_IN],
        description="Ordered list of preferred languages"
    )
    primary_language: LanguageCode = Field(
        default=LanguageCode.HINDI, description="Primary language for responses"
    )
    location: Optional[LocationData] = Field(None, description="User location")
    usage_patterns: UsageAnalytics = Field(
        default_factory=UsageAnalytics, description="Usage analytics"
    )
    privacy_settings: PrivacyConfiguration = Field(
        default_factory=PrivacyConfiguration, description="Privacy configuration"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    @validator('preferred_languages')
    def validate_preferred_languages(cls, v):
        if not v:
            raise ValueError('At least one preferred language must be specified')
        return v


class UserInteraction(BaseModel):
    """Represents a single user interaction."""
    
    interaction_id: UUID = Field(default_factory=uuid4, description="Unique interaction ID")
    user_id: UUID = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction timestamp")
    input_text: str = Field(..., description="User input text")
    input_language: LanguageCode = Field(..., description="Input language")
    response_text: str = Field(..., description="System response text")
    response_language: LanguageCode = Field(..., description="Response language")
    intent: Optional[str] = Field(None, description="Detected intent")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    processing_time: float = Field(..., description="Processing time in seconds")


class ConversationState(BaseModel):
    """Maintains conversation context and state."""
    
    session_id: UUID = Field(default_factory=uuid4, description="Unique session identifier")
    user_id: UUID = Field(..., description="User identifier")
    current_language: LanguageCode = Field(..., description="Current conversation language")
    conversation_history: List[UserInteraction] = Field(
        default_factory=list, description="Conversation history"
    )
    context_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Context variables"
    )
    last_interaction_time: datetime = Field(
        default_factory=datetime.utcnow, description="Last interaction timestamp"
    )
    is_active: bool = Field(default=True, description="Whether session is active")
    
    @property
    def interaction_count(self) -> int:
        """Get the number of interactions in this session."""
        return len(self.conversation_history)
    
    @property
    def session_duration(self) -> float:
        """Get session duration in minutes."""
        if not self.conversation_history:
            return 0.0
        
        start_time = self.conversation_history[0].timestamp
        end_time = self.last_interaction_time
        return (end_time - start_time).total_seconds() / 60.0


class LocalService(BaseModel):
    """Represents a local service or business."""
    
    name: str = Field(..., description="Service name")
    category: str = Field(..., description="Service category")
    address: str = Field(..., description="Service address")
    phone: Optional[str] = Field(None, description="Contact phone number")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Service rating")
    distance_km: Optional[float] = Field(None, description="Distance from user in km")


class WeatherData(BaseModel):
    """Weather information for a location."""
    
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Humidity percentage")
    description: str = Field(..., description="Weather description")
    wind_speed_kmh: float = Field(..., description="Wind speed in km/h")
    precipitation_mm: float = Field(default=0.0, description="Precipitation in mm")
    is_monsoon_season: bool = Field(default=False, description="Whether it's monsoon season")
    air_quality_index: Optional[int] = Field(None, description="Air quality index")


class CulturalEvent(BaseModel):
    """Cultural event or festival information."""
    
    name: str = Field(..., description="Event name")
    date: datetime = Field(..., description="Event date")
    description: str = Field(..., description="Event description")
    significance: str = Field(..., description="Cultural significance")
    regional_relevance: List[str] = Field(
        default_factory=list, description="Relevant regions/states"
    )
    celebration_type: str = Field(..., description="Type of celebration")


class TransportService(BaseModel):
    """Transportation service information."""
    
    service_type: str = Field(..., description="Type of transport service")
    name: str = Field(..., description="Service name")
    route: str = Field(..., description="Route information")
    schedule: Dict[str, Any] = Field(default_factory=dict, description="Schedule information")
    fare: Optional[float] = Field(None, description="Fare information")
    availability: bool = Field(default=True, description="Service availability")


class GovernmentService(BaseModel):
    """Government service information."""
    
    service_name: str = Field(..., description="Government service name")
    department: str = Field(..., description="Government department")
    description: str = Field(..., description="Service description")
    required_documents: List[str] = Field(
        default_factory=list, description="Required documents"
    )
    online_portal: Optional[str] = Field(None, description="Online portal URL")
    processing_time: Optional[str] = Field(None, description="Expected processing time")


class RegionalContextData(BaseModel):
    """Regional context information for a location."""
    
    location: LocationData = Field(..., description="Location information")
    local_services: List[LocalService] = Field(
        default_factory=list, description="Local services and businesses"
    )
    weather_info: Optional[WeatherData] = Field(None, description="Current weather")
    cultural_events: List[CulturalEvent] = Field(
        default_factory=list, description="Upcoming cultural events"
    )
    transport_options: List[TransportService] = Field(
        default_factory=list, description="Available transport services"
    )
    government_services: List[GovernmentService] = Field(
        default_factory=list, description="Available government services"
    )
    local_language: LanguageCode = Field(..., description="Primary local language")
    dialect_info: Optional[str] = Field(None, description="Local dialect information")


class Intent(BaseModel):
    """Represents a detected user intent."""
    
    name: str = Field(..., description="Intent name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")
    category: str = Field(..., description="Intent category")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")


class Entity(BaseModel):
    """Represents an extracted entity from user input."""
    
    name: str = Field(..., description="Entity name")
    value: str = Field(..., description="Entity value")
    type: str = Field(..., description="Entity type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    start_pos: int = Field(..., description="Start position in text")
    end_pos: int = Field(..., description="End position in text")


class Response(BaseModel):
    """System response to user query."""
    
    response_id: UUID = Field(default_factory=uuid4, description="Unique response ID")
    text: str = Field(..., description="Response text")
    language: LanguageCode = Field(..., description="Response language")
    intent: Optional[Intent] = Field(None, description="Detected intent")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    requires_followup: bool = Field(default=False, description="Whether followup is needed")
    suggested_actions: List[str] = Field(
        default_factory=list, description="Suggested user actions"
    )
    external_service_used: Optional[ServiceType] = Field(
        None, description="External service used for response"
    )
    processing_time: float = Field(..., description="Response generation time")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ServiceParameters(BaseModel):
    """Parameters for external service calls."""
    
    service_type: ServiceType = Field(..., description="Type of service")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Service parameters")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retry attempts")


class ServiceResult(BaseModel):
    """Result from external service integration."""
    
    service_type: ServiceType = Field(..., description="Service type")
    success: bool = Field(..., description="Whether the service call succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Service response data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    response_time: float = Field(..., description="Service response time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class CulturalContext(BaseModel):
    """Cultural context information for response formatting."""
    
    region: str = Field(..., description="Cultural region")
    festivals: List[str] = Field(default_factory=list, description="Relevant festivals")
    local_customs: List[str] = Field(default_factory=list, description="Local customs")
    preferred_greetings: List[str] = Field(default_factory=list, description="Preferred greetings")
    cultural_references: Dict[str, str] = Field(
        default_factory=dict, description="Cultural references and meanings"
    )