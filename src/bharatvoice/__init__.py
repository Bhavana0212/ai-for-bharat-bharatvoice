"""
BharatVoice Assistant - AI-powered multilingual voice assistant for India.

This package provides a comprehensive voice assistant system designed specifically
for the Indian market, supporting multiple Indian languages, cultural context
understanding, and integration with Indian digital services.
"""

__version__ = "0.1.0"
__author__ = "BharatVoice Team"
__email__ = "team@bharatvoice.ai"

from bharatvoice.core.models import (
    AudioBuffer,
    ConversationState,
    RecognitionResult,
    RegionalContextData,
    Response,
    UserProfile,
)

__all__ = [
    "AudioBuffer",
    "ConversationState", 
    "RecognitionResult",
    "RegionalContextData",
    "Response",
    "UserProfile",
]