"""
Context Management Service for BharatVoice Assistant.

This package provides conversation state management, enhanced user profile management,
and regional context services with privacy compliance and adaptive learning.
"""

from bharatvoice.services.context_management.conversation_manager import (
    ConversationContextManager,
    ConversationManager,
)
from bharatvoice.services.context_management.service import ContextManagementService
from bharatvoice.services.context_management.user_profile_manager import (
    UserProfileManager,
    ProfileEncryption,
    LanguageLearningEngine,
    LocationContextManager,
)

__all__ = [
    "ContextManagementService",
    "ConversationManager",
    "ConversationContextManager",
    "UserProfileManager",
    "ProfileEncryption",
    "LanguageLearningEngine",
    "LocationContextManager",
]