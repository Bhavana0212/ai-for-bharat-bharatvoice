"""
Learning and Adaptation Services for BharatVoice Assistant.

This package provides advanced adaptive learning mechanisms, system extensibility,
and personalization features for the voice assistant.
"""

from .adaptive_learning_service import AdaptiveLearningService
from .vocabulary_learner import VocabularyLearner
from .accent_adapter import AccentAdapter
from .preference_learner import PreferenceLearner
from .feedback_processor import FeedbackProcessor
from .response_style_adapter import ResponseStyleAdapter
from .model_manager import ModelManager
from .plugin_manager import PluginManager
from .ab_testing_framework import ABTestingFramework
from .system_extensibility_service import SystemExtensibilityService

__all__ = [
    "AdaptiveLearningService",
    "VocabularyLearner", 
    "AccentAdapter",
    "PreferenceLearner",
    "FeedbackProcessor",
    "ResponseStyleAdapter",
    "ModelManager",
    "PluginManager", 
    "ABTestingFramework",
    "SystemExtensibilityService"
]