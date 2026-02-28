<<<<<<< HEAD
"""
Response Generation Service Package.

This package provides Natural Language Understanding (NLU) and response generation
capabilities for the BharatVoice Assistant, with specialized support for Indian
cultural context, colloquial terms, and multilingual processing.
"""

from .nlu_service import (
    NLUService,
    ColloquialTermMapper,
    IndianEntityExtractor,
    IndianIntentClassifier,
    CulturalContextInterpreter,
    IntentCategory,
    EntityType
)

from .nlu_interface import NLUInterface

__all__ = [
    'NLUService',
    'ColloquialTermMapper',
    'IndianEntityExtractor',
    'IndianIntentClassifier',
    'CulturalContextInterpreter',
    'IntentCategory',
    'EntityType',
    'NLUInterface'
=======
"""
Response Generation Service Package.

This package provides Natural Language Understanding (NLU) and response generation
capabilities for the BharatVoice Assistant, with specialized support for Indian
cultural context, colloquial terms, and multilingual processing.
"""

from .nlu_service import (
    NLUService,
    ColloquialTermMapper,
    IndianEntityExtractor,
    IndianIntentClassifier,
    CulturalContextInterpreter,
    IntentCategory,
    EntityType
)

from .nlu_interface import NLUInterface

__all__ = [
    'NLUService',
    'ColloquialTermMapper',
    'IndianEntityExtractor',
    'IndianIntentClassifier',
    'CulturalContextInterpreter',
    'IntentCategory',
    'EntityType',
    'NLUInterface'
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
]