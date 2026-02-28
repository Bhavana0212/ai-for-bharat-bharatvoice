<<<<<<< HEAD
"""
Voice Processing Service for audio input/output and speech processing.

This module provides comprehensive audio processing capabilities including:
- Real-time audio stream processing
- Voice Activity Detection (VAD) using WebRTC VAD
- Background noise filtering using spectral subtraction
- Text-to-Speech synthesis with Indian language support
- Audio format conversion and preprocessing utilities
"""

from bharatvoice.services.voice_processing.audio_processor import (
    AudioFormatConverter,
    AudioProcessor,
    RealTimeAudioProcessor,
)
from bharatvoice.services.voice_processing.service import (
    VoiceProcessingService,
    create_voice_processing_service,
)
from bharatvoice.services.voice_processing.tts_engine import (
    AdaptiveTTSEngine,
    TTSEngine,
)

__all__ = [
    "AudioProcessor",
    "AudioFormatConverter", 
    "RealTimeAudioProcessor",
    "TTSEngine",
    "AdaptiveTTSEngine",
    "VoiceProcessingService",
    "create_voice_processing_service",
=======
"""
Voice Processing Service for audio input/output and speech processing.

This module provides comprehensive audio processing capabilities including:
- Real-time audio stream processing
- Voice Activity Detection (VAD) using WebRTC VAD
- Background noise filtering using spectral subtraction
- Text-to-Speech synthesis with Indian language support
- Audio format conversion and preprocessing utilities
"""

from bharatvoice.services.voice_processing.audio_processor import (
    AudioFormatConverter,
    AudioProcessor,
    RealTimeAudioProcessor,
)
from bharatvoice.services.voice_processing.service import (
    VoiceProcessingService,
    create_voice_processing_service,
)
from bharatvoice.services.voice_processing.tts_engine import (
    AdaptiveTTSEngine,
    TTSEngine,
)

__all__ = [
    "AudioProcessor",
    "AudioFormatConverter", 
    "RealTimeAudioProcessor",
    "TTSEngine",
    "AdaptiveTTSEngine",
    "VoiceProcessingService",
    "create_voice_processing_service",
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
]