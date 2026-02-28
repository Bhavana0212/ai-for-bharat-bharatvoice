<<<<<<< HEAD
"""
Voice processing endpoints for BharatVoice Assistant.

This module provides endpoints for speech recognition, text-to-speech synthesis,
and voice activity detection with support for multiple Indian languages.
"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import structlog

from bharatvoice.config import get_settings, Settings
from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    VoiceActivityResult,
)


logger = structlog.get_logger(__name__)
router = APIRouter()


class SpeechRecognitionRequest(BaseModel):
    """Speech recognition request model."""
    
    language: Optional[LanguageCode] = None
    enable_code_switching: bool = True
    max_alternatives: int = 3
    filter_profanity: bool = True


class SpeechRecognitionResponse(BaseModel):
    """Speech recognition response model."""
    
    request_id: str
    result: RecognitionResult
    processing_time: float


class TextToSpeechRequest(BaseModel):
    """Text-to-speech synthesis request model."""
    
    text: str
    language: LanguageCode
    accent: AccentType = AccentType.STANDARD
    speed: float = 1.0
    pitch: float = 1.0


class TextToSpeechResponse(BaseModel):
    """Text-to-speech synthesis response model."""
    
    request_id: str
    audio_url: str
    duration: float
    format: AudioFormat


class VoiceActivityRequest(BaseModel):
    """Voice activity detection request model."""
    
    sensitivity: float = 0.5
    min_speech_duration: float = 0.1


@router.post("/recognize", response_model=SpeechRecognitionResponse)
async def recognize_speech(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    enable_code_switching: bool = Form(True),
    max_alternatives: int = Form(3),
    settings: Settings = Depends(get_settings)
):
    """
    Recognize speech from audio file.
    
    Args:
        audio_file: Audio file to process
        language: Target language (auto-detect if None)
        enable_code_switching: Enable code-switching detection
        max_alternatives: Maximum alternative transcriptions
        settings: Application settings
        
    Returns:
        Speech recognition result
    """
    try:
        request_id = str(uuid4())
        
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # TODO: Implement actual speech recognition
        # This is a placeholder implementation
        
        # Create mock recognition result
        mock_result = RecognitionResult(
            transcribed_text="नमस्ते, मैं BharatVoice Assistant हूं। How can I help you today?",
            confidence=0.95,
            detected_language=LanguageCode.HINDI,
            code_switching_points=[],
            alternative_transcriptions=[],
            processing_time=1.2
        )
        
        logger.info(
            "Speech recognition completed",
            request_id=request_id,
            language=language,
            confidence=mock_result.confidence,
            text_length=len(mock_result.transcribed_text)
        )
        
        return SpeechRecognitionResponse(
            request_id=request_id,
            result=mock_result,
            processing_time=mock_result.processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Speech recognition error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Speech recognition failed"
        )


@router.post("/synthesize", response_model=TextToSpeechResponse)
async def synthesize_speech(
    request: TextToSpeechRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Synthesize speech from text.
    
    Args:
        request: Text-to-speech request
        settings: Application settings
        
    Returns:
        Text-to-speech synthesis result
    """
    try:
        request_id = str(uuid4())
        
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long (maximum 5000 characters)"
            )
        
        # TODO: Implement actual text-to-speech synthesis
        # This is a placeholder implementation
        
        # Calculate estimated duration (rough estimate: 150 words per minute)
        word_count = len(request.text.split())
        estimated_duration = (word_count / 150) * 60  # seconds
        
        # Mock audio URL (in production, this would be a real audio file)
        audio_url = f"/audio/tts/{request_id}.wav"
        
        logger.info(
            "Text-to-speech synthesis completed",
            request_id=request_id,
            language=request.language,
            accent=request.accent,
            text_length=len(request.text),
            duration=estimated_duration
        )
        
        return TextToSpeechResponse(
            request_id=request_id,
            audio_url=audio_url,
            duration=estimated_duration,
            format=AudioFormat.WAV
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text-to-speech synthesis error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Speech synthesis failed"
        )


@router.post("/voice-activity", response_model=VoiceActivityResult)
async def detect_voice_activity(
    audio_file: UploadFile = File(...),
    sensitivity: float = Form(0.5),
    min_speech_duration: float = Form(0.1),
    settings: Settings = Depends(get_settings)
):
    """
    Detect voice activity in audio file.
    
    Args:
        audio_file: Audio file to analyze
        sensitivity: Detection sensitivity (0.0-1.0)
        min_speech_duration: Minimum speech duration in seconds
        settings: Application settings
        
    Returns:
        Voice activity detection result
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # TODO: Implement actual voice activity detection
        # This is a placeholder implementation
        
        # Mock voice activity result
        mock_result = VoiceActivityResult(
            is_speech=True,
            confidence=0.87,
            start_time=0.5,
            end_time=3.2,
            energy_level=0.65
        )
        
        logger.info(
            "Voice activity detection completed",
            is_speech=mock_result.is_speech,
            confidence=mock_result.confidence,
            duration=mock_result.end_time - mock_result.start_time
        )
        
        return mock_result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice activity detection error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Voice activity detection failed"
        )


@router.get("/languages", response_model=List[dict])
async def get_supported_languages():
    """
    Get list of supported languages.
    
    Returns:
        List of supported languages with metadata
    """
    try:
        languages = [
            {
                "code": "hi",
                "name": "Hindi",
                "native_name": "हिन्दी",
                "region": "North India",
                "accuracy_target": 0.90
            },
            {
                "code": "en-IN",
                "name": "English (India)",
                "native_name": "English",
                "region": "Pan-India",
                "accuracy_target": 0.90
            },
            {
                "code": "ta",
                "name": "Tamil",
                "native_name": "தமிழ்",
                "region": "Tamil Nadu",
                "accuracy_target": 0.85
            },
            {
                "code": "te",
                "name": "Telugu",
                "native_name": "తెలుగు",
                "region": "Andhra Pradesh, Telangana",
                "accuracy_target": 0.85
            },
            {
                "code": "bn",
                "name": "Bengali",
                "native_name": "বাংলা",
                "region": "West Bengal",
                "accuracy_target": 0.85
            },
            {
                "code": "mr",
                "name": "Marathi",
                "native_name": "मराठी",
                "region": "Maharashtra",
                "accuracy_target": 0.85
            },
            {
                "code": "gu",
                "name": "Gujarati",
                "native_name": "ગુજરાતી",
                "region": "Gujarat",
                "accuracy_target": 0.85
            },
            {
                "code": "kn",
                "name": "Kannada",
                "native_name": "ಕನ್ನಡ",
                "region": "Karnataka",
                "accuracy_target": 0.85
            },
            {
                "code": "ml",
                "name": "Malayalam",
                "native_name": "മലയാളം",
                "region": "Kerala",
                "accuracy_target": 0.85
            },
            {
                "code": "pa",
                "name": "Punjabi",
                "native_name": "ਪੰਜਾਬੀ",
                "region": "Punjab",
                "accuracy_target": 0.85
            },
            {
                "code": "or",
                "name": "Odia",
                "native_name": "ଓଡ଼ିଆ",
                "region": "Odisha",
                "accuracy_target": 0.85
            }
        ]
        
        return languages
    
    except Exception as e:
        logger.error("Error fetching supported languages", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch supported languages"
=======
"""
Voice processing endpoints for BharatVoice Assistant.

This module provides endpoints for speech recognition, text-to-speech synthesis,
and voice activity detection with support for multiple Indian languages.
"""

from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import structlog

from bharatvoice.config import get_settings, Settings
from bharatvoice.core.models import (
    AccentType,
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    RecognitionResult,
    VoiceActivityResult,
)


logger = structlog.get_logger(__name__)
router = APIRouter()


class SpeechRecognitionRequest(BaseModel):
    """Speech recognition request model."""
    
    language: Optional[LanguageCode] = None
    enable_code_switching: bool = True
    max_alternatives: int = 3
    filter_profanity: bool = True


class SpeechRecognitionResponse(BaseModel):
    """Speech recognition response model."""
    
    request_id: str
    result: RecognitionResult
    processing_time: float


class TextToSpeechRequest(BaseModel):
    """Text-to-speech synthesis request model."""
    
    text: str
    language: LanguageCode
    accent: AccentType = AccentType.STANDARD
    speed: float = 1.0
    pitch: float = 1.0


class TextToSpeechResponse(BaseModel):
    """Text-to-speech synthesis response model."""
    
    request_id: str
    audio_url: str
    duration: float
    format: AudioFormat


class VoiceActivityRequest(BaseModel):
    """Voice activity detection request model."""
    
    sensitivity: float = 0.5
    min_speech_duration: float = 0.1


@router.post("/recognize", response_model=SpeechRecognitionResponse)
async def recognize_speech(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    enable_code_switching: bool = Form(True),
    max_alternatives: int = Form(3),
    settings: Settings = Depends(get_settings)
):
    """
    Recognize speech from audio file.
    
    Args:
        audio_file: Audio file to process
        language: Target language (auto-detect if None)
        enable_code_switching: Enable code-switching detection
        max_alternatives: Maximum alternative transcriptions
        settings: Application settings
        
    Returns:
        Speech recognition result
    """
    try:
        request_id = str(uuid4())
        
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # TODO: Implement actual speech recognition
        # This is a placeholder implementation
        
        # Create mock recognition result
        mock_result = RecognitionResult(
            transcribed_text="नमस्ते, मैं BharatVoice Assistant हूं। How can I help you today?",
            confidence=0.95,
            detected_language=LanguageCode.HINDI,
            code_switching_points=[],
            alternative_transcriptions=[],
            processing_time=1.2
        )
        
        logger.info(
            "Speech recognition completed",
            request_id=request_id,
            language=language,
            confidence=mock_result.confidence,
            text_length=len(mock_result.transcribed_text)
        )
        
        return SpeechRecognitionResponse(
            request_id=request_id,
            result=mock_result,
            processing_time=mock_result.processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Speech recognition error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Speech recognition failed"
        )


@router.post("/synthesize", response_model=TextToSpeechResponse)
async def synthesize_speech(
    request: TextToSpeechRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Synthesize speech from text.
    
    Args:
        request: Text-to-speech request
        settings: Application settings
        
    Returns:
        Text-to-speech synthesis result
    """
    try:
        request_id = str(uuid4())
        
        # Validate text length
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400,
                detail="Text too long (maximum 5000 characters)"
            )
        
        # TODO: Implement actual text-to-speech synthesis
        # This is a placeholder implementation
        
        # Calculate estimated duration (rough estimate: 150 words per minute)
        word_count = len(request.text.split())
        estimated_duration = (word_count / 150) * 60  # seconds
        
        # Mock audio URL (in production, this would be a real audio file)
        audio_url = f"/audio/tts/{request_id}.wav"
        
        logger.info(
            "Text-to-speech synthesis completed",
            request_id=request_id,
            language=request.language,
            accent=request.accent,
            text_length=len(request.text),
            duration=estimated_duration
        )
        
        return TextToSpeechResponse(
            request_id=request_id,
            audio_url=audio_url,
            duration=estimated_duration,
            format=AudioFormat.WAV
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text-to-speech synthesis error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Speech synthesis failed"
        )


@router.post("/voice-activity", response_model=VoiceActivityResult)
async def detect_voice_activity(
    audio_file: UploadFile = File(...),
    sensitivity: float = Form(0.5),
    min_speech_duration: float = Form(0.1),
    settings: Settings = Depends(get_settings)
):
    """
    Detect voice activity in audio file.
    
    Args:
        audio_file: Audio file to analyze
        sensitivity: Detection sensitivity (0.0-1.0)
        min_speech_duration: Minimum speech duration in seconds
        settings: Application settings
        
    Returns:
        Voice activity detection result
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # TODO: Implement actual voice activity detection
        # This is a placeholder implementation
        
        # Mock voice activity result
        mock_result = VoiceActivityResult(
            is_speech=True,
            confidence=0.87,
            start_time=0.5,
            end_time=3.2,
            energy_level=0.65
        )
        
        logger.info(
            "Voice activity detection completed",
            is_speech=mock_result.is_speech,
            confidence=mock_result.confidence,
            duration=mock_result.end_time - mock_result.start_time
        )
        
        return mock_result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice activity detection error", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Voice activity detection failed"
        )


@router.get("/languages", response_model=List[dict])
async def get_supported_languages():
    """
    Get list of supported languages.
    
    Returns:
        List of supported languages with metadata
    """
    try:
        languages = [
            {
                "code": "hi",
                "name": "Hindi",
                "native_name": "हिन्दी",
                "region": "North India",
                "accuracy_target": 0.90
            },
            {
                "code": "en-IN",
                "name": "English (India)",
                "native_name": "English",
                "region": "Pan-India",
                "accuracy_target": 0.90
            },
            {
                "code": "ta",
                "name": "Tamil",
                "native_name": "தமிழ்",
                "region": "Tamil Nadu",
                "accuracy_target": 0.85
            },
            {
                "code": "te",
                "name": "Telugu",
                "native_name": "తెలుగు",
                "region": "Andhra Pradesh, Telangana",
                "accuracy_target": 0.85
            },
            {
                "code": "bn",
                "name": "Bengali",
                "native_name": "বাংলা",
                "region": "West Bengal",
                "accuracy_target": 0.85
            },
            {
                "code": "mr",
                "name": "Marathi",
                "native_name": "मराठी",
                "region": "Maharashtra",
                "accuracy_target": 0.85
            },
            {
                "code": "gu",
                "name": "Gujarati",
                "native_name": "ગુજરાતી",
                "region": "Gujarat",
                "accuracy_target": 0.85
            },
            {
                "code": "kn",
                "name": "Kannada",
                "native_name": "ಕನ್ನಡ",
                "region": "Karnataka",
                "accuracy_target": 0.85
            },
            {
                "code": "ml",
                "name": "Malayalam",
                "native_name": "മലയാളം",
                "region": "Kerala",
                "accuracy_target": 0.85
            },
            {
                "code": "pa",
                "name": "Punjabi",
                "native_name": "ਪੰਜਾਬੀ",
                "region": "Punjab",
                "accuracy_target": 0.85
            },
            {
                "code": "or",
                "name": "Odia",
                "native_name": "ଓଡ଼ିଆ",
                "region": "Odisha",
                "accuracy_target": 0.85
            }
        ]
        
        return languages
    
    except Exception as e:
        logger.error("Error fetching supported languages", exc_info=e)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch supported languages"
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        )