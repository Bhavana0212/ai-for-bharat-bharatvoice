"""
Accessibility API endpoints for BharatVoice Assistant.

This module provides REST API endpoints for managing accessibility features,
settings, and voice-guided help system.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.accessibility import (
    get_accessibility_manager,
    AccessibilitySettings,
    VolumeLevel,
    SpeechRate,
    InteractionMode,
    AccessibilityMode
)
from bharatvoice.utils.error_handler import handle_error, ErrorCode


logger = structlog.get_logger(__name__)
router = APIRouter()


class AccessibilitySettingsRequest(BaseModel):
    """Request model for updating accessibility settings."""
    
    volume_level: Optional[int] = Field(None, ge=0, le=6, description="Volume level (0-6)")
    speech_rate: Optional[float] = Field(None, ge=0.5, le=2.0, description="Speech rate (0.5-2.0)")
    listening_timeout: Optional[float] = Field(None, ge=5.0, le=120.0, description="Listening timeout in seconds")
    max_recognition_attempts: Optional[int] = Field(None, ge=1, le=10, description="Maximum recognition attempts")
    interaction_mode: Optional[str] = Field(None, description="Interaction mode")
    accessibility_mode: Optional[str] = Field(None, description="Accessibility mode")
    preferred_language: Optional[str] = Field(None, description="Preferred language code")
    enable_visual_indicators: Optional[bool] = Field(None, description="Enable visual indicators")
    enable_voice_guided_help: Optional[bool] = Field(None, description="Enable voice-guided help")
    enable_confirmation_prompts: Optional[bool] = Field(None, description="Enable confirmation prompts")
    enable_detailed_feedback: Optional[bool] = Field(None, description="Enable detailed feedback")
    enable_audio_descriptions: Optional[bool] = Field(None, description="Enable audio descriptions")
    enable_tutorial_mode: Optional[bool] = Field(None, description="Enable tutorial mode")


class VolumeControlRequest(BaseModel):
    """Request model for volume control."""
    
    action: str = Field(..., description="Volume action: 'up', 'down', 'mute', 'max', or specific level")


class HelpRequest(BaseModel):
    """Request model for help system."""
    
    topic: Optional[str] = Field(None, description="Specific help topic")
    language: Optional[str] = Field(None, description="Language for help text")


class AccessibilityResponse(BaseModel):
    """Response model for accessibility operations."""
    
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


@router.get("/settings", response_model=AccessibilityResponse)
async def get_accessibility_settings():
    """
    Get current accessibility settings.
    
    Returns:
        Current accessibility settings and status
    """
    try:
        manager = get_accessibility_manager()
        report = manager.get_accessibility_report()
        
        return AccessibilityResponse(
            success=True,
            message="Accessibility settings retrieved successfully",
            data=report
        )
        
    except Exception as e:
        logger.error("Failed to get accessibility settings", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.put("/settings", response_model=AccessibilityResponse)
async def update_accessibility_settings(settings: AccessibilitySettingsRequest):
    """
    Update accessibility settings.
    
    Args:
        settings: New accessibility settings
        
    Returns:
        Updated settings confirmation
    """
    try:
        manager = get_accessibility_manager()
        
        # Convert request to settings dictionary
        settings_dict = {}
        
        if settings.volume_level is not None:
            settings_dict["volume_level"] = VolumeLevel(settings.volume_level)
        
        if settings.speech_rate is not None:
            # Find closest speech rate
            rates = [SpeechRate.VERY_SLOW, SpeechRate.SLOW, SpeechRate.NORMAL, SpeechRate.FAST, SpeechRate.VERY_FAST]
            closest_rate = min(rates, key=lambda x: abs(x.value - settings.speech_rate))
            settings_dict["speech_rate"] = closest_rate
        
        if settings.listening_timeout is not None:
            settings_dict["listening_timeout"] = settings.listening_timeout
        
        if settings.max_recognition_attempts is not None:
            settings_dict["max_recognition_attempts"] = settings.max_recognition_attempts
        
        if settings.interaction_mode is not None:
            try:
                settings_dict["interaction_mode"] = InteractionMode(settings.interaction_mode)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid interaction mode: {settings.interaction_mode}")
        
        if settings.accessibility_mode is not None:
            try:
                settings_dict["accessibility_mode"] = AccessibilityMode(settings.accessibility_mode)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid accessibility mode: {settings.accessibility_mode}")
        
        if settings.preferred_language is not None:
            try:
                settings_dict["preferred_language"] = LanguageCode(settings.preferred_language)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid language code: {settings.preferred_language}")
        
        # Boolean settings
        for field in ["enable_visual_indicators", "enable_voice_guided_help", "enable_confirmation_prompts",
                     "enable_detailed_feedback", "enable_audio_descriptions", "enable_tutorial_mode"]:
            value = getattr(settings, field)
            if value is not None:
                settings_dict[field] = value
        
        # Update settings
        manager.update_settings(settings_dict)
        
        return AccessibilityResponse(
            success=True,
            message="Accessibility settings updated successfully",
            data={"updated_settings": list(settings_dict.keys())}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update accessibility settings", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.post("/volume", response_model=AccessibilityResponse)
async def control_volume(volume_request: VolumeControlRequest):
    """
    Control volume level.
    
    Args:
        volume_request: Volume control request
        
    Returns:
        New volume level
    """
    try:
        manager = get_accessibility_manager()
        new_level = manager.adjust_volume(volume_request.action)
        
        return AccessibilityResponse(
            success=True,
            message=f"Volume adjusted to level {new_level.value}",
            data={
                "volume_level": new_level.value,
                "volume_name": new_level.name,
                "action": volume_request.action
            }
        )
        
    except Exception as e:
        logger.error("Failed to control volume", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.post("/mode", response_model=AccessibilityResponse)
async def switch_interaction_mode(mode: str = Body(..., embed=True)):
    """
    Switch interaction mode.
    
    Args:
        mode: New interaction mode
        
    Returns:
        Mode switch confirmation
    """
    try:
        manager = get_accessibility_manager()
        new_mode = manager.switch_interaction_mode(mode)
        
        return AccessibilityResponse(
            success=True,
            message=f"Switched to {new_mode.value.replace('_', ' ')} mode",
            data={
                "interaction_mode": new_mode.value,
                "mode_name": new_mode.name
            }
        )
        
    except Exception as e:
        logger.error("Failed to switch interaction mode", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.get("/help", response_model=AccessibilityResponse)
async def get_help(
    topic: Optional[str] = Query(None, description="Help topic"),
    language: Optional[str] = Query(None, description="Language code")
):
    """
    Get voice-guided help.
    
    Args:
        topic: Specific help topic
        language: Language for help text
        
    Returns:
        Help text and available topics
    """
    try:
        manager = get_accessibility_manager()
        
        # Update language if provided
        if language:
            try:
                lang_code = LanguageCode(language)
                manager.voice_help.language = lang_code
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid language code: {language}")
        
        help_text = manager.provide_help(topic)
        
        # Get available help topics
        available_topics = list(manager.voice_help.help_topics.get(
            manager.voice_help.language.value,
            manager.voice_help.help_topics.get(LanguageCode.ENGLISH_INDIA.value, {})
        ).keys())
        
        return AccessibilityResponse(
            success=True,
            message="Help provided successfully",
            data={
                "help_text": help_text,
                "topic": topic,
                "language": manager.voice_help.language.value,
                "available_topics": available_topics
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to provide help", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.post("/tutorial/start", response_model=AccessibilityResponse)
async def start_tutorial():
    """
    Start the voice-guided tutorial.
    
    Returns:
        First tutorial step
    """
    try:
        manager = get_accessibility_manager()
        first_step = manager.start_tutorial()
        
        return AccessibilityResponse(
            success=True,
            message="Tutorial started successfully",
            data={
                "tutorial_text": first_step,
                "step": 0,
                "total_steps": manager.voice_help.get_total_tutorial_steps()
            }
        )
        
    except Exception as e:
        logger.error("Failed to start tutorial", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.post("/tutorial/next", response_model=AccessibilityResponse)
async def next_tutorial_step():
    """
    Get the next tutorial step.
    
    Returns:
        Next tutorial step or completion message
    """
    try:
        manager = get_accessibility_manager()
        next_step = manager.get_next_tutorial_step()
        
        return AccessibilityResponse(
            success=True,
            message="Next tutorial step retrieved",
            data={
                "tutorial_text": next_step,
                "step": manager.current_tutorial_step,
                "total_steps": manager.voice_help.get_total_tutorial_steps(),
                "completed": not manager.settings.enable_tutorial_mode
            }
        )
        
    except Exception as e:
        logger.error("Failed to get next tutorial step", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.get("/status", response_model=AccessibilityResponse)
async def get_accessibility_status():
    """
    Get current accessibility status.
    
    Returns:
        Current system status with accessibility information
    """
    try:
        manager = get_accessibility_manager()
        status_message = manager.get_status_message()
        
        return AccessibilityResponse(
            success=True,
            message="Accessibility status retrieved",
            data={
                "status_message": status_message,
                "is_listening": manager.is_listening,
                "is_speaking": manager.is_speaking,
                "is_processing": manager.is_processing,
                "current_mode": manager.current_mode.value,
                "volume_level": manager.settings.volume_level.value,
                "tutorial_active": manager.settings.enable_tutorial_mode
            }
        )
        
    except Exception as e:
        logger.error("Failed to get accessibility status", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.get("/indicators/{state}", response_model=AccessibilityResponse)
async def get_visual_indicator(state: str):
    """
    Get visual indicator for a specific state.
    
    Args:
        state: System state (listening, processing, speaking, error, etc.)
        
    Returns:
        Visual indicator configuration
    """
    try:
        manager = get_accessibility_manager()
        indicator = manager.visual_indicators.get_indicator(state)
        
        return AccessibilityResponse(
            success=True,
            message=f"Visual indicator for '{state}' retrieved",
            data={
                "state": state,
                "indicator": indicator,
                "enabled": manager.settings.enable_visual_indicators
            }
        )
        
    except Exception as e:
        logger.error("Failed to get visual indicator", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.get("/presets", response_model=AccessibilityResponse)
async def get_accessibility_presets():
    """
    Get predefined accessibility presets for different user needs.
    
    Returns:
        Available accessibility presets
    """
    try:
        presets = {
            "hearing_impaired": {
                "name": "Hearing Impaired",
                "description": "Enhanced visual indicators and text-based interaction",
                "settings": {
                    "enable_visual_indicators": True,
                    "interaction_mode": "text_only",
                    "enable_detailed_feedback": True,
                    "volume_level": 0,  # Muted
                }
            },
            "vision_impaired": {
                "name": "Vision Impaired",
                "description": "Enhanced audio feedback and voice guidance",
                "settings": {
                    "enable_voice_guided_help": True,
                    "enable_audio_descriptions": True,
                    "enable_detailed_feedback": True,
                    "speech_rate": 0.8,  # Slightly slower
                    "volume_level": 5,  # Higher volume
                }
            },
            "motor_impaired": {
                "name": "Motor Impaired",
                "description": "Extended timeouts and simplified interactions",
                "settings": {
                    "listening_timeout": 60.0,  # Longer timeout
                    "max_recognition_attempts": 5,  # More attempts
                    "enable_confirmation_prompts": True,
                    "speech_rate": 0.7,  # Slower speech
                }
            },
            "cognitive_support": {
                "name": "Cognitive Support",
                "description": "Step-by-step guidance and simplified language",
                "settings": {
                    "enable_step_by_step_guidance": True,
                    "enable_tutorial_mode": True,
                    "speech_rate": 0.8,  # Slower speech
                    "enable_confirmation_prompts": True,
                    "use_formal_language": False,  # Simpler language
                }
            },
            "elderly_friendly": {
                "name": "Elderly Friendly",
                "description": "Larger text, slower speech, and patient interaction",
                "settings": {
                    "speech_rate": 0.7,  # Slower speech
                    "volume_level": 4,  # Higher volume
                    "listening_timeout": 45.0,  # Longer timeout
                    "large_text_mode": True,
                    "enable_confirmation_prompts": True,
                    "max_recognition_attempts": 4,
                }
            }
        }
        
        return AccessibilityResponse(
            success=True,
            message="Accessibility presets retrieved",
            data={"presets": presets}
        )
        
    except Exception as e:
        logger.error("Failed to get accessibility presets", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)


@router.post("/presets/{preset_name}", response_model=AccessibilityResponse)
async def apply_accessibility_preset(preset_name: str):
    """
    Apply a predefined accessibility preset.
    
    Args:
        preset_name: Name of the preset to apply
        
    Returns:
        Preset application confirmation
    """
    try:
        # Get presets (reuse the logic from get_accessibility_presets)
        presets_response = await get_accessibility_presets()
        presets = presets_response.data["presets"]
        
        if preset_name not in presets:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
        
        preset = presets[preset_name]
        manager = get_accessibility_manager()
        
        # Convert preset settings to proper types
        settings_dict = {}
        for key, value in preset["settings"].items():
            if key == "volume_level":
                settings_dict[key] = VolumeLevel(value)
            elif key == "interaction_mode":
                settings_dict[key] = InteractionMode(value)
            elif key == "speech_rate":
                # Find closest speech rate
                rates = [SpeechRate.VERY_SLOW, SpeechRate.SLOW, SpeechRate.NORMAL, SpeechRate.FAST, SpeechRate.VERY_FAST]
                closest_rate = min(rates, key=lambda x: abs(x.value - value))
                settings_dict[key] = closest_rate
            else:
                settings_dict[key] = value
        
        # Apply settings
        manager.update_settings(settings_dict)
        
        return AccessibilityResponse(
            success=True,
            message=f"Applied '{preset['name']}' accessibility preset",
            data={
                "preset_name": preset_name,
                "preset_description": preset["description"],
                "applied_settings": list(settings_dict.keys())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to apply accessibility preset", exc_info=e)
        error = handle_error(e)
        raise HTTPException(status_code=500, detail=error.message)