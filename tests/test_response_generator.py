"""
Tests for Response Generator functionality.

This module tests the comprehensive response generation system including
multilingual output, Indian localization, grammatical correctness,
and natural code-switching capabilities.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any

from bharatvoice.core.models import (
    LanguageCode,
    Intent,
    Entity,
    Response,
    ConversationState,
    RegionalContextData,
    LocationData,
    WeatherData,
    CulturalEvent,
    UserProfile
)
from bharatvoice.services.response_generation.response_generator import (
    MultilingualResponseGenerator,
    IndianLocalizationEngine,
    CodeSwitchingEngine,
    ResponseStyle,
    LocalizationFormat,
    LocalizedValue,
    CodeSwitchingPoint
)
from bharatvoice.services.response_generation.nlu_service import IntentCategory, EntityType


class TestIndianLocalizationEngine:
    """Test Indian localization functionality."""
    
    @pytest.fixture
    def localization_engine(self):
        return IndianLocalizationEngine()
    
    @pytest.mark.asyncio
    async def test_currency_localization_usd_to_inr(self, localization_engine):
        """Test USD to INR currency localization."""
        result = await localization_engine.localize_currency(100.0, "USD")
        
        assert isinstance(result, LocalizedValue)
        assert result.format_type == LocalizationFormat.CURRENCY
        assert result.locale == "en-IN"
        assert "₹" in result.localized_value
        assert result.explanation is not None
        assert "Converted from USD to INR" in result.explanation
    
    @pytest.mark.asyncio
    async def test_currency_localization_inr_no_conversion(self, localization_engine):
        """Test INR currency localization (no conversion needed)."""
        result = await localization_engine.lo