"""
Comprehensive deployment and integration tests for BharatVoice Assistant.

This module provides end-to-end testing for deployment scenarios including:
- Complete voice interaction workflows
- Multilingual conversation flows
- Indian service integrations
- Offline/online mode transitions
- Performance under realistic Indian network conditions
"""

import asyncio
import json
import time
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import structlog

from bharatvoice.main import create_app
from bharatvoice.core.models import LanguageCode, AudioBuffer, ConversationState
from bharatvoice.config import get_settings


logger = structlog.get_logger(__name__)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndVoiceInteraction:
    """
    Comprehensive end-to-end voice interaction testing.
    
    Tests complete voice workflows from audio input to response synthesis,
    including multilingual processing and cultural context understanding.
    """
    
    @pytest.fixture
    def voice_interaction_client(self):
        """Create client configured for voice interaction testing."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def sample_hindi_audio(self) -> AudioBuffer:
        """Sample Hindi audio buffer for testing."""
        # Simulate 2-second Hindi audio: "नमस्ते, मुझे ट्रेन की जानकारी चाहिए"
        return AudioBuffer(
            data=[0.1 * i for i in range(32000)],  # 2 seconds at 16kHz
            sample_rate=16000,
            channels=1,
            duration=2.0,
            metadata={
                "language": "hi",
                "text": "नमस्ते, मुझे ट्रेन की जानकारी चाहिए",
                "intent": "train_inquiry"
            }
        )
    
    @pytest.fixture
    def sample_english_audio(self) -> AudioBuffer:
        """Sample English audio buffer for testing."""
        # Simulate English audio: "Hello, I need train information"
        return AudioBuffer(
            data=[0.1 * i for i in range(24000)],  # 1.5 seconds at 16kHz
            sample_rate=16000,
            channels=1,
            duration=1.5,
            metadata={
                "language": "en-IN",
                "text": "Hello, I need train information",
                "intent": "train_inquiry"
            }
        )
    
    @pytest.fixture
    def sample_code_switched_audio(self) -> AudioBuffer:
        """Sample code-switched Hindi-English audio."""
        # Simulate: "Hello, मुझे Delhi से Mumbai का train schedule चाहिए"
        return AudioBuffer(
            data=[0.1 * i for i in range(40000)],  # 2.5 seconds at 16kHz
            sample_rate=16000,
            channels=1,
            duration=2.5,
            metadata={
                "language": "hi-en",
                "text": "Hello, मुझे Delhi से Mumbai का train schedule चाहिए",
                "intent": "train_schedule_inquiry",
                "code_switching": True
            }
        )
    
    async def test_complete_hindi_voice_workflow(self, voice_interaction_client, sample_hindi_audio):
        """
        Test complete voice workflow in Hindi.
        
        Workflow:
        1. Audio input processing
        2. Speech recognition
        3. Intent understanding
        4. Service integration
        5. Response generation
        6. Speech synthesis
        """
        # Mock external services for controlled testing
        with patch('bharatvoice.services.voice_processing.service.VoiceProcessingService') as mock_voice, \
             patch('bharatvoice.services.language_engine.service.LanguageEngineService') as mock_lang, \
             patch('bharatvoice.services.response_generation.response_generator.ResponseGenerator') as mock_response, \
             patch('bharatvoice.services.external_integrations.service_manager.ServiceManager') as mock_external:
            
            # Configure mocks for Hindi workflow
            mock_voice_instance = AsyncMock()
            mock_voice.return_value = mock_voice_instance
            mock_voice_instance.process_audio_stream.return_value = {
                "processed": True,
                "quality_score": 0.95,
                "noise_level": 0.1
            }
            mock_voice_instance.synthesize_speech.return_value = AudioBuffer(
                data=[0.1] * 16000,
                sample_rate=16000,
                channels=1,
                duration=1.0
            )
            
            mock_lang_instance = AsyncMock()
            mock_lang.return_value = mock_lang_instance
            mock_lang_instance.recognize_speech.return_value = {
                "text": "नमस्ते, मुझे ट्रेन की जानकारी चाहिए",
                "language": LanguageCode.HINDI,
                "confidence": 0.92
            }
            
            mock_response_instance = AsyncMock()
            mock_response.return_value = mock_response_instance
            mock_response_instance.process_query.return_value = {
                "intent": "train_inquiry",
                "entities": [
                    {"type": "greeting", "value": "नमस्ते"},
                    {"type": "service", "value": "ट्रेन"}
                ],
                "confidence": 0.89
            }
            mock_response_instance.generate_response.return_value = {
                "text": "नमस्कार! मैं आपकी ट्रेन की जानकारी में मदद कर सकता हूं। कृपया बताएं कि आप कहां से कहां जाना चाहते हैं?",
                "language": LanguageCode.HINDI,
                "cultural_context": "formal_respectful"
            }
            
            # Step 1: Submit audio for processing
            audio_data = {
                "audio_data": sample_hindi_audio.data[:1000],  # Truncate for API
                "sample_rate": sample_hindi_audio.sample_rate,
                "language_hint": "hi"
            }
            
            response = voice_interaction_client.post("/voice/process", json=audio_data)
            assert response.status_code == 200
            
            process_result = response.json()
            assert "session_id" in process_result
            assert "status" in process_result
            assert process_result["status"] == "processing"
            
            session_id = process_result["session_id"]
            
            # Step 2: Check processing status and get results
            await asyncio.sleep(0.1)  # Simulate processing time
            
            status_response = voice_interaction_client.get(f"/voice/status/{session_id}")
            assert status_response.status_code == 200
            
            status_result = status_response.json()
            assert status_result["status"] == "completed"
            assert "recognition_result" in status_result
            assert "response" in status_result
            
            # Verify recognition results
            recognition = status_result["recognition_result"]
            assert recognition["text"] == "नमस्ते, मुझे ट्रेन की जानकारी चाहिए"
            assert recognition["language"] == "hi"
            assert recognition["confidence"] > 0.8
            
            # Verify response generation
            response_data = status_result["response"]
            assert "नमस्कार" in response_data["text"]
            assert "ट्रेन" in response_data["text"]
            assert response_data["language"] == "hi"
            
            # Step 3: Get synthesized audio response
            audio_response = voice_interaction_client.get(f"/voice/audio/{session_id}")
            assert audio_response.status_code == 200
            assert audio_response.headers["content-type"] == "audio/wav"
    
    async def test_multilingual_conversation_flow(self, voice_interaction_client):
        """
        Test multilingual conversation flow with language switching.
        
        Simulates a conversation that switches between Hindi and English.
        """
        conversation_steps = [
            {
                "input": "Hello, मुझे help चाहिए",
                "expected_language": "hi-en",
                "expected_response_lang": "hi-en"
            },
            {
                "input": "I want to book a train ticket",
                "expected_language": "en-IN",
                "expected_response_lang": "en-IN"
            },
            {
                "input": "Delhi से Mumbai जाना है",
                "expected_language": "hi",
                "expected_response_lang": "hi"
            }
        ]
        
        session_id = None
        
        for step_num, step in enumerate(conversation_steps):
            with patch('bharatvoice.services.language_engine.service.LanguageEngineService') as mock_lang, \
                 patch('bharatvoice.services.response_generation.response_generator.ResponseGenerator') as mock_response:
                
                # Configure mocks for this step
                mock_lang_instance = AsyncMock()
                mock_lang.return_value = mock_lang_instance
                mock_lang_instance.recognize_speech.return_value = {
                    "text": step["input"],
                    "language": step["expected_language"],
                    "confidence": 0.88
                }
                mock_lang_instance.detect_code_switching.return_value = [
                    {"start": 0, "end": 5, "language": "en"},
                    {"start": 6, "end": 15, "language": "hi"}
                ] if step["expected_language"] == "hi-en" else []
                
                mock_response_instance = AsyncMock()
                mock_response.return_value = mock_response_instance
                mock_response_instance.generate_response.return_value = {
                    "text": f"Response for step {step_num + 1}",
                    "language": step["expected_response_lang"],
                    "maintains_language_context": True
                }
                
                # Submit conversation step
                conversation_data = {
                    "text": step["input"],
                    "session_id": session_id,
                    "maintain_context": True
                }
                
                response = voice_interaction_client.post("/voice/conversation", json=conversation_data)
                assert response.status_code == 200
                
                result = response.json()
                if session_id is None:
                    session_id = result["session_id"]
                
                assert result["recognized_language"] == step["expected_language"]
                assert result["response_language"] == step["expected_response_lang"]
                assert "conversation_context" in result
    
    async def test_cultural_context_understanding(self, voice_interaction_client):
        """
        Test cultural context understanding in voice interactions.
        
        Tests recognition and appropriate response to Indian cultural contexts.
        """
        cultural_test_cases = [
            {
                "input": "आज दिवाली है, शुभकामनाएं",
                "expected_context": "festival_greeting",
                "expected_response_type": "festival_acknowledgment"
            },
            {
                "input": "गुरुजी, मुझे सिखाइए",
                "expected_context": "respectful_learning",
                "expected_response_type": "respectful_teaching"
            },
            {
                "input": "भाई साहब, ट्रेन कब आएगी?",
                "expected_context": "informal_respectful",
                "expected_response_type": "helpful_informative"
            }
        ]
        
        for test_case in cultural_test_cases:
            with patch('bharatvoice.services.response_generation.nlu_service.NLUService') as mock_nlu, \
                 patch('bharatvoice.services.response_generation.response_generator.ResponseGenerator') as mock_response:
                
                mock_nlu_instance = AsyncMock()
                mock_nlu.return_value = mock_nlu_instance
                mock_nlu_instance.analyze_cultural_context.return_value = {
                    "context_type": test_case["expected_context"],
                    "formality_level": "respectful",
                    "cultural_markers": ["festival", "respect", "relationship"]
                }
                
                mock_response_instance = AsyncMock()
                mock_response.return_value = mock_response_instance
                mock_response_instance.format_cultural_response.return_value = {
                    "text": "Culturally appropriate response",
                    "response_type": test_case["expected_response_type"],
                    "maintains_cultural_context": True
                }
                
                cultural_data = {
                    "text": test_case["input"],
                    "analyze_cultural_context": True
                }
                
                response = voice_interaction_client.post("/voice/cultural-analysis", json=cultural_data)
                assert response.status_code == 200
                
                result = response.json()
                assert result["cultural_context"]["context_type"] == test_case["expected_context"]
                assert result["response"]["response_type"] == test_case["expected_response_type"]
                assert result["response"]["maintains_cultural_context"] is True


@pytest.mark.integration
@pytest.mark.slow
class TestIndianServiceIntegration:
    """
    Test integration with Indian-specific services and platforms.
    
    Validates connectivity, data processing, and response handling
    for Indian Railways, weather services, and government platforms.
    """
    
    @pytest.fixture
    def service_integration_client(self):
        """Create client for service integration testing."""
        app = create_app()
        return TestClient(app)
    
    async def test_indian_railways_integration(self, service_integration_client):
        """
        Test Indian Railways API integration.
        
        Tests train schedule queries, booking information,
        and natural language processing for railway queries.
        """
        with patch('bharatvoice.services.external_integrations.indian_railways_service.IndianRailwaysService') as mock_railways:
            mock_railways_instance = AsyncMock()
            mock_railways.return_value = mock_railways_instance
            
            # Mock successful railway service response
            mock_railways_instance.search_trains.return_value = {
                "trains": [
                    {
                        "train_number": "12951",
                        "train_name": "Mumbai Rajdhani Express",
                        "departure": "16:55",
                        "arrival": "08:35",
                        "duration": "15:40",
                        "availability": "Available"
                    }
                ],
                "status": "success"
            }
            
            # Test railway query
            railway_query = {
                "query": "Delhi से Mumbai की train",
                "from_station": "NDLS",
                "to_station": "CSTM",
                "date": "2024-01-15"
            }
            
            response = service_integration_client.post("/external/railways/search", json=railway_query)
            assert response.status_code == 200
            
            result = response.json()
            assert "trains" in result
            assert len(result["trains"]) > 0
            assert result["trains"][0]["train_name"] == "Mumbai Rajdhani Express"
            
            # Test error handling
            mock_railways_instance.search_trains.side_effect = Exception("API Error")
            
            error_response = service_integration_client.post("/external/railways/search", json=railway_query)
            assert error_response.status_code == 503  # Service unavailable
            
            error_result = error_response.json()
            assert "error" in error_result
            assert "fallback" in error_result
    
    async def test_weather_service_integration(self, service_integration_client):
        """
        Test weather service integration with Indian weather patterns.
        
        Tests monsoon information, temperature in Celsius,
        and location-based weather queries.
        """
        with patch('bharatvoice.services.external_integrations.weather_service.WeatherService') as mock_weather:
            mock_weather_instance = AsyncMock()
            mock_weather.return_value = mock_weather_instance
            
            # Mock weather service response with Indian context
            mock_weather_instance.get_weather.return_value = {
                "location": "Mumbai",
                "temperature": 28,
                "unit": "celsius",
                "humidity": 85,
                "monsoon_status": "active",
                "rainfall_mm": 15.2,
                "weather_description": "Heavy monsoon showers",
                "air_quality_index": 156,
                "uv_index": 3
            }
            
            weather_query = {
                "location": "Mumbai",
                "include_monsoon": True,
                "language": "hi"
            }
            
            response = service_integration_client.post("/external/weather/current", json=weather_query)
            assert response.status_code == 200
            
            result = response.json()
            assert result["temperature"] == 28
            assert result["unit"] == "celsius"
            assert result["monsoon_status"] == "active"
            assert "rainfall_mm" in result
            assert "air_quality_index" in result
    
    async def test_digital_india_integration(self, service_integration_client):
        """
        Test Digital India platform integration.
        
        Tests government service discovery and information retrieval.
        """
        with patch('bharatvoice.services.external_integrations.digital_india_service.DigitalIndiaService') as mock_digital:
            mock_digital_instance = AsyncMock()
            mock_digital.return_value = mock_digital_instance
            
            # Mock Digital India service response
            mock_digital_instance.search_services.return_value = {
                "services": [
                    {
                        "service_name": "Passport Application",
                        "department": "Ministry of External Affairs",
                        "description": "Apply for Indian passport online",
                        "url": "https://passportindia.gov.in",
                        "documents_required": ["Aadhaar", "Birth Certificate"],
                        "processing_time": "30 days"
                    }
                ],
                "total_results": 1
            }
            
            digital_query = {
                "service_type": "passport",
                "language": "hi"
            }
            
            response = service_integration_client.post("/external/digital-india/search", json=digital_query)
            assert response.status_code == 200
            
            result = response.json()
            assert "services" in result
            assert len(result["services"]) > 0
            assert "Passport Application" in result["services"][0]["service_name"]
    
    async def test_service_integration_fallbacks(self, service_integration_client):
        """
        Test fallback mechanisms when external services fail.
        
        Ensures graceful degradation and user-friendly error messages.
        """
        # Test multiple service failures
        with patch('bharatvoice.services.external_integrations.service_manager.ServiceManager') as mock_manager:
            mock_manager_instance = AsyncMock()
            mock_manager.return_value = mock_manager_instance
            
            # Simulate service failures
            mock_manager_instance.get_service_status.return_value = {
                "railways": "unavailable",
                "weather": "degraded",
                "digital_india": "unavailable"
            }
            
            mock_manager_instance.get_fallback_response.return_value = {
                "message": "कुछ सेवाएं अभी उपलब्ध नहीं हैं। कृपया बाद में पुनः प्रयास करें।",
                "available_services": ["basic_info"],
                "retry_after": 300
            }
            
            fallback_query = {
                "query": "train information",
                "language": "hi"
            }
            
            response = service_integration_client.post("/external/query", json=fallback_query)
            assert response.status_code == 200
            
            result = response.json()
            assert "fallback" in result
            assert "available_services" in result
            assert result["retry_after"] == 300


@pytest.mark.integration
@pytest.mark.slow
class TestOfflineOnlineTransitions:
    """
    Test offline/online mode transitions and data synchronization.
    
    Validates seamless switching between offline and online modes,
    data sync integrity, and user experience continuity.
    """
    
    @pytest.fixture
    def offline_client(self):
        """Create client configured for offline testing."""
        app = create_app()
        return TestClient(app)
    
    async def test_offline_mode_activation(self, offline_client):
        """
        Test automatic offline mode activation when network is unavailable.
        """
        with patch('bharatvoice.services.offline_sync.network_monitor.NetworkMonitor') as mock_network:
            mock_network_instance = AsyncMock()
            mock_network.return_value = mock_network_instance
            
            # Simulate network unavailability
            mock_network_instance.is_online.return_value = False
            mock_network_instance.get_connection_quality.return_value = {
                "status": "offline",
                "last_online": "2024-01-15T10:30:00Z",
                "offline_duration": 300
            }
            
            # Test offline mode detection
            response = offline_client.get("/voice/network-status")
            assert response.status_code == 200
            
            result = response.json()
            assert result["mode"] == "offline"
            assert result["offline_capabilities"]["basic_recognition"] is True
            assert result["offline_capabilities"]["cached_responses"] is True
    
    async def test_offline_voice_processing(self, offline_client):
        """
        Test voice processing capabilities in offline mode.
        """
        with patch('bharatvoice.services.offline_sync.offline_voice_processor.OfflineVoiceProcessor') as mock_offline:
            mock_offline_instance = AsyncMock()
            mock_offline.return_value = mock_offline_instance
            
            # Mock offline processing capabilities
            mock_offline_instance.process_offline_query.return_value = {
                "text": "ट्रेन की जानकारी",
                "intent": "train_inquiry",
                "response": "ऑफलाइन मोड में बेसिक जानकारी उपलब्ध है।",
                "confidence": 0.75,
                "source": "offline_cache"
            }
            
            offline_query = {
                "audio_data": [0.1] * 1000,
                "sample_rate": 16000,
                "offline_mode": True
            }
            
            response = offline_client.post("/voice/process-offline", json=offline_query)
            assert response.status_code == 200
            
            result = response.json()
            assert result["source"] == "offline_cache"
            assert result["confidence"] > 0.7
            assert "ऑफलाइन" in result["response"]
    
    async def test_online_mode_restoration(self, offline_client):
        """
        Test restoration to online mode and data synchronization.
        """
        with patch('bharatvoice.services.offline_sync.data_sync_manager.DataSyncManager') as mock_sync, \
             patch('bharatvoice.services.offline_sync.network_monitor.NetworkMonitor') as mock_network:
            
            mock_sync_instance = AsyncMock()
            mock_sync.return_value = mock_sync_instance
            
            mock_network_instance = AsyncMock()
            mock_network.return_value = mock_network_instance
            
            # Simulate network restoration
            mock_network_instance.is_online.return_value = True
            mock_network_instance.get_connection_quality.return_value = {
                "status": "online",
                "speed_mbps": 2.5,
                "latency_ms": 150,
                "quality": "good"
            }
            
            # Mock data synchronization
            mock_sync_instance.sync_offline_data.return_value = {
                "synced_queries": 5,
                "synced_responses": 3,
                "conflicts_resolved": 1,
                "sync_duration": 2.3,
                "status": "completed"
            }
            
            # Test online restoration
            response = offline_client.post("/voice/restore-online")
            assert response.status_code == 200
            
            result = response.json()
            assert result["mode"] == "online"
            assert result["sync_result"]["status"] == "completed"
            assert result["sync_result"]["synced_queries"] == 5
    
    async def test_data_sync_conflict_resolution(self, offline_client):
        """
        Test conflict resolution during data synchronization.
        """
        with patch('bharatvoice.services.offline_sync.data_sync_manager.DataSyncManager') as mock_sync:
            mock_sync_instance = AsyncMock()
            mock_sync.return_value = mock_sync_instance
            
            # Mock sync conflicts
            mock_sync_instance.resolve_sync_conflicts.return_value = {
                "conflicts": [
                    {
                        "type": "user_preference",
                        "offline_value": "hi",
                        "online_value": "en-IN",
                        "resolution": "merge",
                        "final_value": ["hi", "en-IN"]
                    }
                ],
                "resolution_strategy": "user_preference_priority",
                "conflicts_resolved": 1
            }
            
            sync_request = {
                "conflict_resolution": "user_preference_priority",
                "preserve_offline_changes": True
            }
            
            response = offline_client.post("/voice/sync-conflicts", json=sync_request)
            assert response.status_code == 200
            
            result = response.json()
            assert result["conflicts_resolved"] == 1
            assert result["resolution_strategy"] == "user_preference_priority"


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceUnderIndianNetworkConditions:
    """
    Test performance under realistic Indian network conditions.
    
    Simulates various network scenarios common in India including
    slow connections, intermittent connectivity, and high latency.
    """
    
    @pytest.fixture
    def performance_client(self):
        """Create client for performance testing."""
        app = create_app()
        return TestClient(app)
    
    async def test_slow_network_performance(self, performance_client):
        """
        Test performance under slow network conditions (2G/3G).
        
        Simulates network speeds common in rural India.
        """
        with patch('bharatvoice.utils.performance_monitor.PerformanceMonitor') as mock_perf:
            mock_perf_instance = AsyncMock()
            mock_perf.return_value = mock_perf_instance
            
            # Simulate slow network conditions
            mock_perf_instance.simulate_network_conditions.return_value = {
                "bandwidth_kbps": 64,  # 2G speed
                "latency_ms": 800,
                "packet_loss": 0.05,
                "jitter_ms": 200
            }
            
            # Test voice processing under slow conditions
            start_time = time.time()
            
            slow_network_query = {
                "audio_data": [0.1] * 500,  # Smaller payload for slow network
                "sample_rate": 8000,  # Lower quality for bandwidth
                "optimize_for_slow_network": True
            }
            
            response = performance_client.post("/voice/process", json=slow_network_query)
            
            processing_time = time.time() - start_time
            
            assert response.status_code == 200
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            result = response.json()
            assert "optimized_for_slow_network" in result
            assert result["optimized_for_slow_network"] is True
    
    async def test_intermittent_connectivity(self, performance_client):
        """
        Test handling of intermittent connectivity.
        
        Simulates connection drops and reconnections common in mobile networks.
        """
        connection_states = [
            {"online": True, "duration": 2.0},
            {"online": False, "duration": 1.0},
            {"online": True, "duration": 3.0},
            {"online": False, "duration": 0.5},
            {"online": True, "duration": 2.0}
        ]
        
        with patch('bharatvoice.services.offline_sync.network_monitor.NetworkMonitor') as mock_network:
            mock_network_instance = AsyncMock()
            mock_network.return_value = mock_network_instance
            
            for state in connection_states:
                mock_network_instance.is_online.return_value = state["online"]
                
                # Test query during this connection state
                query = {
                    "text": "मुझे मदद चाहिए",
                    "handle_intermittent_connection": True
                }
                
                response = performance_client.post("/voice/conversation", json=query)
                
                if state["online"]:
                    assert response.status_code == 200
                    result = response.json()
                    assert "response" in result
                else:
                    # Should either succeed with cached response or gracefully handle offline
                    assert response.status_code in [200, 202]  # 202 for queued processing
                
                await asyncio.sleep(0.1)  # Simulate state duration
    
    async def test_high_latency_performance(self, performance_client):
        """
        Test performance under high latency conditions.
        
        Simulates satellite or congested network conditions.
        """
        with patch('bharatvoice.utils.performance_monitor.PerformanceMonitor') as mock_perf:
            mock_perf_instance = AsyncMock()
            mock_perf.return_value = mock_perf_instance
            
            # Simulate high latency
            mock_perf_instance.get_network_metrics.return_value = {
                "latency_ms": 2000,  # 2 second latency
                "bandwidth_mbps": 1.0,
                "stability": "poor"
            }
            
            # Test with timeout handling
            high_latency_query = {
                "text": "weather information",
                "timeout_ms": 5000,
                "enable_caching": True
            }
            
            start_time = time.time()
            response = performance_client.post("/voice/query", json=high_latency_query)
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 6.0  # Should respect timeout
            
            result = response.json()
            assert "cached_response" in result or "live_response" in result
    
    async def test_concurrent_user_load(self, performance_client):
        """
        Test performance under concurrent user load.
        
        Simulates multiple users accessing the system simultaneously.
        """
        async def simulate_user_request(user_id: int) -> Dict[str, Any]:
            """Simulate a single user request."""
            user_query = {
                "text": f"User {user_id} query",
                "user_id": f"test_user_{user_id}",
                "session_id": f"session_{user_id}"
            }
            
            start_time = time.time()
            response = performance_client.post("/voice/conversation", json=user_query)
            response_time = time.time() - start_time
            
            return {
                "user_id": user_id,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            }
        
        # Simulate 10 concurrent users
        concurrent_users = 10
        tasks = [simulate_user_request(i) for i in range(concurrent_users)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        average_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        
        assert len(successful_requests) >= concurrent_users * 0.8  # 80% success rate
        assert average_response_time < 5.0  # Average response under 5 seconds
    
    async def test_bandwidth_optimization(self, performance_client):
        """
        Test bandwidth optimization features.
        
        Validates data compression and efficient payload handling.
        """
        # Test with different payload sizes
        payload_sizes = [
            {"size": "small", "audio_samples": 1000},
            {"size": "medium", "audio_samples": 5000},
            {"size": "large", "audio_samples": 10000}
        ]
        
        for payload in payload_sizes:
            optimization_query = {
                "audio_data": [0.1] * payload["audio_samples"],
                "sample_rate": 16000,
                "enable_compression": True,
                "bandwidth_optimization": True
            }
            
            response = performance_client.post("/voice/process-optimized", json=optimization_query)
            assert response.status_code == 200
            
            result = response.json()
            assert "compression_ratio" in result
            assert "optimized_payload_size" in result
            assert result["compression_ratio"] > 0.5  # At least 50% compression


@pytest.mark.integration
@pytest.mark.slow
class TestDeploymentHealthChecks:
    """
    Test deployment health checks and system readiness.
    
    Validates that all system components are properly initialized
    and functioning in a deployment environment.
    """
    
    @pytest.fixture
    def deployment_client(self):
        """Create client for deployment testing."""
        app = create_app()
        return TestClient(app)
    
    def test_system_startup_health(self, deployment_client):
        """Test system health immediately after startup."""
        response = deployment_client.get("/health/")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "services" in health_data
        
        # Check critical services
        critical_services = ["database", "redis", "voice_processing", "language_engine"]
        for service in critical_services:
            assert service in health_data["services"]
    
    def test_readiness_probe(self, deployment_client):
        """Test Kubernetes readiness probe endpoint."""
        response = deployment_client.get("/health/ready")
        assert response.status_code == 200
        
        readiness_data = response.json()
        assert readiness_data["status"] == "ready"
    
    def test_liveness_probe(self, deployment_client):
        """Test Kubernetes liveness probe endpoint."""
        response = deployment_client.get("/health/live")
        assert response.status_code == 200
        
        liveness_data = response.json()
        assert liveness_data["status"] == "alive"
    
    def test_metrics_endpoint(self, deployment_client):
        """Test Prometheus metrics endpoint."""
        response = deployment_client.get("/health/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        
        # Check for essential metrics
        essential_metrics = [
            "requests_total",
            "request_duration_seconds",
            "active_sessions",
            "memory_usage_bytes"
        ]
        
        for metric in essential_metrics:
            assert metric in metrics_data
    
    def test_gateway_status(self, deployment_client):
        """Test gateway load balancing status."""
        response = deployment_client.get("/gateway/status")
        assert response.status_code == 200
        
        gateway_data = response.json()
        assert gateway_data["status"] == "operational"
        assert "load_balancer" in gateway_data
        assert "services" in gateway_data
    
    def test_service_discovery(self, deployment_client):
        """Test service discovery and routing."""
        response = deployment_client.get("/gateway/services")
        assert response.status_code == 200
        
        services_data = response.json()
        assert "services" in services_data
        assert services_data["total_services"] > 0


if __name__ == "__main__":
    """Run deployment integration tests."""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "integration",
        "--durations=10"
    ])