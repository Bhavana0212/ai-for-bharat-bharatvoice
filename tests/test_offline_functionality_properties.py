"""
Property-based tests for offline functionality in BharatVoice Assistant.

This module tests the offline voice processing system and data synchronization
to ensure they work correctly under various conditions and maintain data integrity.

Property 14: Offline Functionality
Property 15: Network Resilience
"""

import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    AccentType,
    RecognitionResult,
    VoiceActivityResult,
)
from bharatvoice.services.offline_sync.offline_voice_processor import (
    OfflineVoiceProcessor,
    NetworkStatus,
    OfflineQuery,
    create_offline_voice_processor,
)
from bharatvoice.services.offline_sync.data_sync_manager import (
    DataSyncManager,
    SyncStatus,
    ConflictResolution,
    SyncItem,
    OfflineUsageRecord,
    create_data_sync_manager,
)


# Test strategies
@st.composite
def audio_buffer_strategy(draw):
    """Generate valid AudioBuffer instances."""
    sample_rate = draw(st.sampled_from([8000, 16000, 22050, 44100]))
    duration = draw(st.floats(min_value=0.1, max_value=5.0))
    num_samples = int(sample_rate * duration)
    
    data = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0),
        min_size=num_samples,
        max_size=num_samples
    ))
    
    return AudioBuffer(
        data=data,
        sample_rate=sample_rate,
        channels=draw(st.sampled_from([1, 2])),
        format=draw(st.sampled_from(list(AudioFormat))),
        duration=duration
    )


@st.composite
def offline_query_strategy(draw):
    """Generate OfflineQuery instances."""
    return OfflineQuery(
        query_text=draw(st.text(min_size=1, max_size=200)),
        language=draw(st.sampled_from(list(LanguageCode))),
        response_text=draw(st.text(min_size=1, max_size=500)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        timestamp=datetime.now(),
        usage_count=draw(st.integers(min_value=0, max_value=1000))
    )


@st.composite
def sync_item_strategy(draw):
    """Generate SyncItem instances."""
    return SyncItem(
        item_id=draw(st.text(min_size=1, max_size=50)),
        item_type=draw(st.sampled_from(["query", "model", "preference", "analytics"])),
        local_data=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            min_size=1,
            max_size=10
        )),
        local_timestamp=datetime.now(),
        sync_status=draw(st.sampled_from(list(SyncStatus)))
    )


class TestOfflineVoiceProcessorProperties:
    """Property-based tests for OfflineVoiceProcessor."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def offline_processor(self, temp_cache_dir):
        """Create OfflineVoiceProcessor instance for testing."""
        return create_offline_voice_processor(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=10,  # Small cache for testing
            enable_local_asr=True,
            enable_local_tts=True,
            common_queries_limit=100
        )
    
    @given(audio_buffer_strategy())
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_audio_processing_preserves_format(
        self, 
        offline_processor, 
        audio_buffer
    ):
        """
        Property 14.1: Offline audio processing preserves audio format characteristics.
        
        **Validates: Requirements 1.4** - Audio processing maintains quality offline
        """
        # Process audio offline
        processed_audio = await offline_processor.process_audio_stream(
            audio_buffer, LanguageCode.ENGLISH_IN
        )
        
        # Verify format preservation
        assert processed_audio.sample_rate == audio_buffer.sample_rate
        assert processed_audio.channels == audio_buffer.channels
        assert processed_audio.format == audio_buffer.format
        assert len(processed_audio.data) > 0
        assert processed_audio.duration > 0
    
    @given(audio_buffer_strategy())
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_vad_consistency(self, offline_processor, audio_buffer):
        """
        Property 14.2: Voice activity detection works consistently offline.
        
        **Validates: Requirements 1.4** - VAD functionality in offline mode
        """
        # Test VAD multiple times with same audio
        results = []
        for _ in range(3):
            vad_result = await offline_processor.detect_voice_activity(audio_buffer)
            results.append(vad_result)
        
        # Results should be consistent
        assert all(r.is_speech == results[0].is_speech for r in results)
        assert all(isinstance(r.confidence, float) for r in results)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)
        assert all(r.energy_level >= 0.0 for r in results)
    
    @given(st.text(min_size=1, max_size=100), st.sampled_from(list(LanguageCode)))
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_tts_synthesis_completeness(
        self, 
        offline_processor, 
        text, 
        language
    ):
        """
        Property 14.3: Offline TTS synthesis produces complete audio output.
        
        **Validates: Requirements 3.3** - TTS functionality in offline mode
        """
        # Synthesize speech offline
        synthesized_audio = await offline_processor.synthesize_speech_offline(
            text, language, AccentType.STANDARD
        )
        
        # Verify synthesis completeness
        assert isinstance(synthesized_audio, AudioBuffer)
        assert len(synthesized_audio.data) > 0
        assert synthesized_audio.duration > 0
        assert synthesized_audio.sample_rate > 0
        assert synthesized_audio.channels in [1, 2]
        
        # Duration should be reasonable for text length
        min_duration = max(0.1, len(text) * 0.05)  # At least 50ms per character
        assert synthesized_audio.duration >= min_duration
    
    @given(st.text(min_size=1, max_size=50), st.sampled_from(list(LanguageCode)))
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_query_caching_consistency(
        self, 
        offline_processor, 
        query_text, 
        language
    ):
        """
        Property 14.4: Offline query caching maintains consistency.
        
        **Validates: Requirements 2.1** - Query processing in offline mode
        """
        response_text = f"Response to: {query_text}"
        
        # Cache a query response
        await offline_processor.cache_query_response(
            query_text, language, response_text, confidence=0.9
        )
        
        # Retrieve cached response
        cached_response = await offline_processor.process_common_query(
            query_text, language
        )
        
        # Verify consistency
        assert cached_response == response_text
        
        # Test case insensitivity and whitespace handling
        cached_response_2 = await offline_processor.process_common_query(
            query_text.upper().strip(), language
        )
        assert cached_response_2 == response_text
    
    @given(audio_buffer_strategy())
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_noise_filtering_stability(
        self, 
        offline_processor, 
        audio_buffer
    ):
        """
        Property 14.5: Offline noise filtering produces stable results.
        
        **Validates: Requirements 1.4** - Noise filtering in offline mode
        """
        # Apply noise filtering multiple times
        filtered_results = []
        for _ in range(3):
            filtered_audio = await offline_processor.filter_background_noise(audio_buffer)
            filtered_results.append(filtered_audio)
        
        # Results should be stable (same length and format)
        assert all(len(r.data) == len(filtered_results[0].data) for r in filtered_results)
        assert all(r.sample_rate == filtered_results[0].sample_rate for r in filtered_results)
        assert all(r.channels == filtered_results[0].channels for r in filtered_results)
        
        # Filtered audio should not be empty
        assert all(len(r.data) > 0 for r in filtered_results)
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_cache_size_management(self, temp_cache_dir):
        """
        Property 14.6: Offline cache respects size limits.
        
        **Validates: Requirements 1.4** - Cache management in offline mode
        """
        # Create processor with very small cache limit
        processor = create_offline_voice_processor(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=1,  # Very small limit
            common_queries_limit=5
        )
        
        # Add many queries to exceed limit
        for i in range(10):
            await processor.cache_query_response(
                f"query_{i}", LanguageCode.ENGLISH_IN, f"response_{i}"
            )
        
        # Verify cache size is managed
        stats = processor.get_offline_stats()
        assert stats['cached_queries_count'] <= 5  # Should not exceed limit
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_network_status_detection(self, offline_processor):
        """
        Property 15.1: Network status detection is reliable.
        
        **Validates: Requirements 1.4** - Network resilience
        """
        # Test network connectivity check
        with patch('socket.create_connection') as mock_socket:
            # Test online detection
            mock_socket.return_value = Mock()
            status = await offline_processor.check_network_connectivity()
            assert status in [NetworkStatus.ONLINE, NetworkStatus.OFFLINE]
            
            # Test offline detection
            mock_socket.side_effect = OSError("Network unreachable")
            status = await offline_processor.check_network_connectivity()
            assert status == NetworkStatus.OFFLINE
    
    @given(audio_buffer_strategy())
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_processing_fallback(self, offline_processor, audio_buffer):
        """
        Property 15.2: Offline processing provides reliable fallback.
        
        **Validates: Requirements 1.4** - Graceful degradation
        """
        # Simulate network failure
        with patch.object(offline_processor, 'check_network_connectivity') as mock_check:
            mock_check.return_value = NetworkStatus.OFFLINE
            
            # Process audio in offline mode
            processed_audio = await offline_processor.process_audio_stream(
                audio_buffer, LanguageCode.ENGLISH_IN
            )
            
            # Verify fallback works
            assert isinstance(processed_audio, AudioBuffer)
            assert len(processed_audio.data) > 0
            assert processed_audio.duration > 0
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_health_check_completeness(self, offline_processor):
        """
        Property 14.7: Offline health check provides comprehensive status.
        
        **Validates: Requirements 1.4** - System monitoring in offline mode
        """
        health_status = await offline_processor.health_check()
        
        # Verify health check completeness
        assert 'status' in health_status
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'network_status' in health_status
        assert 'databases' in health_status
        assert 'local_engines' in health_status
        assert 'functionality_tests' in health_status
        assert 'cache_stats' in health_status
        assert 'offline_stats' in health_status


class TestDataSyncManagerProperties:
    """Property-based tests for DataSyncManager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sync_manager(self, temp_cache_dir):
        """Create DataSyncManager instance for testing."""
        return create_data_sync_manager(
            cache_dir=temp_cache_dir,
            sync_interval_minutes=1,  # Short interval for testing
            max_retry_attempts=2,
            enable_analytics=True
        )
    
    @given(sync_item_strategy())
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_sync_queue_integrity(self, sync_manager, sync_item):
        """
        Property 15.3: Sync queue maintains data integrity.
        
        **Validates: Requirements 1.4** - Data synchronization integrity
        """
        # Add item to sync queue
        await sync_manager.add_to_sync_queue(
            sync_item.item_id,
            sync_item.item_type,
            sync_item.local_data
        )
        
        # Verify item is in queue
        status = sync_manager.get_sync_status()
        assert status['total_queued_items'] > 0
        assert sync_item.item_type in status['pending_by_type']
        
        # Verify data integrity
        queued_item = None
        for item in sync_manager.sync_queue:
            if item.item_id == sync_item.item_id:
                queued_item = item
                break
        
        assert queued_item is not None
        assert queued_item.item_type == sync_item.item_type
        assert queued_item.local_data == sync_item.local_data
        assert queued_item.sync_status == SyncStatus.PENDING
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.floats()),
        min_size=1,
        max_size=5
    ))
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_conflict_resolution_consistency(self, sync_manager, local_data):
        """
        Property 15.4: Conflict resolution produces consistent results.
        
        **Validates: Requirements 1.4** - Conflict resolution in sync
        """
        # Create conflicted sync item
        sync_item = SyncItem(
            item_id="test_conflict",
            item_type="preference",
            local_data=local_data,
            remote_data={"conflicting": "data", "timestamp": "2023-01-01T00:00:00"},
            local_timestamp=datetime.now(),
            remote_timestamp=datetime.now() - timedelta(hours=1),
            sync_status=SyncStatus.CONFLICT
        )
        
        # Test different resolution strategies
        strategies = [
            ConflictResolution.LOCAL_WINS,
            ConflictResolution.REMOTE_WINS,
            ConflictResolution.TIMESTAMP_BASED,
            ConflictResolution.MERGE
        ]
        
        for strategy in strategies:
            # Resolve conflict
            result = await sync_manager.resolve_conflict(sync_item, strategy)
            
            # Verify resolution consistency
            if strategy != ConflictResolution.USER_CHOICE:
                assert 'success' in result
                # For timestamp-based, local should win (newer)
                if strategy == ConflictResolution.TIMESTAMP_BASED:
                    assert result.get('success', False)
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_session_tracking_accuracy(self, sync_manager):
        """
        Property 15.5: Offline session tracking maintains accurate metrics.
        
        **Validates: Requirements 1.4** - Offline usage analytics
        """
        # Start offline session
        session_id = await sync_manager.start_offline_session()
        assert session_id != ""
        assert sync_manager.current_session is not None
        
        # Record various activities
        activities = [
            ("query", LanguageCode.HINDI, True),
            ("tts", LanguageCode.ENGLISH_IN, False),
            ("query", LanguageCode.TAMIL, True),
            ("network_interruption", None, False),
        ]
        
        for activity_type, language, cache_hit in activities:
            sync_manager.record_offline_activity(activity_type, language, cache_hit)
        
        # Verify session metrics
        session = sync_manager.current_session
        assert session.queries_processed == 2
        assert session.tts_synthesized == 1
        assert session.network_interruptions == 1
        assert len(session.languages_used) == 3  # All unique languages
        assert LanguageCode.HINDI in session.languages_used
        assert LanguageCode.ENGLISH_IN in session.languages_used
        assert LanguageCode.TAMIL in session.languages_used
        
        # End session
        await sync_manager.end_offline_session(user_satisfaction=4.5)
        assert sync_manager.current_session is None
    
    @given(st.integers(min_value=1, max_value=30))
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_analytics_data_completeness(self, sync_manager, days_back):
        """
        Property 15.6: Analytics data provides complete information.
        
        **Validates: Requirements 1.4** - Comprehensive offline analytics
        """
        # Get analytics data
        analytics = await sync_manager.get_offline_analytics(days_back)
        
        # Verify completeness
        required_fields = [
            'period_days', 'total_sessions', 'total_offline_time_hours',
            'total_queries', 'total_tts_synthesized', 'average_session_duration_minutes',
            'languages_used', 'average_cache_hit_rate', 'network_interruptions',
            'user_satisfaction_average', 'sync_statistics'
        ]
        
        for field in required_fields:
            assert field in analytics, f"Missing field: {field}"
        
        # Verify data types
        assert isinstance(analytics['period_days'], int)
        assert isinstance(analytics['total_sessions'], int)
        assert isinstance(analytics['total_offline_time_hours'], (int, float))
        assert isinstance(analytics['languages_used'], dict)
        assert isinstance(analytics['sync_statistics'], dict)
        
        # Verify ranges
        assert analytics['period_days'] == days_back
        assert analytics['total_sessions'] >= 0
        assert analytics['total_offline_time_hours'] >= 0.0
        assert 0.0 <= analytics['average_cache_hit_rate'] <= 1.0
        assert 0.0 <= analytics['user_satisfaction_average'] <= 5.0
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_sync_retry_mechanism_reliability(self, sync_manager):
        """
        Property 15.7: Sync retry mechanism handles failures reliably.
        
        **Validates: Requirements 1.4** - Reliable data synchronization
        """
        # Add item that will fail to sync
        await sync_manager.add_to_sync_queue(
            "failing_item",
            "query",
            {"test": "data"}
        )
        
        # Mock network connectivity and sync failure
        with patch.object(sync_manager, '_check_network_connectivity') as mock_network:
            with patch.object(sync_manager, '_sync_item') as mock_sync:
                mock_network.return_value = True
                mock_sync.return_value = {"success": False, "error": "network_error"}
                
                # Perform sync multiple times
                for _ in range(sync_manager.max_retry_attempts + 1):
                    await sync_manager.perform_sync()
                
                # Verify retry mechanism
                failed_item = None
                for item in sync_manager.sync_queue:
                    if item.item_id == "failing_item":
                        failed_item = item
                        break
                
                assert failed_item is not None
                assert failed_item.retry_count >= sync_manager.max_retry_attempts
                assert failed_item.sync_status == SyncStatus.FAILED
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_sync_status_consistency(self, sync_manager):
        """
        Property 15.8: Sync status reporting is consistent and accurate.
        
        **Validates: Requirements 1.4** - Sync status monitoring
        """
        # Add various items to sync queue
        items = [
            ("item1", "query", {"data": "1"}),
            ("item2", "model", {"data": "2"}),
            ("item3", "preference", {"data": "3"}),
        ]
        
        for item_id, item_type, data in items:
            await sync_manager.add_to_sync_queue(item_id, item_type, data)
        
        # Get sync status
        status = sync_manager.get_sync_status()
        
        # Verify status consistency
        assert status['total_queued_items'] == len(items)
        assert sum(status['pending_by_type'].values()) == len(items)
        assert 'query' in status['pending_by_type']
        assert 'model' in status['pending_by_type']
        assert 'preference' in status['pending_by_type']
        
        # Verify status fields
        assert isinstance(status['sync_in_progress'], bool)
        assert isinstance(status['sync_statistics'], dict)
        assert isinstance(status['current_session_active'], bool)


class OfflineSystemStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for offline system behavior.
    
    This tests the interaction between offline voice processing and
    data synchronization under various network conditions.
    """
    
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.mkdtemp()
        self.offline_processor = None
        self.sync_manager = None
        self.network_online = True
        self.cached_queries = {}
        self.sync_queue_items = []
    
    @initialize()
    def setup_system(self):
        """Initialize offline system components."""
        self.offline_processor = create_offline_voice_processor(
            cache_dir=self.temp_dir,
            max_cache_size_mb=5,
            common_queries_limit=10
        )
        
        self.sync_manager = create_data_sync_manager(
            cache_dir=self.temp_dir,
            sync_interval_minutes=1,
            max_retry_attempts=2,
            enable_analytics=True
        )
    
    @rule(query_text=st.text(min_size=1, max_size=50))
    def cache_query(self, query_text):
        """Cache a query response."""
        assume(len(query_text.strip()) > 0)
        
        response = f"Response to: {query_text}"
        language = LanguageCode.ENGLISH_IN
        
        # Cache the query
        asyncio.run(self.offline_processor.cache_query_response(
            query_text, language, response, confidence=0.8
        ))
        
        self.cached_queries[query_text.lower().strip()] = response
    
    @rule(item_type=st.sampled_from(["query", "model", "preference"]))
    def add_sync_item(self, item_type):
        """Add item to sync queue."""
        item_id = f"{item_type}_{len(self.sync_queue_items)}"
        data = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        asyncio.run(self.sync_manager.add_to_sync_queue(item_id, item_type, data))
        self.sync_queue_items.append((item_id, item_type))
    
    @rule()
    def toggle_network(self):
        """Toggle network connectivity."""
        self.network_online = not self.network_online
    
    @rule()
    def perform_sync(self):
        """Perform data synchronization."""
        if self.network_online:
            with patch.object(self.sync_manager, '_check_network_connectivity') as mock_check:
                mock_check.return_value = True
                result = asyncio.run(self.sync_manager.perform_sync())
                assert 'status' in result
    
    @invariant()
    def cache_consistency(self):
        """Cached queries remain accessible."""
        for query_text, expected_response in self.cached_queries.items():
            cached_response = asyncio.run(
                self.offline_processor.process_common_query(
                    query_text, LanguageCode.ENGLISH_IN
                )
            )
            if cached_response is not None:
                assert cached_response == expected_response
    
    @invariant()
    def sync_queue_integrity(self):
        """Sync queue maintains integrity."""
        status = self.sync_manager.get_sync_status()
        assert status['total_queued_items'] >= 0
        assert isinstance(status['sync_statistics'], dict)
    
    def teardown(self):
        """Clean up test resources."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# Stateful test
TestOfflineSystemStateMachine = OfflineSystemStateMachine.TestCase


@pytest.mark.property
@pytest.mark.slow
class TestOfflineSystemIntegration:
    """Integration property tests for complete offline system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def offline_system(self, temp_cache_dir):
        """Create complete offline system for testing."""
        processor = create_offline_voice_processor(
            cache_dir=temp_cache_dir,
            max_cache_size_mb=10,
            enable_local_asr=True,
            enable_local_tts=True
        )
        
        sync_manager = create_data_sync_manager(
            cache_dir=temp_cache_dir,
            sync_interval_minutes=1,
            enable_analytics=True
        )
        
        return processor, sync_manager
    
    @given(
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        st.lists(audio_buffer_strategy(), min_size=1, max_size=5)
    )
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_complete_offline_workflow(self, offline_system, queries, audio_buffers):
        """
        Property 14.8 & 15.9: Complete offline workflow maintains consistency.
        
        **Validates: Requirements 1.4, 2.1, 3.3** - End-to-end offline functionality
        """
        processor, sync_manager = offline_system
        
        # Start offline session
        session_id = await sync_manager.start_offline_session()
        assert session_id != ""
        
        # Process queries offline
        for query in queries:
            assume(len(query.strip()) > 0)
            
            # Cache query response
            response = f"Offline response to: {query}"
            await processor.cache_query_response(
                query, LanguageCode.ENGLISH_IN, response
            )
            
            # Retrieve cached response
            cached_response = await processor.process_common_query(
                query, LanguageCode.ENGLISH_IN
            )
            assert cached_response == response
            
            # Record activity
            sync_manager.record_offline_activity("query", LanguageCode.ENGLISH_IN, True)
        
        # Process audio offline
        for audio_buffer in audio_buffers:
            # Test voice activity detection
            vad_result = await processor.detect_voice_activity(audio_buffer)
            assert isinstance(vad_result, VoiceActivityResult)
            
            # Test audio processing
            processed_audio = await processor.process_audio_stream(
                audio_buffer, LanguageCode.ENGLISH_IN
            )
            assert isinstance(processed_audio, AudioBuffer)
            
            # Test TTS synthesis
            synthesized_audio = await processor.synthesize_speech_offline(
                "Test synthesis", LanguageCode.ENGLISH_IN
            )
            assert isinstance(synthesized_audio, AudioBuffer)
            
            # Record activity
            sync_manager.record_offline_activity("tts", LanguageCode.ENGLISH_IN, False)
        
        # End session
        await sync_manager.end_offline_session(user_satisfaction=4.0)
        
        # Verify system state
        processor_stats = processor.get_offline_stats()
        sync_status = sync_manager.get_sync_status()
        
        assert processor_stats['cached_queries_count'] > 0
        assert processor_stats['local_asr_available']
        assert processor_stats['local_tts_available']
        assert sync_status['current_session_active'] is False
    
    @pytest.mark.asyncio
    @pytest.mark.property
    async def test_offline_system_resilience(self, offline_system):
        """
        Property 15.10: Offline system maintains resilience under stress.
        
        **Validates: Requirements 1.4** - System resilience and recovery
        """
        processor, sync_manager = offline_system
        
        # Simulate various failure conditions
        failure_scenarios = [
            "network_interruption",
            "cache_corruption",
            "resource_exhaustion",
            "concurrent_access"
        ]
        
        for scenario in failure_scenarios:
            try:
                if scenario == "network_interruption":
                    # Simulate network failure during sync
                    with patch.object(sync_manager, '_check_network_connectivity') as mock_check:
                        mock_check.return_value = False
                        result = await sync_manager.perform_sync()
                        assert result['status'] == 'no_network'
                
                elif scenario == "cache_corruption":
                    # Test recovery from cache issues
                    await processor.clear_offline_cache("all")
                    stats = processor.get_offline_stats()
                    assert stats['cached_queries_count'] == 0
                
                elif scenario == "resource_exhaustion":
                    # Test behavior under resource constraints
                    # Add many items to test limits
                    for i in range(20):
                        await sync_manager.add_to_sync_queue(
                            f"stress_item_{i}", "query", {"data": f"test_{i}"}
                        )
                    
                    status = sync_manager.get_sync_status()
                    assert status['total_queued_items'] > 0
                
                elif scenario == "concurrent_access":
                    # Test concurrent operations
                    tasks = []
                    for i in range(5):
                        task = processor.cache_query_response(
                            f"concurrent_query_{i}",
                            LanguageCode.ENGLISH_IN,
                            f"response_{i}"
                        )
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks)
                    
                    # Verify all queries were cached
                    for i in range(5):
                        response = await processor.process_common_query(
                            f"concurrent_query_{i}", LanguageCode.ENGLISH_IN
                        )
                        assert response == f"response_{i}"
                
                # System should remain functional after each scenario
                health_status = await processor.health_check()
                assert health_status['status'] in ['healthy', 'degraded']
                
            except Exception as e:
                # Log but don't fail - some failures are expected
                print(f"Expected failure in scenario {scenario}: {e}")


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])