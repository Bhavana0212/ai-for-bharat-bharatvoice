"""
Property-based tests for Streamlit Web Interface offline functionality.

This module tests the offline detection, caching, and queue management
functionality of the Streamlit web interface to ensure they work correctly
under various conditions.

Property 18: Offline Detection
Property 19: Feature Disabling in Offline Mode
Property 20: Feature Re-enabling on Connection Restore
Property 21: Cache Indicator Display
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

# Import the functions from app.py
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from app import (
    check_backend_health,
    update_connection_status,
    monitor_connection,
    queue_for_offline_processing,
    process_offline_queue,
    process_with_cache,
    cache_response,
    get_cached_response,
    clear_cache,
    log_action,
    CACHE_TTL
)


# Test strategies
@st.composite
def operation_data_strategy(draw):
    """Generate operation data for offline queue."""
    operation_type = draw(st.sampled_from(['transcribe', 'respond', 'tts']))
    
    if operation_type == 'transcribe':
        return {
            'audio_data': draw(st.binary(min_size=100, max_size=1000)),
            'language': draw(st.sampled_from(['hi', 'en-IN', 'ta', 'te', 'bn']))
        }
    elif operation_type == 'respond':
        return {
            'text': draw(st.text(min_size=1, max_size=200)),
            'language': draw(st.sampled_from(['hi', 'en-IN', 'ta', 'te', 'bn']))
        }
    else:  # tts
        return {
            'text': draw(st.text(min_size=1, max_size=200)),
            'language': draw(st.sampled_from(['hi', 'en-IN', 'ta', 'te', 'bn']))
        }


@st.composite
def cache_key_strategy(draw):
    """Generate cache keys."""
    prefix = draw(st.sampled_from(['transcription', 'response', 'tts', 'query']))
    suffix = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd')
    )))
    return f"{prefix}_{suffix}"


class TestOfflineDetectionProperties:
    """Property-based tests for offline detection functionality."""
    
    @pytest.fixture
    def mock_session_state(self):
        """Create mock Streamlit session state."""
        session_state = MagicMock()
        session_state.is_online = True
        session_state.last_health_check = 0
        session_state.action_history = []
        return session_state
    
    @pytest.mark.property
    def test_backend_health_check_reliability(self):
        """
        Property 18: Offline Detection - Backend health check is reliable.
        
        **Validates: Requirements 7.1** - Connection failure detection
        """
        # Test with successful connection
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = check_backend_health()
            assert result is True
        
        # Test with connection failure
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            result = check_backend_health()
            assert result is False
        
        # Test with non-200 status code
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            result = check_backend_health()
            assert result is False
    
    @given(st.booleans())
    @pytest.mark.property
    def test_connection_status_update_consistency(self, is_online):
        """
        Property 18: Offline Detection - Connection status updates are consistent.
        
        **Validates: Requirements 7.1** - Connection status tracking
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = not is_online  # Different from new status
            mock_st.session_state.action_history = []
            
            with patch('app.check_backend_health') as mock_health:
                mock_health.return_value = is_online
                
                # Import and patch the module-level st
                import app
                app.st = mock_st
                
                update_connection_status()
                
                # Verify status was updated
                assert mock_st.session_state.is_online == is_online
                
                # Verify appropriate message was displayed
                if is_online:
                    mock_st.success.assert_called_once()
                else:
                    mock_st.error.assert_called_once()
    
    @given(st.floats(min_value=0, max_value=60))
    @pytest.mark.property
    def test_connection_monitoring_interval(self, elapsed_time):
        """
        Property 18: Offline Detection - Connection monitoring respects interval.
        
        **Validates: Requirements 7.1, 7.3** - Periodic health checks
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.last_health_check = time.time() - elapsed_time
            mock_st.session_state.is_online = True
            
            with patch('app.update_connection_status') as mock_update:
                with patch('app.time.time') as mock_time:
                    mock_time.return_value = time.time()
                    
                    import app
                    app.st = mock_st
                    
                    monitor_connection()
                    
                    # Should only update if more than 30 seconds elapsed
                    if elapsed_time > 30:
                        mock_update.assert_called_once()
                    else:
                        mock_update.assert_not_called()


class TestOfflineModeFeatureDisabling:
    """Property-based tests for feature disabling in offline mode."""
    
    @given(st.booleans())
    @pytest.mark.property
    def test_feature_availability_based_on_connection(self, is_online):
        """
        Property 19: Feature Disabling in Offline Mode - Features disabled when offline.
        
        **Validates: Requirements 7.2** - Feature disabling
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = is_online
            
            import app
            app.st = mock_st
            
            from app import render_offline_indicator
            
            render_offline_indicator()
            
            # Verify warning is shown only when offline
            if not is_online:
                mock_st.warning.assert_called_once()
            else:
                mock_st.warning.assert_not_called()
    
    @pytest.mark.property
    def test_offline_indicator_message_completeness(self):
        """
        Property 19: Feature Disabling in Offline Mode - Offline indicator shows complete info.
        
        **Validates: Requirements 7.1, 7.2** - User feedback
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = False
            
            import app
            app.st = mock_st
            
            from app import render_offline_indicator
            
            render_offline_indicator()
            
            # Verify warning was called
            mock_st.warning.assert_called_once()
            
            # Get the warning message
            warning_message = mock_st.warning.call_args[0][0]
            
            # Verify message contains key information
            assert 'Offline Mode' in warning_message or 'ऑफ़लाइन मोड' in warning_message
            assert 'Disabled Features' in warning_message or 'अक्षम सुविधाएँ' in warning_message
            assert 'Available Features' in warning_message or 'उपलब्ध सुविधाएँ' in warning_message
            assert 'Speech recognition' in warning_message or 'वाक् पहचान' in warning_message


class TestOfflineQueueManagement:
    """Property-based tests for offline queue management."""
    
    @given(
        st.sampled_from(['transcribe', 'respond', 'tts']),
        operation_data_strategy()
    )
    @pytest.mark.property
    def test_queue_operation_integrity(self, operation, data):
        """
        Property 20: Feature Re-enabling on Connection Restore - Queue maintains integrity.
        
        **Validates: Requirements 7.3** - Offline queue management
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.offline_queue = []
            mock_st.session_state.action_history = []
            
            import app
            app.st = mock_st
            
            queue_for_offline_processing(operation, data)
            
            # Verify item was added to queue
            assert len(mock_st.session_state.offline_queue) == 1
            
            # Verify queue item structure
            queue_item = mock_st.session_state.offline_queue[0]
            assert queue_item['operation'] == operation
            assert queue_item['data'] == data
            assert 'timestamp' in queue_item
            
            # Verify info message was displayed
            mock_st.info.assert_called()
    
    @given(st.lists(
        st.tuples(
            st.sampled_from(['transcribe', 'respond', 'tts']),
            operation_data_strategy()
        ),
        min_size=1,
        max_size=10
    ))
    @pytest.mark.property
    def test_queue_processing_completeness(self, operations):
        """
        Property 20: Feature Re-enabling on Connection Restore - All queued items processed.
        
        **Validates: Requirements 7.3** - Queue processing
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = True
            mock_st.session_state.action_history = []
            
            # Create queue with operations
            mock_st.session_state.offline_queue = [
                {
                    'operation': op,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                for op, data in operations
            ]
            
            import app
            app.st = mock_st
            
            process_offline_queue()
            
            # Verify queue was cleared
            assert len(mock_st.session_state.offline_queue) == 0
            
            # Verify success message was displayed
            mock_st.success.assert_called()
    
    @pytest.mark.property
    def test_queue_not_processed_when_offline(self):
        """
        Property 20: Feature Re-enabling on Connection Restore - Queue not processed when offline.
        
        **Validates: Requirements 7.3** - Conditional processing
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = False
            mock_st.session_state.offline_queue = [
                {
                    'operation': 'transcribe',
                    'data': {'audio_data': b'test', 'language': 'hi'},
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            import app
            app.st = mock_st
            
            process_offline_queue()
            
            # Verify queue was NOT cleared
            assert len(mock_st.session_state.offline_queue) == 1
            
            # Verify no success message
            mock_st.success.assert_not_called()


class TestCacheAwareProcessing:
    """Property-based tests for cache-aware processing."""
    
    @given(
        cache_key_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats()),
            min_size=1,
            max_size=5
        ),
        st.integers(min_value=60, max_value=7200)
    )
    @pytest.mark.property
    def test_cache_hit_indicator_display(self, cache_key, cached_value, ttl):
        """
        Property 21: Cache Indicator Display - Cached responses show indicator.
        
        **Validates: Requirements 7.5** - Cache indicator
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.cache = {}
            mock_st.session_state.action_history = []
            
            import app
            app.st = mock_st
            
            # Cache a value
            cache_response(cache_key, cached_value, ttl)
            
            # Process with cache (should hit cache)
            processor_called = False
            
            def mock_processor():
                nonlocal processor_called
                processor_called = True
                return cached_value
            
            result = process_with_cache(cache_key, mock_processor, ttl)
            
            # Verify cache was hit
            assert result == cached_value
            assert not processor_called  # Processor should not be called
            
            # Verify cache indicator was displayed
            mock_st.info.assert_called()
            info_message = mock_st.info.call_args[0][0]
            assert 'cache' in info_message.lower() or 'कैश' in info_message
    
    @given(
        cache_key_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats()),
            min_size=1,
            max_size=5
        )
    )
    @pytest.mark.property
    def test_cache_miss_processing(self, cache_key, value):
        """
        Property 21: Cache Indicator Display - Cache miss triggers processing.
        
        **Validates: Requirements 7.5** - Cache behavior
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.cache = {}
            mock_st.session_state.action_history = []
            
            import app
            app.st = mock_st
            
            # Process with cache (should miss cache)
            processor_called = False
            
            def mock_processor():
                nonlocal processor_called
                processor_called = True
                return value
            
            result = process_with_cache(cache_key, mock_processor)
            
            # Verify processor was called
            assert processor_called
            assert result == value
            
            # Verify result was cached
            cached = get_cached_response(cache_key)
            assert cached == value
    
    @given(
        cache_key_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers()),
            min_size=1,
            max_size=5
        )
    )
    @pytest.mark.property
    def test_cache_expiration_behavior(self, cache_key, value):
        """
        Property 21: Cache Indicator Display - Expired cache entries are removed.
        
        **Validates: Requirements 7.5** - Cache TTL
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.cache = {}
            
            import app
            app.st = mock_st
            
            # Cache with very short TTL
            cache_response(cache_key, value, ttl=0)
            
            # Wait a moment
            time.sleep(0.1)
            
            # Try to retrieve (should be expired)
            cached = get_cached_response(cache_key)
            
            # Verify cache entry was removed
            assert cached is None
            assert cache_key not in mock_st.session_state.cache


class OfflineSystemStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for Streamlit offline system behavior.
    
    This tests the interaction between connection monitoring, offline queue,
    and cache management under various conditions.
    """
    
    def __init__(self):
        super().__init__()
        self.is_online = True
        self.offline_queue = []
        self.cache = {}
        self.last_health_check = 0
    
    @rule()
    def toggle_connection(self):
        """Toggle network connectivity."""
        self.is_online = not self.is_online
    
    @rule(
        operation=st.sampled_from(['transcribe', 'respond', 'tts']),
        data=operation_data_strategy()
    )
    def queue_operation(self, operation, data):
        """Queue an operation for offline processing."""
        if not self.is_online:
            queue_item = {
                'operation': operation,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.offline_queue.append(queue_item)
    
    @rule()
    def process_queue(self):
        """Process offline queue if online."""
        if self.is_online and self.offline_queue:
            self.offline_queue.clear()
    
    @rule(
        cache_key=cache_key_strategy(),
        value=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(),
            min_size=1,
            max_size=3
        )
    )
    def cache_value(self, cache_key, value):
        """Cache a value."""
        self.cache[cache_key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': 3600
        }
    
    @rule(cache_key=cache_key_strategy())
    def retrieve_cache(self, cache_key):
        """Retrieve cached value."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] <= entry['ttl']:
                return entry['value']
            else:
                del self.cache[cache_key]
        return None
    
    @invariant()
    def queue_only_when_offline(self):
        """Queue should only grow when offline."""
        if self.is_online:
            assert len(self.offline_queue) == 0
    
    @invariant()
    def cache_integrity(self):
        """Cache entries should be valid."""
        for key, entry in list(self.cache.items()):
            assert 'value' in entry
            assert 'timestamp' in entry
            assert 'ttl' in entry
            assert isinstance(entry['timestamp'], float)
            assert isinstance(entry['ttl'], (int, float))


# Stateful test
TestOfflineSystemStateMachine = OfflineSystemStateMachine.TestCase


@pytest.mark.property
class TestOfflineSystemIntegration:
    """Integration property tests for complete offline system."""
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['transcribe', 'respond', 'tts']),
                operation_data_strategy()
            ),
            min_size=1,
            max_size=5
        ),
        st.lists(
            st.tuples(
                cache_key_strategy(),
                st.dictionaries(
                    st.text(min_size=1, max_size=10),
                    st.integers(),
                    min_size=1,
                    max_size=3
                )
            ),
            min_size=1,
            max_size=5
        )
    )
    @pytest.mark.property
    def test_complete_offline_workflow(self, operations, cache_items):
        """
        Property 18-21: Complete offline workflow maintains consistency.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.5** - End-to-end offline functionality
        """
        with patch('app.st') as mock_st:
            mock_st.session_state = MagicMock()
            mock_st.session_state.is_online = False
            mock_st.session_state.offline_queue = []
            mock_st.session_state.cache = {}
            mock_st.session_state.action_history = []
            
            import app
            app.st = mock_st
            
            # Queue operations while offline
            for operation, data in operations:
                queue_for_offline_processing(operation, data)
            
            # Verify all operations were queued
            assert len(mock_st.session_state.offline_queue) == len(operations)
            
            # Cache some values
            for cache_key, value in cache_items:
                cache_response(cache_key, value)
            
            # Verify all values were cached
            for cache_key, value in cache_items:
                cached = get_cached_response(cache_key)
                assert cached == value
            
            # Go online
            mock_st.session_state.is_online = True
            
            # Process queue
            process_offline_queue()
            
            # Verify queue was cleared
            assert len(mock_st.session_state.offline_queue) == 0
            
            # Verify cache still works
            for cache_key, value in cache_items:
                cached = get_cached_response(cache_key)
                assert cached == value


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "property"])
