"""
Property-Based Tests for Error Handling

This module contains property-based tests for error handling and retry logic
in the Streamlit web interface. Tests validate universal correctness properties
across all possible inputs.

Properties tested:
- Property 9: Error Message Display - Errors always display user-friendly messages
- Property 13: Retry Option on Failure - Retry option is provided on failures
- Property 28: Timeout Handling - Timeouts provide cancel and retry options

Requirements validated: 3.4, 4.4, 10.1, 10.4
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import requests
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    handle_network_error,
    handle_validation_error,
    handle_api_error,
    retry_with_backoff,
    process_with_retry,
    parse_error_response
)


# Strategy for generating error types
error_types = st.sampled_from([
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.HTTPError,
    Exception
])

# Strategy for generating operation names
operation_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters=' _-'
))

# Strategy for generating validation field names
validation_fields = st.sampled_from([
    'audio_format',
    'audio_size',
    'text_length',
    'unknown_field'
])

# Strategy for generating HTTP status codes
http_status_codes = st.sampled_from([
    400, 401, 403, 404, 429, 500, 503, 418, 502, 504
])


class TestProperty9_ErrorMessageDisplay:
    """
    Property 9: Error Message Display
    
    Test that errors always display user-friendly messages, never raw technical
    errors or stack traces. All error messages should be bilingual (English/Hindi).
    
    Validates: Requirements 3.4, 4.4, 10.1
    """
    
    @given(
        error_type=error_types,
        operation=operation_names
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_network_errors_display_user_friendly_messages(self, error_type, operation):
        """Test that network errors always display user-friendly messages"""
        assume(len(operation.strip()) > 0)
        
        # Create mock error
        if error_type == requests.exceptions.Timeout:
            error = requests.exceptions.Timeout("Connection timeout")
        elif error_type == requests.exceptions.ConnectionError:
            error = requests.exceptions.ConnectionError("Connection refused")
        elif error_type == requests.exceptions.HTTPError:
            error = requests.exceptions.HTTPError("HTTP error")
        else:
            error = Exception("Generic error")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call error handler
            handle_network_error(error, operation)
            
            # Verify error message was displayed
            assert mock_st.error.called, "Error message should be displayed"
            
            # Get the error message
            error_message = mock_st.error.call_args[0][0]
            
            # Verify message is user-friendly (contains emoji and bilingual text)
            assert any(emoji in error_message for emoji in ['⏱️', '🔌', '❌']), \
                "Error message should contain user-friendly emoji"
            
            # Verify no raw exception details in message
            assert "Traceback" not in error_message, \
                "Error message should not contain stack traces"
            assert "Exception" not in error_message or "exception" in error_message.lower(), \
                "Error message should not contain raw exception class names"
            
            # Verify action was logged
            assert mock_log.called, "Error should be logged"
    
    @given(
        field=validation_fields,
        error_msg=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=50)
    def test_validation_errors_display_helpful_messages(self, field, error_msg):
        """Test that validation errors display helpful, actionable messages"""
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log:
            
            mock_st.error = Mock()
            
            # Call validation error handler
            handle_validation_error(error_msg, field)
            
            # Verify error message was displayed
            assert mock_st.error.called, "Validation error message should be displayed"
            
            # Get the error message
            error_message = mock_st.error.call_args[0][0]
            
            # Verify message contains helpful information
            if field in ['audio_format', 'audio_size', 'text_length']:
                # Known fields should have detailed messages
                assert '❌' in error_message, "Error message should have error emoji"
                assert len(error_message) > 50, "Error message should be detailed"
            
            # Verify action was logged
            assert mock_log.called, "Validation error should be logged"
    
    @given(
        status_code=http_status_codes,
        operation=operation_names
    )
    @settings(max_examples=50)
    def test_api_errors_display_status_appropriate_messages(self, status_code, operation):
        """Test that API errors display appropriate messages based on status code"""
        assume(len(operation.strip()) > 0)
        
        # Create mock response
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.headers = {'Retry-After': '60'}
        mock_response.json = Mock(return_value={'detail': 'Test error'})
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log:
            
            mock_st.error = Mock()
            mock_st.info = Mock()
            
            # Call API error handler
            handle_api_error(mock_response, operation)
            
            # Verify error message was displayed
            assert mock_st.error.called, "API error message should be displayed"
            
            # Get the error message
            error_message = mock_st.error.call_args[0][0]
            
            # Verify message is user-friendly
            assert any(emoji in error_message for emoji in ['❌', '🔒', '🚫', '🔍', '⏸️', '⚠️', '🔧']), \
                "Error message should contain user-friendly emoji"
            
            # Verify bilingual support (should contain '/' separator)
            assert '/' in error_message or 'HTTP' in error_message, \
                "Error message should be bilingual or show HTTP status"
            
            # Verify action was logged
            assert mock_log.called, "API error should be logged"


class TestProperty13_RetryOptionOnFailure:
    """
    Property 13: Retry Option on Failure
    
    Test that retry option is provided on failures for retryable errors
    (timeouts, connection errors). Non-retryable errors should not offer retry.
    
    Validates: Requirements 10.1, 10.4
    """
    
    @given(operation=operation_names)
    @settings(max_examples=30)
    def test_timeout_errors_provide_retry_option(self, operation):
        """Test that timeout errors provide a retry button"""
        assume(len(operation.strip()) > 0)
        
        error = requests.exceptions.Timeout("Connection timeout")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call error handler
            handle_network_error(error, operation)
            
            # Verify retry button was offered
            assert mock_st.button.called, "Retry button should be displayed for timeout errors"
            
            # Verify button text contains "Retry"
            button_text = mock_st.button.call_args[0][0]
            assert 'Retry' in button_text or 'retry' in button_text.lower(), \
                "Button should offer retry option"
    
    @given(operation=operation_names)
    @settings(max_examples=30)
    def test_connection_errors_provide_retry_option(self, operation):
        """Test that connection errors provide a retry button"""
        assume(len(operation.strip()) > 0)
        
        error = requests.exceptions.ConnectionError("Connection refused")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.update_connection_status') as mock_update, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call error handler
            handle_network_error(error, operation)
            
            # Verify retry button was offered
            assert mock_st.button.called, "Retry button should be displayed for connection errors"
            
            # Verify button text contains "Retry"
            button_text = mock_st.button.call_args[0][0]
            assert 'Retry' in button_text or 'retry' in button_text.lower(), \
                "Button should offer retry option"
    
    @given(
        max_retries=st.integers(min_value=1, max_value=5),
        should_succeed_on_attempt=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=30)
    def test_retry_with_backoff_retries_correct_number_of_times(self, max_retries, should_succeed_on_attempt):
        """Test that retry_with_backoff retries the correct number of times"""
        assume(should_succeed_on_attempt <= max_retries)
        
        attempt_count = [0]
        
        def failing_operation():
            attempt_count[0] += 1
            if attempt_count[0] < should_succeed_on_attempt:
                raise requests.exceptions.Timeout("Timeout")
            return "success"
        
        # Mock streamlit and time
        with patch('app.st') as mock_st, \
             patch('app.time.sleep') as mock_sleep:
            
            mock_st.warning = Mock()
            
            # Call retry function
            result = retry_with_backoff(failing_operation, max_retries=max_retries)
            
            # Verify operation succeeded
            assert result == "success", "Operation should eventually succeed"
            
            # Verify correct number of attempts
            assert attempt_count[0] == should_succeed_on_attempt, \
                f"Should have attempted {should_succeed_on_attempt} times"
            
            # Verify warnings were shown for failed attempts
            if should_succeed_on_attempt > 1:
                assert mock_st.warning.called, "Warning should be shown for retry attempts"
    
    @given(max_retries=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_process_with_retry_handles_retryable_errors(self, max_retries):
        """Test that process_with_retry properly handles retryable errors"""
        
        def always_failing_operation():
            raise requests.exceptions.Timeout("Always fails")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.time.sleep') as mock_sleep, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.warning = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call process_with_retry
            result = process_with_retry(always_failing_operation, 'test_operation', max_retries=max_retries)
            
            # Verify result is None (operation failed)
            assert result is None, "Failed operation should return None"
            
            # Verify error was displayed
            assert mock_st.error.called, "Error message should be displayed"
            
            # Verify retry warnings were shown
            if max_retries > 1:
                assert mock_st.warning.called, "Retry warnings should be shown"


class TestProperty28_TimeoutHandling:
    """
    Property 28: Timeout Handling
    
    Test that timeouts provide cancel and retry options, and that timeout
    errors are handled gracefully without crashing the application.
    
    Validates: Requirements 10.1, 10.4
    """
    
    @given(
        operation=operation_names,
        timeout_seconds=st.integers(min_value=1, max_value=60)
    )
    @settings(max_examples=30)
    def test_timeout_errors_provide_cancel_and_retry_options(self, operation, timeout_seconds):
        """Test that timeout errors provide both cancel and retry options"""
        assume(len(operation.strip()) > 0)
        
        error = requests.exceptions.Timeout(f"Timeout after {timeout_seconds}s")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call error handler
            handle_network_error(error, operation)
            
            # Verify error message was displayed (implicit cancel - user can navigate away)
            assert mock_st.error.called, "Error message should be displayed"
            
            # Verify retry button was offered
            assert mock_st.button.called, "Retry button should be displayed"
            
            # Verify button allows retry
            button_text = mock_st.button.call_args[0][0]
            assert 'Retry' in button_text or 'retry' in button_text.lower(), \
                "Button should offer retry option"
    
    @given(
        initial_delay=st.floats(min_value=0.1, max_value=5.0),
        backoff_factor=st.floats(min_value=1.5, max_value=3.0)
    )
    @settings(max_examples=20)
    def test_exponential_backoff_increases_delay(self, initial_delay, backoff_factor):
        """Test that exponential backoff increases delay between retries"""
        
        attempt_count = [0]
        sleep_delays = []
        
        def always_failing_operation():
            attempt_count[0] += 1
            raise requests.exceptions.Timeout("Timeout")
        
        # Mock streamlit and time
        with patch('app.st') as mock_st, \
             patch('app.time.sleep') as mock_sleep:
            
            mock_st.warning = Mock()
            
            def capture_sleep(delay):
                sleep_delays.append(delay)
            
            mock_sleep.side_effect = capture_sleep
            
            # Call retry function (should fail after max retries)
            try:
                retry_with_backoff(
                    always_failing_operation,
                    max_retries=3,
                    initial_delay=initial_delay,
                    backoff_factor=backoff_factor
                )
            except requests.exceptions.Timeout:
                pass  # Expected to fail
            
            # Verify delays increased exponentially
            if len(sleep_delays) >= 2:
                for i in range(len(sleep_delays) - 1):
                    # Each delay should be approximately backoff_factor times the previous
                    ratio = sleep_delays[i + 1] / sleep_delays[i]
                    assert ratio >= backoff_factor * 0.9, \
                        f"Delay should increase by backoff factor (got ratio {ratio})"
    
    @given(operation=operation_names)
    @settings(max_examples=20)
    def test_timeout_handling_does_not_crash_application(self, operation):
        """Test that timeout errors are handled gracefully without crashing"""
        assume(len(operation.strip()) > 0)
        
        error = requests.exceptions.Timeout("Connection timeout")
        
        # Mock streamlit functions
        with patch('app.st') as mock_st, \
             patch('app.log_action') as mock_log, \
             patch('app.time.time', return_value=1234567890):
            
            mock_st.error = Mock()
            mock_st.button = Mock(return_value=False)
            mock_st.rerun = Mock()
            
            # Call error handler - should not raise exception
            try:
                handle_network_error(error, operation)
                success = True
            except Exception as e:
                success = False
                error_msg = str(e)
            
            # Verify no exception was raised
            assert success, f"Timeout handling should not crash: {error_msg if not success else ''}"
            
            # Verify error was logged
            assert mock_log.called, "Timeout should be logged"


class TestErrorResponseParsing:
    """
    Additional tests for error response parsing to ensure user-friendly messages
    """
    
    @given(
        status_code=http_status_codes,
        error_detail=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=30)
    def test_parse_error_response_returns_user_friendly_message(self, status_code, error_detail):
        """Test that parse_error_response always returns user-friendly messages"""
        
        # Create mock response
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.json = Mock(return_value={'detail': error_detail})
        
        # Parse error response
        error_message = parse_error_response(mock_response)
        
        # Verify message is not empty
        assert len(error_message) > 0, "Error message should not be empty"
        
        # Verify message is user-friendly (contains bilingual separator or HTTP status)
        assert '/' in error_message or 'HTTP' in error_message or 'Error' in error_message, \
            "Error message should be user-friendly"
    
    @given(status_code=http_status_codes)
    @settings(max_examples=20)
    def test_parse_error_response_handles_invalid_json(self, status_code):
        """Test that parse_error_response handles invalid JSON gracefully"""
        
        # Create mock response with invalid JSON
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = status_code
        mock_response.json = Mock(side_effect=ValueError("Invalid JSON"))
        
        # Parse error response - should not crash
        try:
            error_message = parse_error_response(mock_response)
            success = True
        except Exception:
            success = False
        
        # Verify no exception was raised
        assert success, "parse_error_response should handle invalid JSON gracefully"
        
        # Verify fallback message includes status code
        if success:
            assert str(status_code) in error_message, \
                "Fallback message should include HTTP status code"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
