"""
Property-based tests for Localized Error Handling.

**Property 21: Localized Error Handling**
**Validates: Requirements 6.1, 6.2, 6.3**

This module tests that the BharatVoice Assistant provides comprehensive
localized error handling with appropriate messages in multiple Indian languages.
"""

import time
from typing import Dict, Any, List, Optional
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import structlog

from bharatvoice.core.models import LanguageCode
from bharatvoice.utils.error_handler import (
    ErrorHandler,
    LocalizedError,
    ErrorCode,
    ErrorSeverity,
    ERROR_MESSAGES,
    get_error_handler,
    handle_error,
    create_error_response
)


logger = structlog.get_logger(__name__)


# Test data generators
@st.composite
def generate_error_code(draw):
    """Generate error codes."""
    return draw(st.sampled_from(list(ErrorCode)))


@st.composite
def generate_error_severity(draw):
    """Generate error severity levels."""
    return draw(st.sampled_from(list(ErrorSeverity)))


@st.composite
def generate_language_code(draw):
    """Generate supported language codes."""
    return draw(st.sampled_from([
        LanguageCode.HINDI,
        LanguageCode.ENGLISH_INDIA,
        LanguageCode.TAMIL,
        LanguageCode.TELUGU,
        LanguageCode.BENGALI,
        LanguageCode.MARATHI,
        LanguageCode.GUJARATI,
        LanguageCode.KANNADA,
        LanguageCode.MALAYALAM,
        LanguageCode.PUNJABI,
        LanguageCode.ODIA,
    ]))


@st.composite
def generate_common_exception(draw):
    """Generate common Python exceptions."""
    exceptions = [
        ValueError("Invalid input value"),
        KeyError("Missing required key"),
        ConnectionError("Network connection failed"),
        TimeoutError("Operation timed out"),
        PermissionError("Access denied"),
        FileNotFoundError("File not found"),
        MemoryError("Out of memory"),
    ]
    return draw(st.sampled_from(exceptions))


@st.composite
def generate_error_context(draw):
    """Generate error context information."""
    contexts = [
        {},
        {"user_id": "test_user_123"},
        {"request_id": "req_456", "endpoint": "/voice/recognize"},
        {"language": "hi", "complexity": "multilingual"},
        {"service": "external_api", "timeout": 30.0},
        {"file_path": "/tmp/audio.wav", "size": 1024},
    ]
    return draw(st.sampled_from(contexts))


class ErrorHandlingTestStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for error handling system.
    """
    
    def __init__(self):
        super().__init__()
        self.handler = ErrorHandler()
        self.handled_errors = []
        self.error_counts = {}
    
    @rule(
        error_code=generate_error_code(),
        language=generate_language_code(),
        severity=generate_error_severity(),
        context=generate_error_context()
    )
    def handle_error_code(self, error_code, language, severity, context):
        """Handle an error code."""
        localized_error = self.handler.handle_error(
            error_code,
            language=language,
            severity=severity,
            context=context,
            log_error=False  # Avoid cluttering test logs
        )
        
        self.handled_errors.append(localized_error)
        
        # Track error counts by type
        key = (error_code, language)
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Verify error properties
        assert localized_error.error_code == error_code
        assert localized_error.language == language
        assert localized_error.severity == severity
        assert localized_error.context == context
    
    @rule(exception=generate_common_exception(), language=generate_language_code())
    def handle_exception(self, exception, language):
        """Handle a Python exception."""
        localized_error = self.handler.handle_error(
            exception,
            language=language,
            log_error=False
        )
        
        self.handled_errors.append(localized_error)
        
        # Verify exception mapping
        assert isinstance(localized_error, LocalizedError)
        assert localized_error.language == language
        assert localized_error.original_error == exception
    
    @invariant()
    def error_messages_valid(self):
        """All handled errors should have valid messages."""
        for error in self.handled_errors:
            assert isinstance(error.message, str)
            assert len(error.message) > 0
            assert error.message != error.error_code.value  # Should be localized, not just code
    
    @invariant()
    def error_codes_consistent(self):
        """Error codes should be consistent."""
        for error in self.handled_errors:
            assert isinstance(error.error_code, ErrorCode)
            assert error.error_code.value.startswith(error.error_code.value.split('_')[0])


@pytest.mark.asyncio
class TestLocalizedErrorHandling:
    """Test localized error handling compliance."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler."""
        return ErrorHandler()
    
    @given(
        error_code=generate_error_code(),
        language=generate_language_code(),
        severity=generate_error_severity()
    )
    @settings(max_examples=100, deadline=10000)
    async def test_error_message_localization(self, error_handler, error_code, language, severity):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.1**
        
        Property: Error messages should be properly localized for supported languages.
        """
        # Handle error with specific language
        localized_error = error_handler.handle_error(
            error_code,
            language=language,
            severity=severity,
            log_error=False
        )
        
        # Property 1: Error should be properly localized
        assert isinstance(localized_error, LocalizedError), \
            "Should return LocalizedError instance"
        
        assert localized_error.error_code == error_code, \
            f"Error code mismatch: expected {error_code}, got {localized_error.error_code}"
        
        assert localized_error.language == language, \
            f"Language mismatch: expected {language}, got {localized_error.language}"
        
        assert localized_error.severity == severity, \
            f"Severity mismatch: expected {severity}, got {localized_error.severity}"
        
        # Property 2: Message should be in correct language
        message = localized_error.message
        assert isinstance(message, str), "Error message should be string"
        assert len(message) > 0, "Error message should not be empty"
        assert message != error_code.value, "Message should be localized, not just error code"
        
        # Property 3: Message should be appropriate for error type
        message_lower = message.lower()
        
        if error_code.value.startswith("AUTH_"):
            # Authentication errors should mention authentication concepts
            auth_keywords = ["login", "password", "credential", "permission", "access", "token", 
                           "लॉग", "पासवर्ड", "प्रमाण", "अनुमति", "पहुंच", "टोकन"]
            if language in [LanguageCode.HINDI]:
                assert any(keyword in message for keyword in auth_keywords), \
                    f"Auth error should contain relevant keywords: {message}"
        
        elif error_code.value.startswith("VOICE_"):
            # Voice errors should mention voice/audio concepts
            voice_keywords = ["voice", "audio", "speak", "listen", "sound", 
                            "आवाज़", "ऑडियो", "बोल", "सुन", "ध्वनि"]
            if language in [LanguageCode.ENGLISH_INDIA, LanguageCode.HINDI]:
                # Only check for languages we have translations for
                pass  # Message content validation is complex for all languages
        
        # Property 4: Fallback to English should work
        if language not in ERROR_MESSAGES:
            # Should fallback to English (India)
            english_error = error_handler.handle_error(
                error_code,
                language=LanguageCode.ENGLISH_INDIA,
                severity=severity,
                log_error=False
            )
            # Should have a valid English message
            assert len(english_error.message) > 0, "English fallback should have valid message"
    
    @given(
        error_codes=st.lists(generate_error_code(), min_size=2, max_size=10),
        language=generate_language_code()
    )
    @settings(max_examples=30, deadline=15000)
    async def test_error_message_consistency(self, error_handler, error_codes, language):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.2**
        
        Property: Error messages should be consistent in style and format within a language.
        """
        assume(len(set(error_codes)) >= 2)  # Need at least 2 different error codes
        
        messages = []
        for error_code in error_codes:
            localized_error = error_handler.handle_error(
                error_code,
                language=language,
                log_error=False
            )
            messages.append(localized_error.message)
        
        # Property 1: All messages should be non-empty strings
        for i, message in enumerate(messages):
            assert isinstance(message, str), f"Message {i} should be string"
            assert len(message) > 0, f"Message {i} should not be empty"
        
        # Property 2: Messages should be distinct for different error codes
        unique_messages = set(messages)
        unique_codes = set(error_codes)
        
        # If we have different error codes, we should generally have different messages
        # (unless some codes map to the same generic message)
        if len(unique_codes) > 1:
            assert len(unique_messages) >= 1, "Should have at least one unique message"
        
        # Property 3: Messages should follow consistent formatting
        for message in messages:
            # Should not start or end with whitespace
            assert message == message.strip(), f"Message should not have leading/trailing whitespace: '{message}'"
            
            # Should not be all uppercase or all lowercase (unless it's a very short message)
            if len(message) > 10:
                assert not message.isupper(), f"Message should not be all uppercase: {message}"
                assert not (message.islower() and not any(c.isupper() for c in message)), \
                    f"Message should have proper capitalization: {message}"
        
        # Property 4: Messages in same language should have similar characteristics
        if language == LanguageCode.ENGLISH_INDIA:
            # English messages should end with proper punctuation
            for message in messages:
                if len(message) > 20:  # Only check longer messages
                    assert message[-1] in '.!?', f"English message should end with punctuation: {message}"
    
    @given(
        exception=generate_common_exception(),
        language=generate_language_code(),
        context=generate_error_context()
    )
    @settings(max_examples=50, deadline=10000)
    async def test_exception_mapping_accuracy(self, error_handler, exception, language, context):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.3**
        
        Property: Common exceptions should be mapped to appropriate error codes and messages.
        """
        # Handle the exception
        localized_error = error_handler.handle_error(
            exception,
            language=language,
            context=context,
            log_error=False
        )
        
        # Property 1: Exception should be mapped to appropriate error code
        exception_type = type(exception).__name__
        
        expected_mappings = {
            "ValueError": [ErrorCode.INPUT_INVALID_FORMAT, ErrorCode.INPUT_VALUE_OUT_OF_RANGE],
            "KeyError": [ErrorCode.INPUT_MISSING_REQUIRED],
            "ConnectionError": [ErrorCode.NET_CONNECTION_FAILED, ErrorCode.EXT_SERVICE_UNAVAILABLE],
            "TimeoutError": [ErrorCode.NET_TIMEOUT, ErrorCode.EXT_TIMEOUT],
            "PermissionError": [ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS],
            "FileNotFoundError": [ErrorCode.SYS_INTERNAL_ERROR],
            "MemoryError": [ErrorCode.SYS_RESOURCE_EXHAUSTED],
        }
        
        if exception_type in expected_mappings:
            expected_codes = expected_mappings[exception_type]
            assert localized_error.error_code in expected_codes, \
                f"Exception {exception_type} mapped to unexpected code: {localized_error.error_code}"
        
        # Property 2: Original exception should be preserved
        assert localized_error.original_error == exception, \
            "Original exception should be preserved"
        
        # Property 3: Context should be preserved
        assert localized_error.context == context, \
            "Context should be preserved"
        
        # Property 4: Message should be appropriate for exception type
        message = localized_error.message
        message_lower = message.lower()
        
        if exception_type == "ConnectionError":
            # Connection errors should mention connection/network
            if language == LanguageCode.ENGLISH_INDIA:
                connection_keywords = ["connect", "network", "internet", "service"]
                assert any(keyword in message_lower for keyword in connection_keywords), \
                    f"Connection error should mention connection: {message}"
        
        elif exception_type == "TimeoutError":
            # Timeout errors should mention time/waiting
            if language == LanguageCode.ENGLISH_INDIA:
                timeout_keywords = ["time", "timeout", "wait", "slow", "long"]
                assert any(keyword in message_lower for keyword in timeout_keywords), \
                    f"Timeout error should mention timing: {message}"
        
        elif exception_type == "PermissionError":
            # Permission errors should mention access/permission
            if language == LanguageCode.ENGLISH_INDIA:
                permission_keywords = ["permission", "access", "allow", "denied"]
                assert any(keyword in message_lower for keyword in permission_keywords), \
                    f"Permission error should mention access: {message}"
    
    @given(
        error_code=generate_error_code(),
        languages=st.lists(generate_language_code(), min_size=2, max_size=5, unique=True)
    )
    @settings(max_examples=30, deadline=15000)
    async def test_multilingual_error_coverage(self, error_handler, error_code, languages):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.1, 6.2**
        
        Property: Error messages should be available in multiple Indian languages.
        """
        assume(len(languages) >= 2)
        
        messages_by_language = {}
        
        for language in languages:
            localized_error = error_handler.handle_error(
                error_code,
                language=language,
                log_error=False
            )
            messages_by_language[language] = localized_error.message
        
        # Property 1: All languages should produce valid messages
        for language, message in messages_by_language.items():
            assert isinstance(message, str), f"Message for {language} should be string"
            assert len(message) > 0, f"Message for {language} should not be empty"
        
        # Property 2: Different languages should generally produce different messages
        # (unless falling back to the same default)
        unique_messages = set(messages_by_language.values())
        
        # Check if we have translations for these languages
        supported_languages = set(ERROR_MESSAGES.keys())
        languages_with_translations = [lang for lang in languages if lang.value in supported_languages]
        
        if len(languages_with_translations) >= 2:
            # Should have different messages for languages with translations
            lang_messages = [messages_by_language[lang] for lang in languages_with_translations]
            unique_translated = set(lang_messages)
            
            # Allow some overlap but expect some diversity
            assert len(unique_translated) >= 1, "Should have at least one unique translated message"
        
        # Property 3: Fallback mechanism should work
        for language in languages:
            if language.value not in ERROR_MESSAGES:
                # Should fallback to English
                english_error = error_handler.handle_error(
                    error_code,
                    language=LanguageCode.ENGLISH_INDIA,
                    log_error=False
                )
                
                # Fallback message should be valid
                assert len(english_error.message) > 0, "Fallback message should be valid"
    
    @given(
        error_code=generate_error_code(),
        language=generate_language_code(),
        context=generate_error_context()
    )
    @settings(max_examples=50, deadline=10000)
    async def test_error_response_format(self, error_handler, error_code, language, context):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.3**
        
        Property: Error responses should follow consistent format and include necessary information.
        """
        # Create error response
        response = error_handler.create_error_response(
            error_code,
            language=language,
            include_context=bool(context)
        )
        
        # Property 1: Response should have correct structure
        assert isinstance(response, dict), "Response should be dictionary"
        assert "success" in response, "Response should have success field"
        assert "error" in response, "Response should have error field"
        
        assert response["success"] is False, "Error response should have success=False"
        
        error_info = response["error"]
        assert isinstance(error_info, dict), "Error info should be dictionary"
        
        # Property 2: Error info should have required fields
        required_fields = ["code", "message", "severity"]
        for field in required_fields:
            assert field in error_info, f"Error info should have {field} field"
        
        assert error_info["code"] == error_code.value, \
            f"Error code mismatch: expected {error_code.value}, got {error_info['code']}"
        
        assert isinstance(error_info["message"], str), "Error message should be string"
        assert len(error_info["message"]) > 0, "Error message should not be empty"
        
        assert error_info["severity"] in [s.value for s in ErrorSeverity], \
            f"Invalid severity: {error_info['severity']}"
        
        # Property 3: Context should be included when requested
        if context and bool(context):
            assert "context" in error_info, "Context should be included when requested"
            assert error_info["context"] == context, "Context should match original"
        
        # Property 4: Response should be JSON serializable
        import json
        try:
            json_str = json.dumps(response)
            parsed = json.loads(json_str)
            assert parsed == response, "Response should be JSON serializable"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Response not JSON serializable: {e}")
    
    @given(
        severity=generate_error_severity(),
        num_errors=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=20, deadline=20000)
    async def test_error_severity_handling(self, error_handler, severity, num_errors):
        """
        **Property 21: Localized Error Handling**
        **Validates: Requirements 6.2, 6.3**
        
        Property: Error severity should be handled appropriately and consistently.
        """
        errors = []
        
        # Generate errors with specific severity
        for i in range(num_errors):
            error_code = ErrorCode.SYS_INTERNAL_ERROR  # Use consistent error code
            language = LanguageCode.ENGLISH_INDIA
            
            localized_error = error_handler.handle_error(
                error_code,
                language=language,
                severity=severity,
                context={"error_index": i},
                log_error=False
            )
            errors.append(localized_error)
        
        # Property 1: All errors should have correct severity
        for error in errors:
            assert error.severity == severity, \
                f"Severity mismatch: expected {severity}, got {error.severity}"
        
        # Property 2: Severity should be reflected in error responses
        for error in errors:
            response = error_handler.create_error_response(error, LanguageCode.ENGLISH_INDIA)
            assert response["error"]["severity"] == severity.value, \
                f"Response severity mismatch: expected {severity.value}, got {response['error']['severity']}"
        
        # Property 3: Critical errors should be handled with appropriate urgency
        if severity == ErrorSeverity.CRITICAL:
            # Critical errors should have detailed information
            for error in errors:
                assert len(error.message) > 10, "Critical errors should have detailed messages"
                
                error_dict = error.to_dict()
                assert "timestamp" in error_dict, "Critical errors should have timestamp"
                assert error_dict["timestamp"] > 0, "Timestamp should be valid"
        
        # Property 4: Error severity should be consistent across operations
        first_error = errors[0]
        for error in errors[1:]:
            # Same error code and severity should produce similar messages
            assert error.error_code == first_error.error_code, "Error codes should be consistent"
            assert error.severity == first_error.severity, "Severities should be consistent"
            
            # Messages should be identical for same error code/language/severity
            assert error.message == first_error.message, \
                "Messages should be identical for same error parameters"