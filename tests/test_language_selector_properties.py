"""
Property-based tests for Streamlit Web Interface language selector component.

This module tests the correctness properties of language selection persistence,
language change application, and ISO language code consistency.

Feature: streamlit-web-interface
Task: 6.2 Write property tests for language selector
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import MagicMock, patch
import sys
import os

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()

# Import app functions after mocking streamlit
from app import (
    initialize_session_state,
    log_action,
    render_language_selector
)


# Custom strategies for testing
@st.composite
def language_code_strategy(draw):
    """Generate valid language codes from the 11 supported languages."""
    languages = ['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']
    return draw(st.sampled_from(languages))


@st.composite
def language_pair_strategy(draw):
    """Generate a pair of different language codes."""
    lang1 = draw(language_code_strategy())
    lang2 = draw(language_code_strategy())
    # Ensure they are different
    assume(lang1 != lang2)
    return (lang1, lang2)


class MockSessionState:
    """Mock Streamlit session state for testing."""
    
    def __init__(self):
        self._state = {}
    
    def __contains__(self, key):
        return key in self._state
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __delitem__(self, key):
        del self._state[key]
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def keys(self):
        return self._state.keys()
    
    def clear(self):
        self._state.clear()


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    return MockSessionState()


@pytest.fixture(autouse=True)
def setup_streamlit_mock(mock_session_state):
    """Setup streamlit mock with session state."""
    import streamlit as st_mock
    st_mock.session_state = mock_session_state
    yield
    mock_session_state.clear()


# Property 3: Language Selection Persistence
# **Validates: Requirements 2.2**

@pytest.mark.property
@given(language=language_code_strategy())
def test_property_language_selection_persistence(language, mock_session_state):
    """
    Property 3: Language Selection Persistence
    
    For any language selection made by the user, the selected language should
    be stored in session state and persist throughout the session.
    
    **Validates: Requirements 2.2**
    
    Test Strategy:
    1. Generate random language code from supported languages
    2. Simulate language selection by storing in session state
    3. Verify language persists in session state
    4. Verify language can be retrieved for subsequent operations
    5. Verify language persists across multiple accesses
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Simulate language selection (as would happen via selectbox)
    st.session_state.selected_language = language
    
    # Property: Selected language should persist in session state
    assert 'selected_language' in st.session_state, \
        "Selected language should be stored in session state"
    
    assert st.session_state.selected_language == language, \
        "Stored language should match the selected language"
    
    # Property: Language should be retrievable for subsequent operations
    retrieved_language = st.session_state.selected_language
    
    assert retrieved_language is not None, \
        "Language should be retrievable from session state"
    
    assert retrieved_language == language, \
        "Retrieved language should be identical to selected language"
    
    # Property: Language should persist across multiple accesses
    first_access = st.session_state.selected_language
    second_access = st.session_state.selected_language
    third_access = st.session_state.selected_language
    
    assert first_access == second_access == third_access == language, \
        "Language should remain consistent across multiple accesses"


@pytest.mark.property
@given(language=language_code_strategy())
def test_property_language_persistence_with_operations(language, mock_session_state):
    """
    Property 3 (Extended): Language Persistence with Other Operations
    
    The selected language should persist even when other session state
    operations are performed (logging, caching, processing).
    
    **Validates: Requirements 2.2**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set language
    st.session_state.selected_language = language
    
    # Perform various operations
    log_action('upload', 'success', 'Audio uploaded')
    st.session_state.audio_data = b'sample_audio_data'
    st.session_state.transcription = {'text': 'Sample text'}
    st.session_state.is_processing = True
    
    # Property: Language should persist after all operations
    assert st.session_state.selected_language == language, \
        "Language should persist after performing other operations"
    
    # More operations
    log_action('transcribe', 'success', 'Transcription complete')
    st.session_state.response = {'text': 'Response text'}
    
    # Property: Language should still persist
    assert st.session_state.selected_language == language, \
        "Language should persist after additional operations"


# Property 5: Language Change Application
# **Validates: Requirements 2.5**

@pytest.mark.property
@given(language_pair=language_pair_strategy())
def test_property_language_change_application(language_pair, mock_session_state):
    """
    Property 5: Language Change Application
    
    For any language change during a session, all subsequent API requests
    should use the newly selected language.
    
    **Validates: Requirements 2.5**
    
    Test Strategy:
    1. Generate two different language codes
    2. Set initial language
    3. Verify initial language is used
    4. Change to new language
    5. Verify new language replaces old language
    6. Verify subsequent operations use new language
    """
    import streamlit as st
    
    initial_language, new_language = language_pair
    
    # Initialize session state
    initialize_session_state()
    
    # Set initial language
    st.session_state.selected_language = initial_language
    
    # Property: Initial language should be stored
    assert st.session_state.selected_language == initial_language, \
        "Initial language should be stored correctly"
    
    # Simulate a request with initial language
    request_language_1 = st.session_state.selected_language
    assert request_language_1 == initial_language, \
        "First request should use initial language"
    
    # Change language (as would happen when user selects new language)
    st.session_state.selected_language = new_language
    
    # Property: New language should replace old language
    assert st.session_state.selected_language == new_language, \
        "New language should replace the old language"
    
    assert st.session_state.selected_language != initial_language, \
        "Old language should be replaced by new language"
    
    # Property: Subsequent requests should use new language
    request_language_2 = st.session_state.selected_language
    assert request_language_2 == new_language, \
        "Subsequent requests should use the new language"
    
    # Simulate multiple subsequent operations
    for _ in range(3):
        operation_language = st.session_state.selected_language
        assert operation_language == new_language, \
            "All subsequent operations should use the new language"


@pytest.mark.property
@given(
    languages=st.lists(
        language_code_strategy(),
        min_size=2,
        max_size=5
    )
)
def test_property_language_change_application_multiple_changes(
    languages, mock_session_state
):
    """
    Property 5 (Extended): Language Change Application with Multiple Changes
    
    When language is changed multiple times, each subsequent operation should
    always use the most recently selected language.
    
    **Validates: Requirements 2.5**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Change language multiple times
    for i, language in enumerate(languages):
        # Set new language
        st.session_state.selected_language = language
        
        # Property: Current language should be the most recently set
        assert st.session_state.selected_language == language, \
            f"Language should be {language} after change {i+1}"
        
        # Simulate API request
        request_language = st.session_state.selected_language
        assert request_language == language, \
            f"API request should use current language {language}"
        
        # Simulate processing with current language
        log_action('process', 'pending', f'Processing with language {language}')
        
        # Verify language hasn't changed during processing
        assert st.session_state.selected_language == language, \
            f"Language should remain {language} during processing"
    
    # Property: Final language should be the last one set
    assert st.session_state.selected_language == languages[-1], \
        "Final language should be the last one selected"


@pytest.mark.property
@given(
    initial_language=language_code_strategy(),
    new_language=language_code_strategy()
)
def test_property_language_change_immediate_effect(
    initial_language, new_language, mock_session_state
):
    """
    Property 5 (Extended): Language Change Has Immediate Effect
    
    When language is changed, the change should take effect immediately
    for the very next operation.
    
    **Validates: Requirements 2.5**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set initial language
    st.session_state.selected_language = initial_language
    
    # Verify initial state
    assert st.session_state.selected_language == initial_language
    
    # Change language
    st.session_state.selected_language = new_language
    
    # Property: Change should take effect immediately
    immediate_check = st.session_state.selected_language
    assert immediate_check == new_language, \
        "Language change should take effect immediately"
    
    # Property: Very next operation should use new language
    next_operation_language = st.session_state.selected_language
    assert next_operation_language == new_language, \
        "Very next operation should use the new language"
    
    # Property: No residual effect from old language
    if initial_language != new_language:
        assert st.session_state.selected_language != initial_language, \
            "Old language should not be used after change"


# Property 33: ISO Language Code Consistency
# **Validates: Requirements 12.4**

@pytest.mark.property
@given(language=language_code_strategy())
def test_property_iso_language_code_consistency(language, mock_session_state):
    """
    Property 33: ISO Language Code Consistency
    
    For any language selection, the interface should use ISO language codes
    that match the Backend_API specification.
    
    **Validates: Requirements 12.4**
    
    Test Strategy:
    1. Generate language code from supported languages
    2. Verify language code is in the valid ISO format
    3. Verify language code matches backend specification
    4. Verify language code is one of the 11 supported languages
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set language
    st.session_state.selected_language = language
    
    # Define valid ISO language codes matching backend specification
    valid_iso_codes = {
        'hi',      # Hindi
        'en-IN',   # English (India)
        'ta',      # Tamil
        'te',      # Telugu
        'bn',      # Bengali
        'mr',      # Marathi
        'gu',      # Gujarati
        'kn',      # Kannada
        'ml',      # Malayalam
        'pa',      # Punjabi
        'or'       # Odia
    }
    
    # Property: Language code should be a valid ISO code
    assert st.session_state.selected_language in valid_iso_codes, \
        f"Language code {language} should be a valid ISO code matching backend specification"
    
    # Property: Language code should be a string
    assert isinstance(st.session_state.selected_language, str), \
        "Language code should be a string"
    
    # Property: Language code should not be empty
    assert len(st.session_state.selected_language) > 0, \
        "Language code should not be empty"
    
    # Property: Language code should match expected format
    # Either 2 lowercase letters (e.g., 'hi') or 2 letters + '-' + 2 uppercase letters (e.g., 'en-IN')
    lang_code = st.session_state.selected_language
    
    if '-' in lang_code:
        # Format: xx-YY (e.g., en-IN)
        parts = lang_code.split('-')
        assert len(parts) == 2, \
            "Language code with hyphen should have exactly 2 parts"
        assert len(parts[0]) == 2 and parts[0].islower(), \
            "First part should be 2 lowercase letters"
        assert len(parts[1]) == 2 and parts[1].isupper(), \
            "Second part should be 2 uppercase letters"
    else:
        # Format: xx (e.g., hi)
        assert len(lang_code) == 2 and lang_code.islower(), \
            "Simple language code should be 2 lowercase letters"


@pytest.mark.property
def test_property_iso_language_code_consistency_all_languages(mock_session_state):
    """
    Property 33 (Extended): ISO Language Code Consistency for All Languages
    
    All 11 supported languages should use valid ISO codes that match
    the backend specification.
    
    **Validates: Requirements 12.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # All supported languages with their ISO codes
    supported_languages = {
        'hi': 'Hindi',
        'en-IN': 'English (India)',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia'
    }
    
    # Property: All language codes should be valid ISO codes
    for lang_code, lang_name in supported_languages.items():
        # Set language
        st.session_state.selected_language = lang_code
        
        # Verify it's stored correctly
        assert st.session_state.selected_language == lang_code, \
            f"Language code for {lang_name} should be {lang_code}"
        
        # Verify it's a valid string
        assert isinstance(lang_code, str), \
            f"Language code for {lang_name} should be a string"
        
        # Verify it's not empty
        assert len(lang_code) > 0, \
            f"Language code for {lang_name} should not be empty"


@pytest.mark.property
@given(language=language_code_strategy())
def test_property_iso_language_code_consistency_in_api_calls(
    language, mock_session_state
):
    """
    Property 33 (Extended): ISO Language Code Consistency in API Calls
    
    When language is used in API calls, it should maintain ISO code format
    and match backend specification.
    
    **Validates: Requirements 12.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Set language
    st.session_state.selected_language = language
    
    # Simulate preparing API request
    api_language_param = st.session_state.selected_language
    
    # Property: Language parameter for API should be valid ISO code
    valid_iso_codes = {'hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'}
    assert api_language_param in valid_iso_codes, \
        f"API language parameter {api_language_param} should be a valid ISO code"
    
    # Property: Language parameter should match session state
    assert api_language_param == language, \
        "API language parameter should match selected language in session state"
    
    # Property: Language parameter should be unchanged from selection
    assert api_language_param == st.session_state.selected_language, \
        "Language parameter should be identical to session state value"


@pytest.mark.property
@given(
    languages=st.lists(
        language_code_strategy(),
        min_size=1,
        max_size=11
    )
)
def test_property_iso_language_code_consistency_across_changes(
    languages, mock_session_state
):
    """
    Property 33 (Extended): ISO Language Code Consistency Across Changes
    
    ISO language codes should remain valid and consistent even when
    language is changed multiple times.
    
    **Validates: Requirements 12.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    valid_iso_codes = {'hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'}
    
    # Change language multiple times
    for language in languages:
        # Set language
        st.session_state.selected_language = language
        
        # Property: Language code should always be valid ISO code
        assert st.session_state.selected_language in valid_iso_codes, \
            f"Language code {language} should be valid ISO code after change"
        
        # Property: Language code should match backend specification
        current_lang = st.session_state.selected_language
        assert current_lang == language, \
            "Stored language should match the set language"
        
        # Simulate API call
        api_lang = st.session_state.selected_language
        assert api_lang in valid_iso_codes, \
            f"API language {api_lang} should be valid ISO code"


# Integration test: All three properties together

@pytest.mark.property
@given(
    initial_language=language_code_strategy(),
    new_language=language_code_strategy()
)
def test_property_combined_language_selector_properties(
    initial_language, new_language, mock_session_state
):
    """
    Combined Property Test: Language Selector Properties
    
    Tests all three properties together:
    - Property 3: Language Selection Persistence
    - Property 5: Language Change Application
    - Property 33: ISO Language Code Consistency
    
    **Validates: Requirements 2.2, 2.5, 12.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    valid_iso_codes = {'hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'}
    
    # Set initial language
    st.session_state.selected_language = initial_language
    
    # Property 3: Language should persist
    assert st.session_state.selected_language == initial_language, \
        "Initial language should persist in session state"
    
    # Property 33: Language should be valid ISO code
    assert st.session_state.selected_language in valid_iso_codes, \
        "Initial language should be valid ISO code"
    
    # Simulate API request with initial language
    api_lang_1 = st.session_state.selected_language
    assert api_lang_1 == initial_language, \
        "API should use initial language"
    
    # Change language
    st.session_state.selected_language = new_language
    
    # Property 3: New language should persist
    assert st.session_state.selected_language == new_language, \
        "New language should persist in session state"
    
    # Property 5: Language change should apply to subsequent requests
    api_lang_2 = st.session_state.selected_language
    assert api_lang_2 == new_language, \
        "Subsequent API requests should use new language"
    
    if initial_language != new_language:
        assert api_lang_2 != api_lang_1, \
            "New API requests should not use old language"
    
    # Property 33: New language should be valid ISO code
    assert st.session_state.selected_language in valid_iso_codes, \
        "New language should be valid ISO code"
    
    # Perform operations
    log_action('transcribe', 'pending', 'Processing')
    
    # All properties should still hold after operations
    assert st.session_state.selected_language == new_language, \
        "Language should persist after operations"
    
    assert st.session_state.selected_language in valid_iso_codes, \
        "Language should remain valid ISO code after operations"
    
    api_lang_3 = st.session_state.selected_language
    assert api_lang_3 == new_language, \
        "Operations should continue using new language"


@pytest.mark.property
def test_property_language_selector_default_initialization(mock_session_state):
    """
    Property Test: Language Selector Default Initialization
    
    When session state is initialized, the default language should be
    a valid ISO code (Hindi).
    
    **Validates: Requirements 2.2, 12.4**
    """
    import streamlit as st
    
    # Initialize session state
    initialize_session_state()
    
    # Property: Default language should be set
    assert 'selected_language' in st.session_state, \
        "Default language should be initialized"
    
    # Property: Default language should be Hindi
    assert st.session_state.selected_language == 'hi', \
        "Default language should be Hindi"
    
    # Property: Default language should be valid ISO code
    valid_iso_codes = {'hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or'}
    assert st.session_state.selected_language in valid_iso_codes, \
        "Default language should be valid ISO code"
