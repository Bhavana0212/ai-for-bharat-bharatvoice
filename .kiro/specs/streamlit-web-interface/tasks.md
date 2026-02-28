# Implementation Plan: Streamlit Web Interface

## Overview

This plan implements a single-file Streamlit web application (`app.py`) that provides a browser-based interface for the BharatVoice AI system. The implementation follows an incremental approach, building core functionality first, then adding UI components, API integration, error handling, and deployment configuration.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `app.py` in project root
  - Create `requirements.txt` with dependencies: streamlit, requests, audio-recorder-streamlit, python-dotenv
  - Create `.streamlit/config.toml` for Streamlit configuration
  - Create `.env` file for environment variables (BACKEND_URL, DEBUG, CACHE_TTL, REQUEST_TIMEOUT)
  - _Requirements: 11.1, 11.2, 11.3_

- [x] 2. Implement core session state management and initialization
  - [x] 2.1 Create session state initialization function
    - Initialize all session state variables (selected_language, audio_data, transcription, response, tts_audio, action_history, cache, is_processing, is_online)
    - Set default values for user preferences
    - _Requirements: 2.2, 6.1, 7.1_
  
  - [x] 2.2 Create action logging functions
    - Implement `log_action()` to record user interactions with timestamp, type, status, and details
    - Implement action history size limiting (keep last 50 actions)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 2.3 Create cache management functions
    - Implement `cache_response()` with TTL support
    - Implement `get_cached_response()` with expiration checking
    - Implement `clear_cache()` function
    - _Requirements: 7.5_

- [x] 2.4 Write property tests for session state management
  - **Property 2: Recording State Persistence** - Test that audio data persists in session state after recording stops
  - **Property 3: Language Selection Persistence** - Test that selected language persists throughout session
  - **Validates: Requirements 1.4, 2.2**

- [x] 3. Implement API client layer
  - [x] 3.1 Create BharatVoiceAPIClient class
    - Initialize with base_url and timeout from environment variables
    - Create requests session for connection pooling
    - _Requirements: 11.2, 12.5_
  
  - [x] 3.2 Implement speech recognition API method
    - Create `recognize_speech()` method with multipart/form-data encoding
    - Send audio file and language parameters to `/api/voice/recognize` endpoint
    - Parse and return transcription response
    - _Requirements: 3.1, 12.1, 12.2_
  
  - [x] 3.3 Implement response generation API method
    - Create `generate_response()` method with JSON payload
    - Send transcribed text and language to response generation endpoint
    - Parse and return AI response
    - _Requirements: 4.1, 12.2_
  
  - [x] 3.4 Implement text-to-speech API method
    - Create `synthesize_speech()` method with JSON payload
    - Send text and language to `/api/voice/synthesize` endpoint
    - Fetch and return audio file from audio_url
    - Handle base64 decoding if needed
    - _Requirements: 5.1, 12.3_
  
  - [x] 3.5 Implement health check method
    - Create `check_health()` method to ping `/api/health` endpoint
    - Return boolean indicating backend availability
    - _Requirements: 7.1_

- [x] 3.6 Write property tests for API client
  - **Property 6: Speech Recognition API Integration** - Test that API calls include correct audio and language data
  - **Property 4: Language Propagation to API** - Test that selected language is included in all API requests
  - **Property 31: JSON Response Validation** - Test that JSON responses are validated before use
  - **Validates: Requirements 3.1, 2.3, 12.1, 12.2**

- [x] 4. Implement error handling and retry logic
  - [x] 4.1 Create error handling functions
    - Implement `handle_network_error()` for timeout and connection errors
    - Implement `handle_validation_error()` for input validation errors
    - Implement `handle_api_error()` for HTTP error responses with status code mapping
    - _Requirements: 10.1, 10.2, 10.3, 10.5_
  
  - [x] 4.2 Implement retry logic with exponential backoff
    - Create `retry_with_backoff()` function with configurable max retries and delay
    - Create `process_with_retry()` wrapper for API operations
    - _Requirements: 10.1, 10.4_
  
  - [x] 4.3 Implement response parsing functions
    - Create `parse_transcription_response()` to extract transcription data
    - Create `parse_error_response()` to map technical errors to user-friendly messages
    - _Requirements: 3.2, 10.3, 12.2_

- [x] 4.4 Write property tests for error handling
  - **Property 9: Error Message Display** - Test that errors always display user-friendly messages
  - **Property 13: Retry Option on Failure** - Test that retry option is provided on failures
  - **Property 28: Timeout Handling** - Test that timeouts provide cancel and retry options
  - **Validates: Requirements 3.4, 4.4, 10.1, 10.4**

- [x] 5. Checkpoint - Ensure core infrastructure is working
  - Verify session state initialization works correctly
  - Verify API client can be instantiated with environment variables
  - Verify error handling functions work with mock errors
  - Ask the user if questions arise

- [x] 6. Implement UI components - Language Selector
  - [x] 6.1 Create `render_language_selector()` function
    - Display selectbox with all 11 supported languages (hi, en-IN, ta, te, bn, mr, gu, kn, ml, pa, or)
    - Show language names in native scripts with English translations
    - Store selection in `st.session_state.selected_language`
    - _Requirements: 2.1, 2.2, 9.2_

- [x] 6.2 Write property tests for language selector
  - **Property 3: Language Selection Persistence** - Test language persists in session state
  - **Property 5: Language Change Application** - Test that language changes apply to subsequent requests
  - **Property 33: ISO Language Code Consistency** - Test that ISO codes match backend specification
  - **Validates: Requirements 2.2, 2.5, 12.4**

- [x] 7. Implement UI components - Audio Input
  - [x] 7.1 Create `render_audio_uploader()` function
    - Display file uploader accepting WAV, MP3, M4A, OGG formats
    - Validate file size (max 10MB) and display error if exceeded
    - Store audio data in `st.session_state.audio_data`
    - Log upload action to action history
    - _Requirements: 1.1, 1.5, 6.1, 9.3_
  
  - [x] 7.2 Create `render_voice_recorder()` function
    - Use `audio_recorder_streamlit` component for browser recording
    - Configure with 16kHz sample rate and 2-second pause threshold
    - Display recording status indicator
    - Store recorded audio in `st.session_state.audio_data`
    - Log recording action to action history
    - _Requirements: 1.2, 1.3, 1.4, 6.1, 9.3_

- [x] 7.3 Write property tests for audio input
  - **Property 1: Audio File Size Validation** - Test that files over 10MB are rejected
  - **Property 2: Recording State Persistence** - Test that recorded audio persists in session state
  - **Validates: Requirements 1.5, 1.4**

- [-] 8. Implement UI components - Display Components
  - [x] 8.1 Create `render_transcription_display()` function
    - Display transcribed text in info box
    - Show confidence score, detected language, and processing time in columns
    - Only render when transcription exists in session state
    - _Requirements: 3.2, 9.4_
  
  - [x] 8.2 Create `render_response_display()` function
    - Display AI response text in success box
    - Show suggested actions as buttons if available
    - Only render when response exists in session state
    - _Requirements: 4.2, 9.4_
  
  - [x] 8.3 Create `render_audio_player()` function
    - Display audio player widget with TTS audio
    - Handle base64 decoding if audio is encoded
    - Support auto-play option based on user preference
    - Only render when TTS audio exists in session state
    - _Requirements: 5.2, 5.3, 9.4_
  
  - [x] 8.4 Create `render_action_log()` function
    - Display in sidebar with expandable entries
    - Show most recent 10 actions in reverse chronological order
    - Display timestamp, type, status, and details for each action
    - _Requirements: 6.5, 9.5_

- [ ] 8.5 Write property tests for display components
  - **Property 7: Transcription Display** - Test that transcription results are displayed correctly
  - **Property 12: Response Display** - Test that AI responses are displayed correctly
  - **Property 15: Audio Player Display** - Test that audio player appears when TTS audio is available
  - **Property 17: Action Log Display Limit** - Test that only 10 most recent actions are shown
  - **Validates: Requirements 3.2, 4.2, 5.2, 6.5**

- [-] 9. Implement UI components - Progress Indicators
  - [x] 9.1 Create `render_progress_indicator()` function
    - Display spinner with operation message
    - Show progress bar if progress percentage is available
    - Display elapsed time for operations over 3 seconds
    - _Requirements: 8.1, 8.2_
  
  - [x] 9.2 Create status message display functions
    - Implement success message display with 2-second auto-dismiss
    - Implement error message display with details
    - Implement warning message display for non-critical issues
    - _Requirements: 8.3, 8.4_

- [ ] 9.3 Write property tests for progress indicators
  - **Property 8: Progress Indicator During Processing** - Test that loading state displays during operations
  - **Property 22: Extended Processing Time Feedback** - Test that operations over 3 seconds show time estimate
  - **Property 23: Success Message Display** - Test that success messages display for 2 seconds
  - **Property 24: Detailed Error Messages** - Test that error messages include details
  - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

- [x] 10. Checkpoint - Ensure all UI components render correctly
  - Verify language selector displays all 11 languages
  - Verify audio uploader and recorder components work
  - Verify display components show data from session state
  - Verify action log displays in sidebar
  - Ask the user if questions arise

- [x] 11. Implement offline detection and caching
  - [x] 11.1 Create connection monitoring functions
    - Implement `check_backend_health()` to ping backend health endpoint
    - Implement `update_connection_status()` to update `is_online` state
    - Implement `monitor_connection()` to check every 30 seconds
    - _Requirements: 7.1, 7.3_
  
  - [x] 11.2 Create offline mode UI components
    - Implement `render_offline_indicator()` to show offline status warning
    - Display list of disabled features in offline mode
    - Display list of available features in offline mode
    - _Requirements: 7.1, 7.2_
  
  - [x] 11.3 Create offline queue management
    - Implement `queue_for_offline_processing()` to queue operations
    - Implement `process_offline_queue()` to process when connection restored
    - Store queue in session state
    - _Requirements: 7.3_
  
  - [x] 11.4 Implement cache-aware processing
    - Create `process_with_cache()` wrapper function
    - Check cache before making API calls
    - Display cache indicator when showing cached responses
    - _Requirements: 7.5_

- [x] 11.5 Write property tests for offline functionality
  - **Property 18: Offline Detection** - Test that connection failures are detected
  - **Property 19: Feature Disabling in Offline Mode** - Test that online features are disabled when offline
  - **Property 20: Feature Re-enabling on Connection Restore** - Test that features re-enable when connection restored
  - **Property 21: Cache Indicator Display** - Test that cached responses show cache indicator
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.5**

- [x] 12. Implement main processing workflow
  - [x] 12.1 Create audio processing orchestration function
    - Implement `process_audio()` to coordinate transcription, response generation, and TTS
    - Set `is_processing` flag during operations
    - Track operation start time for progress indicators
    - Handle errors at each step with appropriate recovery
    - _Requirements: 3.1, 4.1, 5.1_
  
  - [x] 12.2 Implement transcription processing
    - Call API client `recognize_speech()` method
    - Parse response and store in session state
    - Display progress indicator during processing
    - Log action to action history
    - Handle errors and provide retry option
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 12.3 Implement response generation processing
    - Automatically trigger after successful transcription
    - Call API client `generate_response()` method
    - Parse response and store in session state
    - Display progress indicator during processing
    - Log action to action history
    - Handle errors and provide retry option
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 12.4 Implement TTS processing
    - Automatically trigger after successful response generation
    - Call API client `synthesize_speech()` method
    - Store audio in session state
    - Display progress indicator during processing
    - Log action to action history
    - Gracefully degrade to text-only if TTS fails
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

- [x] 12.5 Write property tests for processing workflow
  - **Property 11: Automatic Response Generation** - Test that response generation triggers after transcription
  - **Property 14: Automatic TTS Request** - Test that TTS triggers after response generation
  - **Property 16: Graceful TTS Degradation** - Test that text displays even if TTS fails
  - **Property 10: Action Logging Completeness** - Test that all operations are logged
  - **Validates: Requirements 4.1, 5.1, 5.5, 6.1, 6.2, 6.3, 6.4**

- [x] 13. Implement main application layout and entry point
  - [x] 13.1 Create main application function
    - Set page configuration (title, icon, layout)
    - Initialize session state
    - Monitor connection status
    - Render offline indicator if needed
    - _Requirements: 9.1, 11.2_
  
  - [x] 13.2 Implement UI layout structure
    - Display app title and description
    - Render language selector at top
    - Render audio input components (uploader and recorder) in columns
    - Add "Process Audio" button to trigger workflow
    - Render transcription, response, and audio player in sequence
    - Render action log in sidebar
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [x] 13.3 Wire process button to workflow
    - Check if audio data exists in session state
    - Validate audio data before processing
    - Call `process_audio()` orchestration function
    - Handle processing state and disable button during processing
    - _Requirements: 1.1, 1.2, 3.1_

- [x] 13.4 Write integration tests for main workflow
  - **Property 34: API Data Round-Trip Consistency** - Test that data structures remain consistent through request-response cycles
  - **Property 32: Base64 Audio Decoding** - Test that base64 audio is correctly decoded
  - **Validates: Requirements 12.3, 12.5**

- [x] 14. Checkpoint - Ensure complete workflow functions end-to-end
  - Test audio upload → transcription → response → TTS workflow
  - Test audio recording → transcription → response → TTS workflow
  - Test error handling at each step
  - Test offline mode transitions
  - Ask the user if questions arise

- [-] 15. Add configuration and environment setup
  - [x] 15.1 Create Streamlit configuration file
    - Create `.streamlit/config.toml` with server, browser, and theme settings
    - Configure CORS and XSRF protection
    - Set custom theme colors
    - _Requirements: 11.2_
  
  - [x] 15.2 Create environment configuration
    - Create `.env.example` with all required variables
    - Document each environment variable with comments
    - Set sensible defaults for local development
    - _Requirements: 11.2, 11.3_
  
  - [x] 15.3 Add configuration loading logic
    - Load environment variables using python-dotenv
    - Validate required configuration on startup
    - Display configuration errors clearly
    - _Requirements: 11.2_

- [x] 16. Add deployment configuration
  - [x] 16.1 Create AWS Amplify configuration
    - Create `amplify.yml` with build phases and artifact configuration
    - Configure cache paths for Streamlit cache directory
    - _Requirements: 11.4_
  
  - [x] 16.2 Create Docker configuration (optional)
    - Create `Dockerfile` with Python 3.9 base image
    - Create `docker-compose.yml` for local testing with backend
    - Add health check configuration
    - _Requirements: 11.4_
  
  - [x] 16.3 Add deployment documentation
    - Document local development setup steps
    - Document AWS Amplify deployment steps
    - Document Docker deployment steps
    - Document environment variable configuration for production
    - _Requirements: 11.1, 11.3, 11.4, 11.5_

- [-] 17. Add logging and monitoring
  - [x] 17.1 Implement application logging
    - Configure Python logging with file and console handlers
    - Log important events (startup, API calls, errors)
    - Add log rotation for production
    - _Requirements: 10.5_
  
  - [x] 17.2 Implement metrics tracking
    - Create `track_api_call()` function to record API metrics
    - Store metrics in session state for display
    - Log metrics to file for analysis
    - _Requirements: 8.1, 8.2_
  
  - [x] 17.3 Add debug mode
    - Create debug information panel in sidebar when DEBUG=true
    - Display session state, backend URL, and connection status
    - Add cache inspection tools
    - _Requirements: 11.2_

- [ ] 17.4 Write property tests for logging
  - **Property 10: Action Logging Completeness** - Test that all user interactions are logged
  - **Property 29: Critical Error Logging and Reporting** - Test that critical errors are logged with details
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 10.5**

- [-] 18. Add input validation and security
  - [x] 18.1 Implement audio file validation
    - Validate file format against allowed types
    - Validate file size against 10MB limit
    - Validate audio file integrity
    - Display helpful error messages for validation failures
    - _Requirements: 1.1, 1.5, 10.2_
  
  - [x] 18.2 Implement input sanitization
    - Sanitize file names before storage
    - Validate language codes against allowed list
    - Sanitize text inputs if any
    - _Requirements: 12.4_
  
  - [x] 18.3 Add security headers and configuration
    - Enable XSRF protection in Streamlit config
    - Configure secure session handling
    - Document security best practices
    - _Requirements: 11.2_

- [ ] 18.4 Write property tests for validation
  - **Property 1: Audio File Size Validation** - Test file size validation with various sizes
  - **Property 25: Network Error Handling** - Test network error handling with user-friendly messages
  - **Property 26: Invalid Audio File Feedback** - Test invalid file handling with helpful feedback
  - **Property 27: Backend Error Message Localization** - Test error message localization
  - **Validates: Requirements 1.5, 10.1, 10.2, 10.3**

- [x] 19. Final integration and polish
  - [x] 19.1 Add bilingual UI labels
    - Add Hindi translations for all UI labels
    - Use format "English / हिन्दी" for all labels
    - Ensure proper Unicode rendering
    - _Requirements: 2.1, 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [x] 19.2 Optimize performance
    - Add caching decorators for expensive operations
    - Optimize session state updates
    - Minimize unnecessary reruns
    - _Requirements: 7.5_
  
  - [x] 19.3 Add user experience enhancements
    - Add keyboard shortcuts for common actions
    - Add tooltips for UI components
    - Improve loading state transitions
    - Add confirmation dialogs for destructive actions
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 20. Final checkpoint - Complete end-to-end testing
  - Test complete workflow with all 11 languages
  - Test error handling for all error types
  - Test offline mode and cache functionality
  - Test deployment configuration locally
  - Verify all requirements are met
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- The implementation is designed to be incremental, with checkpoints to validate progress
- All code should be in a single `app.py` file for simplified deployment
- Property tests validate universal correctness properties across all inputs
- Unit tests (not listed here) should be added for specific examples and edge cases
- The workflow automatically triggers response generation and TTS after transcription
- Offline mode gracefully degrades functionality when backend is unavailable
- All UI labels should be bilingual (English/Hindi) for accessibility
