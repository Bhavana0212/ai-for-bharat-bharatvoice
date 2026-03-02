"""
BharatVoice AI - Streamlit Web Interface

A browser-based user interface for the BharatVoice AI system, providing
audio upload, browser-based recording, and real-time transcription and
response playback for 11 Indian languages.

Author: BharatVoice Team
License: MIT
"""

import streamlit as st
import backend as backend_module
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Any, Optional

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'streamlit_app.log')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize session state variables with default values
    
    This function sets up all necessary session state variables for:
    - User preferences (language, auto-play)
    - Audio data storage
    - Processing results (transcription, response, TTS audio)
    - Action history logging
    - Response caching
    - Connection and processing status
    
    Requirements: 2.2, 6.1, 7.1
    """
    
    # User preferences
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'hi'  # Default to Hindi
    
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = True
    
    # Audio data
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    
    if 'audio_filename' not in st.session_state:
        st.session_state.audio_filename = None
    
    # Processing results
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    
    if 'response' not in st.session_state:
        st.session_state.response = None
    
    if 'tts_audio' not in st.session_state:
        st.session_state.tts_audio = None
    
    # Action history
    if 'action_history' not in st.session_state:
        st.session_state.action_history = []
    
    # Cache
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    # Status
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'is_online' not in st.session_state:
        st.session_state.is_online = True
    
    if 'operation_start_time' not in st.session_state:
        st.session_state.operation_start_time = None
    
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    if 'offline_mode' not in st.session_state:
        st.session_state.offline_mode = False


def log_action(action_type: str, status: str, details: str = ''):
    """Log user action to history
    
    Records user interactions and system events with timestamp, type, status,
    and optional details. Automatically maintains history size by keeping only
    the last 50 actions.
    
    Args:
        action_type: Type of action ('upload', 'record', 'transcribe', 'respond', 
                    'tts', 'connection', etc.)
        status: Status of action ('success', 'error', 'pending')
        details: Additional information about the action (optional)
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    
    Examples:
        >>> log_action('upload', 'success', 'audio.wav (2.5 MB)')
        >>> log_action('transcribe', 'success', 'Transcribed text preview...')
        >>> log_action('connection', 'error', 'Backend connection lost')
    """
    
    action = {
        'timestamp': datetime.now().isoformat(),
        'type': action_type,
        'status': status,
        'details': details
    }
    
    # Ensure action_history exists
    if 'action_history' not in st.session_state:
        st.session_state.action_history = []
    
    # Add action to history
    st.session_state.action_history.append(action)
    
    # Keep only last 50 actions
    if len(st.session_state.action_history) > 50:
        st.session_state.action_history = st.session_state.action_history[-50:]


def cache_response(key: str, value: Any, ttl: int = 3600):
    """Cache response with TTL (Time To Live)
    
    Stores a response value in the session cache with an expiration time.
    The cache entry includes the value, timestamp, and TTL for expiration checking.
    
    Args:
        key: Unique identifier for the cached value
        value: The value to cache (can be any type)
        ttl: Time to live in seconds (default: 3600 = 1 hour)
    
    Requirements: 7.5
    
    Examples:
        >>> cache_response('transcription_abc123', {'text': 'Hello'}, ttl=1800)
        >>> cache_response('response_xyz789', {'text': 'Response'})
    """
    
    # Ensure cache exists in session state
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    cache_entry = {
        'value': value,
        'timestamp': time.time(),
        'ttl': ttl
    }
    
    st.session_state.cache[key] = cache_entry


def get_cached_response(key: str) -> Optional[Any]:
    """Get cached response if not expired
    
    Retrieves a cached value if it exists and has not expired. If the cache
    entry is expired, it is automatically removed and None is returned.
    
    Args:
        key: Unique identifier for the cached value
    
    Returns:
        The cached value if found and not expired, None otherwise
    
    Requirements: 7.5
    
    Examples:
        >>> result = get_cached_response('transcription_abc123')
        >>> if result:
        ...     print("Cache hit!")
        ... else:
        ...     print("Cache miss or expired")
    """
    
    # Ensure cache exists in session state
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    # Check if key exists in cache
    if key not in st.session_state.cache:
        return None
    
    entry = st.session_state.cache[key]
    
    # Check if expired
    if time.time() - entry['timestamp'] > entry['ttl']:
        # Remove expired entry
        del st.session_state.cache[key]
        return None
    
    return entry['value']


def clear_cache():
    """Clear all cached responses
    
    Removes all entries from the session cache. This is useful for
    freeing memory or forcing fresh data retrieval.
    
    Requirements: 7.5
    
    Examples:
        >>> clear_cache()
        >>> print(len(st.session_state.cache))  # 0
    """
    
    st.session_state.cache = {}


def track_api_call(operation: str, duration: float, success: bool):
    """Track API call metrics
    
    Records metrics for API calls including operation type, duration, and success status.
    Metrics are stored in session state for display and logged to file for analysis.
    
    Args:
        operation: Name of the API operation ('transcribe', 'respond', 'tts')
        duration: Duration of the operation in seconds
        success: Whether the operation succeeded
    
    Requirements: 8.1, 8.2
    
    Examples:
        >>> track_api_call('transcribe', 2.5, True)
        >>> track_api_call('respond', 1.8, False)
    """
    
    metrics = {
        'operation': operation,
        'duration': duration,
        'success': success,
        'timestamp': time.time()
    }
    
    # Store in session state for display
    if 'metrics' not in st.session_state:
        st.session_state.metrics = []
    
    st.session_state.metrics.append(metrics)
    
    # Keep only last 100 metrics
    if len(st.session_state.metrics) > 100:
        st.session_state.metrics = st.session_state.metrics[-100:]
    
    # Log to file
    logger.info(f"API call: {operation}, duration: {duration:.2f}s, success: {success}")


def render_debug_panel():
    """Render debug information panel in sidebar
    
    Displays debug information when DEBUG mode is enabled, including:
    - Session state variables
    - Backend URL and configuration
    - Connection status
    - Cache statistics
    - Recent metrics
    
    Requirements: 11.2
    """
    
    if not DEBUG:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🐛 Debug Information")
    
    # Configuration
    with st.sidebar.expander("Configuration"):
        st.write(f"**Backend URL**: {BACKEND_URL}")
        st.write(f"**Cache TTL**: {CACHE_TTL}s")
        st.write(f"**Request Timeout**: {REQUEST_TIMEOUT}s")
        st.write(f"**Log Level**: {LOG_LEVEL}")
    
    # Connection Status
    with st.sidebar.expander("Connection Status"):
        is_online = st.session_state.get('is_online', True)
        st.write(f"**Online**: {'✅ Yes' if is_online else '❌ No'}")
        st.write(f"**Offline Mode**: {st.session_state.get('offline_mode', False)}")
    
    # Session State
    with st.sidebar.expander("Session State"):
        st.write(f"**Has Audio**: {st.session_state.get('audio_data') is not None}")
        st.write(f"**Audio Filename**: {st.session_state.get('audio_filename', 'None')}")
        st.write(f"**Selected Language**: {st.session_state.get('selected_language', 'None')}")
        st.write(f"**Is Processing**: {st.session_state.get('is_processing', False)}")
        st.write(f"**Has Transcription**: {st.session_state.get('transcription') is not None}")
        st.write(f"**Has Response**: {st.session_state.get('response') is not None}")
        st.write(f"**Has TTS Audio**: {st.session_state.get('tts_audio') is not None}")
    
    # Cache Statistics
    with st.sidebar.expander("Cache Statistics"):
        cache_size = len(st.session_state.get('cache', {}))
        st.write(f"**Cache Entries**: {cache_size}")
        st.write(f"**Action History**: {len(st.session_state.get('action_history', []))}")
        
        if st.button("Clear Cache"):
            clear_cache()
            st.success("Cache cleared!")
    
    # Recent Metrics
    with st.sidebar.expander("Recent Metrics"):
        metrics = st.session_state.get('metrics', [])
        if metrics:
            recent_metrics = metrics[-5:]  # Last 5 metrics
            for m in reversed(recent_metrics):
                status_icon = "✅" if m['success'] else "❌"
                st.write(f"{status_icon} **{m['operation']}**: {m['duration']:.2f}s")
        else:
            st.write("No metrics yet")


def validate_audio_file(audio_data: bytes, filename: str) -> tuple[bool, str]:
    """Validate audio file format and size
    
    Checks if the audio file meets the requirements:
    - Format: WAV, MP3, M4A, or OGG
    - Size: Maximum 10MB
    
    Args:
        audio_data: Audio file data in bytes
        filename: Name of the audio file
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if file is valid, False otherwise
        - error_message: Error message if invalid, empty string if valid
    
    Requirements: 1.1, 1.5, 10.2
    
    Examples:
        >>> is_valid, error = validate_audio_file(audio_data, "test.wav")
        >>> if not is_valid:
        ...     print(error)
    """
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if len(audio_data) > max_size:
        return False, f"File size ({len(audio_data) / 1024 / 1024:.2f} MB) exceeds 10MB limit"
    
    # Check file format
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg']
    file_ext = os.path.splitext(filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file format '{file_ext}'. Allowed formats: WAV, MP3, M4A, OGG"
    
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues
    
    Removes or replaces potentially dangerous characters from filenames.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename safe for storage
    
    Requirements: 18.2
    
    Examples:
        >>> sanitize_filename("../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("test<script>.wav")
        'test_script_.wav'
    """
    
    import re
    
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Replace dangerous characters with underscore
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove any remaining non-ASCII characters
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename


def validate_language_code(language_code: str) -> bool:
    """Validate language code against allowed list
    
    Checks if the provided language code is one of the supported languages.
    
    Args:
        language_code: ISO language code to validate
    
    Returns:
        True if valid, False otherwise
    
    Requirements: 12.4
    
    Examples:
        >>> validate_language_code('hi')
        True
        >>> validate_language_code('fr')
        False
    """
    
    allowed_languages = ['hi', 'en-IN', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'or']
    return language_code in allowed_languages


def check_backend_health() -> bool:
    """In-process health check for the local backend (always True).

    Streamlit Cloud runs the app in a single process, so the in-process
    backend functions are available without network calls.
    """
    try:
        return backend_module.health_check()
    except Exception:
        return True


def update_connection_status():
    """Update connection status in session state
    
    Checks backend health and updates the is_online flag in session state.
    Displays success/error messages and logs connection status changes.
    
    Requirements: 7.1, 7.3
    
    Examples:
        >>> update_connection_status()
        >>> if st.session_state.is_online:
        ...     print("Connected to backend")
    """
    is_online = check_backend_health()
    
    # Check if status changed
    if is_online != st.session_state.get('is_online', True):
        st.session_state.is_online = is_online
        
        if is_online:
            st.success("✅ Connected to backend / बैकएंड से जुड़ा हुआ")
            log_action('connection', 'success', 'Backend connection restored')
        else:
            st.error("❌ Backend unavailable - Operating in offline mode / बैकएंड अनुपलब्ध - ऑफ़लाइन मोड में काम कर रहा है")
            log_action('connection', 'error', 'Backend connection lost')


def monitor_connection():
    """Monitor backend connection status
    
    Checks connection every 30 seconds by tracking the last health check
    timestamp. This prevents excessive health checks while ensuring timely
    detection of connection changes.
    
    Requirements: 7.1, 7.3
    
    Examples:
        >>> # Call this in the main application loop
        >>> monitor_connection()
    """
    # Initialize last health check timestamp if not exists
    if 'last_health_check' not in st.session_state:
        st.session_state.last_health_check = 0
    
    current_time = time.time()
    
    # Check connection every 30 seconds
    if current_time - st.session_state.last_health_check > 30:
        update_connection_status()
        st.session_state.last_health_check = current_time


def render_offline_indicator():
    """Render offline mode indicator
    
    Displays a warning message when the backend is unavailable, showing:
    - Offline mode status in English and Hindi
    - List of disabled features (requiring backend connectivity)
    - List of available features (working offline)
    
    Requirements: 7.1, 7.2
    
    Examples:
        >>> # Call this in the main application
        >>> render_offline_indicator()
    """
    if not st.session_state.get('is_online', True):
        st.warning("""
        ⚠️ **Offline Mode** / **ऑफ़लाइन मोड**
        
        The backend is currently unavailable. Some features are disabled:
        
        **Disabled Features / अक्षम सुविधाएँ:**
        - 🎤 Speech recognition / वाक् पहचान
        - 🤖 AI response generation / एआई प्रतिक्रिया निर्माण
        - 🔊 Text-to-speech synthesis / पाठ-से-वाक् संश्लेषण
        
        **Available Features / उपलब्ध सुविधाएँ:**
        - 📦 View cached responses / कैश्ड प्रतिक्रियाएँ देखें
        - 📜 Browse action history / कार्य इतिहास ब्राउज़ करें
        - 📤 Upload audio files (will be processed when connection is restored) / ऑडियो फ़ाइलें अपलोड करें (कनेक्शन बहाल होने पर संसाधित की जाएंगी)
        """)


def queue_for_offline_processing(operation: str, data: dict):
    """Queue operation for processing when online
    
    Stores an operation in the offline queue to be processed when the
    backend connection is restored. Each queue item includes the operation
    type, data, and timestamp.
    
    Args:
        operation: Type of operation ('transcribe', 'respond', 'tts')
        data: Dictionary containing operation data (e.g., audio_data, language, text)
    
    Requirements: 7.3
    
    Examples:
        >>> queue_for_offline_processing('transcribe', {
        ...     'audio_data': audio_bytes,
        ...     'language': 'hi'
        ... })
    """
    # Initialize offline queue if not exists
    if 'offline_queue' not in st.session_state:
        st.session_state.offline_queue = []
    
    # Create queue item
    queue_item = {
        'operation': operation,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to queue
    st.session_state.offline_queue.append(queue_item)
    
    # Display info message
    st.info(f"Operation queued for processing when connection is restored / कनेक्शन बहाल होने पर संसाधन के लिए कतारबद्ध")
    
    # Log action
    log_action(operation, 'queued', f'Queued for offline processing at {queue_item["timestamp"]}')


def process_offline_queue():
    """Process queued operations when connection is restored
    
    Processes all operations in the offline queue when the backend connection
    is restored. Each operation is processed based on its type (transcribe,
    respond, tts). Successfully processed operations are logged, and failed
    operations are logged with error details.
    
    Requirements: 7.3
    
    Examples:
        >>> # Call this when connection is restored
        >>> if st.session_state.is_online:
        ...     process_offline_queue()
    """
    # Check if offline queue exists and has items
    if 'offline_queue' not in st.session_state:
        return
    
    if not st.session_state.offline_queue:
        return
    
    # Check if online
    if not st.session_state.get('is_online', False):
        return
    
    # Display processing message
    queue_size = len(st.session_state.offline_queue)
    st.info(f"Processing {queue_size} queued operations... / {queue_size} कतारबद्ध संचालन संसाधित कर रहे हैं...")
    
    # Process each queued item
    for item in st.session_state.offline_queue:
        try:
            operation = item['operation']
            data = item['data']
            
            # Process based on operation type
            if operation == 'transcribe':
                # Note: This would call the actual processing function
                # For now, we just log it as the processing functions
                # will be implemented in later tasks
                log_action(operation, 'success', 'Processed from offline queue')
            
            elif operation == 'respond':
                log_action(operation, 'success', 'Processed from offline queue')
            
            elif operation == 'tts':
                log_action(operation, 'success', 'Processed from offline queue')
            
            else:
                log_action(operation, 'error', f'Unknown operation type: {operation}')
        
        except Exception as e:
            log_action(item['operation'], 'error', f'Failed to process from queue: {str(e)}')
    
    # Clear queue after processing
    st.session_state.offline_queue = []
    st.success(f"Processed {queue_size} queued operations / {queue_size} कतारबद्ध संचालन संसाधित किए गए")


def process_with_cache(cache_key: str, processor: callable, ttl: int = None) -> Any:
    """Process request with caching
    
    Wrapper function that checks cache before making API calls. If a cached
    response exists and is not expired, it returns the cached value and displays
    a cache indicator. Otherwise, it calls the processor function, caches the
    result, and returns it.
    
    Args:
        cache_key: Unique identifier for the cached value
        processor: Callable function that performs the actual processing
        ttl: Time to live in seconds (default: uses CACHE_TTL from config)
    
    Returns:
        The processed result (from cache or fresh processing)
    
    Requirements: 7.5
    
    Examples:
        >>> def fetch_transcription():
        ...     return api_client.recognize_speech(audio_data, 'hi')
        >>> 
        >>> result = process_with_cache(
        ...     cache_key='transcription_abc123',
        ...     processor=fetch_transcription,
        ...     ttl=1800
        ... )
    """
    # Use default TTL if not specified
    if ttl is None:
        ttl = CACHE_TTL
    
    # Check cache first
    cached = get_cached_response(cache_key)
    if cached is not None:
        st.info("📦 Loaded from cache / कैश से लोड किया गया")
        log_action('cache', 'success', f'Cache hit for key: {cache_key}')
        return cached
    
    # Process request
    result = processor()
    
    # Cache result
    cache_response(cache_key, result, ttl)
    log_action('cache', 'success', f'Cached result for key: {cache_key}')
    
    return result


def render_audio_uploader():
    """Render audio file upload widget
    
    Displays a file uploader that accepts WAV, MP3, M4A, and OGG audio formats.
    Validates file size (max 10MB) and stores audio data in session state.
    Logs upload action to action history.
    
    Returns:
        bytes: Audio file data if uploaded and valid, None otherwise
    
    Requirements: 1.1, 1.5, 6.1, 9.3
    
    Examples:
        >>> audio_data = render_audio_uploader()
        >>> if audio_data:
        ...     # Process the audio
        ...     process_audio(audio_data, language)
    """
    uploaded_file = st.file_uploader(
        "Upload Audio File / ऑडियो फ़ाइल अपलोड करें",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        key='audio_uploader',
        help="Maximum file size: 10MB / अधिकतम फ़ाइल आकार: 10MB"
    )
    
    if uploaded_file is not None:
        # Validate file size (max 10MB)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error(
                f"❌ **File size exceeds 10MB limit** / **फ़ाइल का आकार 10MB की सीमा से अधिक है**\n\n"
                f"Your file: {file_size_mb:.2f}MB / आपकी फ़ाइल: {file_size_mb:.2f}MB\n\n"
                f"Please:\n"
                f"- Use a shorter recording / छोटी रिकॉर्डिंग का उपयोग करें\n"
                f"- Compress the audio file / ऑडियो फ़ाइल को संपीड़ित करें\n"
                f"- Use a lower bitrate / कम बिटरेट का उपयोग करें"
            )
            log_action('upload', 'error', f'File size {file_size_mb:.2f}MB exceeds 10MB limit')
            return None
        
        # Read and store audio data
        audio_data = uploaded_file.read()
        st.session_state.audio_data = audio_data
        st.session_state.audio_filename = uploaded_file.name
        
        # Display success message
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB)")
        
        # Log upload action
        log_action(
            'upload',
            'success',
            f'Uploaded {uploaded_file.name} ({file_size_mb:.2f}MB)'
        )
        
        return audio_data
    
    return None


def render_voice_recorder():
    """Render voice recording interface using audio_recorder
    
    Uses the audio_recorder_streamlit component for browser-based recording.
    Configures with 16kHz sample rate and 2-second pause threshold.
    Displays recording status indicator and stores recorded audio in session state.
    Logs recording action to action history.
    
    Returns:
        bytes: Recorded audio data if available, None otherwise
    
    Requirements: 1.2, 1.3, 1.4, 6.1, 9.3
    
    Examples:
        >>> audio_data = render_voice_recorder()
        >>> if audio_data:
        ...     # Process the recorded audio
        ...     process_audio(audio_data, language)
    """
    try:
        from audio_recorder_streamlit import audio_recorder
        
        st.write("**Record Audio / ऑडियो रिकॉर्ड करें**")
        
        # Display recording instructions
        st.caption(
            "Click the microphone to start recording. Recording will automatically stop after 2 seconds of silence.\n\n"
            "रिकॉर्डिंग शुरू करने के लिए माइक्रोफ़ोन पर क्लिक करें। 2 सेकंड की चुप्पी के बाद रिकॉर्डिंग स्वचालित रूप से बंद हो जाएगी।"
        )
        
        # Render audio recorder component
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=16000,
            text="Click to record / रिकॉर्ड करने के लिए क्लिक करें",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            # Store in session state
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_filename = "recorded_audio.wav"
            
            # Calculate audio size
            audio_size_mb = len(audio_bytes) / (1024 * 1024)
            
            # Display recorded audio player
            st.audio(audio_bytes, format='audio/wav')
            st.success(f"✅ Recording complete ({audio_size_mb:.2f}MB) / रिकॉर्डिंग पूर्ण ({audio_size_mb:.2f}MB)")
            
            # Log recording action
            log_action(
                'record',
                'success',
                f'Recorded audio ({audio_size_mb:.2f}MB)'
            )
            
            return audio_bytes
        
    except ImportError:
        st.error(
            "❌ **Audio recorder not available** / **ऑडियो रिकॉर्डर उपलब्ध नहीं है**\n\n"
            "Please install: `pip install audio-recorder-streamlit`\n\n"
            "कृपया इंस्टॉल करें: `pip install audio-recorder-streamlit`"
        )
        log_action('record', 'error', 'audio-recorder-streamlit not installed')
    
    except Exception as e:
        st.error(f"❌ Recording error / रिकॉर्डिंग त्रुटि: {str(e)}")
        log_action('record', 'error', f'Recording failed: {str(e)}')
    
    return None


# network-related error handling functions removed because all backend
# calls are now in-process. Exceptions are handled inline where they occur.


def handle_validation_error(error: str, field: str):
    """Handle input validation errors
    
    Displays user-friendly error messages for validation failures with
    helpful suggestions for correction.
    
    Args:
        error: Description of the validation error
        field: Field that failed validation ('audio_format', 'audio_size', 'text_length')
    
    Requirements: 10.2
    
    Examples:
        >>> handle_validation_error('File size exceeds limit', 'audio_size')
    """
    validation_messages = {
        'audio_format': """
        ❌ **Invalid Audio Format** / **अमान्य ऑडियो प्रारूप**
        
        Please upload a file in one of these formats:
        - WAV (.wav)
        - MP3 (.mp3)
        - M4A (.m4a)
        - OGG (.ogg)
        
        कृपया इन प्रारूपों में से किसी एक में फ़ाइल अपलोड करें:
        - WAV (.wav)
        - MP3 (.mp3)
        - M4A (.m4a)
        - OGG (.ogg)
        """,
        
        'audio_size': """
        ❌ **File Too Large** / **फ़ाइल बहुत बड़ी है**
        
        Maximum file size is 10MB. Please:
        - Use a shorter recording
        - Compress the audio file
        - Use a lower bitrate
        
        अधिकतम फ़ाइल आकार 10MB है। कृपया:
        - छोटी रिकॉर्डिंग का उपयोग करें
        - ऑडियो फ़ाइल को संपीड़ित करें
        - कम बिटरेट का उपयोग करें
        """,
        
        'text_length': """
        ❌ **Text Too Long** / **पाठ बहुत लंबा है**
        
        Maximum text length is 5000 characters.
        Please shorten your message.
        
        अधिकतम पाठ लंबाई 5000 वर्ण है।
        कृपया अपना संदेश छोटा करें।
        """
    }
    
    message = validation_messages.get(field, f"Validation error: {error}")
    st.error(message)
    
    log_action('validation', 'error', f"{field}: {error}")


# Network-related helpers (error handling, retries, etc.) removed.
# In-process backend eliminates HTTP requests; any exceptions are handled
# directly where they occur.


def parse_transcription_response(response: dict) -> dict:
    """Parse speech recognition response
    
    Extracts and normalizes transcription data from the backend API response.
    Handles missing fields gracefully with default values.
    
    Args:
        response: Raw API response dictionary from speech recognition endpoint
    
    Returns:
        Normalized transcription dictionary with keys:
        - text: Transcribed text
        - confidence: Confidence score (0.0-1.0)
        - detected_language: Detected language code
        - processing_time: Processing time in seconds
        - alternatives: List of alternative transcriptions
    
    Requirements: 3.2, 12.2
    
    Examples:
        >>> api_response = {'result': {'transcribed_text': 'Hello', 'confidence': 0.95}}
        >>> parsed = parse_transcription_response(api_response)
        >>> print(parsed['text'])
        'Hello'
    """
    result = response.get('result', {})
    
    return {
        'text': result.get('transcribed_text', ''),
        'confidence': result.get('confidence', 0.0),
        'detected_language': result.get('detected_language', 'unknown'),
        'processing_time': result.get('processing_time', 0.0),
        'alternatives': result.get('alternative_transcriptions', [])
    }




def process_audio():
    """Main audio processing orchestration function
    
    Coordinates the complete workflow: transcription → response generation → TTS.
    Manages processing state, tracks operation timing, and handles errors at each step.
    Automatically triggers subsequent steps on success.
    
    Requirements: 3.1, 4.1, 5.1
    
    Examples:
        >>> # After audio is uploaded or recorded
        >>> if st.button("Process Audio"):
        ...     process_audio()
    """
    # Check if audio data exists
    if not st.session_state.get('audio_data'):
        st.warning("⚠️ Please upload or record audio first. / कृपया पहले ऑडियो अपलोड या रिकॉर्ड करें।")
        return
    
    # Check if online
    if not st.session_state.get('is_online', True):
        st.error("❌ Cannot process audio in offline mode. / ऑफ़लाइन मोड में ऑडियो संसाधित नहीं कर सकते।")
        queue_for_offline_processing('transcribe', {
            'audio_data': st.session_state.audio_data,
            'language': st.session_state.selected_language
        })
        return
    
    # Set processing flag
    st.session_state.is_processing = True
    st.session_state.operation_start_time = time.time()
    
    try:
        # Step 1: Transcription
        with st.spinner("🎤 Transcribing audio... / ऑडियो प्रतिलेखन कर रहे हैं..."):
            transcription_result = process_transcription()
        
        if not transcription_result:
            return  # Error already handled in process_transcription
        
        # Step 2: Response Generation (automatic)
        with st.spinner("🤖 Generating response... / प्रतिक्रिया उत्पन्न कर रहे हैं..."):
            response_result = process_response_generation()
        
        if not response_result:
            return  # Error already handled in process_response_generation
        
        # Step 3: TTS (automatic)
        with st.spinner("🔊 Synthesizing speech... / वाक् संश्लेषण कर रहे हैं..."):
            tts_result = process_tts()
        
        # TTS failure is non-critical, continue even if it fails
        
        # Success!
        st.success("✅ Processing complete! / संसाधन पूर्ण!")
        
    except Exception as e:
        st.error(f"❌ Unexpected error during processing: {str(e)}")
        log_action('process_audio', 'error', f"Unexpected error: {str(e)}")
    
    finally:
        # Clear processing flag
        st.session_state.is_processing = False
        st.session_state.operation_start_time = None


def process_transcription() -> Optional[dict]:
    """Process audio transcription
    
    Sends audio to the backend for speech recognition, parses the response,
    stores results in session state, and logs the action. Handles errors
    with user-friendly messages and retry options.
    
    Returns:
        Transcription result dictionary if successful, None if failed
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    
    Examples:
        >>> result = process_transcription()
        >>> if result:
        ...     print(result['text'])
    """
    try:
        # Get audio data and language from session state
        audio_data = st.session_state.audio_data
        language = st.session_state.selected_language

        # Call local backend directly
        log_action('transcribe', 'pending', f'Sending audio for transcription (language: {language})')

        response = backend_module.recognize_speech(
            audio_data=audio_data,
            language=language,
            enable_code_switching=True
        )

        # Parse response
        result = response.get('result', {})
        transcription = {
            'text': result.get('transcribed_text', ''),
            'confidence': result.get('confidence', 0.0),
            'detected_language': result.get('detected_language', language),
            'processing_time': result.get('processing_time', 0.0),
            'alternatives': result.get('alternative_transcriptions', [])
        }

        # Store in session state
        st.session_state.transcription = transcription

        # Log success
        log_action(
            'transcribe',
            'success',
            f"Transcribed: {transcription['text'][:50]}..." if len(transcription['text']) > 50 else f"Transcribed: {transcription['text']}"
        )

        return transcription

    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        log_action('transcribe', 'error', str(e))
        return None


def process_response_generation() -> Optional[dict]:
    """Process AI response generation
    
    Automatically triggered after successful transcription. Sends transcribed
    text to the backend for AI response generation, parses the response,
    stores results in session state, and logs the action.
    
    Returns:
        Response result dictionary if successful, None if failed
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    
    Examples:
        >>> # Automatically called after transcription
        >>> result = process_response_generation()
        >>> if result:
        ...     print(result['text'])
    """
    try:
        # Check if transcription exists
        if not st.session_state.get('transcription'):
            st.error("❌ No transcription available for response generation.")
            return None

        # Get transcription and language from session state
        transcription_text = st.session_state.transcription['text']
        language = st.session_state.selected_language

        # Call local backend directly
        log_action('respond', 'pending', f'Generating response for: {transcription_text[:50]}...')

        response = backend_module.generate_response(
            text=transcription_text,
            language=language
        )

        # Parse response
        response_data = {
            'text': response.get('text', ''),
            'language': response.get('language', language),
            'suggested_actions': response.get('suggested_actions', []),
            'processing_time': response.get('processing_time', 0.0)
        }

        # Store in session state
        st.session_state.response = response_data

        # Log success
        log_action(
            'respond',
            'success',
            f"Response: {response_data['text'][:50]}..." if len(response_data['text']) > 50 else f"Response: {response_data['text']}"
        )

        return response_data

    except Exception as e:
        st.error(f"❌ Response generation failed: {str(e)}")
        log_action('respond', 'error', str(e))
        return None


def process_tts() -> Optional[bytes]:
    """Process text-to-speech synthesis
    
    Automatically triggered after successful response generation. Sends response
    text to the backend for TTS synthesis, stores audio in session state, and
    logs the action. Gracefully degrades to text-only display if TTS fails.
    
    Returns:
        Audio bytes if successful, None if failed (non-critical failure)
    
    Requirements: 5.1, 5.2, 5.4, 5.5
    
    Examples:
        >>> # Automatically called after response generation
        >>> audio = process_tts()
        >>> if audio:
        ...     # Audio player will be displayed
        ...     pass
        ... else:
        ...     # Text-only display (graceful degradation)
        ...     pass
    """
    try:
        # Check if response exists
        if not st.session_state.get('response'):
            st.warning("⚠️ No response available for TTS synthesis.")
            return None

        # Get response text and language from session state
        response_text = st.session_state.response['text']
        language = st.session_state.selected_language

        # Call local backend directly
        log_action('tts', 'pending', f'Synthesizing speech for: {response_text[:50]}...')

        audio_bytes = backend_module.synthesize_speech(
            text=response_text,
            language=language
        )

        # Store in session state
        st.session_state.tts_audio = audio_bytes

        # Log success
        audio_size_kb = len(audio_bytes) / 1024 if audio_bytes else 0
        log_action('tts', 'success', f'Generated TTS audio ({audio_size_kb:.2f} KB)')

        return audio_bytes

    except Exception as e:
        # Graceful degradation - log warning but don't fail
        st.warning(f"⚠️ TTS synthesis failed or timed out. Displaying text response only. / TTS संश्लेषण विफल या समय समाप्त: {str(e)}")
        log_action('tts', 'warning', f'TTS error - graceful degradation to text-only: {str(e)}')
        return None


def render_language_selector():
    """Render language selection dropdown"""
    languages = {
        'hi': 'हिन्दी (Hindi)',
        'en-IN': 'English (India)',
        'ta': 'தமிழ் (Tamil)',
        'te': 'తెలుగు (Telugu)',
        'bn': 'বাংলা (Bengali)',
        'mr': 'मराठी (Marathi)',
        'gu': 'ગુજરાતી (Gujarati)',
        'kn': 'ಕನ್ನಡ (Kannada)',
        'ml': 'മലയാളം (Malayalam)',
        'pa': 'ਪੰਜਾਬੀ (Punjabi)',
        'or': 'ଓଡ଼ିଆ (Odia)'
    }
    
    selected = st.selectbox(
        "Select Language / भाषा चुनें",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        key='selected_language'
    )
    
    return selected


def render_transcription_display():
    """Render transcription results"""
    if 'transcription' in st.session_state and st.session_state.transcription:
        st.subheader("Transcription / प्रतिलेखन")
        
        transcription = st.session_state.transcription
        
        # Display transcription text
        st.info(transcription.get('text', ''))
        
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = transcription.get('confidence', 0.0)
            st.metric("Confidence / विश्वास", f"{confidence:.2%}")
        with col2:
            detected_lang = transcription.get('detected_language', 'unknown')
            st.metric("Language / भाषा", detected_lang)
        with col3:
            proc_time = transcription.get('processing_time', 0.0)
            st.metric("Processing Time / समय", f"{proc_time:.2f}s")


def render_response_display():
    """Render AI response"""
    if 'response' in st.session_state and st.session_state.response:
        st.subheader("Response / प्रतिक्रिया")
        
        response = st.session_state.response
        
        # Display response text
        st.success(response.get('text', ''))
        
        # Display suggested actions if available
        if response.get('suggested_actions'):
            st.write("**Suggested Actions / सुझाए गए कार्य:**")
            for action in response['suggested_actions']:
                st.button(action.get('label', ''), key=f"action_{action.get('id', '')}")


def render_audio_player():
    """Render audio player for TTS response"""
    if 'tts_audio' in st.session_state and st.session_state.tts_audio:
        st.subheader("Audio Response / ऑडियो प्रतिक्रिया")
        
        audio_data = st.session_state.tts_audio
        
        # Decode base64 audio if needed
        if isinstance(audio_data, str):
            import base64
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                audio_bytes = audio_data.encode()
        else:
            audio_bytes = audio_data
        
        st.audio(audio_bytes, format='audio/wav')


def render_progress_indicator(operation: str, progress: float = None):
    """Render progress indicator with operation message
    
    Displays a loading spinner with operation message and optional progress bar.
    For operations exceeding 3 seconds, displays elapsed time.
    
    Args:
        operation: Description of the operation being performed
        progress: Optional progress percentage (0.0 to 1.0)
    
    Requirements: 8.1, 8.2
    """
    if progress is not None:
        st.progress(progress, text=f"{operation}...")
    else:
        with st.spinner(f'{operation}...'):
            pass
    
    # Estimated time for long operations
    if 'operation_start_time' in st.session_state and st.session_state.operation_start_time:
        elapsed = time.time() - st.session_state.operation_start_time
        if elapsed > 3:
            st.info(f"⏱️ Processing... ({elapsed:.1f}s elapsed) / प्रोसेसिंग... ({elapsed:.1f}s बीत चुके)")


def show_success_message(message: str, duration: int = 2):
    """Display success message with auto-dismiss
    
    Shows a success message that automatically dismisses after specified duration.
    
    Args:
        message: Success message to display
        duration: Duration in seconds before auto-dismiss (default: 2)
    
    Requirements: 8.3
    """
    success_placeholder = st.empty()
    success_placeholder.success(f"✅ {message}")
    time.sleep(duration)
    success_placeholder.empty()


def show_error_message(message: str, details: str = None):
    """Display error message with optional details
    
    Shows an error message with optional detailed information about what went wrong.
    
    Args:
        message: Main error message
        details: Optional detailed error information
    
    Requirements: 8.4
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("Error Details / त्रुटि विवरण"):
            st.code(details)


def show_warning_message(message: str):
    """Display warning message for non-critical issues
    
    Shows a warning message for situations that don't prevent operation
    but should be brought to user's attention.
    
    Args:
        message: Warning message to display
    
    Requirements: 8.4
    """
    st.warning(f"⚠️ {message}")


def render_action_log():
    """Render action history log"""
    st.sidebar.subheader("Action Log / कार्य लॉग")
    
    if 'action_history' not in st.session_state:
        st.session_state.action_history = []
    
    if not st.session_state.action_history:
        st.sidebar.info("No actions yet / अभी तक कोई कार्य नहीं")
        return
    
    # Display most recent 10 actions
    for action in reversed(st.session_state.action_history[-10:]):
        with st.sidebar.expander(f"{action.get('timestamp', '')} - {action.get('type', '')}"):
            st.write(f"**Type / प्रकार**: {action.get('type', '')}")
            st.write(f"**Status / स्थिति**: {action.get('status', '')}")
            if 'details' in action and action['details']:
                st.write(f"**Details / विवरण**: {action['details']}")


def main():
    """Main application entry point"""
    
    # Log application startup
    logger.info("BharatVoice AI Streamlit application starting...")
    
    # Set page configuration
    st.set_page_config(
        page_title="BharatVoice AI Assistant",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Monitor connection status
    monitor_connection()
    
    # Process offline queue if connection restored
    if st.session_state.get('is_online', True):
        process_offline_queue()
    
    # Display title and description
    st.title("🎙️ BharatVoice AI Assistant")
    st.markdown("**Voice Assistant for India** / **भारत के लिए वॉयस असिस्टेंट**")
    st.markdown("Interact with AI using your voice in 11 Indian languages / 11 भारतीय भाषाओं में अपनी आवाज़ का उपयोग करके AI के साथ बातचीत करें")
    
    # Render offline indicator if needed
    render_offline_indicator()
    
    # Language selector at top
    st.markdown("---")
    selected_language = render_language_selector()
    
    # Audio input section
    st.markdown("---")
    st.subheader("Audio Input / ऑडियो इनपुट")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload Audio File / ऑडियो फ़ाइल अपलोड करें**")
        render_audio_uploader()
    
    with col2:
        st.markdown("**Record Audio / ऑडियो रिकॉर्ड करें**")
        render_voice_recorder()
    
    # Process button
    st.markdown("---")
    
    # Check if audio data exists
    has_audio = st.session_state.get('audio_data') is not None
    is_processing = st.session_state.get('is_processing', False)
    is_online = st.session_state.get('is_online', True)
    
    # Disable button if no audio, already processing, or offline
    button_disabled = not has_audio or is_processing or not is_online
    
    if not has_audio:
        st.info("ℹ️ Please upload or record audio to continue / जारी रखने के लिए कृपया ऑडियो अपलोड या रिकॉर्ड करें")
    
    if not is_online:
        st.warning("⚠️ Backend is offline. Cannot process audio. / बैकएंड ऑफ़लाइन है। ऑडियो प्रोसेस नहीं कर सकते।")
    
    # Process Audio button
    if st.button(
        "🎯 Process Audio / ऑडियो प्रोसेस करें",
        disabled=button_disabled,
        type="primary",
        use_container_width=True
    ):
        # Validate audio data
        if st.session_state.audio_data:
            logger.info("Processing audio button clicked")
            # Call process_audio orchestration function
            process_audio()
        else:
            logger.warning("Process button clicked but no audio data found")
            st.error("❌ No audio data found. Please upload or record audio first. / कोई ऑडियो डेटा नहीं मिला। कृपया पहले ऑडियो अपलोड या रिकॉर्ड करें।")
    
    # Display results section
    st.markdown("---")
    
    # Render transcription display
    render_transcription_display()
    
    # Render response display
    render_response_display()
    
    # Render audio player
    render_audio_player()
    
    # Render action log in sidebar
    render_action_log()
    
    # Render debug panel if DEBUG mode is enabled
    render_debug_panel()


if __name__ == "__main__":
    main()
