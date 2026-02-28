<<<<<<< HEAD
# Voice Processing Service

This module implements the complete audio processing pipeline for BharatVoice Assistant, providing real-time audio processing, voice activity detection, background noise filtering, and text-to-speech synthesis optimized for Indian languages.

## Features

### 🎵 Audio Processing Pipeline
- **Real-time stream processing** with configurable buffer sizes and overlap
- **Language-specific optimizations** for Hindi, Tamil, Telugu, Bengali, and other Indian languages
- **Preemphasis filtering** and audio normalization
- **Format conversion** between different audio formats (WAV, MP3, FLAC, OGG)
- **Feature extraction** (MFCC, spectral features, chroma, zero-crossing rate)

### 🎤 Voice Activity Detection (VAD)
- **WebRTC VAD integration** with configurable aggressiveness levels (0-3)
- **Confidence scoring** based on energy and spectral characteristics
- **Real-time speech boundary detection**
- **Optimized for Indian accents** and speaking patterns

### 🔇 Background Noise Filtering
- **Spectral subtraction** for noise reduction
- **Adaptive noise profile estimation** from initial audio frames
- **High-pass filtering** to remove low-frequency noise
- **Configurable noise reduction strength**

### 🗣️ Text-to-Speech (TTS) Synthesis
- **Multi-language support** for 10+ Indian languages
- **Regional accent adaptation** (North Indian, South Indian, Mumbai, Delhi, etc.)
- **Google TTS integration** with Indian English support
- **Caching system** for improved performance
- **Adaptive TTS** with user preference learning
- **Emotional tone synthesis** (future enhancement)
- **SSML support** (future enhancement)

## Architecture

```
VoiceProcessingService
├── AudioProcessor
│   ├── Real-time stream processing
│   ├── Voice Activity Detection
│   ├── Background noise filtering
│   └── Language-specific optimizations
├── TTSEngine / AdaptiveTTSEngine
│   ├── Multi-language synthesis
│   ├── Regional accent support
│   ├── Caching system
│   └── User preference learning
├── AudioFormatConverter
│   ├── Format conversion utilities
│   ├── Preprocessing for ASR
│   └── Feature extraction
└── RealTimeAudioProcessor
    ├── Streaming audio processing
    ├── Buffer management
    └── Overlap handling
```

## Usage

### Basic Usage

```python
from bharatvoice.services.voice_processing import create_voice_processing_service
from bharatvoice.core.models import AudioBuffer, LanguageCode, AccentType

# Create voice processing service
service = create_voice_processing_service(
    sample_rate=16000,
    vad_aggressiveness=2,
    noise_reduction_factor=0.5,
    enable_adaptive_tts=True
)

# Process audio stream
audio_buffer = AudioBuffer(
    data=audio_samples,
    sample_rate=16000,
    channels=1,
    format=AudioFormat.WAV,
    duration=1.0
)

processed_audio = await service.process_audio_stream(
    audio_buffer, 
    LanguageCode.HINDI
)

# Detect voice activity
vad_result = await service.detect_voice_activity(audio_buffer)
print(f"Speech detected: {vad_result.is_speech}")
print(f"Confidence: {vad_result.confidence}")

# Synthesize speech
synthesized = await service.synthesize_speech(
    "नमस्ते, आप कैसे हैं?",
    LanguageCode.HINDI,
    AccentType.NORTH_INDIAN
)
```

### Real-time Processing

```python
# Process real-time audio stream
audio_chunk = [0.1, 0.2, -0.1, -0.2] * 1024  # Audio samples

processed_audio, vad_results = await service.process_realtime_stream(
    audio_chunk,
    LanguageCode.TAMIL
)

# Handle VAD results
for vad_result in vad_results:
    if vad_result.is_speech:
        print(f"Speech detected with confidence: {vad_result.confidence}")
```

### User-Adaptive TTS

```python
# Update user preferences
service.update_user_tts_preferences("user123", {
    'preferred_accent': AccentType.MUMBAI,
    'speed_preference': 1.1  # Slightly faster
})

# Synthesize with user preferences
user_speech = await service.synthesize_for_user(
    "Your order has been confirmed",
    LanguageCode.ENGLISH_IN,
    "user123"
)

# Record user feedback
service.record_tts_feedback(
    "user123",
    "Your order has been confirmed",
    LanguageCode.ENGLISH_IN,
    4.5,  # Rating out of 5
    "clarity"
)
```

### Audio Preprocessing

```python
# Preprocess for speech recognition
preprocessed = await service.preprocess_for_recognition(audio_buffer)

# Extract audio features
features = await service.extract_audio_features(audio_buffer)
print(f"MFCC features shape: {len(features['mfcc'])}")
print(f"Spectral centroid: {features['spectral_centroid']}")

# Filter background noise
filtered = await service.filter_background_noise(noisy_audio)
```

## Configuration

### AudioProcessor Configuration

```python
audio_processor = AudioProcessor(
    sample_rate=16000,          # Audio sample rate
    frame_duration_ms=30,       # VAD frame duration
    vad_aggressiveness=2,       # VAD sensitivity (0-3)
    noise_reduction_factor=0.5  # Noise reduction strength
)
```

### TTS Engine Configuration

```python
tts_engine = TTSEngine(
    sample_rate=22050  # TTS output sample rate
)

# Or adaptive TTS
adaptive_tts = AdaptiveTTSEngine(
    sample_rate=22050
)
```

### Real-time Processor Configuration

```python
realtime_processor = RealTimeAudioProcessor(
    audio_processor,
    buffer_size=1024,      # Processing buffer size
    overlap_ratio=0.5      # Overlap between buffers
)
```

## Language Support

### Supported Languages
- **Hindi** (`hi`) - Primary language with optimized processing
- **English (Indian)** (`en-IN`) - Indian English with regional TLD
- **Tamil** (`ta`) - Dravidian language optimization
- **Telugu** (`te`) - Dravidian language optimization
- **Bengali** (`bn`) - Eastern Indian language
- **Marathi** (`mr`) - Western Indian language
- **Gujarati** (`gu`) - Western Indian language
- **Kannada** (`kn`) - Southern Indian language
- **Malayalam** (`ml`) - Southern Indian language
- **Punjabi** (`pa`) - Northern Indian language
- **Odia** (`or`) - Eastern Indian language

### Regional Accents
- **Standard** - Neutral accent
- **North Indian** - Delhi, Punjab region
- **South Indian** - Tamil Nadu, Karnataka region
- **West Indian** - Maharashtra, Gujarat region
- **East Indian** - West Bengal, Odisha region
- **City-specific** - Mumbai, Delhi, Bangalore, Chennai, Kolkata

## Performance Optimizations

### Language-Specific Processing
- **Frequency emphasis** tailored to each language's phonetic characteristics
- **Hindi/English**: Mid-frequency emphasis (1000-3000 Hz)
- **Tamil/Telugu**: Higher frequency emphasis (1500-4000 Hz)
- **Bengali/Marathi**: Balanced frequency response (800-3500 Hz)

### Caching System
- **TTS result caching** with configurable cache size
- **LRU eviction** policy for memory management
- **Cache statistics** and monitoring

### Real-time Optimizations
- **Overlap-add processing** for seamless audio streams
- **Buffer management** with configurable sizes
- **Asynchronous processing** for non-blocking operations

## Error Handling

The service includes comprehensive error handling:

- **Graceful degradation** when external services fail
- **Fallback mechanisms** for TTS synthesis
- **Input validation** for audio buffers and parameters
- **Logging** at appropriate levels for debugging

## Health Monitoring

```python
# Check service health
health_status = await service.health_check()
print(f"Service status: {health_status['status']}")

# Get service statistics
stats = service.get_service_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Average processing time: {stats['average_processing_time']:.3f}s")
```

## Testing

The module includes comprehensive unit tests covering:

- **Audio buffer operations** and validation
- **Voice activity detection** accuracy
- **Noise filtering** effectiveness
- **TTS synthesis** functionality
- **Real-time processing** capabilities
- **Error handling** and edge cases
- **Format conversion** utilities
- **Feature extraction** accuracy

Run tests with:
```bash
pytest tests/test_voice_processing.py -v
```

## Dependencies

### Core Dependencies
- **numpy** - Numerical operations and array processing
- **scipy** - Signal processing and filtering
- **librosa** - Audio analysis and feature extraction
- **webrtcvad** - Voice activity detection
- **gtts** - Google Text-to-Speech
- **pydub** - Audio format conversion and manipulation

### Optional Dependencies
- **torch** - For advanced neural TTS models (future)
- **transformers** - For language-specific optimizations (future)

## Future Enhancements

### Planned Features
1. **Neural TTS models** for more natural synthesis
2. **Emotional tone control** in speech synthesis
3. **Advanced noise reduction** using deep learning
4. **Speaker recognition** and adaptation
5. **Multi-speaker TTS** synthesis
6. **Real-time voice conversion** between accents
7. **Prosody control** for better naturalness
8. **Custom voice training** for personalization

### Performance Improvements
1. **GPU acceleration** for real-time processing
2. **Model quantization** for mobile deployment
3. **Streaming TTS** for reduced latency
4. **Edge computing** optimizations
5. **Batch processing** for multiple requests

## Contributing

When contributing to the voice processing service:

1. **Follow the existing architecture** and interfaces
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for API changes
4. **Consider performance implications** of changes
5. **Test with multiple Indian languages** and accents
6. **Ensure backward compatibility** when possible

## License

=======
# Voice Processing Service

This module implements the complete audio processing pipeline for BharatVoice Assistant, providing real-time audio processing, voice activity detection, background noise filtering, and text-to-speech synthesis optimized for Indian languages.

## Features

### 🎵 Audio Processing Pipeline
- **Real-time stream processing** with configurable buffer sizes and overlap
- **Language-specific optimizations** for Hindi, Tamil, Telugu, Bengali, and other Indian languages
- **Preemphasis filtering** and audio normalization
- **Format conversion** between different audio formats (WAV, MP3, FLAC, OGG)
- **Feature extraction** (MFCC, spectral features, chroma, zero-crossing rate)

### 🎤 Voice Activity Detection (VAD)
- **WebRTC VAD integration** with configurable aggressiveness levels (0-3)
- **Confidence scoring** based on energy and spectral characteristics
- **Real-time speech boundary detection**
- **Optimized for Indian accents** and speaking patterns

### 🔇 Background Noise Filtering
- **Spectral subtraction** for noise reduction
- **Adaptive noise profile estimation** from initial audio frames
- **High-pass filtering** to remove low-frequency noise
- **Configurable noise reduction strength**

### 🗣️ Text-to-Speech (TTS) Synthesis
- **Multi-language support** for 10+ Indian languages
- **Regional accent adaptation** (North Indian, South Indian, Mumbai, Delhi, etc.)
- **Google TTS integration** with Indian English support
- **Caching system** for improved performance
- **Adaptive TTS** with user preference learning
- **Emotional tone synthesis** (future enhancement)
- **SSML support** (future enhancement)

## Architecture

```
VoiceProcessingService
├── AudioProcessor
│   ├── Real-time stream processing
│   ├── Voice Activity Detection
│   ├── Background noise filtering
│   └── Language-specific optimizations
├── TTSEngine / AdaptiveTTSEngine
│   ├── Multi-language synthesis
│   ├── Regional accent support
│   ├── Caching system
│   └── User preference learning
├── AudioFormatConverter
│   ├── Format conversion utilities
│   ├── Preprocessing for ASR
│   └── Feature extraction
└── RealTimeAudioProcessor
    ├── Streaming audio processing
    ├── Buffer management
    └── Overlap handling
```

## Usage

### Basic Usage

```python
from bharatvoice.services.voice_processing import create_voice_processing_service
from bharatvoice.core.models import AudioBuffer, LanguageCode, AccentType

# Create voice processing service
service = create_voice_processing_service(
    sample_rate=16000,
    vad_aggressiveness=2,
    noise_reduction_factor=0.5,
    enable_adaptive_tts=True
)

# Process audio stream
audio_buffer = AudioBuffer(
    data=audio_samples,
    sample_rate=16000,
    channels=1,
    format=AudioFormat.WAV,
    duration=1.0
)

processed_audio = await service.process_audio_stream(
    audio_buffer, 
    LanguageCode.HINDI
)

# Detect voice activity
vad_result = await service.detect_voice_activity(audio_buffer)
print(f"Speech detected: {vad_result.is_speech}")
print(f"Confidence: {vad_result.confidence}")

# Synthesize speech
synthesized = await service.synthesize_speech(
    "नमस्ते, आप कैसे हैं?",
    LanguageCode.HINDI,
    AccentType.NORTH_INDIAN
)
```

### Real-time Processing

```python
# Process real-time audio stream
audio_chunk = [0.1, 0.2, -0.1, -0.2] * 1024  # Audio samples

processed_audio, vad_results = await service.process_realtime_stream(
    audio_chunk,
    LanguageCode.TAMIL
)

# Handle VAD results
for vad_result in vad_results:
    if vad_result.is_speech:
        print(f"Speech detected with confidence: {vad_result.confidence}")
```

### User-Adaptive TTS

```python
# Update user preferences
service.update_user_tts_preferences("user123", {
    'preferred_accent': AccentType.MUMBAI,
    'speed_preference': 1.1  # Slightly faster
})

# Synthesize with user preferences
user_speech = await service.synthesize_for_user(
    "Your order has been confirmed",
    LanguageCode.ENGLISH_IN,
    "user123"
)

# Record user feedback
service.record_tts_feedback(
    "user123",
    "Your order has been confirmed",
    LanguageCode.ENGLISH_IN,
    4.5,  # Rating out of 5
    "clarity"
)
```

### Audio Preprocessing

```python
# Preprocess for speech recognition
preprocessed = await service.preprocess_for_recognition(audio_buffer)

# Extract audio features
features = await service.extract_audio_features(audio_buffer)
print(f"MFCC features shape: {len(features['mfcc'])}")
print(f"Spectral centroid: {features['spectral_centroid']}")

# Filter background noise
filtered = await service.filter_background_noise(noisy_audio)
```

## Configuration

### AudioProcessor Configuration

```python
audio_processor = AudioProcessor(
    sample_rate=16000,          # Audio sample rate
    frame_duration_ms=30,       # VAD frame duration
    vad_aggressiveness=2,       # VAD sensitivity (0-3)
    noise_reduction_factor=0.5  # Noise reduction strength
)
```

### TTS Engine Configuration

```python
tts_engine = TTSEngine(
    sample_rate=22050  # TTS output sample rate
)

# Or adaptive TTS
adaptive_tts = AdaptiveTTSEngine(
    sample_rate=22050
)
```

### Real-time Processor Configuration

```python
realtime_processor = RealTimeAudioProcessor(
    audio_processor,
    buffer_size=1024,      # Processing buffer size
    overlap_ratio=0.5      # Overlap between buffers
)
```

## Language Support

### Supported Languages
- **Hindi** (`hi`) - Primary language with optimized processing
- **English (Indian)** (`en-IN`) - Indian English with regional TLD
- **Tamil** (`ta`) - Dravidian language optimization
- **Telugu** (`te`) - Dravidian language optimization
- **Bengali** (`bn`) - Eastern Indian language
- **Marathi** (`mr`) - Western Indian language
- **Gujarati** (`gu`) - Western Indian language
- **Kannada** (`kn`) - Southern Indian language
- **Malayalam** (`ml`) - Southern Indian language
- **Punjabi** (`pa`) - Northern Indian language
- **Odia** (`or`) - Eastern Indian language

### Regional Accents
- **Standard** - Neutral accent
- **North Indian** - Delhi, Punjab region
- **South Indian** - Tamil Nadu, Karnataka region
- **West Indian** - Maharashtra, Gujarat region
- **East Indian** - West Bengal, Odisha region
- **City-specific** - Mumbai, Delhi, Bangalore, Chennai, Kolkata

## Performance Optimizations

### Language-Specific Processing
- **Frequency emphasis** tailored to each language's phonetic characteristics
- **Hindi/English**: Mid-frequency emphasis (1000-3000 Hz)
- **Tamil/Telugu**: Higher frequency emphasis (1500-4000 Hz)
- **Bengali/Marathi**: Balanced frequency response (800-3500 Hz)

### Caching System
- **TTS result caching** with configurable cache size
- **LRU eviction** policy for memory management
- **Cache statistics** and monitoring

### Real-time Optimizations
- **Overlap-add processing** for seamless audio streams
- **Buffer management** with configurable sizes
- **Asynchronous processing** for non-blocking operations

## Error Handling

The service includes comprehensive error handling:

- **Graceful degradation** when external services fail
- **Fallback mechanisms** for TTS synthesis
- **Input validation** for audio buffers and parameters
- **Logging** at appropriate levels for debugging

## Health Monitoring

```python
# Check service health
health_status = await service.health_check()
print(f"Service status: {health_status['status']}")

# Get service statistics
stats = service.get_service_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Average processing time: {stats['average_processing_time']:.3f}s")
```

## Testing

The module includes comprehensive unit tests covering:

- **Audio buffer operations** and validation
- **Voice activity detection** accuracy
- **Noise filtering** effectiveness
- **TTS synthesis** functionality
- **Real-time processing** capabilities
- **Error handling** and edge cases
- **Format conversion** utilities
- **Feature extraction** accuracy

Run tests with:
```bash
pytest tests/test_voice_processing.py -v
```

## Dependencies

### Core Dependencies
- **numpy** - Numerical operations and array processing
- **scipy** - Signal processing and filtering
- **librosa** - Audio analysis and feature extraction
- **webrtcvad** - Voice activity detection
- **gtts** - Google Text-to-Speech
- **pydub** - Audio format conversion and manipulation

### Optional Dependencies
- **torch** - For advanced neural TTS models (future)
- **transformers** - For language-specific optimizations (future)

## Future Enhancements

### Planned Features
1. **Neural TTS models** for more natural synthesis
2. **Emotional tone control** in speech synthesis
3. **Advanced noise reduction** using deep learning
4. **Speaker recognition** and adaptation
5. **Multi-speaker TTS** synthesis
6. **Real-time voice conversion** between accents
7. **Prosody control** for better naturalness
8. **Custom voice training** for personalization

### Performance Improvements
1. **GPU acceleration** for real-time processing
2. **Model quantization** for mobile deployment
3. **Streaming TTS** for reduced latency
4. **Edge computing** optimizations
5. **Batch processing** for multiple requests

## Contributing

When contributing to the voice processing service:

1. **Follow the existing architecture** and interfaces
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for API changes
4. **Consider performance implications** of changes
5. **Test with multiple Indian languages** and accents
6. **Ensure backward compatibility** when possible

## License

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
This module is part of the BharatVoice Assistant project and follows the same licensing terms.