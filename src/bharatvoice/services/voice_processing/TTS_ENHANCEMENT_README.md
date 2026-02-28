<<<<<<< HEAD
# Enhanced Text-to-Speech (TTS) Implementation

## Overview

This document describes the enhanced TTS implementation for the BharatVoice Assistant, which provides advanced speech synthesis capabilities with Indian language support, regional accent adaptation, and quality optimization.

## Key Enhancements

### 1. Quality Optimization System

The enhanced TTS engine includes a comprehensive quality optimization system with three quality levels:

- **High Quality**: 22kHz sample rate, 128k bitrate, full optimization pipeline
- **Medium Quality**: 16kHz sample rate, 96k bitrate, basic optimization
- **Low Quality**: 8kHz sample rate, 64k bitrate, minimal processing

#### Quality Features:
- Audio normalization for consistent volume levels
- Dynamic range compression for better clarity
- Noise gate to reduce background artifacts
- EQ boost for enhanced speech intelligibility

### 2. Advanced Regional Accent Adaptation

Enhanced accent processing with 5 parameters per accent:

- **Speed**: Playback speed adjustment (0.7x - 1.2x)
- **Pitch Shift**: Pitch modification in semitones (-3 to +3)
- **Formant Shift**: Vocal tract resonance adjustment
- **Emphasis Factor**: Dynamic range modification
- **Pause Duration**: Inter-word pause timing

#### Supported Accents:
- Standard Indian English
- Regional accents: North Indian, South Indian, West Indian, East Indian
- City-specific accents: Mumbai, Delhi, Bangalore, Chennai, Kolkata

### 3. Audio Streaming Capabilities

Real-time audio streaming for improved user experience:

```python
async for chunk in tts_engine.synthesize_streaming(
    text="Long text for streaming",
    language=LanguageCode.HINDI,
    accent=AccentType.MUMBAI,
    chunk_duration=0.5  # 500ms chunks
):
    # Process each audio chunk as it becomes available
    play_audio_chunk(chunk)
```

### 4. Multiple Output Formats

Support for various audio formats:
- WAV (uncompressed, high quality)
- MP3 (compressed, good for storage)
- FLAC (lossless compression)
- OGG (open-source alternative)

### 5. Multi-Segment Synthesis

Synthesize multiple text segments with configurable pauses:

```python
segments = ["Welcome to BharatVoice", "How can I help you today?"]
combined_audio = await tts_engine.synthesize_with_pauses(
    segments, 
    language=LanguageCode.HINDI,
    pause_duration=0.8  # 800ms pause between segments
)
```

### 6. Adaptive Learning System

The `AdaptiveTTSEngine` learns from user preferences and feedback:

```python
# Update user preferences
adaptive_tts.update_user_preferences("user123", {
    'preferred_accent': AccentType.BANGALORE,
    'speed_preference': 1.1,
    'volume_preference': 0.9
})

# Record user feedback
adaptive_tts.record_feedback(
    user_id="user123",
    text="Synthesized text",
    language=LanguageCode.TAMIL,
    rating=4.5,  # 0.0 to 5.0 scale
    feedback_type="accent_quality"
)
```

## API Reference

### TTSEngine Class

#### Constructor
```python
TTSEngine(sample_rate: int = 22050, quality: str = 'high')
```

#### Key Methods

##### synthesize_speech()
```python
async def synthesize_speech(
    text: str,
    language: LanguageCode,
    accent: AccentType = AccentType.STANDARD,
    use_cache: bool = True,
    quality_optimize: bool = True
) -> AudioBuffer
```

##### synthesize_streaming()
```python
async def synthesize_streaming(
    text: str,
    language: LanguageCode,
    accent: AccentType = AccentType.STANDARD,
    chunk_duration: float = 0.5
) -> Generator[AudioBuffer, None, None]
```

##### synthesize_to_format()
```python
async def synthesize_to_format(
    text: str,
    language: LanguageCode,
    output_format: AudioFormat,
    accent: AccentType = AccentType.STANDARD,
    bitrate: str = "128k"
) -> bytes
```

### AdaptiveTTSEngine Class

Extends `TTSEngine` with adaptive learning capabilities.

#### Additional Methods

##### synthesize_for_user()
```python
async def synthesize_for_user(
    text: str,
    language: LanguageCode,
    user_id: str,
    accent: Optional[AccentType] = None
) -> AudioBuffer
```

##### update_user_preferences()
```python
def update_user_preferences(
    user_id: str,
    preferences: Dict[str, any]
)
```

##### record_feedback()
```python
def record_feedback(
    user_id: str,
    text: str,
    language: LanguageCode,
    rating: float,
    feedback_type: str = "general"
)
```

## Integration with Voice Processing Service

The enhanced TTS is fully integrated with the `VoiceProcessingService`:

```python
# Create service with adaptive TTS
service = create_voice_processing_service(
    sample_rate=16000,
    enable_adaptive_tts=True
)

# Use enhanced TTS features
audio = await service.synthesize_speech(
    "नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?",
    language=LanguageCode.HINDI,
    accent=AccentType.DELHI,
    quality_optimize=True
)

# Stream synthesis
async for chunk in service.synthesize_streaming(
    "This is a streaming example",
    LanguageCode.ENGLISH_IN,
    AccentType.MUMBAI
):
    # Handle each chunk
    pass
```

## Performance Considerations

### Caching System
- Intelligent caching of synthesized audio
- FIFO cache with configurable size limit
- Cache key includes text, language, accent, and quality settings

### Synthesis Time Estimation
```python
estimated_time = tts_engine.estimate_synthesis_time(
    "Text to synthesize",
    LanguageCode.HINDI
)
# Use estimation for UI progress indicators
```

### Memory Management
- Streaming reduces memory usage for long texts
- Automatic cleanup of temporary files
- Configurable cache size limits

## Language Support

### Fully Supported Languages
- Hindi (hi)
- English (Indian) (en-IN)
- Tamil (ta)
- Telugu (te)
- Bengali (bn)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- Odia (or)

### Regional Variations
Each language supports multiple regional accents and pronunciation patterns specific to different Indian states and cities.

## Error Handling

The enhanced TTS system includes robust error handling:

- Graceful fallback to silence for synthesis failures
- Automatic language fallback (unsupported → English)
- Network timeout handling for gTTS
- Format conversion error recovery

## Testing

Comprehensive test suite includes:
- Unit tests for all new methods
- Quality optimization validation
- Accent configuration testing
- Streaming functionality tests
- Adaptive learning system tests
- Integration tests with voice processing service

Run tests with:
```bash
python -m pytest tests/test_voice_processing.py::TestTTSEngine -v
python -m pytest tests/test_voice_processing.py::TestAdaptiveTTSEngine -v
```

## Dependencies

### Required
- gtts >= 2.4.0 (Google Text-to-Speech)
- pydub >= 0.25.0 (Audio processing)
- scipy >= 1.11.0 (Signal processing)
- numpy >= 1.24.0 (Numerical operations)

### Optional
- ffmpeg (for additional audio format support)
- libsndfile (for FLAC support)

## Configuration Examples

### High-Quality Production Setup
```python
tts_engine = TTSEngine(
    sample_rate=22050,
    quality='high'
)
```

### Low-Latency Setup
```python
tts_engine = TTSEngine(
    sample_rate=16000,
    quality='medium'
)
```

### Adaptive Learning Setup
```python
adaptive_tts = AdaptiveTTSEngine(
    sample_rate=22050,
    quality='high'
)

# Configure for specific user
adaptive_tts.update_user_preferences("user123", {
    'preferred_accent': AccentType.SOUTH_INDIAN,
    'speed_preference': 0.9,
    'volume_preference': 0.8
})
```

## Future Enhancements

Planned improvements include:
- Emotional TTS synthesis
- SSML (Speech Synthesis Markup Language) support
- Custom voice model training
- Real-time voice cloning
- Advanced prosody control
- Multi-speaker synthesis

## Troubleshooting

### Common Issues

1. **gTTS Network Errors**
   - Check internet connectivity
   - Verify gTTS service availability
   - Implement retry logic with exponential backoff

2. **Audio Format Issues**
   - Ensure ffmpeg is installed for MP3/OGG support
   - Check file permissions for audio output
   - Verify codec availability

3. **Performance Issues**
   - Adjust cache size for memory constraints
   - Use lower quality settings for faster synthesis
   - Implement streaming for long texts

4. **Accent Not Applied**
   - Verify accent configuration exists
   - Check audio processing pipeline
   - Ensure pydub dependencies are installed

=======
# Enhanced Text-to-Speech (TTS) Implementation

## Overview

This document describes the enhanced TTS implementation for the BharatVoice Assistant, which provides advanced speech synthesis capabilities with Indian language support, regional accent adaptation, and quality optimization.

## Key Enhancements

### 1. Quality Optimization System

The enhanced TTS engine includes a comprehensive quality optimization system with three quality levels:

- **High Quality**: 22kHz sample rate, 128k bitrate, full optimization pipeline
- **Medium Quality**: 16kHz sample rate, 96k bitrate, basic optimization
- **Low Quality**: 8kHz sample rate, 64k bitrate, minimal processing

#### Quality Features:
- Audio normalization for consistent volume levels
- Dynamic range compression for better clarity
- Noise gate to reduce background artifacts
- EQ boost for enhanced speech intelligibility

### 2. Advanced Regional Accent Adaptation

Enhanced accent processing with 5 parameters per accent:

- **Speed**: Playback speed adjustment (0.7x - 1.2x)
- **Pitch Shift**: Pitch modification in semitones (-3 to +3)
- **Formant Shift**: Vocal tract resonance adjustment
- **Emphasis Factor**: Dynamic range modification
- **Pause Duration**: Inter-word pause timing

#### Supported Accents:
- Standard Indian English
- Regional accents: North Indian, South Indian, West Indian, East Indian
- City-specific accents: Mumbai, Delhi, Bangalore, Chennai, Kolkata

### 3. Audio Streaming Capabilities

Real-time audio streaming for improved user experience:

```python
async for chunk in tts_engine.synthesize_streaming(
    text="Long text for streaming",
    language=LanguageCode.HINDI,
    accent=AccentType.MUMBAI,
    chunk_duration=0.5  # 500ms chunks
):
    # Process each audio chunk as it becomes available
    play_audio_chunk(chunk)
```

### 4. Multiple Output Formats

Support for various audio formats:
- WAV (uncompressed, high quality)
- MP3 (compressed, good for storage)
- FLAC (lossless compression)
- OGG (open-source alternative)

### 5. Multi-Segment Synthesis

Synthesize multiple text segments with configurable pauses:

```python
segments = ["Welcome to BharatVoice", "How can I help you today?"]
combined_audio = await tts_engine.synthesize_with_pauses(
    segments, 
    language=LanguageCode.HINDI,
    pause_duration=0.8  # 800ms pause between segments
)
```

### 6. Adaptive Learning System

The `AdaptiveTTSEngine` learns from user preferences and feedback:

```python
# Update user preferences
adaptive_tts.update_user_preferences("user123", {
    'preferred_accent': AccentType.BANGALORE,
    'speed_preference': 1.1,
    'volume_preference': 0.9
})

# Record user feedback
adaptive_tts.record_feedback(
    user_id="user123",
    text="Synthesized text",
    language=LanguageCode.TAMIL,
    rating=4.5,  # 0.0 to 5.0 scale
    feedback_type="accent_quality"
)
```

## API Reference

### TTSEngine Class

#### Constructor
```python
TTSEngine(sample_rate: int = 22050, quality: str = 'high')
```

#### Key Methods

##### synthesize_speech()
```python
async def synthesize_speech(
    text: str,
    language: LanguageCode,
    accent: AccentType = AccentType.STANDARD,
    use_cache: bool = True,
    quality_optimize: bool = True
) -> AudioBuffer
```

##### synthesize_streaming()
```python
async def synthesize_streaming(
    text: str,
    language: LanguageCode,
    accent: AccentType = AccentType.STANDARD,
    chunk_duration: float = 0.5
) -> Generator[AudioBuffer, None, None]
```

##### synthesize_to_format()
```python
async def synthesize_to_format(
    text: str,
    language: LanguageCode,
    output_format: AudioFormat,
    accent: AccentType = AccentType.STANDARD,
    bitrate: str = "128k"
) -> bytes
```

### AdaptiveTTSEngine Class

Extends `TTSEngine` with adaptive learning capabilities.

#### Additional Methods

##### synthesize_for_user()
```python
async def synthesize_for_user(
    text: str,
    language: LanguageCode,
    user_id: str,
    accent: Optional[AccentType] = None
) -> AudioBuffer
```

##### update_user_preferences()
```python
def update_user_preferences(
    user_id: str,
    preferences: Dict[str, any]
)
```

##### record_feedback()
```python
def record_feedback(
    user_id: str,
    text: str,
    language: LanguageCode,
    rating: float,
    feedback_type: str = "general"
)
```

## Integration with Voice Processing Service

The enhanced TTS is fully integrated with the `VoiceProcessingService`:

```python
# Create service with adaptive TTS
service = create_voice_processing_service(
    sample_rate=16000,
    enable_adaptive_tts=True
)

# Use enhanced TTS features
audio = await service.synthesize_speech(
    "नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?",
    language=LanguageCode.HINDI,
    accent=AccentType.DELHI,
    quality_optimize=True
)

# Stream synthesis
async for chunk in service.synthesize_streaming(
    "This is a streaming example",
    LanguageCode.ENGLISH_IN,
    AccentType.MUMBAI
):
    # Handle each chunk
    pass
```

## Performance Considerations

### Caching System
- Intelligent caching of synthesized audio
- FIFO cache with configurable size limit
- Cache key includes text, language, accent, and quality settings

### Synthesis Time Estimation
```python
estimated_time = tts_engine.estimate_synthesis_time(
    "Text to synthesize",
    LanguageCode.HINDI
)
# Use estimation for UI progress indicators
```

### Memory Management
- Streaming reduces memory usage for long texts
- Automatic cleanup of temporary files
- Configurable cache size limits

## Language Support

### Fully Supported Languages
- Hindi (hi)
- English (Indian) (en-IN)
- Tamil (ta)
- Telugu (te)
- Bengali (bn)
- Marathi (mr)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- Odia (or)

### Regional Variations
Each language supports multiple regional accents and pronunciation patterns specific to different Indian states and cities.

## Error Handling

The enhanced TTS system includes robust error handling:

- Graceful fallback to silence for synthesis failures
- Automatic language fallback (unsupported → English)
- Network timeout handling for gTTS
- Format conversion error recovery

## Testing

Comprehensive test suite includes:
- Unit tests for all new methods
- Quality optimization validation
- Accent configuration testing
- Streaming functionality tests
- Adaptive learning system tests
- Integration tests with voice processing service

Run tests with:
```bash
python -m pytest tests/test_voice_processing.py::TestTTSEngine -v
python -m pytest tests/test_voice_processing.py::TestAdaptiveTTSEngine -v
```

## Dependencies

### Required
- gtts >= 2.4.0 (Google Text-to-Speech)
- pydub >= 0.25.0 (Audio processing)
- scipy >= 1.11.0 (Signal processing)
- numpy >= 1.24.0 (Numerical operations)

### Optional
- ffmpeg (for additional audio format support)
- libsndfile (for FLAC support)

## Configuration Examples

### High-Quality Production Setup
```python
tts_engine = TTSEngine(
    sample_rate=22050,
    quality='high'
)
```

### Low-Latency Setup
```python
tts_engine = TTSEngine(
    sample_rate=16000,
    quality='medium'
)
```

### Adaptive Learning Setup
```python
adaptive_tts = AdaptiveTTSEngine(
    sample_rate=22050,
    quality='high'
)

# Configure for specific user
adaptive_tts.update_user_preferences("user123", {
    'preferred_accent': AccentType.SOUTH_INDIAN,
    'speed_preference': 0.9,
    'volume_preference': 0.8
})
```

## Future Enhancements

Planned improvements include:
- Emotional TTS synthesis
- SSML (Speech Synthesis Markup Language) support
- Custom voice model training
- Real-time voice cloning
- Advanced prosody control
- Multi-speaker synthesis

## Troubleshooting

### Common Issues

1. **gTTS Network Errors**
   - Check internet connectivity
   - Verify gTTS service availability
   - Implement retry logic with exponential backoff

2. **Audio Format Issues**
   - Ensure ffmpeg is installed for MP3/OGG support
   - Check file permissions for audio output
   - Verify codec availability

3. **Performance Issues**
   - Adjust cache size for memory constraints
   - Use lower quality settings for faster synthesis
   - Implement streaming for long texts

4. **Accent Not Applied**
   - Verify accent configuration exists
   - Check audio processing pipeline
   - Ensure pydub dependencies are installed

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
For additional support, refer to the main BharatVoice documentation or contact the development team.