# Language Engine Service

The Language Engine Service is the core multilingual processing component of BharatVoice Assistant, providing advanced speech recognition, language detection, and translation capabilities optimized for Indian languages.

## Features

### 🎯 Multilingual ASR (Automatic Speech Recognition)
- **Whisper-based recognition** with support for 10+ Indian languages
- **High accuracy** speech-to-text conversion for Hindi, Tamil, Telugu, Bengali, and more
- **Confidence scoring** for transcription quality assessment
- **Alternative transcriptions** with multiple recognition hypotheses
- **Real-time processing** capabilities for live audio streams

### 🌐 Language Detection & Code-Switching
- **Automatic language detection** from audio and text
- **Code-switching detection** for mixed-language conversations
- **Language boundary identification** within single utterances
- **Confidence scoring** for language detection results
- **Support for Indian English** with regional accent recognition

### 🔄 Translation Engine
- **Neural machine translation** between Indian languages
- **Cultural context preservation** in translations
- **Semantic meaning validation** for translation quality
- **Caching system** for improved performance
- **Batch translation** support for multiple texts

### 🎭 Regional Accent Adaptation
- **Accent-aware recognition** for different Indian regions
- **Model adaptation** based on regional speech patterns
- **Dynamic accent switching** during conversations
- **User preference learning** for personalized recognition

## Architecture

```
LanguageEngineService
├── MultilingualASREngine
│   ├── Whisper Model Integration
│   ├── Language Detection Pipeline
│   ├── Confidence Scoring System
│   └── Alternative Generation
├── Code-Switching Detector
│   ├── Text Segmentation
│   ├── Language Identification
│   └── Switch Point Detection
├── Translation Engine
│   ├── Neural Translation Models
│   ├── Cultural Context Preservation
│   └── Quality Assessment
└── Caching System
    ├── Recognition Result Cache
    ├── Translation Cache
    └── LRU Eviction Policy
```

## Supported Languages

### Primary Languages
- **Hindi** (`hi`) - Primary Indian language with optimized models
- **English (Indian)** (`en-IN`) - Indian English with regional accent support

### Regional Indian Languages
- **Tamil** (`ta`) - Dravidian language with specialized processing
- **Telugu** (`te`) - Dravidian language with regional variants
- **Bengali** (`bn`) - Eastern Indian language with cultural context
- **Marathi** (`mr`) - Western Indian language with Mumbai dialect
- **Gujarati** (`gu`) - Western Indian language with business terminology
- **Kannada** (`kn`) - Southern Indian language with Bangalore accent
- **Malayalam** (`ml`) - Southern Indian language with Kerala dialect
- **Punjabi** (`pa`) - Northern Indian language with regional variants
- **Odia** (`or`) - Eastern Indian language with cultural references

## Usage

### Basic Speech Recognition

```python
from bharatvoice.services.language_engine import create_language_engine_service
from bharatvoice.core.models import AudioBuffer, LanguageCode

# Create language engine service
service = create_language_engine_service(
    asr_model_size="base",  # or "small", "medium", "large"
    device="cpu",           # or "cuda" for GPU acceleration
    enable_caching=True,
    cache_size=1000
)

# Prepare audio buffer
audio_buffer = AudioBuffer(
    data=audio_samples,     # List of float audio samples
    sample_rate=16000,      # 16kHz sample rate
    channels=1,             # Mono audio
    duration=2.5            # Duration in seconds
)

# Recognize speech
result = await service.recognize_speech(audio_buffer)

print(f"Transcription: {result.transcribed_text}")
print(f"Language: {result.detected_language}")
print(f"Confidence: {result.confidence:.3f}")

# Check for code-switching
if result.has_code_switching:
    print("Code-switching detected:")
    for switch in result.code_switching_points:
        print(f"  Position {switch.position}: {switch.from_language} -> {switch.to_language}")

# Alternative transcriptions
for alt in result.alternative_transcriptions:
    print(f"Alternative: {alt.text} ({alt.language}, {alt.confidence:.3f})")
```

### Language Detection

```python
# Detect language from text
text = "नमस्ते, how are you today?"
detected_language = await service.detect_language(text)
print(f"Detected language: {detected_language}")

# Detect code-switching points
code_switches = await service.detect_code_switching(text)
for switch in code_switches:
    print(f"Language switch at position {switch['position']}: "
          f"{switch['from_language']} -> {switch['to_language']}")
```

### Translation

```python
# Translate between Indian languages
hindi_text = "आप कैसे हैं?"
english_translation = await service.translate_text(
    hindi_text,
    source_lang=LanguageCode.HINDI,
    target_lang=LanguageCode.ENGLISH_IN
)
print(f"Translation: {english_translation}")
```

### Batch Processing

```python
# Process multiple audio files
audio_buffers = [audio1, audio2, audio3]
results = await service.batch_recognize_speech(audio_buffers)

for i, result in enumerate(results):
    print(f"Audio {i+1}: {result.transcribed_text}")
```

### Language Confidence Scores

```python
# Get confidence scores for all languages
confidence_scores = await service.get_language_confidence_scores(audio_buffer)

for language, confidence in confidence_scores.items():
    print(f"{language}: {confidence:.3f}")
```

### Recognition with Language Hint

```python
# Provide language hint for better accuracy
result = await service.recognize_with_language_hint(
    audio_buffer,
    language_hint=LanguageCode.TAMIL
)
```

## Configuration

### ASR Engine Configuration

```python
# Configure ASR engine parameters
service = create_language_engine_service(
    asr_model_size="medium",        # Model size: tiny, base, small, medium, large
    device="cuda",                  # Use GPU for faster processing
    enable_caching=True,            # Enable result caching
    cache_size=2000,                # Cache up to 2000 results
    enable_language_detection=True  # Enable automatic language detection
)
```

### Model Sizes and Performance

| Model Size | Parameters | Speed | Accuracy | Memory Usage |
|------------|------------|-------|----------|--------------|
| tiny       | 39M        | ~32x  | Good     | ~1GB         |
| base       | 74M        | ~16x  | Better   | ~1GB         |
| small      | 244M       | ~6x   | Good     | ~2GB         |
| medium     | 769M       | ~2x   | Better   | ~5GB         |
| large      | 1550M      | 1x    | Best     | ~10GB        |

### Device Configuration

```python
# CPU configuration (default)
service = create_language_engine_service(device="cpu")

# GPU configuration (requires CUDA)
service = create_language_engine_service(device="cuda")
```

## Advanced Features

### Regional Accent Adaptation

```python
# Adapt model for regional accent
accent_data = {
    "region": "mumbai",
    "accent_samples": [...],  # Training samples
    "phonetic_variations": {...}
}

adapted_model_id = await service.adapt_to_regional_accent(
    "base_model",
    accent_data
)
```

### Custom Language Models

```python
# Load custom language model (future feature)
custom_model_path = "/path/to/custom/model"
service.load_custom_model(custom_model_path, LanguageCode.HINDI)
```

### Streaming Recognition

```python
# Real-time streaming recognition (future feature)
async def process_audio_stream(audio_stream):
    async for audio_chunk in audio_stream:
        partial_result = await service.recognize_streaming(audio_chunk)
        print(f"Partial: {partial_result.text}")
```

## Performance Optimization

### Caching Strategy

The service implements intelligent caching to improve performance:

- **Recognition Cache**: Stores results for identical audio inputs
- **Translation Cache**: Caches translation results for repeated text
- **LRU Eviction**: Automatically removes least recently used entries
- **Cache Statistics**: Monitors hit rates and performance metrics

### Memory Management

```python
# Clear caches to free memory
service.clear_caches()

# Get cache statistics
stats = service.get_service_stats()
print(f"Cache hit rate: {stats['cache_stats']['cache_hit_rate']:.2%}")
```

### Batch Processing

Process multiple inputs simultaneously for better throughput:

```python
# Batch recognition for better performance
audio_list = [audio1, audio2, audio3, audio4]
results = await service.batch_recognize_speech(audio_list)
```

## Error Handling

The service includes comprehensive error handling:

```python
try:
    result = await service.recognize_speech(audio_buffer)
    if result.confidence < 0.5:
        print("Low confidence result, consider retry")
except Exception as e:
    logger.error(f"Recognition failed: {e}")
    # Service returns empty result on error
```

## Monitoring and Health Checks

### Health Check

```python
# Check service health
health_status = await service.health_check()
print(f"Service status: {health_status['status']}")
print(f"ASR engine: {health_status['asr_engine']['status']}")
```

### Service Statistics

```python
# Get detailed service statistics
stats = service.get_service_stats()
print(f"Total recognitions: {stats['total_recognitions']}")
print(f"Average processing time: {stats['average_recognition_time']:.3f}s")
print(f"Language distribution: {stats['language_distribution']}")
print(f"Code-switching detections: {stats['code_switching_detections']}")
```

## Testing

### Unit Tests

```bash
# Run language engine tests
pytest tests/test_language_engine.py -v

# Run ASR engine tests
pytest tests/test_asr_engine.py -v

# Run with coverage
pytest tests/test_language_engine.py --cov=src/bharatvoice/services/language_engine
```

### Property-Based Tests

The service includes property-based tests to validate:

- **Recognition consistency** across different audio qualities
- **Language detection accuracy** for mixed-language inputs
- **Translation fidelity** between language pairs
- **Confidence score reliability** across different scenarios

### Integration Tests

```python
# Test complete workflow
async def test_complete_workflow():
    service = create_language_engine_service()
    
    # Test recognition
    result = await service.recognize_speech(test_audio)
    assert result.transcribed_text
    assert result.confidence > 0.0
    
    # Test language detection
    language = await service.detect_language(result.transcribed_text)
    assert language in service.get_supported_languages()
    
    # Test translation
    if language != LanguageCode.ENGLISH_IN:
        translation = await service.translate_text(
            result.transcribed_text,
            language,
            LanguageCode.ENGLISH_IN
        )
        assert translation
```

## Dependencies

### Core Dependencies
- **openai-whisper** - Whisper ASR model
- **transformers** - Language detection models
- **langdetect** - Fallback language detection
- **torch** - PyTorch for model inference
- **librosa** - Audio processing utilities
- **soundfile** - Audio file I/O

### Optional Dependencies
- **cuda** - GPU acceleration (if available)
- **onnxruntime** - Optimized inference (future)
- **tensorrt** - NVIDIA TensorRT optimization (future)

## Installation

```bash
# Install core dependencies
pip install openai-whisper transformers langdetect torch librosa soundfile

# Install with GPU support
pip install openai-whisper transformers langdetect torch[cuda] librosa soundfile

# Install development dependencies
pip install -e ".[dev]"
```

## Future Enhancements

### Planned Features
1. **Streaming Recognition** - Real-time audio processing
2. **Custom Model Training** - Fine-tuning for specific domains
3. **Emotion Detection** - Emotional state recognition from speech
4. **Speaker Identification** - Multi-speaker conversation handling
5. **Noise Robustness** - Enhanced performance in noisy environments
6. **Edge Deployment** - Optimized models for mobile/edge devices

### Performance Improvements
1. **Model Quantization** - Reduced memory usage
2. **ONNX Runtime** - Faster inference
3. **TensorRT Optimization** - GPU acceleration
4. **Distributed Processing** - Multi-GPU support
5. **Caching Optimization** - Smarter cache strategies

## Contributing

When contributing to the language engine service:

1. **Follow the existing architecture** and interfaces
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for API changes
4. **Consider performance implications** of changes
5. **Test with multiple Indian languages** and accents
6. **Ensure backward compatibility** when possible

## License

This module is part of the BharatVoice Assistant project and follows the same licensing terms.