# BharatVoice Assistant - API Documentation

## Overview

The BharatVoice Assistant provides a comprehensive REST API for multilingual voice interactions, supporting 10+ Indian languages with advanced features like code-switching, cultural context awareness, and accessibility support.

## Base URL

```
Production: https://api.bharatvoice.ai/v1
Staging: https://staging-api.bharatvoice.ai/v1
Development: http://localhost:8000/v1
```

## Authentication

### JWT Token Authentication

All API endpoints (except public ones) require JWT authentication.

```http
Authorization: Bearer <jwt_token>
```

### Obtaining Access Token

```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user_id": "uuid-string",
  "user_profile": {
    "name": "User Name",
    "preferred_language": "hi",
    "accessibility_settings": {...}
  }
}
```

## Core API Endpoints

### 1. Voice Processing

#### Process Voice Input

Process audio input and return transcription with intent analysis.

```http
POST /voice/process
Authorization: Bearer <token>
Content-Type: multipart/form-data

audio_file: <audio_file>
language: hi (optional)
context: conversation_context (optional)
```

**Response:**
```json
{
  "transcription": {
    "text": "नमस्ते, आज मौसम कैसा है?",
    "confidence": 0.95,
    "language": "hi",
    "code_switching_points": [
      {
        "position": 15,
        "from_language": "hi",
        "to_language": "en",
        "confidence": 0.87
      }
    ]
  },
  "intent": {
    "category": "weather_inquiry",
    "confidence": 0.92,
    "entities": [
      {
        "type": "time",
        "value": "today",
        "confidence": 0.88
      }
    ]
  },
  "cultural_context": {
    "formality_level": "medium",
    "communication_style": "respectful",
    "regional_influence": "north_indian",
    "cultural_references": ["greeting"]
  },
  "response": {
    "text": "आज दिल्ली में मौसम साफ है, तापमान 28°C है।",
    "audio_url": "/audio/response/uuid.mp3",
    "language": "hi"
  },
  "processing_time": 1.2,
  "session_id": "session-uuid"
}
```

#### Text-to-Speech Synthesis

Convert text to speech with accent and quality options.

```http
POST /voice/synthesize
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "नमस्ते, आप कैसे हैं?",
  "language": "hi",
  "accent": "north_indian",
  "quality": "high",
  "speed": 1.0,
  "volume": 0.8
}
```

**Response:**
```json
{
  "audio_url": "/audio/synthesis/uuid.mp3",
  "duration": 3.2,
  "format": "mp3",
  "quality_metrics": {
    "clarity_score": 0.94,
    "naturalness_score": 0.91
  },
  "processing_time": 0.8
}
```

### 2. Language Processing

#### Detect Language

Automatically detect the language of input text or audio.

```http
POST /language/detect
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Hello नमस्ते, how are you आप कैसे हैं?"
}
```

**Response:**
```json
{
  "primary_language": "en",
  "detected_languages": [
    {
      "language": "en",
      "confidence": 0.65,
      "text_span": "Hello, how are you"
    },
    {
      "language": "hi", 
      "confidence": 0.89,
      "text_span": "नमस्ते, आप कैसे हैं?"
    }
  ],
  "code_switching_detected": true,
  "switching_points": [
    {
      "position": 6,
      "from_language": "en",
      "to_language": "hi"
    }
  ]
}
```

#### Translate Text

Translate text between supported languages with cultural context preservation.

```http
POST /language/translate
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "I want to order biryani",
  "source_language": "en",
  "target_language": "hi",
  "preserve_cultural_context": true
}
```

**Response:**
```json
{
  "translated_text": "मुझे बिरयानी ऑर्डर करनी है",
  "source_language": "en",
  "target_language": "hi",
  "confidence": 0.93,
  "cultural_adaptations": [
    {
      "original": "biryani",
      "adapted": "बिरयानी",
      "reason": "food_item_transliteration"
    }
  ],
  "alternative_translations": [
    "मैं बिरयानी मंगवाना चाहता हूं"
  ]
}
```

### 3. Context Management

#### Get User Profile

Retrieve user profile with preferences and conversation history.

```http
GET /context/profile
Authorization: Bearer <token>
```

**Response:**
```json
{
  "user_id": "uuid-string",
  "profile": {
    "name": "राज शर्मा",
    "preferred_languages": ["hi", "en"],
    "location": {
      "city": "Delhi",
      "state": "Delhi",
      "country": "India",
      "coordinates": {
        "latitude": 28.6139,
        "longitude": 77.2090
      }
    },
    "accessibility_settings": {
      "volume_level": "high",
      "speech_rate": "normal",
      "interaction_mode": "voice_primary"
    },
    "cultural_preferences": {
      "formality_preference": "medium",
      "regional_context": "north_indian",
      "festival_notifications": true
    }
  },
  "conversation_stats": {
    "total_interactions": 1247,
    "preferred_topics": ["weather", "food", "travel"],
    "average_session_length": 5.2
  }
}
```

#### Update Regional Context

Update user's regional context for better localization.

```http
PUT /context/regional
Authorization: Bearer <token>
Content-Type: application/json

{
  "location": {
    "city": "Mumbai",
    "state": "Maharashtra",
    "pincode": "400001"
  },
  "cultural_preferences": {
    "regional_context": "mumbai",
    "local_language_preference": "mr",
    "festival_calendar": "maharashtrian"
  }
}
```

### 4. External Service Integration

#### Indian Railways Information

Get train schedules and booking information.

```http
GET /services/railways/trains
Authorization: Bearer <token>
Query Parameters:
  from: NDLS (station code)
  to: CSTM (station code)
  date: 2024-03-15
  class: 2A (optional)
```

**Response:**
```json
{
  "trains": [
    {
      "train_number": "12951",
      "train_name": "Mumbai Rajdhani Express",
      "departure": {
        "station": "NDLS",
        "time": "16:55",
        "date": "2024-03-15"
      },
      "arrival": {
        "station": "CSTM", 
        "time": "08:35",
        "date": "2024-03-16"
      },
      "duration": "15:40",
      "classes_available": ["1A", "2A", "3A"],
      "availability": {
        "2A": "Available",
        "3A": "RAC 15"
      },
      "fare": {
        "2A": 3500,
        "3A": 1800
      }
    }
  ],
  "search_metadata": {
    "total_trains": 12,
    "search_time": 0.8
  }
}
```

#### Weather Information

Get current weather and forecast for Indian cities.

```http
GET /services/weather/current
Authorization: Bearer <token>
Query Parameters:
  city: Delhi
  language: hi (optional)
```

**Response:**
```json
{
  "current_weather": {
    "city": "Delhi",
    "temperature": 28,
    "feels_like": 32,
    "humidity": 65,
    "description": "partly cloudy",
    "description_local": "आंशिक रूप से बादल",
    "wind_speed": 12,
    "visibility": 8,
    "uv_index": 6
  },
  "forecast": [
    {
      "date": "2024-03-16",
      "high": 30,
      "low": 18,
      "description": "sunny",
      "description_local": "धूप"
    }
  ],
  "air_quality": {
    "aqi": 156,
    "category": "moderate",
    "category_local": "मध्यम"
  }
}
```

### 5. Platform Integrations

#### Food Delivery Integration

Search and order food from integrated platforms.

```http
POST /platforms/food/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "biryani",
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090
  },
  "cuisine_preference": ["indian", "mughlai"],
  "budget_range": {
    "min": 200,
    "max": 500
  }
}
```

**Response:**
```json
{
  "restaurants": [
    {
      "id": "rest_123",
      "name": "Biryani Blues",
      "rating": 4.3,
      "delivery_time": "35-40 mins",
      "distance": 2.1,
      "cuisines": ["Biryani", "Mughlai", "North Indian"],
      "items": [
        {
          "id": "item_456",
          "name": "Chicken Biryani",
          "price": 320,
          "rating": 4.5,
          "description": "Aromatic basmati rice with tender chicken"
        }
      ],
      "offers": [
        {
          "title": "20% OFF",
          "description": "Up to ₹100 off on orders above ₹400"
        }
      ]
    }
  ],
  "total_results": 45,
  "search_time": 1.1
}
```

### 6. Accessibility Features

#### Update Accessibility Settings

Configure accessibility preferences for the user.

```http
PUT /accessibility/settings
Authorization: Bearer <token>
Content-Type: application/json

{
  "volume_level": "high",
  "speech_rate": "slow",
  "listening_timeout": 60,
  "max_recognition_attempts": 5,
  "interaction_mode": "voice_primary",
  "enable_visual_indicators": true,
  "enable_voice_guided_help": true,
  "enable_confirmation_prompts": true
}
```

#### Get Voice-Guided Help

Request help on specific topics with voice guidance.

```http
GET /accessibility/help
Authorization: Bearer <token>
Query Parameters:
  topic: voice_commands
  language: hi
```

**Response:**
```json
{
  "help_content": {
    "topic": "voice_commands",
    "title": "आवाज़ कमांड्स की सहायता",
    "content": "आप निम्नलिखित आवाज़ कमांड्स का उपयोग कर सकते हैं...",
    "audio_url": "/audio/help/voice_commands_hi.mp3",
    "examples": [
      {
        "command": "मौसम बताओ",
        "description": "आज के मौसम की जानकारी पाने के लिए"
      }
    ]
  },
  "navigation": {
    "previous_topic": "basic_commands",
    "next_topic": "language_switching"
  }
}
```

### 7. Performance and Analytics

#### Get Performance Metrics

Retrieve system performance metrics (admin only).

```http
GET /admin/metrics
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "performance": {
    "response_times": {
      "simple_queries": {
        "average": 1.2,
        "p95": 2.1,
        "p99": 3.8
      },
      "complex_queries": {
        "average": 3.4,
        "p95": 5.2,
        "p99": 8.1
      }
    },
    "throughput": {
      "requests_per_second": 45.2,
      "concurrent_users": 128
    },
    "error_rates": {
      "total_error_rate": 0.02,
      "timeout_rate": 0.005
    }
  },
  "system_health": {
    "cpu_usage": 65.2,
    "memory_usage": 78.5,
    "disk_usage": 45.1,
    "database_connections": 15
  }
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "The uploaded audio file format is not supported",
    "message_local": "अपलोड की गई ऑडियो फ़ाइल का प्रारूप समर्थित नहीं है",
    "details": {
      "supported_formats": ["mp3", "wav", "flac", "m4a"],
      "received_format": "avi"
    },
    "request_id": "req_uuid_123",
    "timestamp": "2024-03-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Valid JWT token required |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `INVALID_AUDIO_FORMAT` | 400 | Unsupported audio file format |
| `LANGUAGE_NOT_SUPPORTED` | 400 | Requested language not supported |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `SERVICE_UNAVAILABLE` | 503 | External service temporarily unavailable |
| `PROCESSING_TIMEOUT` | 408 | Request processing timeout |
| `INVALID_REQUEST_FORMAT` | 400 | Malformed request body |

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Authentication endpoints**: 5 requests per minute
- **Voice processing**: 60 requests per minute
- **Text processing**: 100 requests per minute
- **General API**: 1000 requests per hour

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1647345600
```

## Webhooks

### Voice Processing Completion

For long-running voice processing tasks, you can register webhooks:

```http
POST /webhooks/register
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/voice-complete",
  "events": ["voice.processing.complete", "voice.processing.failed"],
  "secret": "webhook_secret_key"
}
```

**Webhook Payload:**
```json
{
  "event": "voice.processing.complete",
  "timestamp": "2024-03-15T10:30:00Z",
  "data": {
    "session_id": "session_uuid",
    "processing_time": 2.3,
    "result": {
      "transcription": "...",
      "intent": "...",
      "response": "..."
    }
  },
  "signature": "sha256=..."
}
```

## SDK and Integration Examples

### Python SDK

```python
from bharatvoice import BharatVoiceClient

client = BharatVoiceClient(
    api_key="your_api_key",
    base_url="https://api.bharatvoice.ai/v1"
)

# Process voice input
with open("audio.wav", "rb") as audio_file:
    result = client.voice.process(
        audio_file=audio_file,
        language="hi",
        context={"user_id": "user_123"}
    )
    
print(f"Transcription: {result.transcription.text}")
print(f"Intent: {result.intent.category}")
```

### JavaScript SDK

```javascript
import { BharatVoiceClient } from '@bharatvoice/sdk';

const client = new BharatVoiceClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.bharatvoice.ai/v1'
});

// Process voice input
const audioFile = document.getElementById('audio-input').files[0];
const result = await client.voice.process({
  audioFile,
  language: 'hi',
  context: { userId: 'user_123' }
});

console.log('Transcription:', result.transcription.text);
console.log('Intent:', result.intent.category);
```

### cURL Examples

```bash
# Process voice input
curl -X POST "https://api.bharatvoice.ai/v1/voice/process" \
  -H "Authorization: Bearer your_jwt_token" \
  -F "audio_file=@audio.wav" \
  -F "language=hi"

# Get weather information
curl -X GET "https://api.bharatvoice.ai/v1/services/weather/current?city=Delhi&language=hi" \
  -H "Authorization: Bearer your_jwt_token"

# Translate text
curl -X POST "https://api.bharatvoice.ai/v1/language/translate" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How are you?",
    "source_language": "en",
    "target_language": "hi"
  }'
```

## Testing and Development

### Sandbox Environment

Use the sandbox environment for testing:

```
Sandbox URL: https://sandbox-api.bharatvoice.ai/v1
```

### Test Data

Sample audio files and test data are available at:
- Hindi: `/test-data/audio/hindi_sample.wav`
- English: `/test-data/audio/english_sample.wav`
- Code-switching: `/test-data/audio/mixed_sample.wav`

### API Testing Tools

- **Postman Collection**: Import from `/docs/postman/bharatvoice-api.json`
- **OpenAPI Spec**: Available at `/docs/openapi.json`
- **Interactive Docs**: Visit `/docs` for Swagger UI

## Support and Resources

- **API Status**: https://status.bharatvoice.ai
- **Developer Portal**: https://developers.bharatvoice.ai
- **Community Forum**: https://community.bharatvoice.ai
- **Support Email**: api-support@bharatvoice.ai
- **GitHub**: https://github.com/bharatvoice/assistant

## Changelog

### v1.0.0 (Current)
- Initial API release
- Support for 10+ Indian languages
- Voice processing and synthesis
- Cultural context recognition
- External service integrations
- Accessibility features

---

**Note**: This API is designed specifically for the Indian market with deep cultural understanding and multilingual support. For the most up-to-date documentation, visit our [developer portal](https://developers.bharatvoice.ai).