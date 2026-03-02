"""
Local backend module for Streamlit deployment.
Provides in-process implementations of recognize_speech, generate_response, and synthesize_speech.
These are lightweight placeholders that use simple local logic so the Streamlit app can run without an external service.
"""
from uuid import uuid4
import io
import wave
import math
import struct
from typing import Optional


def recognize_speech(audio_data: bytes, language: Optional[str] = None, enable_code_switching: bool = True) -> dict:
    """Mock/local speech recognition.
    Returns a dict similar to the former API JSON response.
    """
    request_id = str(uuid4())

    text = "नमस्ते, मैं BharatVoice Assistant हूं। How can I help you today?"
    if language and language.startswith('en'):
        text = "Hello, I am BharatVoice Assistant. How can I help you today?"

    result = {
        'transcribed_text': text,
        'confidence': 0.95,
        'detected_language': language or ('hi' if not language else language),
        'code_switching_points': [],
        'alternative_transcriptions': [],
        'processing_time': 0.5
    }

    return {
        'request_id': request_id,
        'result': result,
        'processing_time': result['processing_time']
    }


def generate_response(text: str, language: Optional[str] = None, context: Optional[dict] = None) -> dict:
    """Mock/local response generation. Returns a simple echo-style response."""
    request_id = str(uuid4())

    # Simple echo / paraphrase response
    response_text = f"{text}"
    if len(response_text.strip()) == 0:
        response_text = "माफ़ कीजिए, मुझे समझ नहीं आया। Could you repeat that?"
    else:
        response_text = response_text if len(response_text) < 200 else response_text[:197] + '...'

    return {
        'request_id': request_id,
        'text': response_text,
        'language': language or 'hi',
        'suggested_actions': [],
        'processing_time': 0.2
    }


def synthesize_speech(text: str, language: Optional[str] = None, accent: str = 'standard', speed: float = 1.0, pitch: float = 1.0) -> bytes:
    """Generate TTS audio using gTTS for natural speech in 11 Indian languages.
    
    Supports:
    - Hindi (hi), English-India (en-IN), Tamil (ta), Telugu (te), Bengali (bn)
    - Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), Punjabi (pa), Odia (or)
    """
    try:
        from gtts import gTTS
    except ImportError:
        # Fallback to tone if gTTS not installed
        return _generate_tone_fallback(text, speed, pitch)
    
    # Map language codes to gTTS language codes (gTTS uses ISO 639-1 codes)
    lang_map = {
        'hi': 'hi',           # Hindi
        'en-IN': 'en',        # English (India)
        'ta': 'ta',           # Tamil
        'te': 'te',           # Telugu
        'bn': 'bn',           # Bengali
        'mr': 'mr',           # Marathi
        'gu': 'gu',           # Gujarati
        'kn': 'kn',           # Kannada
        'ml': 'ml',           # Malayalam
        'pa': 'pa',           # Punjabi
        'or': 'or'            # Odia
    }
    
    gtts_lang = lang_map.get(language, 'hi')
    
    try:
        # Generate TTS using gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Save to BytesIO buffer (in-memory)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"gTTS error: {e}. Using fallback tone.")
        return _generate_tone_fallback(text, speed, pitch)


def _generate_tone_fallback(text: str, speed: float = 1.0, pitch: float = 1.0) -> bytes:
    """Fallback tone generator if gTTS fails."""
    words = max(1, len(text.split()))
    duration_seconds = min(10.0, max(0.5, words / 3.0))  # 0.5-10s

    sample_rate = 22050
    amplitude = 16000
    freq = 440.0
    n_samples = int(sample_rate * duration_seconds)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            value = int(amplitude * math.sin(2.0 * math.pi * freq * pitch * t) * (0.5 + 0.5 * math.sin(0.25 * t)))
            wf.writeframesraw(struct.pack('<h', value))

    return buf.getvalue()


def health_check() -> bool:
    """Local health check for the in-process backend (always True)."""
    return True
