<<<<<<< HEAD
"""
Offline voice processing system for BharatVoice Assistant.

This module provides local speech recognition, TTS synthesis, and basic query
processing capabilities that work without internet connectivity.
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel
from bharatvoice.core.interfaces import AudioProcessor as AudioProcessorInterface
from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    AccentType,
    RecognitionResult,
    VoiceActivityResult,
)
   

logger = logging.getLogger(__name__)


class NetworkStatus(str, Enum):
    """Network connectivity status."""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"


class OfflineQuery(BaseModel):
    """Represents an offline query with cached response."""
    
    query_text: str
    language: LanguageCode
    response_text: str
    confidence: float
    timestamp: datetime
    usage_count: int = 0


class OfflineVoiceModel(BaseModel):
    """Represents a cached voice model for offline TTS."""
    
    language: LanguageCode
    accent: AccentType
    model_data: bytes
    quality: str
    size_mb: float
    last_used: datetime


class OfflineVoiceProcessor(AudioProcessorInterface):
    """
    Offline voice processing system that provides local speech recognition,
    TTS synthesis, and basic query processing without internet connectivity.
    """
    
    def __init__(
        self,
        cache_dir: str = ".bharatvoice_offline",
        max_cache_size_mb: int = 500,
        enable_local_asr: bool = True,
        enable_local_tts: bool = True,
        common_queries_limit: int = 1000
    ):
        """
        Initialize offline voice processor.
        
        Args:
            cache_dir: Directory for offline cache storage
            max_cache_size_mb: Maximum cache size in MB
            enable_local_asr: Whether to enable local ASR
            enable_local_tts: Whether to enable local TTS
            common_queries_limit: Maximum number of cached common queries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_local_asr = enable_local_asr
        self.enable_local_tts = enable_local_tts
        self.common_queries_limit = common_queries_limit
        
        # Initialize offline databases
        self.queries_db_path = self.cache_dir / "offline_queries.db"
        self.models_db_path = self.cache_dir / "offline_models.db"
        
        # Network status tracking
        self.network_status = NetworkStatus.ONLINE
        self.last_connectivity_check = datetime.now()
        
        # Offline caches
        self.common_queries: Dict[str, OfflineQuery] = {}
        self.cached_voice_models: Dict[str, OfflineVoiceModel] = {}
        self.offline_responses: Dict[str, str] = {}
        
        # Local processing components (lightweight versions)
        self.local_asr_engine = None
        self.local_tts_engine = None
        
        # Statistics
        self.offline_stats = {
            'offline_queries_processed': 0,
            'offline_tts_synthesized': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'network_switches': 0,
            'total_offline_time': 0.0
        }
        
        # Initialize offline databases
        self._init_offline_databases()
        
        # Load cached data
        self._load_offline_caches()
        
        # Initialize local processing engines
        if enable_local_asr:
            self._init_local_asr()
        
        if enable_local_tts:
            self._init_local_tts()
        
        logger.info("OfflineVoiceProcessor initialized successfully")
    
    def _init_offline_databases(self):
        """Initialize SQLite databases for offline storage."""
        try:
            # Initialize queries database
            with sqlite3.connect(self.queries_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        language TEXT NOT NULL,
                        response_text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        UNIQUE(query_text, language)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_language 
                    ON offline_queries(query_text, language)
                """)
            
            # Initialize models database
            with sqlite3.connect(self.models_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        language TEXT NOT NULL,
                        accent TEXT NOT NULL,
                        model_data BLOB NOT NULL,
                        quality TEXT NOT NULL,
                        size_mb REAL NOT NULL,
                        last_used TEXT NOT NULL,
                        UNIQUE(language, accent, quality)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_lang_accent 
                    ON offline_models(language, accent)
                """)
            
            logger.info("Offline databases initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing offline databases: {e}")
    
    def _load_offline_caches(self):
        """Load cached data from databases into memory."""
        try:
            # Load common queries
            with sqlite3.connect(self.queries_db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_text, language, response_text, confidence, 
                           timestamp, usage_count
                    FROM offline_queries 
                    ORDER BY usage_count DESC 
                    LIMIT ?
                """, (self.common_queries_limit,))
                
                for row in cursor.fetchall():
                    query_key = f"{row[0]}:{row[1]}"
                    self.common_queries[query_key] = OfflineQuery(
                        query_text=row[0],
                        language=LanguageCode(row[1]),
                        response_text=row[2],
                        confidence=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        usage_count=row[5]
                    )
            
            # Load voice models metadata (not the actual model data for memory efficiency)
            with sqlite3.connect(self.models_db_path) as conn:
                cursor = conn.execute("""
                    SELECT language, accent, quality, size_mb, last_used
                    FROM offline_models
                """)
                
                for row in cursor.fetchall():
                    model_key = f"{row[0]}:{row[1]}:{row[2]}"
                    self.cached_voice_models[model_key] = OfflineVoiceModel(
                        language=LanguageCode(row[0]),
                        accent=AccentType(row[1]),
                        model_data=b"",  # Load on demand
                        quality=row[2],
                        size_mb=row[3],
                        last_used=datetime.fromisoformat(row[4])
                    )
            
            logger.info(
                f"Loaded {len(self.common_queries)} cached queries and "
                f"{len(self.cached_voice_models)} voice models"
            )
            
        except Exception as e:
            logger.error(f"Error loading offline caches: {e}")
    
    def _init_local_asr(self):
        """Initialize lightweight local ASR engine."""
        try:
            # For offline ASR, we'll use a simplified approach
            # In production, this could use a lightweight model like Whisper tiny
            self.local_asr_engine = {
                'model_type': 'lightweight',
                'supported_languages': [
                    LanguageCode.HINDI,
                    LanguageCode.ENGLISH_IN,
                    LanguageCode.TAMIL,
                    LanguageCode.BENGALI
                ],
                'confidence_threshold': 0.6
            }
            
            logger.info("Local ASR engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing local ASR: {e}")
            self.local_asr_engine = None
    
    def _init_local_tts(self):
        """Initialize lightweight local TTS engine."""
        try:
            # For offline TTS, we'll use cached voice models
            self.local_tts_engine = {
                'engine_type': 'cached_models',
                'supported_languages': list(LanguageCode),
                'default_quality': 'medium',
                'cache_enabled': True
            }
            
            logger.info("Local TTS engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing local TTS: {e}")
            self.local_tts_engine = None
    
    async def check_network_connectivity(self) -> NetworkStatus:
        """
        Check current network connectivity status.
        
        Returns:
            Current network status
        """
        try:
            import socket
            
            # Try to connect to a reliable server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            if self.network_status != NetworkStatus.ONLINE:
                self.offline_stats['network_switches'] += 1
                logger.info("Network connectivity restored")
            
            self.network_status = NetworkStatus.ONLINE
            
        except (socket.error, OSError):
            if self.network_status != NetworkStatus.OFFLINE:
                self.offline_stats['network_switches'] += 1
                logger.info("Network connectivity lost - switching to offline mode")
            
            self.network_status = NetworkStatus.OFFLINE
        
        self.last_connectivity_check = datetime.now()
        return self.network_status
    
    async def process_audio_stream(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio stream with offline capabilities.
        
        Args:
            audio_data: Input audio buffer
            language: Target language for processing
            
        Returns:
            Processed audio buffer
        """
        try:
            # Check network status
            network_status = await self.check_network_connectivity()
            
            if network_status == NetworkStatus.OFFLINE:
                # Use offline processing
                return await self._process_audio_offline(audio_data, language)
            else:
                # Try online processing first, fallback to offline
                try:
                    # This would call the online voice processing service
                    # For now, we'll simulate online processing
                    return await self._process_audio_offline(audio_data, language)
                except Exception as e:
                    logger.warning(f"Online processing failed, using offline: {e}")
                    return await self._process_audio_offline(audio_data, language)
        
        except Exception as e:
            logger.error(f"Error in process_audio_stream: {e}")
            raise
    
    async def _process_audio_offline(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio using offline capabilities.
        
        Args:
            audio_data: Input audio buffer
            language: Target language
            
        Returns:
            Processed audio buffer
        """
        try:
            # Apply basic noise reduction
            processed_data = self._apply_basic_noise_reduction(audio_data.data)
            
            # Create processed audio buffer
            processed_audio = AudioBuffer(
                data=processed_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(processed_data) / audio_data.sample_rate
            )
            
            logger.debug(f"Processed audio offline for language: {language}")
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error in offline audio processing: {e}")
            return audio_data
    
    def _apply_basic_noise_reduction(self, audio_data: List[float]) -> List[float]:
        """
        Apply basic noise reduction to audio data.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Noise-reduced audio samples
        """
        try:
            # Convert to numpy array for processing
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Apply simple high-pass filter to remove low-frequency noise
            # This is a basic implementation - production would use more sophisticated methods
            if len(audio_array) > 1:
                # Simple difference filter
                filtered = np.diff(audio_array, prepend=audio_array[0])
                
                # Normalize
                if np.max(np.abs(filtered)) > 0:
                    filtered = filtered / np.max(np.abs(filtered)) * 0.8
                
                return filtered.tolist()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_data
    
    async def recognize_speech_offline(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech using offline ASR capabilities.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result
        """
        try:
            if not self.local_asr_engine:
                raise RuntimeError("Local ASR engine not available")
            
            # Simulate offline speech recognition
            # In production, this would use a lightweight ASR model
            
            # For common queries, check cache first
            audio_hash = self._generate_audio_hash(audio)
            cached_result = self._get_cached_recognition(audio_hash)
            
            if cached_result:
                self.offline_stats['cache_hits'] += 1
                logger.debug("Using cached recognition result")
                return cached_result
            
            self.offline_stats['cache_misses'] += 1
            
            # Simulate recognition processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate mock result for offline processing
            # In production, this would use actual ASR
            result = RecognitionResult(
                transcribed_text="[Offline Recognition]",
                confidence=0.7,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.1
            )
            
            # Cache the result
            self._cache_recognition_result(audio_hash, result)
            
            self.offline_stats['offline_queries_processed'] += 1
            logger.info("Speech recognized offline")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in offline speech recognition: {e}")
            return RecognitionResult(
                transcribed_text="",
                confidence=0.0,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.0
            )
    
    async def synthesize_speech_offline(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech using offline TTS capabilities.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        try:
            if not self.local_tts_engine:
                raise RuntimeError("Local TTS engine not available")
            
            # Check for cached voice model
            model_key = f"{language.value}:{accent.value}:medium"
            
            if model_key in self.cached_voice_models:
                # Use cached model for synthesis
                return await self._synthesize_with_cached_model(text, model_key)
            else:
                # Generate basic synthesis
                return await self._generate_basic_synthesis(text, language)
        
        except Exception as e:
            logger.error(f"Error in offline speech synthesis: {e}")
            # Return silence as fallback
            return AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
    
    async def _synthesize_with_cached_model(
        self, 
        text: str, 
        model_key: str
    ) -> AudioBuffer:
        """
        Synthesize speech using cached voice model.
        
        Args:
            text: Text to synthesize
            model_key: Cached model key
            
        Returns:
            Synthesized audio buffer
        """
        try:
            # Load model data from database if needed
            model_info = self.cached_voice_models[model_key]
            
            if not model_info.model_data:
                # Load model data from database
                with sqlite3.connect(self.models_db_path) as conn:
                    cursor = conn.execute("""
                        SELECT model_data FROM offline_models 
                        WHERE language = ? AND accent = ? AND quality = ?
                    """, (model_info.language.value, model_info.accent.value, model_info.quality))
                    
                    row = cursor.fetchone()
                    if row:
                        model_info.model_data = row[0]
            
            # Simulate synthesis with cached model
            await asyncio.sleep(0.05)  # Simulate processing time
            
            # Generate synthetic audio (in production, use actual model)
            duration = len(text) * 0.1  # Rough estimate
            sample_count = int(16000 * duration)
            
            # Generate simple tone pattern for text
            audio_data = self._generate_synthetic_audio(text, sample_count)
            
            # Update model usage
            model_info.last_used = datetime.now()
            
            self.offline_stats['offline_tts_synthesized'] += 1
            
            return AudioBuffer(
                data=audio_data,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing with cached model: {e}")
            return await self._generate_basic_synthesis(text, LanguageCode.ENGLISH_IN)
    
    async def _generate_basic_synthesis(
        self, 
        text: str, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Generate basic speech synthesis without cached models.
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Basic synthesized audio buffer
        """
        try:
            # Generate very basic synthesis (tone patterns)
            duration = max(0.5, len(text) * 0.08)  # Minimum 0.5 seconds
            sample_count = int(16000 * duration)
            
            audio_data = self._generate_synthetic_audio(text, sample_count)
            
            return AudioBuffer(
                data=audio_data,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error in basic synthesis: {e}")
            # Return silence
            return AudioBuffer(
                data=[0.0] * 1600,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
    
    def _generate_synthetic_audio(self, text: str, sample_count: int) -> List[float]:
        """
        Generate synthetic audio data for text.
        
        Args:
            text: Input text
            sample_count: Number of audio samples to generate
            
        Returns:
            List of audio samples
        """
        try:
            # Generate simple tone pattern based on text
            # This is a very basic implementation for offline fallback
            
            # Use text hash to generate consistent tone pattern
            text_hash = hash(text) % 1000
            base_freq = 200 + (text_hash % 300)  # 200-500 Hz range
            
            # Generate sine wave pattern
            samples = []
            for i in range(sample_count):
                t = i / 16000.0  # Time in seconds
                
                # Create varying frequency based on text characteristics
                freq_variation = base_freq + 50 * np.sin(2 * np.pi * t * 2)
                
                # Generate sample
                sample = 0.3 * np.sin(2 * np.pi * freq_variation * t)
                
                # Add some envelope to make it sound more natural
                envelope = np.exp(-t * 2) if t < 0.5 else np.exp(-(t - 0.5) * 1)
                sample *= envelope
                
                samples.append(float(sample))
            
            return samples
            
        except Exception as e:
            logger.error(f"Error generating synthetic audio: {e}")
            return [0.0] * sample_count
    
    async def process_common_query(self, query_text: str, language: LanguageCode) -> Optional[str]:
        """
        Process common query using offline cache.
        
        Args:
            query_text: Query text
            language: Query language
            
        Returns:
            Cached response if available, None otherwise
        """
        try:
            query_key = f"{query_text.lower().strip()}:{language.value}"
            
            if query_key in self.common_queries:
                query = self.common_queries[query_key]
                query.usage_count += 1
                
                # Update usage count in database
                with sqlite3.connect(self.queries_db_path) as conn:
                    conn.execute("""
                        UPDATE offline_queries 
                        SET usage_count = usage_count + 1 
                        WHERE query_text = ? AND language = ?
                    """, (query_text, language.value))
                
                self.offline_stats['cache_hits'] += 1
                logger.debug(f"Found cached response for query: {query_text[:50]}...")
                
                return query.response_text
            
            self.offline_stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error processing common query: {e}")
            return None
    
    async def cache_query_response(
        self, 
        query_text: str, 
        language: LanguageCode, 
        response_text: str,
        confidence: float = 1.0
    ):
        """
        Cache a query-response pair for offline use.
        
        Args:
            query_text: Query text
            language: Query language
            response_text: Response text
            confidence: Response confidence score
        """
        try:
            query_key = f"{query_text.lower().strip()}:{language.value}"
            
            # Create offline query object
            offline_query = OfflineQuery(
                query_text=query_text,
                language=language,
                response_text=response_text,
                confidence=confidence,
                timestamp=datetime.now(),
                usage_count=1
            )
            
            # Add to memory cache
            self.common_queries[query_key] = offline_query
            
            # Store in database
            with sqlite3.connect(self.queries_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO offline_queries 
                    (query_text, language, response_text, confidence, timestamp, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query_text, language.value, response_text, 
                    confidence, datetime.now().isoformat(), 1
                ))
            
            # Manage cache size
            await self._manage_cache_size()
            
            logger.debug(f"Cached query response: {query_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching query response: {e}")
    
    async def cache_voice_model(
        self, 
        language: LanguageCode, 
        accent: AccentType, 
        model_data: bytes,
        quality: str = "medium"
    ):
        """
        Cache a voice model for offline TTS.
        
        Args:
            language: Model language
            accent: Model accent
            model_data: Serialized model data
            quality: Model quality level
        """
        try:
            model_key = f"{language.value}:{accent.value}:{quality}"
            size_mb = len(model_data) / (1024 * 1024)
            
            # Create voice model object
            voice_model = OfflineVoiceModel(
                language=language,
                accent=accent,
                model_data=model_data,
                quality=quality,
                size_mb=size_mb,
                last_used=datetime.now()
            )
            
            # Add to memory cache
            self.cached_voice_models[model_key] = voice_model
            
            # Store in database
            with sqlite3.connect(self.models_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO offline_models 
                    (language, accent, model_data, quality, size_mb, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    language.value, accent.value, model_data, 
                    quality, size_mb, datetime.now().isoformat()
                ))
            
            # Manage cache size
            await self._manage_cache_size()
            
            logger.info(f"Cached voice model: {language.value}-{accent.value} ({size_mb:.1f}MB)")
            
        except Exception as e:
            logger.error(f"Error caching voice model: {e}")
    
    async def _manage_cache_size(self):
        """Manage cache size to stay within limits."""
        try:
            # Calculate current cache size
            total_size_mb = 0
            
            # Check models cache size
            with sqlite3.connect(self.models_db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_mb) FROM offline_models")
                result = cursor.fetchone()
                if result and result[0]:
                    total_size_mb += result[0]
            
            # If over limit, remove least recently used items
            if total_size_mb > self.max_cache_size_mb:
                logger.info(f"Cache size ({total_size_mb:.1f}MB) exceeds limit ({self.max_cache_size_mb}MB)")
                
                # Remove oldest voice models
                with sqlite3.connect(self.models_db_path) as conn:
                    # Get models sorted by last_used
                    cursor = conn.execute("""
                        SELECT language, accent, quality, size_mb 
                        FROM offline_models 
                        ORDER BY last_used ASC
                    """)
                    
                    for row in cursor.fetchall():
                        if total_size_mb <= self.max_cache_size_mb * 0.8:  # Keep 20% buffer
                            break
                        
                        # Remove this model
                        conn.execute("""
                            DELETE FROM offline_models 
                            WHERE language = ? AND accent = ? AND quality = ?
                        """, (row[0], row[1], row[2]))
                        
                        # Remove from memory cache
                        model_key = f"{row[0]}:{row[1]}:{row[2]}"
                        if model_key in self.cached_voice_models:
                            del self.cached_voice_models[model_key]
                        
                        total_size_mb -= row[3]
                        logger.debug(f"Removed cached model: {model_key}")
                
                logger.info(f"Cache size reduced to {total_size_mb:.1f}MB")
            
            # Limit number of cached queries
            if len(self.common_queries) > self.common_queries_limit:
                # Remove least used queries
                sorted_queries = sorted(
                    self.common_queries.items(),
                    key=lambda x: x[1].usage_count
                )
                
                queries_to_remove = len(self.common_queries) - self.common_queries_limit
                
                for i in range(queries_to_remove):
                    query_key = sorted_queries[i][0]
                    query = sorted_queries[i][1]
                    
                    # Remove from database
                    with sqlite3.connect(self.queries_db_path) as conn:
                        conn.execute("""
                            DELETE FROM offline_queries 
                            WHERE query_text = ? AND language = ?
                        """, (query.query_text, query.language.value))
                    
                    # Remove from memory
                    del self.common_queries[query_key]
                
                logger.info(f"Removed {queries_to_remove} least used queries")
            
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")
    
    def _generate_audio_hash(self, audio: AudioBuffer) -> str:
        """Generate hash for audio buffer."""
        import hashlib
        
        # Create hash based on audio characteristics
        sample_size = min(100, len(audio.data))
        audio_sample = audio.data[:sample_size]
        
        key_data = f"{audio.sample_rate}:{audio.channels}:{hash(tuple(audio_sample))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_recognition(self, audio_hash: str) -> Optional[RecognitionResult]:
        """Get cached recognition result."""
        # This would be implemented with a proper cache
        # For now, return None to indicate no cache hit
        return None
    
    def _cache_recognition_result(self, audio_hash: str, result: RecognitionResult):
        """Cache recognition result."""
        # This would be implemented with a proper cache
        # For now, just log the caching attempt
        logger.debug(f"Would cache recognition result for hash: {audio_hash}")
    
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame using offline methods.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        try:
            # Simple energy-based VAD for offline use
            audio_array = np.array(audio_frame.data)
            
            # Calculate energy
            energy = np.sum(audio_array ** 2) / len(audio_array)
            
            # Simple threshold-based detection
            energy_threshold = 0.01
            is_speech = energy > energy_threshold
            
            # Calculate confidence based on energy level
            confidence = min(1.0, energy / (energy_threshold * 5))
            
            return VoiceActivityResult(
                is_speech=is_speech,
                confidence=confidence,
                energy_level=float(energy),
                processing_time=0.001
            )
            
        except Exception as e:
            logger.error(f"Error in offline VAD: {e}")
            return VoiceActivityResult(
                is_speech=False,
                confidence=0.0,
                energy_level=0.0,
                processing_time=0.0
            )
    
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise using offline methods.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        try:
            filtered_data = self._apply_basic_noise_reduction(audio_data.data)
            
            return AudioBuffer(
                data=filtered_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(filtered_data) / audio_data.sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error in offline noise filtering: {e}")
            return audio_data
    
    def get_offline_stats(self) -> Dict[str, Any]:
        """
        Get offline processing statistics.
        
        Returns:
            Dictionary with offline statistics
        """
        stats = self.offline_stats.copy()
        stats.update({
            'network_status': self.network_status.value,
            'cached_queries_count': len(self.common_queries),
            'cached_models_count': len(self.cached_voice_models),
            'cache_dir_size_mb': self._get_cache_dir_size_mb(),
            'local_asr_available': self.local_asr_engine is not None,
            'local_tts_available': self.local_tts_engine is not None
        })
        
        return stats
    
    def _get_cache_dir_size_mb(self) -> float:
        """Get cache directory size in MB."""
        try:
            total_size = 0
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0.0
    
    async def clear_offline_cache(self, cache_type: str = "all"):
        """
        Clear offline cache.
        
        Args:
            cache_type: Type of cache to clear ("queries", "models", "all")
        """
        try:
            if cache_type in ["queries", "all"]:
                # Clear queries cache
                self.common_queries.clear()
                
                with sqlite3.connect(self.queries_db_path) as conn:
                    conn.execute("DELETE FROM offline_queries")
                
                logger.info("Cleared offline queries cache")
            
            if cache_type in ["models", "all"]:
                # Clear models cache
                self.cached_voice_models.clear()
                
                with sqlite3.connect(self.models_db_path) as conn:
                    conn.execute("DELETE FROM offline_models")
                
                logger.info("Cleared offline models cache")
            
            # Reset statistics
            if cache_type == "all":
                self.offline_stats = {
                    'offline_queries_processed': 0,
                    'offline_tts_synthesized': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'network_switches': 0,
                    'total_offline_time': 0.0
                }
            
        except Exception as e:
            logger.error(f"Error clearing offline cache: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of offline voice processor.
        
        Returns:
            Health check result
        """
        try:
            # Check network connectivity
            network_status = await self.check_network_connectivity()
            
            # Check cache databases
            queries_db_ok = self.queries_db_path.exists()
            models_db_ok = self.models_db_path.exists()
            
            # Check local engines
            local_asr_ok = self.local_asr_engine is not None
            local_tts_ok = self.local_tts_engine is not None
            
            # Test basic functionality
            test_audio = AudioBuffer(
                data=[0.1] * 1600,  # 0.1 seconds of test audio
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
            
            # Test VAD
            vad_result = await self.detect_voice_activity(test_audio)
            vad_ok = vad_result is not None
            
            # Test noise filtering
            filtered_audio = await self.filter_background_noise(test_audio)
            noise_filter_ok = filtered_audio is not None
            
            overall_status = (
                "healthy" if all([
                    queries_db_ok, models_db_ok, vad_ok, noise_filter_ok
                ]) else "degraded"
            )
            
            return {
                'status': overall_status,
                'network_status': network_status.value,
                'databases': {
                    'queries_db': 'ok' if queries_db_ok else 'error',
                    'models_db': 'ok' if models_db_ok else 'error'
                },
                'local_engines': {
                    'asr': 'available' if local_asr_ok else 'unavailable',
                    'tts': 'available' if local_tts_ok else 'unavailable'
                },
                'functionality_tests': {
                    'vad': 'ok' if vad_ok else 'error',
                    'noise_filter': 'ok' if noise_filter_ok else 'error'
                },
                'cache_stats': {
                    'queries_count': len(self.common_queries),
                    'models_count': len(self.cached_voice_models),
                    'cache_size_mb': self._get_cache_dir_size_mb()
                },
                'offline_stats': self.get_offline_stats()
            }
            
        except Exception as e:
            logger.error(f"Offline voice processor health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'offline_stats': self.get_offline_stats()
            }


# Factory function for creating offline voice processor
def create_offline_voice_processor(
    cache_dir: str = ".bharatvoice_offline",
    max_cache_size_mb: int = 500,
    enable_local_asr: bool = True,
    enable_local_tts: bool = True,
    common_queries_limit: int = 1000
) -> OfflineVoiceProcessor:
    """
    Factory function to create an offline voice processor instance.
    
    Args:
        cache_dir: Directory for offline cache storage
        max_cache_size_mb: Maximum cache size in MB
        enable_local_asr: Whether to enable local ASR
        enable_local_tts: Whether to enable local TTS
        common_queries_limit: Maximum number of cached common queries
        
    Returns:
        Configured OfflineVoiceProcessor instance
    """
    return OfflineVoiceProcessor(
        cache_dir=cache_dir,
        max_cache_size_mb=max_cache_size_mb,
        enable_local_asr=enable_local_asr,
        enable_local_tts=enable_local_tts,
        common_queries_limit=common_queries_limit
=======
"""
Offline voice processing system for BharatVoice Assistant.

This module provides local speech recognition, TTS synthesis, and basic query
processing capabilities that work without internet connectivity.
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel
from bharatvoice.core.interfaces import AudioProcessor as AudioProcessorInterface
from bharatvoice.core.models import (
    AudioBuffer,
    AudioFormat,
    LanguageCode,
    AccentType,
    RecognitionResult,
    VoiceActivityResult,
)
   

logger = logging.getLogger(__name__)


class NetworkStatus(str, Enum):
    """Network connectivity status."""
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"


class OfflineQuery(BaseModel):
    """Represents an offline query with cached response."""
    
    query_text: str
    language: LanguageCode
    response_text: str
    confidence: float
    timestamp: datetime
    usage_count: int = 0


class OfflineVoiceModel(BaseModel):
    """Represents a cached voice model for offline TTS."""
    
    language: LanguageCode
    accent: AccentType
    model_data: bytes
    quality: str
    size_mb: float
    last_used: datetime


class OfflineVoiceProcessor(AudioProcessorInterface):
    """
    Offline voice processing system that provides local speech recognition,
    TTS synthesis, and basic query processing without internet connectivity.
    """
    
    def __init__(
        self,
        cache_dir: str = ".bharatvoice_offline",
        max_cache_size_mb: int = 500,
        enable_local_asr: bool = True,
        enable_local_tts: bool = True,
        common_queries_limit: int = 1000
    ):
        """
        Initialize offline voice processor.
        
        Args:
            cache_dir: Directory for offline cache storage
            max_cache_size_mb: Maximum cache size in MB
            enable_local_asr: Whether to enable local ASR
            enable_local_tts: Whether to enable local TTS
            common_queries_limit: Maximum number of cached common queries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_local_asr = enable_local_asr
        self.enable_local_tts = enable_local_tts
        self.common_queries_limit = common_queries_limit
        
        # Initialize offline databases
        self.queries_db_path = self.cache_dir / "offline_queries.db"
        self.models_db_path = self.cache_dir / "offline_models.db"
        
        # Network status tracking
        self.network_status = NetworkStatus.ONLINE
        self.last_connectivity_check = datetime.now()
        
        # Offline caches
        self.common_queries: Dict[str, OfflineQuery] = {}
        self.cached_voice_models: Dict[str, OfflineVoiceModel] = {}
        self.offline_responses: Dict[str, str] = {}
        
        # Local processing components (lightweight versions)
        self.local_asr_engine = None
        self.local_tts_engine = None
        
        # Statistics
        self.offline_stats = {
            'offline_queries_processed': 0,
            'offline_tts_synthesized': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'network_switches': 0,
            'total_offline_time': 0.0
        }
        
        # Initialize offline databases
        self._init_offline_databases()
        
        # Load cached data
        self._load_offline_caches()
        
        # Initialize local processing engines
        if enable_local_asr:
            self._init_local_asr()
        
        if enable_local_tts:
            self._init_local_tts()
        
        logger.info("OfflineVoiceProcessor initialized successfully")
    
    def _init_offline_databases(self):
        """Initialize SQLite databases for offline storage."""
        try:
            # Initialize queries database
            with sqlite3.connect(self.queries_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        language TEXT NOT NULL,
                        response_text TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        UNIQUE(query_text, language)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_language 
                    ON offline_queries(query_text, language)
                """)
            
            # Initialize models database
            with sqlite3.connect(self.models_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        language TEXT NOT NULL,
                        accent TEXT NOT NULL,
                        model_data BLOB NOT NULL,
                        quality TEXT NOT NULL,
                        size_mb REAL NOT NULL,
                        last_used TEXT NOT NULL,
                        UNIQUE(language, accent, quality)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_lang_accent 
                    ON offline_models(language, accent)
                """)
            
            logger.info("Offline databases initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing offline databases: {e}")
    
    def _load_offline_caches(self):
        """Load cached data from databases into memory."""
        try:
            # Load common queries
            with sqlite3.connect(self.queries_db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_text, language, response_text, confidence, 
                           timestamp, usage_count
                    FROM offline_queries 
                    ORDER BY usage_count DESC 
                    LIMIT ?
                """, (self.common_queries_limit,))
                
                for row in cursor.fetchall():
                    query_key = f"{row[0]}:{row[1]}"
                    self.common_queries[query_key] = OfflineQuery(
                        query_text=row[0],
                        language=LanguageCode(row[1]),
                        response_text=row[2],
                        confidence=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        usage_count=row[5]
                    )
            
            # Load voice models metadata (not the actual model data for memory efficiency)
            with sqlite3.connect(self.models_db_path) as conn:
                cursor = conn.execute("""
                    SELECT language, accent, quality, size_mb, last_used
                    FROM offline_models
                """)
                
                for row in cursor.fetchall():
                    model_key = f"{row[0]}:{row[1]}:{row[2]}"
                    self.cached_voice_models[model_key] = OfflineVoiceModel(
                        language=LanguageCode(row[0]),
                        accent=AccentType(row[1]),
                        model_data=b"",  # Load on demand
                        quality=row[2],
                        size_mb=row[3],
                        last_used=datetime.fromisoformat(row[4])
                    )
            
            logger.info(
                f"Loaded {len(self.common_queries)} cached queries and "
                f"{len(self.cached_voice_models)} voice models"
            )
            
        except Exception as e:
            logger.error(f"Error loading offline caches: {e}")
    
    def _init_local_asr(self):
        """Initialize lightweight local ASR engine."""
        try:
            # For offline ASR, we'll use a simplified approach
            # In production, this could use a lightweight model like Whisper tiny
            self.local_asr_engine = {
                'model_type': 'lightweight',
                'supported_languages': [
                    LanguageCode.HINDI,
                    LanguageCode.ENGLISH_IN,
                    LanguageCode.TAMIL,
                    LanguageCode.BENGALI
                ],
                'confidence_threshold': 0.6
            }
            
            logger.info("Local ASR engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing local ASR: {e}")
            self.local_asr_engine = None
    
    def _init_local_tts(self):
        """Initialize lightweight local TTS engine."""
        try:
            # For offline TTS, we'll use cached voice models
            self.local_tts_engine = {
                'engine_type': 'cached_models',
                'supported_languages': list(LanguageCode),
                'default_quality': 'medium',
                'cache_enabled': True
            }
            
            logger.info("Local TTS engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing local TTS: {e}")
            self.local_tts_engine = None
    
    async def check_network_connectivity(self) -> NetworkStatus:
        """
        Check current network connectivity status.
        
        Returns:
            Current network status
        """
        try:
            import socket
            
            # Try to connect to a reliable server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            if self.network_status != NetworkStatus.ONLINE:
                self.offline_stats['network_switches'] += 1
                logger.info("Network connectivity restored")
            
            self.network_status = NetworkStatus.ONLINE
            
        except (socket.error, OSError):
            if self.network_status != NetworkStatus.OFFLINE:
                self.offline_stats['network_switches'] += 1
                logger.info("Network connectivity lost - switching to offline mode")
            
            self.network_status = NetworkStatus.OFFLINE
        
        self.last_connectivity_check = datetime.now()
        return self.network_status
    
    async def process_audio_stream(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio stream with offline capabilities.
        
        Args:
            audio_data: Input audio buffer
            language: Target language for processing
            
        Returns:
            Processed audio buffer
        """
        try:
            # Check network status
            network_status = await self.check_network_connectivity()
            
            if network_status == NetworkStatus.OFFLINE:
                # Use offline processing
                return await self._process_audio_offline(audio_data, language)
            else:
                # Try online processing first, fallback to offline
                try:
                    # This would call the online voice processing service
                    # For now, we'll simulate online processing
                    return await self._process_audio_offline(audio_data, language)
                except Exception as e:
                    logger.warning(f"Online processing failed, using offline: {e}")
                    return await self._process_audio_offline(audio_data, language)
        
        except Exception as e:
            logger.error(f"Error in process_audio_stream: {e}")
            raise
    
    async def _process_audio_offline(
        self, 
        audio_data: AudioBuffer, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Process audio using offline capabilities.
        
        Args:
            audio_data: Input audio buffer
            language: Target language
            
        Returns:
            Processed audio buffer
        """
        try:
            # Apply basic noise reduction
            processed_data = self._apply_basic_noise_reduction(audio_data.data)
            
            # Create processed audio buffer
            processed_audio = AudioBuffer(
                data=processed_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(processed_data) / audio_data.sample_rate
            )
            
            logger.debug(f"Processed audio offline for language: {language}")
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error in offline audio processing: {e}")
            return audio_data
    
    def _apply_basic_noise_reduction(self, audio_data: List[float]) -> List[float]:
        """
        Apply basic noise reduction to audio data.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Noise-reduced audio samples
        """
        try:
            # Convert to numpy array for processing
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Apply simple high-pass filter to remove low-frequency noise
            # This is a basic implementation - production would use more sophisticated methods
            if len(audio_array) > 1:
                # Simple difference filter
                filtered = np.diff(audio_array, prepend=audio_array[0])
                
                # Normalize
                if np.max(np.abs(filtered)) > 0:
                    filtered = filtered / np.max(np.abs(filtered)) * 0.8
                
                return filtered.tolist()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_data
    
    async def recognize_speech_offline(self, audio: AudioBuffer) -> RecognitionResult:
        """
        Recognize speech using offline ASR capabilities.
        
        Args:
            audio: Audio buffer containing speech
            
        Returns:
            Speech recognition result
        """
        try:
            if not self.local_asr_engine:
                raise RuntimeError("Local ASR engine not available")
            
            # Simulate offline speech recognition
            # In production, this would use a lightweight ASR model
            
            # For common queries, check cache first
            audio_hash = self._generate_audio_hash(audio)
            cached_result = self._get_cached_recognition(audio_hash)
            
            if cached_result:
                self.offline_stats['cache_hits'] += 1
                logger.debug("Using cached recognition result")
                return cached_result
            
            self.offline_stats['cache_misses'] += 1
            
            # Simulate recognition processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate mock result for offline processing
            # In production, this would use actual ASR
            result = RecognitionResult(
                transcribed_text="[Offline Recognition]",
                confidence=0.7,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.1
            )
            
            # Cache the result
            self._cache_recognition_result(audio_hash, result)
            
            self.offline_stats['offline_queries_processed'] += 1
            logger.info("Speech recognized offline")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in offline speech recognition: {e}")
            return RecognitionResult(
                transcribed_text="",
                confidence=0.0,
                detected_language=LanguageCode.ENGLISH_IN,
                code_switching_points=[],
                alternative_transcriptions=[],
                processing_time=0.0
            )
    
    async def synthesize_speech_offline(
        self, 
        text: str, 
        language: LanguageCode, 
        accent: AccentType = AccentType.STANDARD
    ) -> AudioBuffer:
        """
        Synthesize speech using offline TTS capabilities.
        
        Args:
            text: Text to synthesize
            language: Target language
            accent: Regional accent type
            
        Returns:
            Synthesized audio buffer
        """
        try:
            if not self.local_tts_engine:
                raise RuntimeError("Local TTS engine not available")
            
            # Check for cached voice model
            model_key = f"{language.value}:{accent.value}:medium"
            
            if model_key in self.cached_voice_models:
                # Use cached model for synthesis
                return await self._synthesize_with_cached_model(text, model_key)
            else:
                # Generate basic synthesis
                return await self._generate_basic_synthesis(text, language)
        
        except Exception as e:
            logger.error(f"Error in offline speech synthesis: {e}")
            # Return silence as fallback
            return AudioBuffer(
                data=[0.0] * 1600,  # 0.1 seconds of silence
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
    
    async def _synthesize_with_cached_model(
        self, 
        text: str, 
        model_key: str
    ) -> AudioBuffer:
        """
        Synthesize speech using cached voice model.
        
        Args:
            text: Text to synthesize
            model_key: Cached model key
            
        Returns:
            Synthesized audio buffer
        """
        try:
            # Load model data from database if needed
            model_info = self.cached_voice_models[model_key]
            
            if not model_info.model_data:
                # Load model data from database
                with sqlite3.connect(self.models_db_path) as conn:
                    cursor = conn.execute("""
                        SELECT model_data FROM offline_models 
                        WHERE language = ? AND accent = ? AND quality = ?
                    """, (model_info.language.value, model_info.accent.value, model_info.quality))
                    
                    row = cursor.fetchone()
                    if row:
                        model_info.model_data = row[0]
            
            # Simulate synthesis with cached model
            await asyncio.sleep(0.05)  # Simulate processing time
            
            # Generate synthetic audio (in production, use actual model)
            duration = len(text) * 0.1  # Rough estimate
            sample_count = int(16000 * duration)
            
            # Generate simple tone pattern for text
            audio_data = self._generate_synthetic_audio(text, sample_count)
            
            # Update model usage
            model_info.last_used = datetime.now()
            
            self.offline_stats['offline_tts_synthesized'] += 1
            
            return AudioBuffer(
                data=audio_data,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing with cached model: {e}")
            return await self._generate_basic_synthesis(text, LanguageCode.ENGLISH_IN)
    
    async def _generate_basic_synthesis(
        self, 
        text: str, 
        language: LanguageCode
    ) -> AudioBuffer:
        """
        Generate basic speech synthesis without cached models.
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Basic synthesized audio buffer
        """
        try:
            # Generate very basic synthesis (tone patterns)
            duration = max(0.5, len(text) * 0.08)  # Minimum 0.5 seconds
            sample_count = int(16000 * duration)
            
            audio_data = self._generate_synthetic_audio(text, sample_count)
            
            return AudioBuffer(
                data=audio_data,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error in basic synthesis: {e}")
            # Return silence
            return AudioBuffer(
                data=[0.0] * 1600,
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
    
    def _generate_synthetic_audio(self, text: str, sample_count: int) -> List[float]:
        """
        Generate synthetic audio data for text.
        
        Args:
            text: Input text
            sample_count: Number of audio samples to generate
            
        Returns:
            List of audio samples
        """
        try:
            # Generate simple tone pattern based on text
            # This is a very basic implementation for offline fallback
            
            # Use text hash to generate consistent tone pattern
            text_hash = hash(text) % 1000
            base_freq = 200 + (text_hash % 300)  # 200-500 Hz range
            
            # Generate sine wave pattern
            samples = []
            for i in range(sample_count):
                t = i / 16000.0  # Time in seconds
                
                # Create varying frequency based on text characteristics
                freq_variation = base_freq + 50 * np.sin(2 * np.pi * t * 2)
                
                # Generate sample
                sample = 0.3 * np.sin(2 * np.pi * freq_variation * t)
                
                # Add some envelope to make it sound more natural
                envelope = np.exp(-t * 2) if t < 0.5 else np.exp(-(t - 0.5) * 1)
                sample *= envelope
                
                samples.append(float(sample))
            
            return samples
            
        except Exception as e:
            logger.error(f"Error generating synthetic audio: {e}")
            return [0.0] * sample_count
    
    async def process_common_query(self, query_text: str, language: LanguageCode) -> Optional[str]:
        """
        Process common query using offline cache.
        
        Args:
            query_text: Query text
            language: Query language
            
        Returns:
            Cached response if available, None otherwise
        """
        try:
            query_key = f"{query_text.lower().strip()}:{language.value}"
            
            if query_key in self.common_queries:
                query = self.common_queries[query_key]
                query.usage_count += 1
                
                # Update usage count in database
                with sqlite3.connect(self.queries_db_path) as conn:
                    conn.execute("""
                        UPDATE offline_queries 
                        SET usage_count = usage_count + 1 
                        WHERE query_text = ? AND language = ?
                    """, (query_text, language.value))
                
                self.offline_stats['cache_hits'] += 1
                logger.debug(f"Found cached response for query: {query_text[:50]}...")
                
                return query.response_text
            
            self.offline_stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error processing common query: {e}")
            return None
    
    async def cache_query_response(
        self, 
        query_text: str, 
        language: LanguageCode, 
        response_text: str,
        confidence: float = 1.0
    ):
        """
        Cache a query-response pair for offline use.
        
        Args:
            query_text: Query text
            language: Query language
            response_text: Response text
            confidence: Response confidence score
        """
        try:
            query_key = f"{query_text.lower().strip()}:{language.value}"
            
            # Create offline query object
            offline_query = OfflineQuery(
                query_text=query_text,
                language=language,
                response_text=response_text,
                confidence=confidence,
                timestamp=datetime.now(),
                usage_count=1
            )
            
            # Add to memory cache
            self.common_queries[query_key] = offline_query
            
            # Store in database
            with sqlite3.connect(self.queries_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO offline_queries 
                    (query_text, language, response_text, confidence, timestamp, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query_text, language.value, response_text, 
                    confidence, datetime.now().isoformat(), 1
                ))
            
            # Manage cache size
            await self._manage_cache_size()
            
            logger.debug(f"Cached query response: {query_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching query response: {e}")
    
    async def cache_voice_model(
        self, 
        language: LanguageCode, 
        accent: AccentType, 
        model_data: bytes,
        quality: str = "medium"
    ):
        """
        Cache a voice model for offline TTS.
        
        Args:
            language: Model language
            accent: Model accent
            model_data: Serialized model data
            quality: Model quality level
        """
        try:
            model_key = f"{language.value}:{accent.value}:{quality}"
            size_mb = len(model_data) / (1024 * 1024)
            
            # Create voice model object
            voice_model = OfflineVoiceModel(
                language=language,
                accent=accent,
                model_data=model_data,
                quality=quality,
                size_mb=size_mb,
                last_used=datetime.now()
            )
            
            # Add to memory cache
            self.cached_voice_models[model_key] = voice_model
            
            # Store in database
            with sqlite3.connect(self.models_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO offline_models 
                    (language, accent, model_data, quality, size_mb, last_used)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    language.value, accent.value, model_data, 
                    quality, size_mb, datetime.now().isoformat()
                ))
            
            # Manage cache size
            await self._manage_cache_size()
            
            logger.info(f"Cached voice model: {language.value}-{accent.value} ({size_mb:.1f}MB)")
            
        except Exception as e:
            logger.error(f"Error caching voice model: {e}")
    
    async def _manage_cache_size(self):
        """Manage cache size to stay within limits."""
        try:
            # Calculate current cache size
            total_size_mb = 0
            
            # Check models cache size
            with sqlite3.connect(self.models_db_path) as conn:
                cursor = conn.execute("SELECT SUM(size_mb) FROM offline_models")
                result = cursor.fetchone()
                if result and result[0]:
                    total_size_mb += result[0]
            
            # If over limit, remove least recently used items
            if total_size_mb > self.max_cache_size_mb:
                logger.info(f"Cache size ({total_size_mb:.1f}MB) exceeds limit ({self.max_cache_size_mb}MB)")
                
                # Remove oldest voice models
                with sqlite3.connect(self.models_db_path) as conn:
                    # Get models sorted by last_used
                    cursor = conn.execute("""
                        SELECT language, accent, quality, size_mb 
                        FROM offline_models 
                        ORDER BY last_used ASC
                    """)
                    
                    for row in cursor.fetchall():
                        if total_size_mb <= self.max_cache_size_mb * 0.8:  # Keep 20% buffer
                            break
                        
                        # Remove this model
                        conn.execute("""
                            DELETE FROM offline_models 
                            WHERE language = ? AND accent = ? AND quality = ?
                        """, (row[0], row[1], row[2]))
                        
                        # Remove from memory cache
                        model_key = f"{row[0]}:{row[1]}:{row[2]}"
                        if model_key in self.cached_voice_models:
                            del self.cached_voice_models[model_key]
                        
                        total_size_mb -= row[3]
                        logger.debug(f"Removed cached model: {model_key}")
                
                logger.info(f"Cache size reduced to {total_size_mb:.1f}MB")
            
            # Limit number of cached queries
            if len(self.common_queries) > self.common_queries_limit:
                # Remove least used queries
                sorted_queries = sorted(
                    self.common_queries.items(),
                    key=lambda x: x[1].usage_count
                )
                
                queries_to_remove = len(self.common_queries) - self.common_queries_limit
                
                for i in range(queries_to_remove):
                    query_key = sorted_queries[i][0]
                    query = sorted_queries[i][1]
                    
                    # Remove from database
                    with sqlite3.connect(self.queries_db_path) as conn:
                        conn.execute("""
                            DELETE FROM offline_queries 
                            WHERE query_text = ? AND language = ?
                        """, (query.query_text, query.language.value))
                    
                    # Remove from memory
                    del self.common_queries[query_key]
                
                logger.info(f"Removed {queries_to_remove} least used queries")
            
        except Exception as e:
            logger.error(f"Error managing cache size: {e}")
    
    def _generate_audio_hash(self, audio: AudioBuffer) -> str:
        """Generate hash for audio buffer."""
        import hashlib
        
        # Create hash based on audio characteristics
        sample_size = min(100, len(audio.data))
        audio_sample = audio.data[:sample_size]
        
        key_data = f"{audio.sample_rate}:{audio.channels}:{hash(tuple(audio_sample))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_recognition(self, audio_hash: str) -> Optional[RecognitionResult]:
        """Get cached recognition result."""
        # This would be implemented with a proper cache
        # For now, return None to indicate no cache hit
        return None
    
    def _cache_recognition_result(self, audio_hash: str, result: RecognitionResult):
        """Cache recognition result."""
        # This would be implemented with a proper cache
        # For now, just log the caching attempt
        logger.debug(f"Would cache recognition result for hash: {audio_hash}")
    
    async def detect_voice_activity(self, audio_frame: AudioBuffer) -> VoiceActivityResult:
        """
        Detect voice activity in audio frame using offline methods.
        
        Args:
            audio_frame: Audio frame to analyze
            
        Returns:
            Voice activity detection result
        """
        try:
            # Simple energy-based VAD for offline use
            audio_array = np.array(audio_frame.data)
            
            # Calculate energy
            energy = np.sum(audio_array ** 2) / len(audio_array)
            
            # Simple threshold-based detection
            energy_threshold = 0.01
            is_speech = energy > energy_threshold
            
            # Calculate confidence based on energy level
            confidence = min(1.0, energy / (energy_threshold * 5))
            
            return VoiceActivityResult(
                is_speech=is_speech,
                confidence=confidence,
                energy_level=float(energy),
                processing_time=0.001
            )
            
        except Exception as e:
            logger.error(f"Error in offline VAD: {e}")
            return VoiceActivityResult(
                is_speech=False,
                confidence=0.0,
                energy_level=0.0,
                processing_time=0.0
            )
    
    async def filter_background_noise(self, audio_data: AudioBuffer) -> AudioBuffer:
        """
        Filter background noise using offline methods.
        
        Args:
            audio_data: Input audio with noise
            
        Returns:
            Filtered audio buffer
        """
        try:
            filtered_data = self._apply_basic_noise_reduction(audio_data.data)
            
            return AudioBuffer(
                data=filtered_data,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                format=audio_data.format,
                duration=len(filtered_data) / audio_data.sample_rate
            )
            
        except Exception as e:
            logger.error(f"Error in offline noise filtering: {e}")
            return audio_data
    
    def get_offline_stats(self) -> Dict[str, Any]:
        """
        Get offline processing statistics.
        
        Returns:
            Dictionary with offline statistics
        """
        stats = self.offline_stats.copy()
        stats.update({
            'network_status': self.network_status.value,
            'cached_queries_count': len(self.common_queries),
            'cached_models_count': len(self.cached_voice_models),
            'cache_dir_size_mb': self._get_cache_dir_size_mb(),
            'local_asr_available': self.local_asr_engine is not None,
            'local_tts_available': self.local_tts_engine is not None
        })
        
        return stats
    
    def _get_cache_dir_size_mb(self) -> float:
        """Get cache directory size in MB."""
        try:
            total_size = 0
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0.0
    
    async def clear_offline_cache(self, cache_type: str = "all"):
        """
        Clear offline cache.
        
        Args:
            cache_type: Type of cache to clear ("queries", "models", "all")
        """
        try:
            if cache_type in ["queries", "all"]:
                # Clear queries cache
                self.common_queries.clear()
                
                with sqlite3.connect(self.queries_db_path) as conn:
                    conn.execute("DELETE FROM offline_queries")
                
                logger.info("Cleared offline queries cache")
            
            if cache_type in ["models", "all"]:
                # Clear models cache
                self.cached_voice_models.clear()
                
                with sqlite3.connect(self.models_db_path) as conn:
                    conn.execute("DELETE FROM offline_models")
                
                logger.info("Cleared offline models cache")
            
            # Reset statistics
            if cache_type == "all":
                self.offline_stats = {
                    'offline_queries_processed': 0,
                    'offline_tts_synthesized': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'network_switches': 0,
                    'total_offline_time': 0.0
                }
            
        except Exception as e:
            logger.error(f"Error clearing offline cache: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of offline voice processor.
        
        Returns:
            Health check result
        """
        try:
            # Check network connectivity
            network_status = await self.check_network_connectivity()
            
            # Check cache databases
            queries_db_ok = self.queries_db_path.exists()
            models_db_ok = self.models_db_path.exists()
            
            # Check local engines
            local_asr_ok = self.local_asr_engine is not None
            local_tts_ok = self.local_tts_engine is not None
            
            # Test basic functionality
            test_audio = AudioBuffer(
                data=[0.1] * 1600,  # 0.1 seconds of test audio
                sample_rate=16000,
                channels=1,
                format=AudioFormat.WAV,
                duration=0.1
            )
            
            # Test VAD
            vad_result = await self.detect_voice_activity(test_audio)
            vad_ok = vad_result is not None
            
            # Test noise filtering
            filtered_audio = await self.filter_background_noise(test_audio)
            noise_filter_ok = filtered_audio is not None
            
            overall_status = (
                "healthy" if all([
                    queries_db_ok, models_db_ok, vad_ok, noise_filter_ok
                ]) else "degraded"
            )
            
            return {
                'status': overall_status,
                'network_status': network_status.value,
                'databases': {
                    'queries_db': 'ok' if queries_db_ok else 'error',
                    'models_db': 'ok' if models_db_ok else 'error'
                },
                'local_engines': {
                    'asr': 'available' if local_asr_ok else 'unavailable',
                    'tts': 'available' if local_tts_ok else 'unavailable'
                },
                'functionality_tests': {
                    'vad': 'ok' if vad_ok else 'error',
                    'noise_filter': 'ok' if noise_filter_ok else 'error'
                },
                'cache_stats': {
                    'queries_count': len(self.common_queries),
                    'models_count': len(self.cached_voice_models),
                    'cache_size_mb': self._get_cache_dir_size_mb()
                },
                'offline_stats': self.get_offline_stats()
            }
            
        except Exception as e:
            logger.error(f"Offline voice processor health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'offline_stats': self.get_offline_stats()
            }


# Factory function for creating offline voice processor
def create_offline_voice_processor(
    cache_dir: str = ".bharatvoice_offline",
    max_cache_size_mb: int = 500,
    enable_local_asr: bool = True,
    enable_local_tts: bool = True,
    common_queries_limit: int = 1000
) -> OfflineVoiceProcessor:
    """
    Factory function to create an offline voice processor instance.
    
    Args:
        cache_dir: Directory for offline cache storage
        max_cache_size_mb: Maximum cache size in MB
        enable_local_asr: Whether to enable local ASR
        enable_local_tts: Whether to enable local TTS
        common_queries_limit: Maximum number of cached common queries
        
    Returns:
        Configured OfflineVoiceProcessor instance
    """
    return OfflineVoiceProcessor(
        cache_dir=cache_dir,
        max_cache_size_mb=max_cache_size_mb,
        enable_local_asr=enable_local_asr,
        enable_local_tts=enable_local_tts,
        common_queries_limit=common_queries_limit
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    )