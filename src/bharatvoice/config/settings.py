"""
Configuration settings for BharatVoice Assistant.

This module provides centralized configuration management using Pydantic settings
with support for environment variables and configuration files.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="sqlite:///./bharatvoice.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        env="DATABASE_ECHO",
        description="Enable SQL query logging"
    )
    pool_size: int = Field(
        default=10,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=20,
        env="DATABASE_MAX_OVERFLOW",
        description="Maximum database connection overflow"
    )


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=20,
        env="REDIS_MAX_CONNECTIONS",
        description="Maximum Redis connections"
    )
    socket_timeout: float = Field(
        default=5.0,
        env="REDIS_SOCKET_TIMEOUT",
        description="Redis socket timeout in seconds"
    )


class AudioSettings(BaseSettings):
    """Audio processing configuration settings."""
    
    sample_rate: int = Field(
        default=16000,
        env="AUDIO_SAMPLE_RATE",
        description="Default audio sample rate"
    )
    chunk_size: int = Field(
        default=1024,
        env="AUDIO_CHUNK_SIZE",
        description="Audio processing chunk size"
    )
    max_audio_length: int = Field(
        default=300,
        env="AUDIO_MAX_LENGTH",
        description="Maximum audio length in seconds"
    )
    noise_reduction_level: float = Field(
        default=0.5,
        env="AUDIO_NOISE_REDUCTION",
        description="Background noise reduction level (0.0-1.0)"
    )


class SpeechSettings(BaseSettings):
    """Speech recognition and synthesis settings."""
    
    whisper_model: str = Field(
        default="base",
        env="WHISPER_MODEL",
        description="Whisper model size (tiny, base, small, medium, large)"
    )
    whisper_device: str = Field(
        default="cpu",
        env="WHISPER_DEVICE",
        description="Device for Whisper inference (cpu, cuda)"
    )
    tts_engine: str = Field(
        default="gtts",
        env="TTS_ENGINE",
        description="Text-to-speech engine (gtts, coqui)"
    )
    recognition_timeout: float = Field(
        default=30.0,
        env="SPEECH_RECOGNITION_TIMEOUT",
        description="Speech recognition timeout in seconds"
    )


class SecuritySettings(BaseSettings):
    """Security and encryption settings."""
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY",
        description="Secret key for JWT tokens and encryption"
    )
    algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM",
        description="JWT signing algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration time in minutes"
    )
    encryption_key: Optional[str] = Field(
        default=None,
        env="ENCRYPTION_KEY",
        description="AES encryption key for sensitive data"
    )


class ExternalServiceSettings(BaseSettings):
    """External service integration settings."""
    
    # Indian Railways APIs
    indian_railways_api_key: Optional[str] = Field(
        default=None,
        env="INDIAN_RAILWAYS_API_KEY",
        description="Indian Railways API key"
    )
    irctc_connect_api_key: Optional[str] = Field(
        default=None,
        env="IRCTC_CONNECT_API_KEY",
        description="IRCTC Connect API key"
    )
    confirmtkt_api_key: Optional[str] = Field(
        default=None,
        env="CONFIRMTKT_API_KEY",
        description="ConfirmTkt API key"
    )
    
    # Weather Services
    openweathermap_api_key: Optional[str] = Field(
        default=None,
        env="OPENWEATHERMAP_API_KEY",
        description="OpenWeatherMap API key"
    )
    imd_api_key: Optional[str] = Field(
        default=None,
        env="IMD_API_KEY",
        description="Indian Meteorological Department API key"
    )
    accuweather_api_key: Optional[str] = Field(
        default=None,
        env="ACCUWEATHER_API_KEY",
        description="AccuWeather API key"
    )
    weather_underground_api_key: Optional[str] = Field(
        default=None,
        env="WEATHER_UNDERGROUND_API_KEY",
        description="Weather Underground API key"
    )
    
    # Digital India Platform
    digital_india_api_key: Optional[str] = Field(
        default=None,
        env="DIGITAL_INDIA_API_KEY",
        description="Digital India API key"
    )
    digital_india_client_id: Optional[str] = Field(
        default=None,
        env="DIGITAL_INDIA_CLIENT_ID",
        description="Digital India Client ID"
    )
    digital_india_client_secret: Optional[str] = Field(
        default=None,
        env="DIGITAL_INDIA_CLIENT_SECRET",
        description="Digital India Client Secret"
    )
    aadhaar_verification_api_key: Optional[str] = Field(
        default=None,
        env="AADHAAR_VERIFICATION_API_KEY",
        description="Aadhaar verification API key"
    )
    pan_verification_api_key: Optional[str] = Field(
        default=None,
        env="PAN_VERIFICATION_API_KEY",
        description="PAN verification API key"
    )
    passport_api_key: Optional[str] = Field(
        default=None,
        env="PASSPORT_API_KEY",
        description="Passport services API key"
    )
    driving_license_api_key: Optional[str] = Field(
        default=None,
        env="DRIVING_LICENSE_API_KEY",
        description="Driving license verification API key"
    )
    
    # UPI Payment Gateways
    razorpay_key_id: Optional[str] = Field(
        default=None,
        env="RAZORPAY_KEY_ID",
        description="Razorpay Key ID"
    )
    razorpay_key_secret: Optional[str] = Field(
        default=None,
        env="RAZORPAY_KEY_SECRET",
        description="Razorpay Key Secret"
    )
    razorpay_webhook_secret: Optional[str] = Field(
        default=None,
        env="RAZORPAY_WEBHOOK_SECRET",
        description="Razorpay Webhook Secret"
    )
    payu_merchant_key: Optional[str] = Field(
        default=None,
        env="PAYU_MERCHANT_KEY",
        description="PayU Merchant Key"
    )
    payu_merchant_salt: Optional[str] = Field(
        default=None,
        env="PAYU_MERCHANT_SALT",
        description="PayU Merchant Salt"
    )
    paytm_merchant_id: Optional[str] = Field(
        default=None,
        env="PAYTM_MERCHANT_ID",
        description="Paytm Merchant ID"
    )
    paytm_merchant_key: Optional[str] = Field(
        default=None,
        env="PAYTM_MERCHANT_KEY",
        description="Paytm Merchant Key"
    )
    
    # Food Delivery Platforms
    swiggy_partner_id: Optional[str] = Field(
        default=None,
        env="SWIGGY_PARTNER_ID",
        description="Swiggy Partner ID"
    )
    swiggy_api_key: Optional[str] = Field(
        default=None,
        env="SWIGGY_API_KEY",
        description="Swiggy API Key"
    )
    zomato_partner_id: Optional[str] = Field(
        default=None,
        env="ZOMATO_PARTNER_ID",
        description="Zomato Partner ID"
    )
    zomato_api_key: Optional[str] = Field(
        default=None,
        env="ZOMATO_API_KEY",
        description="Zomato API Key"
    )
    uber_eats_client_id: Optional[str] = Field(
        default=None,
        env="UBER_EATS_CLIENT_ID",
        description="Uber Eats Client ID"
    )
    uber_eats_client_secret: Optional[str] = Field(
        default=None,
        env="UBER_EATS_CLIENT_SECRET",
        description="Uber Eats Client Secret"
    )
    
    # Ride Sharing Platforms
    ola_client_id: Optional[str] = Field(
        default=None,
        env="OLA_CLIENT_ID",
        description="Ola Client ID"
    )
    ola_client_secret: Optional[str] = Field(
        default=None,
        env="OLA_CLIENT_SECRET",
        description="Ola Client Secret"
    )
    uber_client_id: Optional[str] = Field(
        default=None,
        env="UBER_CLIENT_ID",
        description="Uber Client ID"
    )
    uber_client_secret: Optional[str] = Field(
        default=None,
        env="UBER_CLIENT_SECRET",
        description="Uber Client Secret"
    )
    rapido_api_key: Optional[str] = Field(
        default=None,
        env="RAPIDO_API_KEY",
        description="Rapido API Key"
    )
    
    # Service Booking Platforms
    urban_company_partner_id: Optional[str] = Field(
        default=None,
        env="URBAN_COMPANY_PARTNER_ID",
        description="Urban Company Partner ID"
    )
    urban_company_api_key: Optional[str] = Field(
        default=None,
        env="URBAN_COMPANY_API_KEY",
        description="Urban Company API Key"
    )
    justdial_api_key: Optional[str] = Field(
        default=None,
        env="JUSTDIAL_API_KEY",
        description="JustDial API Key"
    )
    
    # Entertainment and News APIs
    cricapi_api_key: Optional[str] = Field(
        default=None,
        env="CRICAPI_API_KEY",
        description="CricAPI Key"
    )
    espn_cricinfo_api_key: Optional[str] = Field(
        default=None,
        env="ESPN_CRICINFO_API_KEY",
        description="ESPN Cricinfo API Key"
    )
    news_api_key: Optional[str] = Field(
        default=None,
        env="NEWS_API_KEY",
        description="NewsAPI Key"
    )
    tmdb_api_key: Optional[str] = Field(
        default=None,
        env="TMDB_API_KEY",
        description="The Movie Database API Key"
    )
    
    # Legacy fields for backward compatibility
    weather_api_key: Optional[str] = Field(
        default=None,
        env="WEATHER_API_KEY",
        description="Weather service API key (legacy)"
    )
    upi_gateway_url: Optional[str] = Field(
        default=None,
        env="UPI_GATEWAY_URL",
        description="UPI payment gateway URL (legacy)"
    )
    cricket_api_key: Optional[str] = Field(
        default=None,
        env="CRICKET_API_KEY",
        description="Cricket scores API key (legacy)"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Log message format"
    )
    file_path: Optional[str] = Field(
        default=None,
        env="LOG_FILE_PATH",
        description="Log file path (if None, logs to console)"
    )
    max_file_size: int = Field(
        default=10485760,  # 10MB
        env="LOG_MAX_FILE_SIZE",
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        env="LOG_BACKUP_COUNT",
        description="Number of log file backups to keep"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics settings."""
    
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable Prometheus metrics collection"
    )
    metrics_port: int = Field(
        default=8001,
        env="METRICS_PORT",
        description="Port for metrics endpoint"
    )
    health_check_interval: int = Field(
        default=30,
        env="HEALTH_CHECK_INTERVAL",
        description="Health check interval in seconds"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field(
        default="BharatVoice Assistant",
        env="APP_NAME",
        description="Application name"
    )
    version: str = Field(
        default="0.1.0",
        env="APP_VERSION",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Application environment (development, staging, production)"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        env="HOST",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        env="PORT",
        description="Server port"
    )
    workers: int = Field(
        default=1,
        env="WORKERS",
        description="Number of worker processes"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        env="CORS_METHODS",
        description="Allowed CORS methods"
    )
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    external_services: ExternalServiceSettings = Field(default_factory=ExternalServiceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # Feature flags
    enable_offline_mode: bool = Field(
        default=True,
        env="ENABLE_OFFLINE_MODE",
        description="Enable offline functionality"
    )
    enable_code_switching: bool = Field(
        default=True,
        env="ENABLE_CODE_SWITCHING",
        description="Enable code-switching detection"
    )
    enable_cultural_context: bool = Field(
        default=True,
        env="ENABLE_CULTURAL_CONTEXT",
        description="Enable cultural context understanding"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Cached settings instance
    """
    return Settings()