<<<<<<< HEAD
"""
Localized error handling system for BharatVoice Assistant.

This module provides comprehensive error handling with localized error messages
in multiple Indian languages and cultural context awareness.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, Union
import structlog

from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the system."""
    
    # Authentication errors
    AUTH_INVALID_CREDENTIALS = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"
    AUTH_MFA_REQUIRED = "AUTH_004"
    
    # Voice processing errors
    VOICE_RECOGNITION_FAILED = "VOICE_001"
    VOICE_SYNTHESIS_FAILED = "VOICE_002"
    VOICE_AUDIO_FORMAT_INVALID = "VOICE_003"
    VOICE_AUDIO_TOO_LONG = "VOICE_004"
    VOICE_NOISE_TOO_HIGH = "VOICE_005"
    
    # Language processing errors
    LANG_UNSUPPORTED_LANGUAGE = "LANG_001"
    LANG_TRANSLATION_FAILED = "LANG_002"
    LANG_CODE_SWITCHING_FAILED = "LANG_003"
    LANG_CONTEXT_UNDERSTANDING_FAILED = "LANG_004"
    
    # External service errors
    EXT_SERVICE_UNAVAILABLE = "EXT_001"
    EXT_API_RATE_LIMITED = "EXT_002"
    EXT_INVALID_RESPONSE = "EXT_003"
    EXT_TIMEOUT = "EXT_004"
    
    # System errors
    SYS_INTERNAL_ERROR = "SYS_001"
    SYS_DATABASE_ERROR = "SYS_002"
    SYS_CACHE_ERROR = "SYS_003"
    SYS_RESOURCE_EXHAUSTED = "SYS_004"
    SYS_PERFORMANCE_DEGRADED = "SYS_005"
    
    # User input errors
    INPUT_INVALID_FORMAT = "INPUT_001"
    INPUT_MISSING_REQUIRED = "INPUT_002"
    INPUT_VALUE_OUT_OF_RANGE = "INPUT_003"
    INPUT_UNSUPPORTED_OPERATION = "INPUT_004"
    
    # Network errors
    NET_CONNECTION_FAILED = "NET_001"
    NET_TIMEOUT = "NET_002"
    NET_OFFLINE_MODE = "NET_003"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Localized error messages
ERROR_MESSAGES = {
    # English (India)
    LanguageCode.ENGLISH_INDIA: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "Invalid username or password. Please check your credentials and try again.",
        ErrorCode.AUTH_TOKEN_EXPIRED: "Your session has expired. Please log in again to continue.",
        ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: "You don't have permission to access this feature. Please contact support if you need access.",
        ErrorCode.AUTH_MFA_REQUIRED: "Multi-factor authentication is required. Please complete the verification process.",
        
        ErrorCode.VOICE_RECOGNITION_FAILED: "I couldn't understand what you said. Please speak clearly and try again.",
        ErrorCode.VOICE_SYNTHESIS_FAILED: "I'm having trouble speaking right now. Please try again in a moment.",
        ErrorCode.VOICE_AUDIO_FORMAT_INVALID: "The audio format is not supported. Please use a different audio file.",
        ErrorCode.VOICE_AUDIO_TOO_LONG: "The audio is too long. Please keep your message under 5 minutes.",
        ErrorCode.VOICE_NOISE_TOO_HIGH: "There's too much background noise. Please try speaking in a quieter place.",
        
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "I don't support this language yet. Please try speaking in Hindi or English.",
        ErrorCode.LANG_TRANSLATION_FAILED: "I couldn't translate that properly. Please try rephrasing your request.",
        ErrorCode.LANG_CODE_SWITCHING_FAILED: "I had trouble understanding the mixed languages. Please try using one language at a time.",
        ErrorCode.LANG_CONTEXT_UNDERSTANDING_FAILED: "I didn't quite understand the context. Could you please be more specific?",
        
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "The service is temporarily unavailable. Please try again later.",
        ErrorCode.EXT_API_RATE_LIMITED: "Too many requests right now. Please wait a moment and try again.",
        ErrorCode.EXT_INVALID_RESPONSE: "I received unexpected information from the service. Please try again.",
        ErrorCode.EXT_TIMEOUT: "The service is taking too long to respond. Please try again.",
        
        ErrorCode.SYS_INTERNAL_ERROR: "Something went wrong on our end. Please try again later.",
        ErrorCode.SYS_DATABASE_ERROR: "I'm having trouble accessing my memory. Please try again in a moment.",
        ErrorCode.SYS_CACHE_ERROR: "I'm having trouble with my quick memory. The request might take a bit longer.",
        ErrorCode.SYS_RESOURCE_EXHAUSTED: "I'm handling too many requests right now. Please try again in a few minutes.",
        ErrorCode.SYS_PERFORMANCE_DEGRADED: "I'm running a bit slow right now. Your request might take longer than usual.",
        
        ErrorCode.INPUT_INVALID_FORMAT: "I didn't understand the format of your request. Please try asking differently.",
        ErrorCode.INPUT_MISSING_REQUIRED: "Some required information is missing. Please provide all the details I need.",
        ErrorCode.INPUT_VALUE_OUT_OF_RANGE: "The value you provided is outside the acceptable range.",
        ErrorCode.INPUT_UNSUPPORTED_OPERATION: "I can't perform that operation. Please try a different request.",
        
        ErrorCode.NET_CONNECTION_FAILED: "I can't connect to the internet right now. Some features may not work.",
        ErrorCode.NET_TIMEOUT: "The connection is taking too long. Please check your internet and try again.",
        ErrorCode.NET_OFFLINE_MODE: "You're in offline mode. Some features are limited but I can still help with basic tasks.",
    },
    
    # Hindi
    LanguageCode.HINDI: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "गलत उपयोगकर्ता नाम या पासवर्ड। कृपया अपनी जानकारी जांचें और फिर से कोशिश करें।",
        ErrorCode.AUTH_TOKEN_EXPIRED: "आपका सत्र समाप्त हो गया है। कृपया जारी रखने के लिए फिर से लॉग इन करें।",
        ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: "आपको इस सुविधा का उपयोग करने की अनुमति नहीं है। यदि आपको पहुंच चाहिए तो कृपया सहायता से संपर्क करें।",
        ErrorCode.AUTH_MFA_REQUIRED: "बहु-कारक प्रमाणीकरण आवश्यक है। कृपया सत्यापन प्रक्रिया पूरी करें।",
        
        ErrorCode.VOICE_RECOGNITION_FAILED: "मैं समझ नहीं पाया कि आपने क्या कहा। कृपया स्पष्ट रूप से बोलें और फिर से कोशिश करें।",
        ErrorCode.VOICE_SYNTHESIS_FAILED: "मुझे अभी बोलने में परेशानी हो रही है। कृपया एक क्षण में फिर से कोशिश करें।",
        ErrorCode.VOICE_AUDIO_FORMAT_INVALID: "ऑडियो प्रारूप समर्थित नहीं है। कृपया एक अलग ऑडियो फ़ाइल का उपयोग करें।",
        ErrorCode.VOICE_AUDIO_TOO_LONG: "ऑडियो बहुत लंबा है। कृपया अपना संदेश 5 मिनट से कम रखें।",
        ErrorCode.VOICE_NOISE_TOO_HIGH: "बहुत अधिक पृष्ठभूमि शोर है। कृपया शांत जगह पर बोलने की कोशिश करें।",
        
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "मैं अभी तक इस भाषा का समर्थन नहीं करता। कृपया हिंदी या अंग्रेजी में बोलने की कोशिश करें।",
        ErrorCode.LANG_TRANSLATION_FAILED: "मैं इसका सही अनुवाद नहीं कर पाया। कृपया अपना अनुरोध दूसरे तरीके से कहने की कोशिश करें।",
        ErrorCode.LANG_CODE_SWITCHING_FAILED: "मुझे मिश्रित भाषाओं को समझने में परेशानी हुई। कृपया एक समय में एक भाषा का उपयोग करने की कोशिश करें।",
        ErrorCode.LANG_CONTEXT_UNDERSTANDING_FAILED: "मैं संदर्भ को पूरी तरह से नहीं समझ पाया। क्या आप कृपया अधिक स्पष्ट हो सकते हैं?",
        
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "सेवा अस्थायी रूप से अनुपलब्ध है। कृपया बाद में फिर से कोशिश करें।",
        ErrorCode.EXT_API_RATE_LIMITED: "अभी बहुत सारे अनुरोध हैं। कृपया एक क्षण प्रतीक्षा करें और फिर से कोशिश करें।",
        ErrorCode.EXT_INVALID_RESPONSE: "मुझे सेवा से अप्रत्याशित जानकारी मिली। कृपया फिर से कोशिश करें।",
        ErrorCode.EXT_TIMEOUT: "सेवा का जवाब देने में बहुत समय लग रहा है। कृपया फिर से कोशिश करें।",
        
        ErrorCode.SYS_INTERNAL_ERROR: "हमारी तरफ से कुछ गलत हुआ। कृपया बाद में फिर से कोशिश करें।",
        ErrorCode.SYS_DATABASE_ERROR: "मुझे अपनी मेमोरी तक पहुंचने में परेशानी हो रही है। कृपया एक क्षण में फिर से कोशिश करें।",
        ErrorCode.SYS_CACHE_ERROR: "मुझे अपनी त्वरित मेमोरी के साथ परेशानी हो रही है। अनुरोध में थोड़ा अधिक समय लग सकता है।",
        ErrorCode.SYS_RESOURCE_EXHAUSTED: "मैं अभी बहुत सारे अनुरोधों को संभाल रहा हूं। कृपया कुछ मिनटों में फिर से कोशिश करें।",
        ErrorCode.SYS_PERFORMANCE_DEGRADED: "मैं अभी थोड़ा धीमा चल रहा हूं। आपके अनुरोध में सामान्य से अधिक समय लग सकता है।",
        
        ErrorCode.INPUT_INVALID_FORMAT: "मैं आपके अनुरोध का प्रारूप नहीं समझ पाया। कृपया अलग तरीके से पूछने की कोशिश करें।",
        ErrorCode.INPUT_MISSING_REQUIRED: "कुछ आवश्यक जानकारी गुम है। कृपया मुझे सभी आवश्यक विवरण प्रदान करें।",
        ErrorCode.INPUT_VALUE_OUT_OF_RANGE: "आपके द्वारा प्रदान किया गया मान स्वीकार्य सीमा से बाहर है।",
        ErrorCode.INPUT_UNSUPPORTED_OPERATION: "मैं वह ऑपरेशन नहीं कर सकता। कृपया एक अलग अनुरोध करने की कोशिश करें।",
        
        ErrorCode.NET_CONNECTION_FAILED: "मैं अभी इंटरनेट से कनेक्ट नहीं हो पा रहा। कुछ सुविधाएं काम नहीं कर सकती हैं।",
        ErrorCode.NET_TIMEOUT: "कनेक्शन में बहुत समय लग रहा है। कृपया अपना इंटरनेट जांचें और फिर से कोशिश करें।",
        ErrorCode.NET_OFFLINE_MODE: "आप ऑफ़लाइन मोड में हैं। कुछ सुविधाएं सीमित हैं लेकिन मैं अभी भी बुनियादी कार्यों में मदद कर सकता हूं।",
    },
    
    # Tamil
    LanguageCode.TAMIL: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "தவறான பயனர்பெயர் அல்லது கடவுச்சொல். உங்கள் விவரங்களைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.VOICE_RECOGNITION_FAILED: "நீங்கள் என்ன சொன்னீர்கள் என்று என்னால் புரிந்து கொள்ள முடியவில்லை. தயவுசெய்து தெளிவாகப் பேசி மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "நான் இந்த மொழியை இன்னும் ஆதரிக்கவில்லை. தயவுசெய்து தமிழ் அல்லது ஆங்கிலத்தில் பேச முயற்சிக்கவும்.",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "சேவை தற்காலிகமாக கிடைக்கவில்லை. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.SYS_INTERNAL_ERROR: "எங்கள் பக்கத்தில் ஏதோ தவறு நடந்தது. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.NET_OFFLINE_MODE: "நீங்கள் ஆஃப்லைன் பயன்முறையில் உள்ளீர்கள். சில அம்சங்கள் வரையறுக்கப்பட்டுள்ளன ஆனால் அடிப்படை பணிகளில் நான் இன்னும் உதவ முடியும்.",
    },
    
    # Telugu
    LanguageCode.TELUGU: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "తప్పుడు వినియోగదారు పేరు లేదా పాస్‌వర్డ్. దయచేసి మీ వివరాలను తనిఖీ చేసి మళ్లీ ప్రయత్నించండి.",
        ErrorCode.VOICE_RECOGNITION_FAILED: "మీరు ఏమి చెప్పారో నేను అర్థం చేసుకోలేకపోయాను. దయచేసి స్పష్టంగా మాట్లాడి మళ్లీ ప్రయత్నించండి.",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "నేను ఇంకా ఈ భాషకు మద్దతు ఇవ్వలేదు. దయచేసి తెలుగు లేదా ఇంగ్లీష్‌లో మాట్లాడటానికి ప్రయత్నించండి.",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "సేవ తాత్కాలికంగా అందుబాటులో లేదు. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
        ErrorCode.SYS_INTERNAL_ERROR: "మా వైపు ఏదో తప్పు జరిగింది. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
        ErrorCode.NET_OFFLINE_MODE: "మీరు ఆఫ్‌లైన్ మోడ్‌లో ఉన్నారు. కొన్ని ఫీచర్లు పరిమితం చేయబడ్డాయి కానీ ప్రాథమిక పనులతో నేను ఇంకా సహాయం చేయగలను.",
    },
    
    # Bengali
    LanguageCode.BENGALI: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "ভুল ব্যবহারকারীর নাম বা পাসওয়ার্ড। অনুগ্রহ করে আপনার তথ্য পরীক্ষা করুন এবং আবার চেষ্টা করুন।",
        ErrorCode.VOICE_RECOGNITION_FAILED: "আপনি কী বলেছেন তা আমি বুঝতে পারিনি। অনুগ্রহ করে স্পষ্টভাবে কথা বলুন এবং আবার চেষ্টা করুন।",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "আমি এখনও এই ভাষা সমর্থন করি না। অনুগ্রহ করে বাংলা বা ইংরেজিতে কথা বলার চেষ্টা করুন।",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "সেবা সাময়িকভাবে অনুপলব্ধ। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
        ErrorCode.SYS_INTERNAL_ERROR: "আমাদের দিক থেকে কিছু ভুল হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
        ErrorCode.NET_OFFLINE_MODE: "আপনি অফলাইন মোডে আছেন। কিছু বৈশিষ্ট্য সীমিত কিন্তু আমি এখনও মৌলিক কাজে সাহায্য করতে পারি।",
    },
}


class LocalizedError(Exception):
    """
    Localized error with multi-language support.
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.error_code = error_code
        self.language = language
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        
        # Get localized message
        self.message = self._get_localized_message()
        super().__init__(self.message)
    
    def _get_localized_message(self) -> str:
        """Get localized error message."""
        # Try to get message in requested language
        if self.language in ERROR_MESSAGES:
            lang_messages = ERROR_MESSAGES[self.language]
            if self.error_code in lang_messages:
                return lang_messages[self.error_code]
        
        # Fallback to English (India)
        if LanguageCode.ENGLISH_INDIA in ERROR_MESSAGES:
            eng_messages = ERROR_MESSAGES[LanguageCode.ENGLISH_INDIA]
            if self.error_code in eng_messages:
                return eng_messages[self.error_code]
        
        # Final fallback
        return f"An error occurred: {self.error_code.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "language": self.language.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": time.time()
        }


class ErrorHandler:
    """
    Centralized error handling system with localization support.
    """
    
    def __init__(self):
        self.logger = logger
    
    def handle_error(
        self,
        error: Union[Exception, ErrorCode],
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        log_error: bool = True
    ) -> LocalizedError:
        """
        Handle and localize an error.
        
        Args:
            error: Exception or error code to handle
            language: Target language for error message
            severity: Error severity level
            context: Additional context information
            log_error: Whether to log the error
            
        Returns:
            Localized error instance
        """
        if isinstance(error, ErrorCode):
            localized_error = LocalizedError(
                error_code=error,
                language=language,
                severity=severity,
                context=context
            )
        elif isinstance(error, LocalizedError):
            localized_error = error
        else:
            # Map common exceptions to error codes
            error_code = self._map_exception_to_code(error)
            localized_error = LocalizedError(
                error_code=error_code,
                language=language,
                severity=severity,
                context=context,
                original_error=error
            )
        
        if log_error:
            self._log_error(localized_error)
        
        return localized_error
    
    def _map_exception_to_code(self, error: Exception) -> ErrorCode:
        """Map common exceptions to error codes."""
        error_type = type(error).__name__
        
        mapping = {
            "ConnectionError": ErrorCode.NET_CONNECTION_FAILED,
            "TimeoutError": ErrorCode.NET_TIMEOUT,
            "ValueError": ErrorCode.INPUT_INVALID_FORMAT,
            "KeyError": ErrorCode.INPUT_MISSING_REQUIRED,
            "PermissionError": ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            "FileNotFoundError": ErrorCode.SYS_INTERNAL_ERROR,
            "MemoryError": ErrorCode.SYS_RESOURCE_EXHAUSTED,
            "DatabaseError": ErrorCode.SYS_DATABASE_ERROR,
        }
        
        return mapping.get(error_type, ErrorCode.SYS_INTERNAL_ERROR)
    
    def _log_error(self, error: LocalizedError) -> None:
        """Log error with structured logging."""
        log_data = {
            "error_code": error.error_code.value,
            "severity": error.severity.value,
            "language": error.language.value,
            "message": error.message,
            "context": error.context
        }
        
        if error.original_error:
            log_data["original_error"] = str(error.original_error)
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", **log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", **log_data)
        else:
            self.logger.info("Low severity error occurred", **log_data)
    
    def get_user_friendly_message(
        self,
        error_code: ErrorCode,
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get user-friendly error message.
        
        Args:
            error_code: Error code
            language: Target language
            context: Additional context for message formatting
            
        Returns:
            Localized user-friendly error message
        """
        localized_error = LocalizedError(
            error_code=error_code,
            language=language,
            context=context
        )
        return localized_error.message
    
    def create_error_response(
        self,
        error: Union[Exception, ErrorCode],
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error: Error to handle
            language: Target language
            include_context: Whether to include context in response
            
        Returns:
            Standardized error response dictionary
        """
        localized_error = self.handle_error(error, language)
        
        response = {
            "success": False,
            "error": {
                "code": localized_error.error_code.value,
                "message": localized_error.message,
                "severity": localized_error.severity.value
            }
        }
        
        if include_context and localized_error.context:
            response["error"]["context"] = localized_error.context
        
        return response


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    error: Union[Exception, ErrorCode],
    language: LanguageCode = LanguageCode.ENGLISH_INDIA,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None
) -> LocalizedError:
    """
    Convenience function to handle errors.
    
    Args:
        error: Error to handle
        language: Target language
        severity: Error severity
        context: Additional context
        
    Returns:
        Localized error instance
    """
    handler = get_error_handler()
    return handler.handle_error(error, language, severity, context)


def create_error_response(
    error: Union[Exception, ErrorCode],
    language: LanguageCode = LanguageCode.ENGLISH_INDIA
) -> Dict[str, Any]:
    """
    Convenience function to create error response.
    
    Args:
        error: Error to handle
        language: Target language
        
    Returns:
        Standardized error response
    """
    handler = get_error_handler()
=======
"""
Localized error handling system for BharatVoice Assistant.

This module provides comprehensive error handling with localized error messages
in multiple Indian languages and cultural context awareness.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, Union
import structlog

from bharatvoice.core.models import LanguageCode


logger = structlog.get_logger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the system."""
    
    # Authentication errors
    AUTH_INVALID_CREDENTIALS = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"
    AUTH_MFA_REQUIRED = "AUTH_004"
    
    # Voice processing errors
    VOICE_RECOGNITION_FAILED = "VOICE_001"
    VOICE_SYNTHESIS_FAILED = "VOICE_002"
    VOICE_AUDIO_FORMAT_INVALID = "VOICE_003"
    VOICE_AUDIO_TOO_LONG = "VOICE_004"
    VOICE_NOISE_TOO_HIGH = "VOICE_005"
    
    # Language processing errors
    LANG_UNSUPPORTED_LANGUAGE = "LANG_001"
    LANG_TRANSLATION_FAILED = "LANG_002"
    LANG_CODE_SWITCHING_FAILED = "LANG_003"
    LANG_CONTEXT_UNDERSTANDING_FAILED = "LANG_004"
    
    # External service errors
    EXT_SERVICE_UNAVAILABLE = "EXT_001"
    EXT_API_RATE_LIMITED = "EXT_002"
    EXT_INVALID_RESPONSE = "EXT_003"
    EXT_TIMEOUT = "EXT_004"
    
    # System errors
    SYS_INTERNAL_ERROR = "SYS_001"
    SYS_DATABASE_ERROR = "SYS_002"
    SYS_CACHE_ERROR = "SYS_003"
    SYS_RESOURCE_EXHAUSTED = "SYS_004"
    SYS_PERFORMANCE_DEGRADED = "SYS_005"
    
    # User input errors
    INPUT_INVALID_FORMAT = "INPUT_001"
    INPUT_MISSING_REQUIRED = "INPUT_002"
    INPUT_VALUE_OUT_OF_RANGE = "INPUT_003"
    INPUT_UNSUPPORTED_OPERATION = "INPUT_004"
    
    # Network errors
    NET_CONNECTION_FAILED = "NET_001"
    NET_TIMEOUT = "NET_002"
    NET_OFFLINE_MODE = "NET_003"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Localized error messages
ERROR_MESSAGES = {
    # English (India)
    LanguageCode.ENGLISH_INDIA: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "Invalid username or password. Please check your credentials and try again.",
        ErrorCode.AUTH_TOKEN_EXPIRED: "Your session has expired. Please log in again to continue.",
        ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: "You don't have permission to access this feature. Please contact support if you need access.",
        ErrorCode.AUTH_MFA_REQUIRED: "Multi-factor authentication is required. Please complete the verification process.",
        
        ErrorCode.VOICE_RECOGNITION_FAILED: "I couldn't understand what you said. Please speak clearly and try again.",
        ErrorCode.VOICE_SYNTHESIS_FAILED: "I'm having trouble speaking right now. Please try again in a moment.",
        ErrorCode.VOICE_AUDIO_FORMAT_INVALID: "The audio format is not supported. Please use a different audio file.",
        ErrorCode.VOICE_AUDIO_TOO_LONG: "The audio is too long. Please keep your message under 5 minutes.",
        ErrorCode.VOICE_NOISE_TOO_HIGH: "There's too much background noise. Please try speaking in a quieter place.",
        
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "I don't support this language yet. Please try speaking in Hindi or English.",
        ErrorCode.LANG_TRANSLATION_FAILED: "I couldn't translate that properly. Please try rephrasing your request.",
        ErrorCode.LANG_CODE_SWITCHING_FAILED: "I had trouble understanding the mixed languages. Please try using one language at a time.",
        ErrorCode.LANG_CONTEXT_UNDERSTANDING_FAILED: "I didn't quite understand the context. Could you please be more specific?",
        
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "The service is temporarily unavailable. Please try again later.",
        ErrorCode.EXT_API_RATE_LIMITED: "Too many requests right now. Please wait a moment and try again.",
        ErrorCode.EXT_INVALID_RESPONSE: "I received unexpected information from the service. Please try again.",
        ErrorCode.EXT_TIMEOUT: "The service is taking too long to respond. Please try again.",
        
        ErrorCode.SYS_INTERNAL_ERROR: "Something went wrong on our end. Please try again later.",
        ErrorCode.SYS_DATABASE_ERROR: "I'm having trouble accessing my memory. Please try again in a moment.",
        ErrorCode.SYS_CACHE_ERROR: "I'm having trouble with my quick memory. The request might take a bit longer.",
        ErrorCode.SYS_RESOURCE_EXHAUSTED: "I'm handling too many requests right now. Please try again in a few minutes.",
        ErrorCode.SYS_PERFORMANCE_DEGRADED: "I'm running a bit slow right now. Your request might take longer than usual.",
        
        ErrorCode.INPUT_INVALID_FORMAT: "I didn't understand the format of your request. Please try asking differently.",
        ErrorCode.INPUT_MISSING_REQUIRED: "Some required information is missing. Please provide all the details I need.",
        ErrorCode.INPUT_VALUE_OUT_OF_RANGE: "The value you provided is outside the acceptable range.",
        ErrorCode.INPUT_UNSUPPORTED_OPERATION: "I can't perform that operation. Please try a different request.",
        
        ErrorCode.NET_CONNECTION_FAILED: "I can't connect to the internet right now. Some features may not work.",
        ErrorCode.NET_TIMEOUT: "The connection is taking too long. Please check your internet and try again.",
        ErrorCode.NET_OFFLINE_MODE: "You're in offline mode. Some features are limited but I can still help with basic tasks.",
    },
    
    # Hindi
    LanguageCode.HINDI: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "गलत उपयोगकर्ता नाम या पासवर्ड। कृपया अपनी जानकारी जांचें और फिर से कोशिश करें।",
        ErrorCode.AUTH_TOKEN_EXPIRED: "आपका सत्र समाप्त हो गया है। कृपया जारी रखने के लिए फिर से लॉग इन करें।",
        ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: "आपको इस सुविधा का उपयोग करने की अनुमति नहीं है। यदि आपको पहुंच चाहिए तो कृपया सहायता से संपर्क करें।",
        ErrorCode.AUTH_MFA_REQUIRED: "बहु-कारक प्रमाणीकरण आवश्यक है। कृपया सत्यापन प्रक्रिया पूरी करें।",
        
        ErrorCode.VOICE_RECOGNITION_FAILED: "मैं समझ नहीं पाया कि आपने क्या कहा। कृपया स्पष्ट रूप से बोलें और फिर से कोशिश करें।",
        ErrorCode.VOICE_SYNTHESIS_FAILED: "मुझे अभी बोलने में परेशानी हो रही है। कृपया एक क्षण में फिर से कोशिश करें।",
        ErrorCode.VOICE_AUDIO_FORMAT_INVALID: "ऑडियो प्रारूप समर्थित नहीं है। कृपया एक अलग ऑडियो फ़ाइल का उपयोग करें।",
        ErrorCode.VOICE_AUDIO_TOO_LONG: "ऑडियो बहुत लंबा है। कृपया अपना संदेश 5 मिनट से कम रखें।",
        ErrorCode.VOICE_NOISE_TOO_HIGH: "बहुत अधिक पृष्ठभूमि शोर है। कृपया शांत जगह पर बोलने की कोशिश करें।",
        
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "मैं अभी तक इस भाषा का समर्थन नहीं करता। कृपया हिंदी या अंग्रेजी में बोलने की कोशिश करें।",
        ErrorCode.LANG_TRANSLATION_FAILED: "मैं इसका सही अनुवाद नहीं कर पाया। कृपया अपना अनुरोध दूसरे तरीके से कहने की कोशिश करें।",
        ErrorCode.LANG_CODE_SWITCHING_FAILED: "मुझे मिश्रित भाषाओं को समझने में परेशानी हुई। कृपया एक समय में एक भाषा का उपयोग करने की कोशिश करें।",
        ErrorCode.LANG_CONTEXT_UNDERSTANDING_FAILED: "मैं संदर्भ को पूरी तरह से नहीं समझ पाया। क्या आप कृपया अधिक स्पष्ट हो सकते हैं?",
        
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "सेवा अस्थायी रूप से अनुपलब्ध है। कृपया बाद में फिर से कोशिश करें।",
        ErrorCode.EXT_API_RATE_LIMITED: "अभी बहुत सारे अनुरोध हैं। कृपया एक क्षण प्रतीक्षा करें और फिर से कोशिश करें।",
        ErrorCode.EXT_INVALID_RESPONSE: "मुझे सेवा से अप्रत्याशित जानकारी मिली। कृपया फिर से कोशिश करें।",
        ErrorCode.EXT_TIMEOUT: "सेवा का जवाब देने में बहुत समय लग रहा है। कृपया फिर से कोशिश करें।",
        
        ErrorCode.SYS_INTERNAL_ERROR: "हमारी तरफ से कुछ गलत हुआ। कृपया बाद में फिर से कोशिश करें।",
        ErrorCode.SYS_DATABASE_ERROR: "मुझे अपनी मेमोरी तक पहुंचने में परेशानी हो रही है। कृपया एक क्षण में फिर से कोशिश करें।",
        ErrorCode.SYS_CACHE_ERROR: "मुझे अपनी त्वरित मेमोरी के साथ परेशानी हो रही है। अनुरोध में थोड़ा अधिक समय लग सकता है।",
        ErrorCode.SYS_RESOURCE_EXHAUSTED: "मैं अभी बहुत सारे अनुरोधों को संभाल रहा हूं। कृपया कुछ मिनटों में फिर से कोशिश करें।",
        ErrorCode.SYS_PERFORMANCE_DEGRADED: "मैं अभी थोड़ा धीमा चल रहा हूं। आपके अनुरोध में सामान्य से अधिक समय लग सकता है।",
        
        ErrorCode.INPUT_INVALID_FORMAT: "मैं आपके अनुरोध का प्रारूप नहीं समझ पाया। कृपया अलग तरीके से पूछने की कोशिश करें।",
        ErrorCode.INPUT_MISSING_REQUIRED: "कुछ आवश्यक जानकारी गुम है। कृपया मुझे सभी आवश्यक विवरण प्रदान करें।",
        ErrorCode.INPUT_VALUE_OUT_OF_RANGE: "आपके द्वारा प्रदान किया गया मान स्वीकार्य सीमा से बाहर है।",
        ErrorCode.INPUT_UNSUPPORTED_OPERATION: "मैं वह ऑपरेशन नहीं कर सकता। कृपया एक अलग अनुरोध करने की कोशिश करें।",
        
        ErrorCode.NET_CONNECTION_FAILED: "मैं अभी इंटरनेट से कनेक्ट नहीं हो पा रहा। कुछ सुविधाएं काम नहीं कर सकती हैं।",
        ErrorCode.NET_TIMEOUT: "कनेक्शन में बहुत समय लग रहा है। कृपया अपना इंटरनेट जांचें और फिर से कोशिश करें।",
        ErrorCode.NET_OFFLINE_MODE: "आप ऑफ़लाइन मोड में हैं। कुछ सुविधाएं सीमित हैं लेकिन मैं अभी भी बुनियादी कार्यों में मदद कर सकता हूं।",
    },
    
    # Tamil
    LanguageCode.TAMIL: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "தவறான பயனர்பெயர் அல்லது கடவுச்சொல். உங்கள் விவரங்களைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.VOICE_RECOGNITION_FAILED: "நீங்கள் என்ன சொன்னீர்கள் என்று என்னால் புரிந்து கொள்ள முடியவில்லை. தயவுசெய்து தெளிவாகப் பேசி மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "நான் இந்த மொழியை இன்னும் ஆதரிக்கவில்லை. தயவுசெய்து தமிழ் அல்லது ஆங்கிலத்தில் பேச முயற்சிக்கவும்.",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "சேவை தற்காலிகமாக கிடைக்கவில்லை. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.SYS_INTERNAL_ERROR: "எங்கள் பக்கத்தில் ஏதோ தவறு நடந்தது. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும்.",
        ErrorCode.NET_OFFLINE_MODE: "நீங்கள் ஆஃப்லைன் பயன்முறையில் உள்ளீர்கள். சில அம்சங்கள் வரையறுக்கப்பட்டுள்ளன ஆனால் அடிப்படை பணிகளில் நான் இன்னும் உதவ முடியும்.",
    },
    
    # Telugu
    LanguageCode.TELUGU: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "తప్పుడు వినియోగదారు పేరు లేదా పాస్‌వర్డ్. దయచేసి మీ వివరాలను తనిఖీ చేసి మళ్లీ ప్రయత్నించండి.",
        ErrorCode.VOICE_RECOGNITION_FAILED: "మీరు ఏమి చెప్పారో నేను అర్థం చేసుకోలేకపోయాను. దయచేసి స్పష్టంగా మాట్లాడి మళ్లీ ప్రయత్నించండి.",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "నేను ఇంకా ఈ భాషకు మద్దతు ఇవ్వలేదు. దయచేసి తెలుగు లేదా ఇంగ్లీష్‌లో మాట్లాడటానికి ప్రయత్నించండి.",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "సేవ తాత్కాలికంగా అందుబాటులో లేదు. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
        ErrorCode.SYS_INTERNAL_ERROR: "మా వైపు ఏదో తప్పు జరిగింది. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి.",
        ErrorCode.NET_OFFLINE_MODE: "మీరు ఆఫ్‌లైన్ మోడ్‌లో ఉన్నారు. కొన్ని ఫీచర్లు పరిమితం చేయబడ్డాయి కానీ ప్రాథమిక పనులతో నేను ఇంకా సహాయం చేయగలను.",
    },
    
    # Bengali
    LanguageCode.BENGALI: {
        ErrorCode.AUTH_INVALID_CREDENTIALS: "ভুল ব্যবহারকারীর নাম বা পাসওয়ার্ড। অনুগ্রহ করে আপনার তথ্য পরীক্ষা করুন এবং আবার চেষ্টা করুন।",
        ErrorCode.VOICE_RECOGNITION_FAILED: "আপনি কী বলেছেন তা আমি বুঝতে পারিনি। অনুগ্রহ করে স্পষ্টভাবে কথা বলুন এবং আবার চেষ্টা করুন।",
        ErrorCode.LANG_UNSUPPORTED_LANGUAGE: "আমি এখনও এই ভাষা সমর্থন করি না। অনুগ্রহ করে বাংলা বা ইংরেজিতে কথা বলার চেষ্টা করুন।",
        ErrorCode.EXT_SERVICE_UNAVAILABLE: "সেবা সাময়িকভাবে অনুপলব্ধ। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
        ErrorCode.SYS_INTERNAL_ERROR: "আমাদের দিক থেকে কিছু ভুল হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
        ErrorCode.NET_OFFLINE_MODE: "আপনি অফলাইন মোডে আছেন। কিছু বৈশিষ্ট্য সীমিত কিন্তু আমি এখনও মৌলিক কাজে সাহায্য করতে পারি।",
    },
}


class LocalizedError(Exception):
    """
    Localized error with multi-language support.
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.error_code = error_code
        self.language = language
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        
        # Get localized message
        self.message = self._get_localized_message()
        super().__init__(self.message)
    
    def _get_localized_message(self) -> str:
        """Get localized error message."""
        # Try to get message in requested language
        if self.language in ERROR_MESSAGES:
            lang_messages = ERROR_MESSAGES[self.language]
            if self.error_code in lang_messages:
                return lang_messages[self.error_code]
        
        # Fallback to English (India)
        if LanguageCode.ENGLISH_INDIA in ERROR_MESSAGES:
            eng_messages = ERROR_MESSAGES[LanguageCode.ENGLISH_INDIA]
            if self.error_code in eng_messages:
                return eng_messages[self.error_code]
        
        # Final fallback
        return f"An error occurred: {self.error_code.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "language": self.language.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": time.time()
        }


class ErrorHandler:
    """
    Centralized error handling system with localization support.
    """
    
    def __init__(self):
        self.logger = logger
    
    def handle_error(
        self,
        error: Union[Exception, ErrorCode],
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        log_error: bool = True
    ) -> LocalizedError:
        """
        Handle and localize an error.
        
        Args:
            error: Exception or error code to handle
            language: Target language for error message
            severity: Error severity level
            context: Additional context information
            log_error: Whether to log the error
            
        Returns:
            Localized error instance
        """
        if isinstance(error, ErrorCode):
            localized_error = LocalizedError(
                error_code=error,
                language=language,
                severity=severity,
                context=context
            )
        elif isinstance(error, LocalizedError):
            localized_error = error
        else:
            # Map common exceptions to error codes
            error_code = self._map_exception_to_code(error)
            localized_error = LocalizedError(
                error_code=error_code,
                language=language,
                severity=severity,
                context=context,
                original_error=error
            )
        
        if log_error:
            self._log_error(localized_error)
        
        return localized_error
    
    def _map_exception_to_code(self, error: Exception) -> ErrorCode:
        """Map common exceptions to error codes."""
        error_type = type(error).__name__
        
        mapping = {
            "ConnectionError": ErrorCode.NET_CONNECTION_FAILED,
            "TimeoutError": ErrorCode.NET_TIMEOUT,
            "ValueError": ErrorCode.INPUT_INVALID_FORMAT,
            "KeyError": ErrorCode.INPUT_MISSING_REQUIRED,
            "PermissionError": ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            "FileNotFoundError": ErrorCode.SYS_INTERNAL_ERROR,
            "MemoryError": ErrorCode.SYS_RESOURCE_EXHAUSTED,
            "DatabaseError": ErrorCode.SYS_DATABASE_ERROR,
        }
        
        return mapping.get(error_type, ErrorCode.SYS_INTERNAL_ERROR)
    
    def _log_error(self, error: LocalizedError) -> None:
        """Log error with structured logging."""
        log_data = {
            "error_code": error.error_code.value,
            "severity": error.severity.value,
            "language": error.language.value,
            "message": error.message,
            "context": error.context
        }
        
        if error.original_error:
            log_data["original_error"] = str(error.original_error)
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", **log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", **log_data)
        else:
            self.logger.info("Low severity error occurred", **log_data)
    
    def get_user_friendly_message(
        self,
        error_code: ErrorCode,
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get user-friendly error message.
        
        Args:
            error_code: Error code
            language: Target language
            context: Additional context for message formatting
            
        Returns:
            Localized user-friendly error message
        """
        localized_error = LocalizedError(
            error_code=error_code,
            language=language,
            context=context
        )
        return localized_error.message
    
    def create_error_response(
        self,
        error: Union[Exception, ErrorCode],
        language: LanguageCode = LanguageCode.ENGLISH_INDIA,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error: Error to handle
            language: Target language
            include_context: Whether to include context in response
            
        Returns:
            Standardized error response dictionary
        """
        localized_error = self.handle_error(error, language)
        
        response = {
            "success": False,
            "error": {
                "code": localized_error.error_code.value,
                "message": localized_error.message,
                "severity": localized_error.severity.value
            }
        }
        
        if include_context and localized_error.context:
            response["error"]["context"] = localized_error.context
        
        return response


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    error: Union[Exception, ErrorCode],
    language: LanguageCode = LanguageCode.ENGLISH_INDIA,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None
) -> LocalizedError:
    """
    Convenience function to handle errors.
    
    Args:
        error: Error to handle
        language: Target language
        severity: Error severity
        context: Additional context
        
    Returns:
        Localized error instance
    """
    handler = get_error_handler()
    return handler.handle_error(error, language, severity, context)


def create_error_response(
    error: Union[Exception, ErrorCode],
    language: LanguageCode = LanguageCode.ENGLISH_INDIA
) -> Dict[str, Any]:
    """
    Convenience function to create error response.
    
    Args:
        error: Error to handle
        language: Target language
        
    Returns:
        Standardized error response
    """
    handler = get_error_handler()
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    return handler.create_error_response(error, language)