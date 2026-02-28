<<<<<<< HEAD
"""
Logging configuration for BharatVoice Assistant.

This module provides structured logging setup with support for different
output formats, log levels, and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from bharatvoice.config.settings import LoggingSettings


def setup_logging(logging_config: LoggingSettings) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        logging_config: Logging configuration settings
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format=logging_config.format,
        level=getattr(logging, logging_config.level.upper()),
        stream=sys.stdout
    )
    
    # Setup file logging if specified
    if logging_config.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=logging_config.file_path,
            maxBytes=logging_config.max_file_size,
            backupCount=logging_config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(logging_config.format)
        )
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class RequestContextFilter(logging.Filter):
    """
    Logging filter to add request context to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request context to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record
        """
        # Add request ID and user context if available
        # This would be populated by middleware in a real implementation
        record.request_id = getattr(record, 'request_id', 'unknown')
        record.user_id = getattr(record, 'user_id', 'anonymous')
        
        return True


class PerformanceLogger:
    """
    Logger for performance metrics and timing information.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
    
    def log_processing_time(
        self,
        operation: str,
        duration: float,
        **kwargs: Any
    ) -> None:
        """
        Log processing time for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional context
        """
        self.logger.info(
            "Operation completed",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        **kwargs: Any
    ) -> None:
        """
        Log API request metrics.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            **kwargs: Additional context
        """
        self.logger.info(
            "API request",
            method=method,
            path=path,
            status_code=status_code,
            duration_seconds=duration,
            **kwargs
        )


class SecurityLogger:
    """
    Logger for security-related events.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.security")
    
    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str,
        **kwargs: Any
    ) -> None:
        """
        Log authentication attempt.
        
        Args:
            username: Username attempted
            success: Whether authentication succeeded
            ip_address: Client IP address
            **kwargs: Additional context
        """
        self.logger.info(
            "Authentication attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        **kwargs: Any
    ) -> None:
        """
        Log authorization failure.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being attempted
            **kwargs: Additional context
        """
        self.logger.warning(
            "Authorization failure",
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        **kwargs: Any
    ) -> None:
        """
        Log data access for privacy compliance.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            operation: Operation performed (read, write, delete)
            **kwargs: Additional context
        """
        self.logger.info(
            "Data access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            **kwargs
=======
"""
Logging configuration for BharatVoice Assistant.

This module provides structured logging setup with support for different
output formats, log levels, and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from bharatvoice.config.settings import LoggingSettings


def setup_logging(logging_config: LoggingSettings) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        logging_config: Logging configuration settings
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format=logging_config.format,
        level=getattr(logging, logging_config.level.upper()),
        stream=sys.stdout
    )
    
    # Setup file logging if specified
    if logging_config.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=logging_config.file_path,
            maxBytes=logging_config.max_file_size,
            backupCount=logging_config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(logging_config.format)
        )
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class RequestContextFilter(logging.Filter):
    """
    Logging filter to add request context to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request context to log record.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record
        """
        # Add request ID and user context if available
        # This would be populated by middleware in a real implementation
        record.request_id = getattr(record, 'request_id', 'unknown')
        record.user_id = getattr(record, 'user_id', 'anonymous')
        
        return True


class PerformanceLogger:
    """
    Logger for performance metrics and timing information.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
    
    def log_processing_time(
        self,
        operation: str,
        duration: float,
        **kwargs: Any
    ) -> None:
        """
        Log processing time for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional context
        """
        self.logger.info(
            "Operation completed",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        **kwargs: Any
    ) -> None:
        """
        Log API request metrics.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration: Request duration in seconds
            **kwargs: Additional context
        """
        self.logger.info(
            "API request",
            method=method,
            path=path,
            status_code=status_code,
            duration_seconds=duration,
            **kwargs
        )


class SecurityLogger:
    """
    Logger for security-related events.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.security")
    
    def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str,
        **kwargs: Any
    ) -> None:
        """
        Log authentication attempt.
        
        Args:
            username: Username attempted
            success: Whether authentication succeeded
            ip_address: Client IP address
            **kwargs: Additional context
        """
        self.logger.info(
            "Authentication attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        **kwargs: Any
    ) -> None:
        """
        Log authorization failure.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being attempted
            **kwargs: Additional context
        """
        self.logger.warning(
            "Authorization failure",
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )
    
    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        **kwargs: Any
    ) -> None:
        """
        Log data access for privacy compliance.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            operation: Operation performed (read, write, delete)
            **kwargs: Additional context
        """
        self.logger.info(
            "Data access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            **kwargs
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        )