import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Setup structured logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        json_format: Use JSON format for logs
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
            if json_format
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler with UTF-8 encoding
    import io
    console_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stdout, 'buffer') else sys.stdout
    )
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if json_format:
        console_formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file

        if json_format:
            file_formatter = jsonlogger.JsonFormatter(
                "%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d"
            )
            file_handler.setFormatter(file_formatter)
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return structlog.get_logger(name)


class StructuredLogger:
    """
    Structured logger wrapper for consistent logging across BLUEPRINT.

    Provides a simple interface for logging with structured data.
    """

    def __init__(self, name: str):
        """
        Initialize structured logger.

        Args:
            name: Logger name (usually module name)
        """
        self.logger = structlog.get_logger(name)
        self.name = name

    def log(self, level: str, message: str, **kwargs):
        """
        Log a message with structured data.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Additional structured data
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
