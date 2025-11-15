"""Centralized logging utilities with colored output and level control"""

import logging
import sys


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    # Color for module name
    MODULE_COLOR = Colors.CYAN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for processed module names
        self._module_name_cache: dict[str, str] = {}

    def _get_short_module_name(self, full_name: str) -> str:
        """Extract and cache the short module name (last part, uppercase)"""
        if full_name not in self._module_name_cache:
            # Get last part after final dot
            short_name = full_name.split(".")[-1].upper()
            self._module_name_cache[full_name] = short_name
        return self._module_name_cache[full_name]

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            colored_levelname = f"{self.LEVEL_COLORS[record.levelno]}{levelname}{Colors.RESET}"
            record.levelname = colored_levelname

        # Process and color module name
        short_name = self._get_short_module_name(record.name)
        colored_name = f"{self.MODULE_COLOR}{short_name}{Colors.RESET}"
        record.name = colored_name

        # Format the message
        result = super().format(record)

        # Reset levelname for next use
        record.levelname = levelname

        return result


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Setup a logger with colored output

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Use custom format if provided, otherwise use default
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    # Add colored formatter with time format
    formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get or create a logger with standard configuration

    Args:
        name: Logger name (usually __name__)
        level: Logging level (if None, reads from settings.log_level)

    Returns:
        Logger instance
    """
    if level is None:
        # Read from settings
        try:
            from tracingrag.config import settings

            level = getattr(logging, settings.log_level.upper(), logging.INFO)
        except Exception:
            # Fallback to INFO if settings not available
            level = logging.INFO

    return setup_logger(name, level)


# Convenience function for module-level loggers
def create_module_logger(module_name: str) -> logging.Logger:
    """Create a logger for a module with INFO level by default

    Args:
        module_name: Module name (use __name__)

    Returns:
        Configured logger
    """
    return get_logger(module_name, logging.INFO)
