import logging
import sys
from typing import Type, cast

try:
    from .logging_formatters import ColoredFormatter, FullColoredFormatter
    from .loggers import CustomLogger
except ImportError:
    from logging_formatters import ColoredFormatter, FullColoredFormatter
    from loggers import CustomLogger


def _create_handler(full_color: bool, include_function: bool) -> logging.Handler:
    """Create and configure a StreamHandler with the chosen formatter."""
    handler = logging.StreamHandler(sys.stdout)
    if full_color:
        FormatterClass = FullColoredFormatter
    else:
        FormatterClass = ColoredFormatter
    formatter = FormatterClass(include_function=include_function)
    handler.setFormatter(formatter)
    return handler


def setup_logging(
    level: int = logging.INFO,
    full_color: bool = False,
    include_function: bool = False,
    logger_class: Type[logging.Logger] | None = CustomLogger,
    logger_name: str = "app",
) -> CustomLogger | logging.Logger:
    """
    Configure and return a logger.

    Args:
        level: Logging level (default=logging.INFO)
        full_color: Use full color formatting (default=False)
        include_function: Include function names (default=False)
        logger_class: Logger class to use (default=CustomLogger)
        logger_name: Logger instance name (default="app")

    Returns:
        CustomLogger | logging.Logger: Configured logger instance

    Example:
        ```python
        from typing import cast
        from utils.logging_config import setup_logging, CustomLogger

        # Use custom logger with success() method
        logger = setup_logging(level=logging.DEBUG, logger_class=CustomLogger)
        logger = cast(CustomLogger, logger)  # Type hinting
        logger.success("This is a success message!")

        # Use default Python logger
        logger = setup_logging(logger_class=logging.Logger)
        logger.info("Standard logger without success method")

        # Direct class usage
        logger = setup_logging(logger_class=CustomLogger, logger_name="my_app")
        logger.success("Direct class usage!")
        ```
    """
    if not isinstance(logger_class, type) or not issubclass(logger_class, logging.Logger):
        print(f"Unknown logger_class ({logger_class}) provided, falling back to logging.Logger")
        logger_class = logging.Logger

    # Create logger instance
    logging.setLoggerClass(logger_class)
    logger = logging.getLogger(logger_name)

    logger.setLevel(level)
    logger.handlers.clear()

    # Set up handler
    if not logger.handlers:
        handler = _create_handler(full_color, include_function)
        logger.addHandler(handler)

    # Initial log message
    config_msg = "Logging configured"
    if include_function:
        config_msg += " with function names"
    config_msg += f" successfully ✅"

    if isinstance(logger, CustomLogger):
        logger = cast(CustomLogger, logger)  # Type hinting
        logger.success(config_msg)
    else:
        logger.info(config_msg)

    return logger


def _logging_test(logger: logging.Logger):
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")

    # Use success() only if available
    if isinstance(logger, CustomLogger):
        logger = cast(CustomLogger, logger)  # Type hinting
        logger.success("This is a success message!")

    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")


if __name__ == "__main__":
    print("Regular color logging (CustomLogger):")
    logger = setup_logging(level=logging.DEBUG, logger_class=CustomLogger)
    _logging_test(logger)

    print("\n" + "=" * 50)
    print("Full color logging (CustomLogger):")
    logger = setup_logging(level=logging.DEBUG,
                           full_color=True,
                           logger_class=CustomLogger)
    _logging_test(logger)

    print("\n" + "=" * 50)
    print("Full color logging with tracing and function names (CustomLogger):")
    logger = setup_logging(level=logging.DEBUG,
                           full_color=True,
                           include_function=True,
                           logger_class=CustomLogger)
    _logging_test(logger)

    print("\n" + "=" * 50)
    print("Default Python logger (fallback, no success method):")
    logger = setup_logging(level=logging.DEBUG, logger_class=logging.Logger)
    _logging_test(logger)

    print("\n" + "=" * 50)
    print("Direct class usage:")
    logger = setup_logging(level=logging.DEBUG,
                           logger_class=CustomLogger,
                           logger_name="direct")
    _logging_test(logger)

    print("\n" + "=" * 50)
    print("Unknown logger type (fallback to logging.Logger):")
    logger = setup_logging(level=logging.DEBUG, logger_class=None)
    _logging_test(logger)
