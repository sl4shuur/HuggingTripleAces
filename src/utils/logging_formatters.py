import logging
from colorama import Fore, Style


def _hex_to_ansi(hex_color: str) -> str:
    """Convert hex color to ANSI escape sequence"""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


COLORS = {
    "DEBUG": _hex_to_ansi("#3ACEFF"),     # Sky Blue
    "INFO": _hex_to_ansi("#A1F7FF"),      # Light Blue
    "SUCCESS": _hex_to_ansi("#69FE69"),   # Bright Green
    "WARNING": _hex_to_ansi("#FDF32F"),   # Yellow
    "ERROR": _hex_to_ansi("#F61C1C"),     # Red
    "CRITICAL": _hex_to_ansi("#FF6EFF"),  # Magenta
}


class ColoredFormatter(logging.Formatter):
    """Colored formatter with file path, line number, and function name support"""

    def __init__(self, include_function: bool = False):
        super().__init__()
        self.include_function = include_function

    def format(self, record: logging.LogRecord) -> str:
        log_color = COLORS.get(record.levelname, Fore.WHITE)
        colored_time = f"{log_color}{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}"
        colored_level = f"{log_color}{record.levelname}{Style.RESET_ALL}"

        # Build the formatted message
        message_parts = []

        # Add timestamp
        message_parts.append(f"[{colored_time}]")

        # Add file path, line number, and function name if enabled
        if self.include_function:
            file_info = f"{record.pathname}:{record.lineno} -> {record.funcName}"
            function_info = f"{_hex_to_ansi('#FF9500')}[{file_info}]{Style.RESET_ALL}"
            message_parts.append(function_info)

        # Combine all parts
        header = " ".join(message_parts)
        formatted_record = f"{header}\n{colored_level}: {record.getMessage()}"

        return formatted_record


class FullColoredFormatter(logging.Formatter):
    """Full colored formatter with file path, line number, and function name support"""

    def __init__(self, include_function: bool = False):
        super().__init__()
        self.include_function = include_function

    def format(self, record: logging.LogRecord) -> str:
        log_color = COLORS.get(record.levelname, Fore.WHITE)

        # Build the formatted message
        message_parts = []
        message_parts.append(
            f"[{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}]")

        # Add file path, line number, and function name if enabled
        if self.include_function:
            file_info = f"{record.pathname}:{record.lineno} -> {record.funcName}"
            message_parts.append(f"[{file_info}]")

        # Combine all parts
        header = " ".join(message_parts)
        formatted_record = f"\n{header}\n{record.levelname}: {record.getMessage()}"

        return f"{log_color}{formatted_record}{Style.RESET_ALL}"
