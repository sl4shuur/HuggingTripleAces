import logging

SUCCESS_LEVEL = 69
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class CustomLogger(logging.Logger):
    """Custom logger with a SUCCESS level."""

    def success(self, message: str, *args, **kwargs):
        """Log a message with SUCCESS level"""
        if self.isEnabledFor(SUCCESS_LEVEL):
            # Add stacklevel to get correct file/function info
            # stacklevel=2 means: skip this function and go to the caller
            self._log(SUCCESS_LEVEL, message, args, **kwargs, stacklevel=2)
