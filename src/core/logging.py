import logging
from src.core.config import config


class SystemReminderFilter(logging.Filter):
    """Filter to replace <system_reminder> tags with a placeholder for cleaner logs."""

    def filter(self, record):
        if record.msg:
            record.msg = str(record.msg).replace(
                "<system_reminder>", "[system_reminder]"
            ).replace(
                "</system_reminder>", "[/system_reminder]"
            )
        return True

# Parse log level - extract just the first word to handle comments
log_level = config.log_level.split()[0].upper()

# Validate and set default if invalid
valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
if log_level not in valid_levels:
    log_level = 'INFO'

# Logging Configuration
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Apply system reminder filter to root handler
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(SystemReminderFilter())

# Set non-web loggers to ERROR level only
root_logger.setLevel(logging.ERROR)

# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)