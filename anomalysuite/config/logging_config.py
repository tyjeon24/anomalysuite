"""setup structlog based logger."""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any
from typing import TextIO, MutableMapping

import structlog

LOGGING_LEVEL = logging.INFO

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def set_stream_handler() -> logging.StreamHandler[TextIO]:
    """Set a handler to check logs on the stream."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOGGING_LEVEL)
    return stream_handler


def set_file_handler() -> logging.handlers.TimedRotatingFileHandler:
    """Set a handler to write logs to a file."""
    log_filename = os.path.join(LOG_DIR, datetime.now().astimezone().strftime("%Y-%m-%d.log"))

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filename,
        when="midnight",
        interval=1,  # Every day
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(LOGGING_LEVEL)
    return file_handler


logging.basicConfig(
    level=LOGGING_LEVEL,
    handlers=[
        set_stream_handler(),
        set_file_handler(),
    ],
    format="%(message)s",
)


def add_timestamp_timezone(
    logger: logging.Logger, method_name: Any, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Set a timezone-aware timestamp."""
    event_dict["timestamp"] = datetime.now().astimezone().isoformat()
    return event_dict


def sort_keys(
    logger: logging.Logger, method_name: Any, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Sort keys in dictionary type logger."""
    event_dict = dict(sorted(event_dict.items()))
    return event_dict


structlog.configure(
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
    processors=[
        add_timestamp_timezone,
        structlog.stdlib.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,  # {"filename": "app.py"}
                structlog.processors.CallsiteParameter.FUNC_NAME,  # {"func_name": "my_function"}
                structlog.processors.CallsiteParameter.LINENO,  # {"lineno": 11}
            ]
        ),
        structlog.processors.dict_tracebacks,
        structlog.processors.format_exc_info,
        sort_keys,
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ],
)

logger = structlog.get_logger()
