"""setup structlog based logger."""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any

import structlog

LOGGING_LEVEL = logging.INFO

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def set_stream_handler() -> logging.StreamHandler:
    """Set a handler to check logs on the stream."""
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOGGING_LEVEL)
    return stream_handler


def set_file_handler() -> logging.handlers.TimedRotatingFileHandler:
    """Set a handler to write logs to a file."""
    log_filename = os.path.join(LOG_DIR, datetime.now().astimezone().strftime("%Y-%m-%d.log"))

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_filename,
        when="midnight",  # 자정마다 파일 분할
        interval=1,  # 1일마다 분할
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(LOGGING_LEVEL)  # 파일에 기록할 최소 로그 레벨
    return file_handler


logging.basicConfig(
    level=LOGGING_LEVEL,
    handlers=[
        set_stream_handler(),
        set_file_handler(),
    ],
    format="%(message)s",
)


def add_timestamp_timezone(logger: logging.Logger, method_name: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Set a timezone-aware timestamp."""
    event_dict["timestamp"] = datetime.now().astimezone().isoformat()
    return event_dict


def sort_keys(logger: logging.Logger, method_name: Any, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Sort keys in dictionary type logger."""
    event_dict = dict(sorted(event_dict.items()))
    return event_dict


structlog.configure(
    logger_factory=structlog.stdlib.LoggerFactory(),  # (필수) logging.basicConfig 연동
    cache_logger_on_first_use=True,  # (필수) 성능 최적화, 한 번 생성한 인스턴스 재활용
    processors=[
        add_timestamp_timezone,  # (필수) timezone 시간 : {"timestamp": "2025-01-01T12:00:00.000000+09:00"}
        structlog.stdlib.add_log_level,  # 로그 레벨 : {"level": "error"}
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,  # {"filename": "app.py"}
                structlog.processors.CallsiteParameter.FUNC_NAME,  # {"func_name": "my_function"}
                structlog.processors.CallsiteParameter.LINENO,  # {"lineno": 11}
            ]
        ),
        structlog.processors.dict_tracebacks,  # 에러 트레이스백 json 형태로 작성
        structlog.processors.format_exc_info,  # (필수) logger.error(message, exc_info=True) 설정 시 오류에 대한 exception 항목 추가. { "exception": "Traceback (most recent call..."}
        sort_keys,  # 정렬
        structlog.processors.JSONRenderer(ensure_ascii=False),  # (필수) JSON 문자열로 변환
    ],
)

logger = structlog.get_logger()
