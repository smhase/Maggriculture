"""Project-wide logging helpers.

Levels used across the platform:
  ERROR, WARNING, INFO, DEBUG, TRACE (custom, numeric 5)

High-volume simulations should stay at WARNING or ERROR.
"""

from __future__ import annotations

import logging
from typing import Optional

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LOGGER_NAME = "kaggriculture"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the package logger."""
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def configure_logging(level: str | int = "WARNING") -> None:
    """Configure the root package logger once."""
    logger = get_logger()
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper == "TRACE":
            level_value: int = TRACE
        else:
            level_value = getattr(logging, level_upper, logging.WARNING)
    else:
        level_value = level

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level_value)
    # Avoid duplicate messages via root
    logger.propagate = False


def trace(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Log at TRACE level."""
    if logger.isEnabledFor(TRACE):
        logger.log(TRACE, msg, *args, **kwargs)
