"""Logging utilities."""
import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise application logging without transaction payloads."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
