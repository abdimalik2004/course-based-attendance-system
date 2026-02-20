import logging
from pathlib import Path


_LOGGER_INITIALIZED = False


def setup_logging(log_file: Path):
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: str):
    return logging.getLogger(name)
