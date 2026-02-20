import threading

from utils.logging import get_logger

logger = get_logger(__name__)

_engine = None
_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is None:
        try:
            import pyttsx3
        except ImportError:
            logger.warning("pyttsx3 is not installed; TTS is disabled.")
            return None
        _engine = pyttsx3.init()
    return _engine


def speak_async(text: str, rate: int | None = None):
    def _worker():
        with _lock:
            engine = _get_engine()
            if engine is None:
                return
            if rate is not None:
                engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=_worker, daemon=True).start()
