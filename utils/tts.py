import threading
import platform
import subprocess

from utils.logging import get_logger

logger = get_logger(__name__)

_engine = None
_lock = threading.Lock()


def _speak_with_windows_tts(text: str, rate: int | None = None):
    escaped = text.replace("'", "''")
    speech_rate = 0
    if rate is not None:
        # System.Speech rate is roughly -10..10; map from pyttsx-style WPM to a sane range.
        speech_rate = max(-10, min(10, int(round((rate - 170) / 15))))

    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Rate = {speech_rate}; "
        f"$speaker.Speak('{escaped}')"
    )

    kwargs = {
        "args": ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        "check": True,
        "capture_output": True,
        "text": True,
    }
    if platform.system().lower().startswith("win"):
        kwargs["creationflags"] = 0x08000000
    subprocess.run(**kwargs)


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
            if platform.system().lower().startswith("win"):
                try:
                    _speak_with_windows_tts(text, rate)
                    logger.info("TTS played via Windows speech engine: %s", text)
                    return
                except Exception as exc:
                    logger.warning("Windows TTS failed, falling back to pyttsx3: %s", exc)

            try:
                engine = _get_engine()
                if engine is None:
                    return
                if rate is not None:
                    engine.setProperty("rate", rate)
                engine.say(text)
                engine.runAndWait()
                logger.info("TTS played via pyttsx3: %s", text)
            except Exception as exc:
                logger.warning("TTS playback failed: %s", exc)

    threading.Thread(target=_worker, daemon=True).start()
