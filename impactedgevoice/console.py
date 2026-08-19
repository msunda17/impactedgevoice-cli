"""
console.py — terminal styling + clean logging config.

Goals:
  * Keep the conversation trace (YOU/BOT, system messages) visually prominent.
  * Push every library log line into a muted, gray styling so it's clearly
    "background noise" and never confused with the assistant's voice.
  * Silence the chatty libraries that flood stdout (httpx, huggingface_hub,
    transformers, sentence_transformers, faster_whisper, llama_cpp banners).

ANSI escape codes are used for color/weight. On Windows 10+ they work natively
in the modern terminal; on older shells we fall back to plain text via
``USE_COLOR``.
"""

from __future__ import annotations

import logging
import os
import sys


# ANSI escape sequences — kept tiny on purpose.
RESET   = "\x1b[0m"
DIM     = "\x1b[2m"
BOLD    = "\x1b[1m"
GREY    = "\x1b[90m"
CYAN    = "\x1b[36m"
GREEN   = "\x1b[32m"
YELLOW  = "\x1b[33m"
RED     = "\x1b[31m"
MAGENTA = "\x1b[35m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    # Windows: enable ANSI in modern terminals.
    if os.name == "nt":
        try:
            import ctypes  # noqa: WPS433
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


USE_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


# ---- Public helpers used by the orchestrator / menu --------------------

def style_you(text: str) -> str:
    """Bold cyan for the user's transcribed turn."""
    return _c(BOLD + CYAN, text)


def style_bot(text: str) -> str:
    """Bold green for the assistant's reply."""
    return _c(BOLD + GREEN, text)


def style_system(text: str) -> str:
    """Dim yellow for system notices (router decisions, mode banners)."""
    return _c(YELLOW, text)


def style_separator(text: str) -> str:
    return _c(DIM, text)


def style_muted(text: str) -> str:
    return _c(GREY, text)


# ---- Log formatter -----------------------------------------------------

class _GreyFormatter(logging.Formatter):
    """Render every log record in muted grey so it sits behind the chat trace."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s | %(message)s",
                         datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        s = super().format(record)
        if not USE_COLOR:
            return s
        if record.levelno >= logging.ERROR:
            return f"{RED}{s}{RESET}"
        if record.levelno >= logging.WARNING:
            return f"{YELLOW}{DIM}{s}{RESET}"
        return f"{GREY}{DIM}{s}{RESET}"


_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
    "faster_whisper",
    "llama_cpp",
)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Install the grey formatter and silence the chatty third-party libraries.
    Safe to call multiple times — replaces existing handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove pre-existing handlers (e.g. from logging.basicConfig).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_GreyFormatter())
    root.addHandler(handler)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
