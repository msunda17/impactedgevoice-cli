"""
whisperloop.main — single user-facing entry point.

Behaviour:
  * No args   → interactive menu (Voice / File / Text Q&A / Status / Benchmark)
  * Any args  → forward to the scripted CLI (whisperloop.cli)

Examples:
    python -m whisperloop                           # interactive menu
    python -m whisperloop "summarize paper.pdf"     # one-shot file mode
    python -m whisperloop "what is 2+2?"            # one-shot text mode
"""

import logging
import sys

from whisperloop.console import configure_logging


def main() -> None:
    # Greyed-out, deduplicated log output so the conversation transcript
    # stands out visually. Suppresses chatty libs (httpx, transformers, etc.)
    configure_logging(level=logging.INFO)

    # No CLI args → interactive menu
    if len(sys.argv) == 1:
        from whisperloop.menu import run_menu
        try:
            run_menu()
        except KeyboardInterrupt:
            print("\nWhisperloop gracefully terminated.")
        return

    # Any args → defer to the scripted CLI
    from whisperloop.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
