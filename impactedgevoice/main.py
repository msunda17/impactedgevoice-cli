"""
impactedgevoice.main — single user-facing entry point.

Behaviour:
  * No args   → interactive menu (Voice / File / Text Q&A / Status / Benchmark)
  * Any args  → forward to the scripted CLI (impactedgevoice.cli)

Examples:
    python -m impactedgevoice                           # interactive menu
    python -m impactedgevoice "summarize paper.pdf"     # one-shot file mode
    python -m impactedgevoice "what is 2+2?"            # one-shot text mode
"""

import logging
import sys

from impactedgevoice.console import configure_logging


def main() -> None:
    # Greyed-out, deduplicated log output so the conversation transcript
    # stands out visually. Suppresses chatty libs (httpx, transformers, etc.)
    configure_logging(level=logging.INFO)

    # No CLI args → interactive menu
    if len(sys.argv) == 1:
        from impactedgevoice.menu import run_menu
        try:
            run_menu()
        except KeyboardInterrupt:
            print("\nImpactEdgeVoice gracefully terminated.")
        return

    # Any args → defer to the scripted CLI
    from impactedgevoice.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
