"""
cli.py — Unified ImpactEdgeVoice entry point.

Replaces separate voice/doc CLIs. Single command auto-routes:
    python -m impactedgevoice                       # Live voice mode (default)
    python -m impactedgevoice "summarize foo.pdf"   # Document mode (auto-detected)
    python -m impactedgevoice "what is 2+2?"        # Text mode → 2B
    python -m impactedgevoice "analyze this contract..."  # Text mode → 9B

Flags:
    --force-tier simple|complex   # Override the adaptive decision
    --max-tokens N                # Output budget
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from impactedgevoice.adaptive_router import AdaptiveRouter, InputModality
from impactedgevoice.model_router import TaskTier

logger = logging.getLogger(__name__)


def cmd_voice_mode():
    """Live voice interaction (existing orchestrator)."""
    from impactedgevoice.orchestrator import Orchestrator
    orch = Orchestrator()
    asyncio.run(orch.run())


def cmd_file_mode(file_path: str, max_tokens: int, decision):
    """Document/audio file processing."""
    from impactedgevoice.document_processor import DocumentProcessor

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ('.wav', '.mp3', '.m4a'):
        # Audio file → transcribe + summarize
        from impactedgevoice.asr import ASR
        import numpy as np
        import soundfile as sf

        print(f"Transcribing audio file: {path}")
        audio, sr = sf.read(str(path))
        if sr != 16000:
            print(f"Warning: resampling from {sr}Hz to 16000Hz")
        asr = ASR(model_size="small.en")
        transcript = asr.transcribe(audio.astype(np.float32))
        print(f"\nTranscript:\n{transcript}\n")

        # Now summarize the transcript with 9B
        proc = DocumentProcessor()
        summary = proc.single_pass_summarize(transcript, max_tokens=max_tokens)
        print(f"\n{'=' * 60}\nSummary:\n{'=' * 60}\n{summary}")

    else:
        # Document path
        proc = DocumentProcessor()
        method = "map_reduce" if decision.use_chunking else "single_pass"
        summary = proc.summarize_file(file_path, method=method, max_tokens=max_tokens)
        print(f"\n{'=' * 60}\nSummary ({method}):\n{'=' * 60}\n{summary}")


def cmd_text_mode(query: str, max_tokens: int, kv):
    """Direct text Q&A — no microphone, no TTS."""
    print(f"\n{'=' * 60}\nResponse:\n{'=' * 60}\n", flush=True)
    for tok in kv.generate(query, max_tokens=max_tokens):
        print(tok, end="", flush=True)
    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="ImpactEdgeVoice — adaptive local AI assistant (voice + docs)"
    )
    parser.add_argument(
        'input',
        nargs='?',
        default=None,
        help='File path, text query, or omit for live voice mode'
    )
    parser.add_argument(
        '--force-tier',
        choices=['simple', 'complex'],
        help='Override adaptive routing decision'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Output token budget (default: 512)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show routing decision details'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # No input → live voice mode (the original UX)
    if args.input is None:
        print("→ Voice mode (no input given). Speak to interact.")
        cmd_voice_mode()
        return

    # Use adaptive router to classify
    router = AdaptiveRouter()
    force_tier = None
    if args.force_tier:
        force_tier = TaskTier.SIMPLE if args.force_tier == 'simple' else TaskTier.COMPLEX

    decision, kv = router.route(args.input, force_tier=force_tier)

    if args.verbose:
        print(f"\n=== Routing Decision ===")
        print(f"  Modality:   {decision.modality.value}")
        print(f"  Tier:       {decision.tier.value}")
        print(f"  Complexity: {decision.complexity_score:.2f}")
        print(f"  Load:       {decision.load_score:.2f}")
        print(f"  Reasoning:  {decision.reasoning}")
        print(f"  Est tokens: {decision.estimated_tokens}")
        print(f"========================\n")

    # Dispatch by modality
    if decision.modality == InputModality.VOICE:
        cmd_voice_mode()
    elif decision.modality == InputModality.FILE:
        cmd_file_mode(args.input, args.max_tokens, decision)
    else:
        cmd_text_mode(args.input, args.max_tokens, kv)


if __name__ == "__main__":
    main()
