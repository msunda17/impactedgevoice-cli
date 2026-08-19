"""
menu.py — Interactive mode picker for ImpactEdgeVoice.

Presented at startup when the user runs `python -m impactedgevoice` with no args.
Each mode is dispatched into the appropriate runtime (voice orchestrator,
document processor, text Q&A, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from impactedgevoice.adaptive_router import AdaptiveRouter, InputModality
from impactedgevoice.model_router import ModelRouter, TaskTier

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Mode registry
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Mode:
    key: str            # Single-char shortcut
    label: str          # User-facing name
    description: str    # One-line help
    handler: Callable[[AdaptiveRouter], None]


def _print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║                  W H I S P E R L O O P               ║")
    print("  ║         Local-first adaptive AI assistant            ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()


def _print_menu(modes: list[Mode]) -> None:
    print("  Choose a mode:")
    print()
    for m in modes:
        print(f"    [{m.key}]  {m.label:<30}  {m.description}")
    print(f"    [q]  Quit")
    print()


def _prompt_choice(modes: list[Mode]) -> Optional[Mode]:
    valid_keys = {m.key.lower() for m in modes} | {"q", "quit", "exit"}
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not choice:
            continue
        if choice in ("q", "quit", "exit"):
            return None
        # Allow either single-char key or the full label match
        for m in modes:
            if choice == m.key.lower() or choice == m.label.lower():
                return m
        print(f"    ✗ Unknown option '{choice}'. Try one of: "
              + ", ".join(m.key for m in modes) + ", q.")


# ──────────────────────────────────────────────────────────────────────
#  Mode handlers
# ──────────────────────────────────────────────────────────────────────

def handle_voice(router: AdaptiveRouter) -> None:
    """Live microphone-driven conversation (no seed)."""
    print("\n  → Voice mode. Speak naturally. Ctrl+C to exit.\n")
    from impactedgevoice.orchestrator import Orchestrator
    # Reuse the AdaptiveRouter's simple-tier KV so we don't double-load.
    from impactedgevoice.model_router import TaskTier
    kv = router.router.select_model(query="voice chat", force_tier=TaskTier.SIMPLE)
    orch = Orchestrator(kv=kv)
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        print("\n  ✓ Voice session ended.")


def _sanitize_path(raw: str) -> Path:
    """
    Normalize a path pasted from Windows Explorer or macOS Finder.

    Handles:
    - Surrounding double/single/smart quotes  ("path" 'path' \u201cpath\u201d)
    - Mixed forward/back slashes
    - Leading/trailing whitespace
    - Escaped spaces (path\ with\ spaces)
    """
    s = raw.strip()
    # Strip any combination of surrounding quote characters
    for quote in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019"):
        if s.startswith(quote) and s.endswith(quote) and len(s) > 1:
            s = s[1:-1]
            break
    # Remove any remaining leading/trailing quotes (Windows sometimes doubles them)
    s = s.strip('"\'')
    # Normalize backslashes to forward slashes for Path, then let Path re-normalise
    s = s.replace("\\\\", "\\")   # collapse double-backslash UNC artefacts first
    return Path(s).expanduser().resolve()


def _file_mode_limits(router: AdaptiveRouter) -> Optional[dict]:
    """
    Compute approximate document-size bounds for the current complex-tier model
    so the CLI can display them. Returns None if the model can't be inspected
    without loading it (we avoid loading just to print a help message).
    """
    try:
        from impactedgevoice.document_processor import DocumentProcessor
        # We intentionally don't force-load the complex model here — just read
        # the configured n_ctx ceiling from the router so the CLI is responsive.
        n_ctx = getattr(router.model_router, "n_ctx_complex", 8192)
        reserve = getattr(DocumentProcessor, "CTX_RESERVE_FRACTION", 0.45)
        chunk_tokens = max(512, min(
            getattr(DocumentProcessor, "CHUNK_SIZE", 6000),
            int(n_ctx * (1.0 - reserve)),
        ))
        # Comfortable = single chunk; max = ~12 chunks before recursion kicks in.
        comfortable_chars = chunk_tokens * 4
        max_chars = chunk_tokens * 12 * 4
        return {
            "n_ctx": n_ctx,
            "chunk_tokens": chunk_tokens,
            "comfortable_tokens": chunk_tokens,
            "comfortable_chars": comfortable_chars,
            "comfortable_pages": max(1, comfortable_chars // 2500),
            "max_chars": max_chars,
            "max_pages": max(1, max_chars // 2500),
        }
    except Exception:
        return None


def _check_doc_size(path: Path, limits: Optional[dict]) -> str:
    """
    Pre-flight: read the file's char count and warn if it exceeds limits.
    Returns 'ok', 'warn' or 'abort' (user declined).
    """
    if not limits:
        return "ok"
    try:
        if path.suffix.lower() == ".pdf":
            # Conservative estimate: 2.5KB per page is typical for academic PDFs.
            # We avoid actually parsing here — that's the expensive step.
            size_bytes = path.stat().st_size
            est_chars = max(size_bytes // 2, 1)  # PDFs compress text heavily
        else:
            est_chars = path.stat().st_size

        comfortable = limits["comfortable_chars"]
        ceiling = limits["max_chars"]
        est_pages = est_chars // 2500

        if est_chars > ceiling:
            print(
                f"    ⚠  Large document (~{est_pages} pages, ~{est_chars:,} chars). "
                f"Hard ceiling is ~{limits['max_pages']} pages — recursive "
                "summarization will trigger (slower, lower fidelity)."
            )
            try:
                ans = input("    Continue anyway? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "abort"
            if ans != "y":
                print("    ✗ Aborted.")
                return "abort"
            return "warn"
        elif est_chars > comfortable:
            print(
                f"    ℹ  Document is ~{est_pages} pages — will use map-reduce "
                f"(comfortable single-pass limit is ~{limits['comfortable_pages']} pages)."
            )
            return "warn"
        return "ok"
    except Exception:
        return "ok"


def handle_file(router: AdaptiveRouter) -> None:
    """Upload a document/audio file, summarize it, then chat about it."""
    print("\n  → File mode. Drop a file and ask follow-up questions.")
    print("    Supported: .pdf .txt .md .docx .wav .mp3 .m4a")

    # Show approximate size limits so the user knows what to expect.
    limits = _file_mode_limits(router)
    if limits:
        print(
            f"    Size limits (auto-scaled to model context, {limits['n_ctx']} tokens):"
        )
        print(
            f"      • Comfortable: ≤ {limits['comfortable_pages']} pages "
            f"(≤ {limits['comfortable_chars']:,} chars / "
            f"≤ {limits['comfortable_tokens']:,} tokens)"
        )
        print(
            f"      • Hard ceiling: ~{limits['max_pages']} pages — beyond this we "
            "fall back to recursive summarization (slower, lower fidelity)."
        )
        print(
            f"      • Per-chunk window: {limits['chunk_tokens']:,} tokens"
        )
    print("    Tip: paste the path directly — quotes and backslashes are handled.")

    try:
        path_str = input("    File path: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not path_str.strip():
        print("    ✗ No path provided.")
        return

    path = _sanitize_path(path_str)
    if not path.exists():
        print(f"    ✗ File not found: {path}")
        print(f"    (resolved from: {path_str.strip()!r})")
        return

    # Pre-flight size check for text/PDF. We don't try to count audio.
    if path.suffix.lower() not in (".wav", ".mp3", ".m4a"):
        size_check = _check_doc_size(path, limits)
        if size_check == "abort":
            return

    decision, kv = router.route(path)
    print(f"    [router] {decision.reasoning}")

    suffix = path.suffix.lower()
    if suffix in (".wav", ".mp3", ".m4a"):
        summary, source_text = _summarize_audio(path, router)
    else:
        summary, source_text = _summarize_document(path, decision.use_chunking, router)

    if not summary:
        return

    # Print the summary as the assistant's opening message.
    print()
    print("  " + "─" * 60)
    print(f"  Summary of {path.name}")
    print("  " + "─" * 60)
    for line in summary.splitlines():
        print(f"  {line}")
    print()

    # Prime the conversational KV with a clean (system + 1 turn) state
    # so the model can answer follow-ups grounded in this document.
    _prime_kv_with_document(kv, path.name, summary, source_text)

    print("  → The assistant will now read the summary and listen for your")
    print("    follow-up questions. Speak naturally. Ctrl+C to exit.\n")

    # Hand off to the voice Orchestrator. The summary is spoken aloud as the
    # opening announcement, then the mic takes over for follow-up questions.
    spoken_summary = _shorten_for_speech(summary)
    from impactedgevoice.orchestrator import Orchestrator
    orch = Orchestrator(kv=kv, baseline_checkpoint="primed")
    try:
        asyncio.run(orch.run(opening_announcement=spoken_summary))
    except KeyboardInterrupt:
        print("\n  ✓ Conversation ended.")


def handle_text(router: AdaptiveRouter) -> None:
    """
    Conversational text Q&A: type a prompt, get a spoken reply, then continue
    by voice. Adaptive tier picked on the first turn, then sticky.
    """
    print("\n  → Text-seeded conversation. Type a prompt to start, then speak.")
    print("    Ctrl+C to exit.\n")

    try:
        first = input("  you > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not first or first.lower() in (":exit", ":quit"):
        return

    decision, kv = router.route(first)
    print(f"    [router] {decision.reasoning}\n")

    # Hand off to the voice Orchestrator. The typed prompt is treated as the
    # first "user turn"; the Orchestrator generates a reply with TTS, then the
    # mic takes over for the rest of the conversation.
    from impactedgevoice.orchestrator import Orchestrator
    orch = Orchestrator(kv=kv)
    try:
        asyncio.run(orch.run(initial_user_text=first))
    except KeyboardInterrupt:
        print("\n  ✓ Conversation ended.")


def handle_model_settings(router: AdaptiveRouter) -> None:
    """
    Let the user pick which model to use for the SIMPLE and/or COMPLEX tier.
    Displays all registry candidates with local availability status.
    If the chosen model isn't downloaded yet it will be fetched automatically
    when that tier next loads (via _resolve_model → _prompt_and_download).
    """
    from impactedgevoice.model_router import TaskTier, CANDIDATE_REGISTRY

    print("\n  → Model Settings")
    print("    Choose which model to use for each tier.")
    print("    Models not downloaded locally will be fetched from HuggingFace.")

    while True:
        # Refresh info each loop iteration so overrides are reflected
        info = router.router.get_tier_info()
        models_dir = Path(info['models_dir'])

        print()
        print("    Current configuration:")
        simple_cur = info.get('simple_override') or "(auto — first match in candidate list)"
        complex_cur = info.get('complex_override') or "(auto — first match in candidate list)"
        print(f"      [S]  Simple  tier: {simple_cur}")
        print(f"      [C]  Complex tier: {complex_cur}")
        print(f"      [L]  List available models")
        print(f"      [B]  Back")
        print()

        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ("b", "back", "q", ""):
            return

        elif choice in ("l", "list"):
            print()
            print("    Available GGUFs in models/:")
            for name in info["available_ggufs"]:
                entry = CANDIDATE_REGISTRY.get(name, {})
                desc = entry.get("desc", name)
                tiers = "/".join(entry.get("tiers", []))
                print(f"      • {name}  ({tiers})  — {desc}")
            if not info["available_ggufs"]:
                print("      (none — run: python download_models.py)")

        elif choice in ("s", "simple"):
            chosen = TaskTier.SIMPLE
            fname = ModelRouter.prompt_model_selection(chosen)
            if fname:
                router.router.set_model(chosen, fname)
                print(f"    ✓ Simple tier → {fname}")
                print("      (will download if not present when next loaded)")

        elif choice in ("c", "complex"):
            chosen = TaskTier.COMPLEX
            fname = ModelRouter.prompt_model_selection(chosen)
            if fname:
                router.router.set_model(chosen, fname)
                print(f"    ✓ Complex tier → {fname}")
                print("      (will download if not present when next loaded)")

        else:
            print("    ✗ Unknown option. Enter S, C, L, or B.")


def handle_benchmark(router: AdaptiveRouter) -> None:
    """Run the latency benchmark suite."""
    print("\n  → Benchmark mode.")
    try:
        from bench.bench import run_bench  # type: ignore
    except ImportError:
        print("    ✗ bench module not available.")
        return
    run_bench()


def handle_status(router: AdaptiveRouter) -> None:
    """Show system + model status."""
    import psutil
    vm = psutil.virtual_memory()
    print(f"\n  → System status")
    print(f"    RAM:  {vm.percent:.1f}% used  ({vm.used / 1e9:.1f} / {vm.total / 1e9:.1f} GB)")
    print(f"    CPU:  {psutil.cpu_percent(interval=0.5):.1f}%")

    info = router.router.get_tier_info()
    print(f"\n    Models dir:  {info['models_dir']}")
    print(f"    Available GGUFs:")
    for name in info["available_ggufs"]:
        print(f"      • {name}")
    if not info["available_ggufs"]:
        print("      (none — run: python download_models.py)")

    print(f"\n    Loaded tiers:")
    simple_tag = f"  [{info['simple_override']}]" if info.get('simple_override') else ""
    complex_tag = f"  [{info['complex_override']}]" if info.get('complex_override') else ""
    print(f"      simple : {'● loaded' if info['simple_loaded'] else '○ not yet loaded'}{simple_tag}")
    print(f"      complex: {'● loaded' if info['complex_loaded'] else '○ not yet loaded'}{complex_tag}")
    print()


# ──────────────────────────────────────────────────────────────────────
#  Conversational helpers
# ──────────────────────────────────────────────────────────────────────

# Source-text excerpt fed into the KV so follow-ups stay grounded.
# Larger = more grounded but eats context budget.
_DOC_PRIMER_CHARS = 8000

# Hard cap for the spoken intro — anything longer is annoying to listen to.
_SPOKEN_SUMMARY_CHARS = 600


def _shorten_for_speech(summary: str) -> str:
    """Trim a written summary down to something palatable to speak aloud."""
    s = summary.strip()
    # Drop common preamble lines like "Here is a concise overall summary:"
    lines = [ln for ln in s.splitlines() if ln.strip() and not ln.lower().startswith("here is")]
    s = " ".join(lines)
    if len(s) <= _SPOKEN_SUMMARY_CHARS:
        return s
    # Keep whole sentences only.
    cutoff = s.rfind(". ", 0, _SPOKEN_SUMMARY_CHARS)
    if cutoff < 200:
        cutoff = _SPOKEN_SUMMARY_CHARS
    return s[: cutoff + 1].strip()


def _stream_turn(kv, user_text: str, max_tokens: int = 512) -> None:
    """Generate one assistant turn and stream it to stdout."""
    print("  bot > ", end="", flush=True)
    for tok in kv.generate(user_text, max_tokens=max_tokens):
        print(tok, end="", flush=True)
    print("\n")


def _chat_loop(kv) -> None:
    """
    Multi-turn conversation loop driven by a single KVCacheManager.
    The KV cache preserves conversational state across turns automatically.
    """
    while True:
        try:
            line = input("  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in (":exit", ":quit", ":back"):
            print("  → returning to menu.\n")
            return
        if line.lower() == ":reset":
            try:
                kv.restore_checkpoint("system")
                print("  → KV cache reset to system prompt. Fresh conversation.\n")
            except Exception as e:
                print(f"  ✗ Could not reset: {e}\n")
            continue
        _stream_turn(kv, line)


def _prime_kv_with_document(kv, filename: str, summary: str, source_excerpt: str) -> None:
    """
    Reset the KV to just the system prompt, then issue ONE primer turn so
    follow-up questions are grounded in the document. We do this rather than
    trusting whatever map-reduce intermediate state may be in the KV.
    """
    try:
        kv.restore_checkpoint("system")
    except Exception as e:
        logger.warning("Could not reset KV before priming: %s", e)

    excerpt = source_excerpt[:_DOC_PRIMER_CHARS]
    truncated_note = "" if len(source_excerpt) <= _DOC_PRIMER_CHARS else \
        f"\n\n[… {len(source_excerpt) - _DOC_PRIMER_CHARS} more characters omitted]"

    primer = (
        f"I have just read the file '{filename}'.\n\n"
        f"--- DOCUMENT SUMMARY ---\n{summary}\n\n"
        f"--- DOCUMENT EXCERPT ---\n{excerpt}{truncated_note}\n\n"
        "Acknowledge briefly that you have read the document, then wait for "
        "the user's questions. Keep this content in mind for follow-ups."
    )

    # Run the primer turn silently — we already showed the user the summary.
    for _ in kv.generate(primer, max_tokens=80):
        pass

    # Save a checkpoint AFTER the primer so the live pruner can rollback to the
    # system+document baseline (not just system) when the KV cache fills up.
    try:
        kv.save_checkpoint("primed")
    except Exception as e:
        logger.debug("Could not save 'primed' checkpoint: %s", e)


# ──────────────────────────────────────────────────────────────────────
#  File-mode summarization
# ──────────────────────────────────────────────────────────────────────

def _summarize_document(
    path: Path, use_chunking: bool, router: AdaptiveRouter
) -> tuple[str, str]:
    """Returns (summary, source_text). Uses the shared model router."""
    from impactedgevoice.document_processor import DocumentProcessor
    # Share the underlying ModelRouter so we don't double-load the LLM.
    proc = DocumentProcessor(model_router=router.router)
    method = "map_reduce" if use_chunking else "single_pass"
    print(f"    Summarizing ({method})...")
    summary = proc.summarize_file(str(path), method=method, max_tokens=1024)

    # Re-read source text for priming follow-up Q&A.
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        source_text = proc._extract_pdf(path)
    elif suffix in (".txt", ".md", ".rst"):
        source_text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        source_text = ""
    return summary, source_text


def _summarize_audio(
    path: Path, router: AdaptiveRouter
) -> tuple[str, str]:
    """Transcribe audio, summarize, and return (summary, transcript)."""
    import numpy as np
    try:
        import soundfile as sf
    except ImportError:
        print("    ✗ soundfile not installed. Run: pip install soundfile")
        return "", ""
    from impactedgevoice.asr import ASR
    from impactedgevoice.document_processor import DocumentProcessor

    print("    Transcribing audio...")
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != 16000:
        print(f"    [warn] sample rate {sr}Hz != 16000Hz; transcription may suffer.")
    asr = ASR(model_size="small.en")
    transcript = asr.transcribe(audio.astype(np.float32))
    print(f"    Transcript ({len(transcript)} chars).")

    proc = DocumentProcessor(model_router=router.router)
    print("    Summarizing transcript...")
    summary = proc.single_pass_summarize(transcript, max_tokens=512)
    return summary, transcript


# ──────────────────────────────────────────────────────────────────────
#  Voice-assistant sub-menu (the three conversation triggers)
# ──────────────────────────────────────────────────────────────────────

def _build_assistant_triggers() -> list[Mode]:
    """The three ways to *start* a conversation. All are just openers."""
    return [
        Mode("1", "Voice",  "Begin speaking immediately (live mic)",                 handle_voice),
        Mode("2", "File",   "Drop a file → summary → continue the conversation",     handle_file),
        Mode("3", "Text",   "Type a prompt → continue the conversation",             handle_text),
    ]


def handle_voice_assistant(router: AdaptiveRouter) -> None:
    """
    Single entry-point to the voice assistant. The three sub-modes only
    differ in HOW the conversation begins; they all drop into the same
    conversational loop afterwards.
    """
    triggers = _build_assistant_triggers()
    print("\n  → Voice Assistant. Pick how to start the conversation:\n")
    for m in triggers:
        print(f"    [{m.key}]  {m.label:<8}  {m.description}")
    print(f"    [b]  Back     Return to main menu")
    print()

    valid_keys = {m.key.lower() for m in triggers} | {"b", "back", "q"}
    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            continue
        if choice in ("b", "back", "q"):
            return
        match = next((m for m in triggers if choice == m.key.lower()
                      or choice == m.label.lower()), None)
        if match is None:
            print(f"    ✗ Unknown option '{choice}'. "
                  + "Try: " + ", ".join(m.key for m in triggers) + ", b.")
            continue
        match.handler(router)
        return  # after the conversation ends, drop back to main menu


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def build_modes() -> list[Mode]:
    return [
        Mode("1", "Trigger Voice Assistant", "Start a conversation (voice / file / text)", handle_voice_assistant),
        Mode("2", "Model Settings",          "Choose simple / complex model + auto-download", handle_model_settings),
        Mode("3", "Show Resource Usage",     "RAM, CPU, and loaded model state",           handle_status),
        Mode("4", "Start Benchmark Suite",   "Run latency benchmark",                      handle_benchmark),
    ]


def run_menu() -> None:
    """Top-level interactive loop."""
    _print_banner()
    router = AdaptiveRouter()
    modes = build_modes()

    while True:
        _print_menu(modes)
        choice = _prompt_choice(modes)
        if choice is None:
            print("  ✓ Goodbye.\n")
            return
        try:
            choice.handler(router)
        except KeyboardInterrupt:
            print("\n  ✓ Mode interrupted, returning to menu.\n")
        except Exception as e:
            logger.exception("Mode handler crashed")
            print(f"\n  ✗ Error: {e}\n")
