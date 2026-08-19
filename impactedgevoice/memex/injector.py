"""
injector.py — Build the MEMEX context block for prefilling.

Layout (token-budgeted):
  [MEMEX recall block] — top-k recalled summaries with relative time tags.
  Capped at MAX_MEMEX_TOKENS so prefill latency stays bounded.
"""

from __future__ import annotations

import time
from typing import Iterable

from impactedgevoice.memex.storage import MemoryRow

# Hard cap to keep prefill latency bounded.
MAX_MEMEX_CHARS = 1800  # ~450 tokens at 4 chars/token


def _relative_time(ts: float, now: float) -> str:
    delta = max(0.0, now - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = int(delta / 60)
        return f"{m}m ago"
    if delta < 86400:
        h = int(delta / 3600)
        return f"{h}h ago"
    d = int(delta / 86400)
    if d < 30:
        return f"{d}d ago"
    return f"{d // 30}mo ago"


def build_memex_block(
    memories: Iterable[MemoryRow],
    live_summaries: Iterable[str] = (),
    max_chars: int = MAX_MEMEX_CHARS,
) -> str:
    """
    Build a string to prepend to the user's turn text. Format:

        [MEMEX]
        - (2d ago | topics: kv_cache, barge_in) Summary text…
        - (45m ago | topics: pdf, summarization) Summary text…
        [LIVE SESSION CONTEXT]
        - earlier in this session: Summary text…
        [/MEMEX]

    Returns "" if there's nothing to inject.
    """
    parts: list[str] = []
    now = time.time()

    mem_lines: list[str] = []
    for m in memories:
        rel = _relative_time(m.timestamp, now)
        topics = ", ".join(m.topics[:4]) if m.topics else ""
        meta = f"{rel} | topics: {topics}" if topics else rel
        line = f"- ({meta}) {m.summary.strip()}"
        mem_lines.append(line)

    live_lines = [f"- earlier this session: {s.strip()}" for s in live_summaries if s.strip()]

    if not mem_lines and not live_lines:
        return ""

    parts.append("[MEMEX — relevant prior memories]")
    parts.extend(mem_lines)
    if live_lines:
        parts.append("[LIVE SESSION CONTEXT — pruned from active conversation]")
        parts.extend(live_lines)
    parts.append("[/MEMEX]")

    block = "\n".join(parts)
    if len(block) <= max_chars:
        return block

    # Truncate at line boundaries to stay under budget.
    out: list[str] = ["[MEMEX — relevant prior memories]"]
    used = len(out[0]) + 1
    for line in mem_lines + live_lines:
        if used + len(line) + 1 > max_chars - 20:
            break
        out.append(line)
        used += len(line) + 1
    out.append("[/MEMEX]")
    return "\n".join(out)
