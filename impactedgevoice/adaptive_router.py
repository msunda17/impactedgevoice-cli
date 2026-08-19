"""
adaptive_router.py — Unified entry point with dynamic load-aware routing.

Replaces separate voice/document CLIs. Single dispatcher that:
  1. Classifies input modality (voice, file, text)
  2. Estimates task complexity (length, keywords, semantic signals)
  3. Estimates current system load (memory, queue depth)
  4. Routes to the optimal model tier
"""

import logging
import os
import psutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from impactedgevoice.model_router import ModelRouter, TaskTier
from impactedgevoice.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


class InputModality(Enum):
    VOICE = "voice"           # Live microphone stream
    FILE = "file"             # Path to document/audio file
    TEXT = "text"             # Direct text query


@dataclass
class RouteDecision:
    """The result of routing analysis — all factors that informed the choice."""
    modality: InputModality
    tier: TaskTier
    estimated_tokens: int
    complexity_score: float    # 0.0 (trivial) - 1.0 (heavy)
    load_score: float          # 0.0 (idle) - 1.0 (saturated)
    reasoning: str
    use_chunking: bool = False
    use_streaming_tts: bool = True


class AdaptiveRouter:
    """
    Load-aware adaptive router. Replaces the dual-CLI design.
    
    Key decisions:
      * Modality detection: voice / file / text
      * Complexity scoring: combines length, keywords, file size
      * Load awareness: degrades to smaller model if RAM/CPU saturated
      * Lazy loading: only loads models actually used
    """

    # Thresholds (tunable)
    COMPLEX_TOKEN_THRESHOLD = 800      # Above this → 9B
    HIGH_LOAD_RAM_PCT = 75.0           # Above 75% RAM → force smaller model
    HIGH_LOAD_CPU_PCT = 90.0           # Sustained high CPU → degrade
    FILE_SIZE_LARGE_KB = 50            # Files >50KB → chunking + 9B

    COMPLEX_KEYWORDS = {
        'summarize', 'summary', 'document', 'paper', 'article',
        'analyze', 'compare', 'contrast', 'enumerate', 'list the',
        'explain in detail', 'pros and cons', 'why does', 'how does',
        'implications', 'consequences', 'differences between',
    }

    def __init__(self, model_router: Optional[ModelRouter] = None):
        self.router = model_router or ModelRouter()

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def route(
        self,
        user_input: Union[str, Path],
        force_tier: Optional[TaskTier] = None,
    ) -> tuple[RouteDecision, KVCacheManager]:
        """
        Classify input, decide tier, return (decision, KVCacheManager).
        
        Single entry point for all interaction modes.
        """
        modality = self._detect_modality(user_input)
        complexity = self._score_complexity(user_input, modality)
        load = self._score_load()

        # Decision logic: combine complexity and load
        if force_tier:
            tier = force_tier
            reasoning = f"Forced tier: {force_tier.value}"
        else:
            tier, reasoning = self._decide_tier(complexity, load, modality)

        # Estimate token count for prefill budgeting
        est_tokens = self._estimate_tokens(user_input, modality)

        decision = RouteDecision(
            modality=modality,
            tier=tier,
            estimated_tokens=est_tokens,
            complexity_score=complexity,
            load_score=load,
            reasoning=reasoning,
            use_chunking=(est_tokens > 5000),
            use_streaming_tts=(modality == InputModality.VOICE),
        )

        logger.info(f"[AdaptiveRouter] {decision}")
        kv = self.router.select_model(
            query=str(user_input)[:200],
            force_tier=tier,
        )
        return decision, kv

    # ──────────────────────────────────────────────────────────────
    #  Modality detection
    # ──────────────────────────────────────────────────────────────

    def _detect_modality(self, user_input: Union[str, Path]) -> InputModality:
        """Decide if input is voice (live), file (path), or text."""
        if isinstance(user_input, Path):
            return InputModality.FILE

        s = str(user_input).strip()

        # Heuristic: looks like a file path?
        if len(s) < 260 and (
            s.endswith(('.pdf', '.txt', '.md', '.wav', '.mp3', '.docx'))
            or (os.path.sep in s and os.path.exists(s))
        ):
            return InputModality.FILE

        # Special token for live voice mode
        if s.lower() in ('voice', 'mic', 'listen', '--voice'):
            return InputModality.VOICE

        return InputModality.TEXT

    # ──────────────────────────────────────────────────────────────
    #  Complexity scoring
    # ──────────────────────────────────────────────────────────────

    def _score_complexity(self, user_input: Union[str, Path], modality: InputModality) -> float:
        """Score 0.0 - 1.0 for how 'heavy' this task is."""
        if modality == InputModality.VOICE:
            return 0.2  # Live voice is bounded by what user can say

        if modality == InputModality.FILE:
            try:
                size_kb = Path(str(user_input)).stat().st_size / 1024
                # Files >500KB are definitely complex
                return min(1.0, size_kb / 500)
            except (OSError, ValueError):
                return 0.5

        # Text mode: weighted keyword + length signal
        text = str(user_input).lower()
        score = 0.0

        # Keyword presence (each adds ~0.15)
        keyword_hits = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in text)
        score += min(0.6, keyword_hits * 0.15)

        # Length (up to 0.4 for >500 words)
        word_count = len(text.split())
        score += min(0.4, word_count / 1250)

        return min(1.0, score)

    # ──────────────────────────────────────────────────────────────
    #  System load scoring
    # ──────────────────────────────────────────────────────────────

    def _score_load(self) -> float:
        """Score 0.0 - 1.0 for current system pressure."""
        try:
            ram_pct = psutil.virtual_memory().percent
            cpu_pct = psutil.cpu_percent(interval=0.1)
            # Weighted average — RAM matters more for model loading
            return min(1.0, (ram_pct * 0.7 + cpu_pct * 0.3) / 100)
        except Exception as e:
            logger.warning(f"[AdaptiveRouter] Load scoring failed: {e}")
            return 0.5

    # ──────────────────────────────────────────────────────────────
    #  Tier decision logic
    # ──────────────────────────────────────────────────────────────

    def _decide_tier(
        self,
        complexity: float,
        load: float,
        modality: InputModality,
    ) -> tuple[TaskTier, str]:
        """Combine complexity + load to pick a tier."""
        # Files always start at COMPLEX (need long-context model)
        if modality == InputModality.FILE:
            return TaskTier.COMPLEX, f"File modality → 9B (load={load:.2f})"

        # High system load → degrade complex queries to simple
        if load > 0.85:
            return TaskTier.SIMPLE, (
                f"High load ({load:.2f}) → degraded to 2B "
                f"despite complexity={complexity:.2f}"
            )

        # Complexity threshold
        if complexity > 0.5:
            return TaskTier.COMPLEX, (
                f"Complex query (score={complexity:.2f}) → 9B"
            )

        return TaskTier.SIMPLE, (
            f"Simple query (complexity={complexity:.2f}, load={load:.2f}) → 2B"
        )

    # ──────────────────────────────────────────────────────────────
    #  Token estimation
    # ──────────────────────────────────────────────────────────────

    def _estimate_tokens(self, user_input: Union[str, Path], modality: InputModality) -> int:
        """Rough token count estimate (1 token ≈ 0.75 words for English)."""
        if modality == InputModality.VOICE:
            return 50  # Typical voice query

        if modality == InputModality.FILE:
            try:
                size_bytes = Path(str(user_input)).stat().st_size
                # Crude: ~5 bytes per token average
                return int(size_bytes / 5)
            except (OSError, ValueError):
                return 1000

        return int(len(str(user_input).split()) / 0.75)
