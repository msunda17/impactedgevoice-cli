"""
summarizer.py — Memory compaction via a dedicated lightweight LLM.

Hybrid design:
  * Structured prompt enforces output schema (entities, intent, key facts, topics).
  * LLM decides which spans are salient — we don't hardcode token weights.
  * Output is parsed into a MemorySummary dataclass.

Why a DEDICATED model (not the shared conversation KV):
  * llama_cpp.Llama.__call__() internally invokes self.reset(), wiping the KV cache.
  * Running summarization on the conversation model corrupts the conversation state.
  * The dedicated summarizer owns its own llama_cpp.Llama instance (no checkpoints,
    no persistent KV needed — each call is stateless).
  * Falls back to extractive summarization if the dedicated model is unavailable.

Why hybrid (not pure rules, not pure freeform):
  * Rule-based (TF-IDF, NER) misses intent and paraphrasing.
  * Pure freeform LLM produces variable-length, schema-violating outputs.
  * Schema-guided prompt + LLM = bounded length + guaranteed fields + semantic flexibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from whisperloop.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)

# Candidate model filenames for the dedicated summarizer, searched in models/
# Prefer the smallest available — 1B Q4 is plenty for extractive-like compaction.
_SUMMARIZER_CANDIDATES = [
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
]
_MODELS_DIR = Path("models")


@dataclass
class MemorySummary:
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0=trivial, 1=critical


_SUMMARY_PROMPT = """You are a memory compaction assistant. Compact the following turn into a brief structured summary.

Output ONLY valid JSON with this exact schema:
{{"summary": "<60-120 words: user intent + key facts + named entities + numbers>",
  "topics": ["<3-5 short topic tags, lowercase, snake_case>"],
  "importance": <float 0.0-1.0; 0.3=chitchat, 0.5=normal Q&A, 0.7=technical, 0.9=user explicitly asked to remember>}}

Guidelines:
- Capture WHO/WHAT/WHEN/WHERE/HOW from the conversation.
- Preserve named entities, file names, technical terms, numbers verbatim.
- Drop pleasantries, hedging, repetition.
- Topics should enable future keyword recall.

USER TURN: {user}

ASSISTANT RESPONSE: {response}

JSON:"""


_SESSION_PROMPT = """Summarize this entire conversation session into a single high-level summary.

Output ONLY valid JSON:
{{"summary": "<100-200 words covering main themes, conclusions, action items>",
  "topics": ["<5-10 topic tags>"],
  "importance": <float 0.0-1.0>}}

CONVERSATION SUMMARIES (chronological):
{summaries}

JSON:"""


class Summarizer:
    """
    LLM-driven compaction using a DEDICATED summarizer model.

    The dedicated model is a separate llama_cpp.Llama instance that never
    shares state with the conversation KV cache. Each summarization call
    is fully stateless (llama_cpp resets internally on each __call__).

    Falls back to extractive summarization if no model file is found.
    """

    def __init__(self, kv_provider=None):
        """
        Args:
            kv_provider: Unused — kept for API compatibility. The dedicated
                summarizer model is loaded independently of the conversation KV.
        """
        self._kv_provider = kv_provider  # kept for compat, not used
        self._summarizer_llm = self._load_summarizer_model()

    @staticmethod
    def _load_summarizer_model():
        """Find and load the smallest available GGUF as a dedicated summarizer."""
        for name in _SUMMARIZER_CANDIDATES:
            path = _MODELS_DIR / name
            if path.exists():
                try:
                    import llama_cpp
                    llm = llama_cpp.Llama(
                        model_path=str(path),
                        n_gpu_layers=-1,
                        n_ctx=2048,      # enough for turn + session summaries
                        logits_all=False,
                        verbose=False,
                    )
                    logger.info("[MEMEX] Dedicated summarizer loaded: %s", name)
                    return llm
                except Exception as e:
                    logger.warning("[MEMEX] Could not load summarizer %s: %s", name, e)
        logger.info("[MEMEX] No dedicated summarizer model found — using extractive fallback")
        return None

    def summarize_turn(
        self, user_text: str, response_text: str
    ) -> MemorySummary:
        """Compact one (user, response) pair into ~80 tokens + topics."""
        if self._summarizer_llm is None:
            return self._extractive_fallback(user_text, response_text)

        try:
            prompt = _SUMMARY_PROMPT.format(
                user=_truncate(user_text, 800),
                response=_truncate(response_text, 1600),
            )
            raw = self._llm_call(prompt, max_tokens=220)
            return self._parse_json(raw) or self._extractive_fallback(user_text, response_text)
        except Exception as e:
            logger.debug("[MEMEX] Turn summarization fell back to extractive: %s", e)
            return self._extractive_fallback(user_text, response_text)

    def summarize_session(self, turn_summaries: list[str]) -> MemorySummary:
        """Roll up many turn summaries into a session-level summary."""
        if not turn_summaries:
            return MemorySummary(summary="(empty session)", topics=[], importance=0.0)
        if self._summarizer_llm is None:
            return MemorySummary(
                summary=" | ".join(turn_summaries[:5])[:1000],
                topics=[],
                importance=0.5,
            )
        try:
            joined = "\n".join(f"- {s}" for s in turn_summaries[-50:])
            prompt = _SESSION_PROMPT.format(summaries=_truncate(joined, 4000))
            raw = self._llm_call(prompt, max_tokens=300)
            return self._parse_json(raw) or MemorySummary(
                summary=" | ".join(turn_summaries[:5])[:1000], importance=0.5
            )
        except Exception as e:
            logger.debug("[MEMEX] Session summarization fell back: %s", e)
            return MemorySummary(summary=" | ".join(turn_summaries[:5])[:1000])

    def summarize_pruned_block(self, conversation_excerpts: list[tuple[str, str]]) -> MemorySummary:
        """
        Compact a block of (user, assistant) pairs that are about to be evicted
        from the live KV cache. Used by the in-session pruner.
        """
        if not conversation_excerpts:
            return MemorySummary(summary="", importance=0.4)

        joined = "\n\n".join(
            f"USER: {_truncate(u, 200)}\nASSISTANT: {_truncate(a, 400)}"
            for u, a in conversation_excerpts
        )
        if self._summarizer_llm is None:
            return self._extractive_fallback("(pruned block)", joined)
        try:
            prompt = (
                "Summarize this conversation block into a JSON object so it can be "
                "rehydrated later as concise context. Preserve named entities, decisions, "
                "and technical terms.\n\n"
                'Output ONLY: {"summary": "<120-200 words>", "topics": [...], "importance": <0-1>}\n\n'
                f"CONVERSATION BLOCK:\n{_truncate(joined, 4000)}\n\nJSON:"
            )
            raw = self._llm_call(prompt, max_tokens=320)
            return self._parse_json(raw) or self._extractive_fallback("block", joined)
        except Exception as e:
            logger.debug("[MEMEX] Pruned-block summarization fell back: %s", e)
            return self._extractive_fallback("block", joined)

    # ---- internals ----------------------------------------------------

    def _llm_call(self, prompt: str, max_tokens: int = 220) -> str:
        """
        Call the DEDICATED summarizer model. This is isolated from the
        conversation KV cache — the dedicated llama_cpp.Llama instance
        resets its own internal state on each call, which is safe because
        we never care about its KV history.

        Raises RuntimeError if no dedicated model was loaded (callers catch
        this and fall back to extractive).
        """
        if self._summarizer_llm is None:
            raise RuntimeError("No dedicated summarizer model available")
        output = self._summarizer_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,    # low temp for structured JSON output
            stop=["}"],         # stop after closing brace
            echo=False,
        )
        # llama_cpp returns stop token in text sometimes; ensure JSON closes
        text = output["choices"][0]["text"].strip()
        if text and not text.endswith("}"):
            text += "}"
        return text

    @staticmethod
    def _parse_json(raw: str) -> Optional[MemorySummary]:
        if not raw:
            return None
        # Find the first JSON object in the output
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return MemorySummary(
                summary=str(obj.get("summary", ""))[:1500],
                topics=[str(t).strip().lower() for t in obj.get("topics", [])][:8],
                importance=max(0.0, min(1.0, float(obj.get("importance", 0.5)))),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug("[MEMEX] JSON parse failed: %s", e)
            return None

    @staticmethod
    def _extractive_fallback(user: str, response: str) -> MemorySummary:
        """No-LLM fallback: take leading sentences + simple keyword topics."""
        u = (user or "").strip()
        r = (response or "").strip()
        first_user = u.split(".")[0][:200]
        first_resp = " ".join(r.split(".")[:2])[:400]
        summary = f"User asked: {first_user}. Response: {first_resp}".strip()
        # Cheap topic extraction: longest words >= 4 chars, non-stopword-ish
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", (u + " " + r).lower())
        stopwords = {
            "this", "that", "with", "from", "have", "been", "were", "what",
            "when", "where", "which", "their", "would", "could", "about",
            "your", "they", "them", "then", "than", "into", "more", "some",
            "does", "doing", "like", "just", "very", "also",
        }
        seen = []
        for w in candidates:
            if w not in stopwords and w not in seen:
                seen.append(w)
            if len(seen) >= 5:
                break
        return MemorySummary(summary=summary[:1000], topics=seen, importance=0.4)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"
