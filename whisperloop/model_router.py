"""
model_router.py — Adaptive model selection based on task complexity.

Routes queries to appropriately-sized models:
  * 1B (Llama 3.2): Simple voice Q&A, short responses, low latency priority
  * 3B (Llama 3.2): Complex reasoning, document summarization, long-context tasks

Usage:
    router = ModelRouter()
    kv_manager = router.select_model("Summarize this 10-page document...")
    # Returns the 3B KVCacheManager instance
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

from whisperloop.kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


# ── Candidate registry ────────────────────────────────────────────────
# All known models, keyed by filename. Each entry has the HuggingFace
# repo, a human description, approximate size, and which tier(s) it
# is suitable for.
CANDIDATE_REGISTRY: dict[str, dict] = {
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "desc": "Llama 3.2 1B Q4  — fastest, ~0.8 GB",
        "size_gb": 0.8,
        "tiers": ["simple"],
    },
    "Llama-3.2-1B-Instruct-Q8_0.gguf": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "desc": "Llama 3.2 1B Q8  — higher quality, ~1.3 GB",
        "size_gb": 1.3,
        "tiers": ["simple"],
    },
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "desc": "Llama 3.2 3B Q4  — balanced speed/quality, ~2.0 GB",
        "size_gb": 2.0,
        "tiers": ["simple", "complex"],
    },
    "Llama-3.2-3B-Instruct-Q8_0.gguf": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "desc": "Llama 3.2 3B Q8  — best 3B quality, ~3.4 GB",
        "size_gb": 3.4,
        "tiers": ["simple", "complex"],
    },
    "Qwen_Qwen3.5-2B-Q4_K_M.gguf": {
        "repo": "bartowski/Qwen_Qwen3.5-2B-GGUF",
        "desc": "Qwen 3.5 2B Q4   — alternative simple tier, ~1.3 GB",
        "size_gb": 1.3,
        "tiers": ["simple"],
    },
    "Qwen_Qwen3.5-9B-Q4_K_M.gguf": {
        "repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "desc": "Qwen 3.5 9B Q4   — strong complex tier, ~5.4 GB",
        "size_gb": 5.4,
        "tiers": ["complex"],
    },
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "desc": "Qwen 2.5 7B Q4   — 128K context, ~4.4 GB",
        "size_gb": 4.4,
        "tiers": ["complex"],
    },
}


class TaskTier(Enum):
    SIMPLE = "simple"      # 1B model: quick Q&A, chitchat, facts
    COMPLEX = "complex"    # 3B model: reasoning, docs, summarization


class ModelRouter:
    """
    Maintains multiple KVCacheManager instances (one per model size)
    and routes tasks to the appropriate one.
    """

    # Preferred model filenames in priority order (first found in models/ wins)
    # Llama 3.2 preferred; also accepts Qwen if present
    SIMPLE_CANDIDATES = [
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
    ]
    COMPLEX_CANDIDATES = [
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ]
    MODELS_DIR = Path("models")

    def __init__(
        self,
        n_ctx_simple: int = 4096,
        n_ctx_complex: int = 8192,
        simple_model: Optional[str] = None,
        complex_model: Optional[str] = None,
    ):
        """
        Args:
            simple_model: Override filename for the simple tier (must exist in
                models/ or will be downloaded interactively).
            complex_model: Override filename for the complex tier.
        """
        self._simple_kv: Optional[KVCacheManager] = None
        self._complex_kv: Optional[KVCacheManager] = None
        self.n_ctx_simple = n_ctx_simple
        self.n_ctx_complex = n_ctx_complex
        # User-chosen overrides: prepended to the candidate lists
        self._simple_override: Optional[str] = simple_model
        self._complex_override: Optional[str] = complex_model

    def _resolve_model(self, candidates: list[str]) -> str:
        """
        Return the path of the first candidate found in MODELS_DIR.
        If none are found, offer an interactive picker + auto-download.
        """
        models_dir = self.MODELS_DIR.resolve()
        for name in candidates:
            p = models_dir / name
            if p.exists():
                return str(p)
        # Nothing found — prompt user to pick + download
        logger.warning("[Router] No model found in %s for candidates: %s", models_dir, candidates)
        chosen = self._prompt_and_download(candidates)
        if chosen:
            return chosen
        # Hard failure — nothing was chosen or downloaded
        available = sorted(models_dir.glob("*.gguf")) if models_dir.exists() else []
        avail_str = "\n    ".join(p.name for p in available) if available else "(none)"
        raise FileNotFoundError(
            f"No model available in {models_dir}.\n"
            f"  Available:\n    {avail_str}\n"
            f"  Run: python download_models.py"
        )

    def _prompt_and_download(self, preferred_candidates: list[str]) -> Optional[str]:
        """
        Interactive: show all known candidates for this tier, let the user
        choose one, and download it if it isn't already present.
        Returns the resolved path on success, None if user cancels.
        """
        models_dir = self.MODELS_DIR.resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        # All registry entries relevant to the tier implied by preferred_candidates
        tier_set = {"simple", "complex"}
        for name in preferred_candidates:
            if name in CANDIDATE_REGISTRY:
                tier_set = set(CANDIDATE_REGISTRY[name]["tiers"])
                break

        options = [
            (fname, info)
            for fname, info in CANDIDATE_REGISTRY.items()
            if any(t in info["tiers"] for t in tier_set)
        ]

        print()
        print("  No model found locally. Available candidates:")
        print()
        for i, (fname, info) in enumerate(options, 1):
            exists = (models_dir / fname).exists()
            tag = "  [downloaded]" if exists else f"  ({info['size_gb']} GB to download)"
            print(f"    [{i}]  {info['desc']}{tag}")
        print(f"    [0]  Cancel")
        print()

        while True:
            try:
                raw = input("  Choose model > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if raw == "0" or raw.lower() in ("q", "cancel"):
                return None
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    break
            except ValueError:
                pass
            print(f"    Enter a number between 1 and {len(options)}, or 0 to cancel.")

        fname, info = options[idx]
        target = models_dir / fname
        if target.exists():
            print(f"  ✓ Using existing {fname}")
            return str(target)

        print(f"  Downloading {fname} (~{info['size_gb']} GB) …")
        ok = self._download_model(info["repo"], fname, str(models_dir))
        if ok and target.exists():
            print(f"  ✓ Download complete: {fname}")
            return str(target)
        print(f"  ✗ Download failed. Place {fname} in {models_dir}/ manually.")
        return None

    @staticmethod
    def _download_model(repo: str, filename: str, local_dir: str) -> bool:
        """Download a GGUF from HuggingFace via huggingface-cli."""
        try:
            result = subprocess.run(
                [
                    "huggingface-cli", "download", repo, filename,
                    "--local-dir", local_dir,
                ],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error("[Router] huggingface-cli failed: %s", e)
            return False
        except FileNotFoundError:
            logger.error(
                "[Router] huggingface-cli not found. "
                "Install with: pip install huggingface-hub"
            )
            return False

    def set_model(self, tier: TaskTier, filename: str) -> None:
        """
        Override the model for a tier at runtime.
        Unloads any already-loaded KV for that tier so the next load() picks
        up the new file.
        """
        if tier == TaskTier.SIMPLE:
            self._simple_override = filename
            self._simple_kv = None
        else:
            self._complex_override = filename
            self._complex_kv = None
        logger.info("[Router] %s tier overridden → %s", tier.value.upper(), filename)

    def _candidates_for(self, tier: TaskTier) -> list[str]:
        """Return the effective candidate list, user override first."""
        if tier == TaskTier.SIMPLE:
            override = self._simple_override
            base = self.SIMPLE_CANDIDATES
        else:
            override = self._complex_override
            base = self.COMPLEX_CANDIDATES
        if override:
            # Put the override first so _resolve_model finds it immediately
            return [override] + [c for c in base if c != override]
        return base

    def load(self, tier: TaskTier = TaskTier.SIMPLE) -> None:
        """Eager-load a specific model tier."""
        if tier == TaskTier.SIMPLE and self._simple_kv is None:
            model_path = self._resolve_model(self._candidates_for(TaskTier.SIMPLE))
            logger.info(f"[Router] Loading SIMPLE model: {Path(model_path).name}")
            self._simple_kv = KVCacheManager(
                model_path=model_path,
                n_ctx=self.n_ctx_simple,
                n_gpu_layers=-1
            )
            self._simple_kv.load()
            self._simple_kv.warm_up()
            logger.info("[Router] SIMPLE model ready")

        elif tier == TaskTier.COMPLEX and self._complex_kv is None:
            try:
                model_path = self._resolve_model(self._candidates_for(TaskTier.COMPLEX))
            except FileNotFoundError:
                # No dedicated complex model — fall back to whatever simple model exists
                logger.warning(
                    "[Router] No COMPLEX model found; falling back to SIMPLE tier."
                )
                self.load(TaskTier.SIMPLE)
                self._complex_kv = self._simple_kv
                return
            logger.info(f"[Router] Loading COMPLEX model: {Path(model_path).name}")
            self._complex_kv = KVCacheManager(
                model_path=model_path,
                n_ctx=self.n_ctx_complex,
                n_gpu_layers=-1
            )
            self._complex_kv.load()
            self._complex_kv.warm_up()
            logger.info("[Router] COMPLEX model ready")

    def select_model(
        self,
        query: str,
        task_hint: Optional[str] = None,
        force_tier: Optional[TaskTier] = None,
    ) -> KVCacheManager:
        """
        Select the appropriate KV cache manager for this task.

        Args:
            query: The user query (used for heuristic classification if no hint)
            task_hint: Optional override ('summarize', 'qa', 'chat')
            force_tier: Force a specific tier

        Returns:
            KVCacheManager instance (2B or 9B)
        """
        if force_tier:
            tier = force_tier
        elif task_hint:
            tier = self._tier_from_hint(task_hint)
        else:
            tier = self._classify_query(query)

        # Lazy-load on first use
        self.load(tier)

        if tier == TaskTier.SIMPLE:
            return self._simple_kv
        return self._complex_kv

    def _tier_from_hint(self, hint: str) -> TaskTier:
        """Map task hints to tiers."""
        complex_hints = {'summarize', 'document', 'analyze', 'explain', 'reason', 'compare'}
        if any(h in hint.lower() for h in complex_hints):
            return TaskTier.COMPLEX
        return TaskTier.SIMPLE

    def _classify_query(self, query: str) -> TaskTier:
        """
        Heuristic classification based on query characteristics.
        Uses simple keyword + length-based rules.
        """
        query_lower = query.lower()

        # Complex task indicators
        complex_keywords = [
            'summarize', 'summary', 'document', 'paper', 'article',
            'explain in detail', 'compare and contrast', 'analyze',
            'why does', 'how does', 'what are the implications',
            'list the', 'enumerate', 'steps to', 'pros and cons'
        ]

        # Check for complex keywords
        if any(kw in query_lower for kw in complex_keywords):
            return TaskTier.COMPLEX

        # Long queries (>100 words) likely need bigger model
        word_count = len(query.split())
        if word_count > 100:
            return TaskTier.COMPLEX

        # Check if query implies multi-step reasoning
        reasoning_indicators = ['and then', 'after that', 'if', 'because', 'therefore']
        if any(ri in query_lower for ri in reasoning_indicators):
            return TaskTier.COMPLEX

        return TaskTier.SIMPLE

    def get_tier_info(self) -> dict:
        """Return loaded model info for diagnostics."""
        models_dir = self.MODELS_DIR.resolve()
        available = [p.name for p in sorted(models_dir.glob("*.gguf"))] if models_dir.exists() else []
        return {
            "simple_loaded": self._simple_kv is not None,
            "complex_loaded": self._complex_kv is not None,
            "simple_override": self._simple_override,
            "complex_override": self._complex_override,
            "available_ggufs": available,
            "models_dir": str(models_dir),
        }

    @staticmethod
    def prompt_model_selection(tier: TaskTier) -> Optional[str]:
        """
        Standalone interactive picker — lets the user browse all registry
        candidates for a tier and returns the chosen filename (or None).
        Used by the menu 'Model Settings' option before the router loads.
        Does NOT download — that happens inside _resolve_model → _prompt_and_download.
        """
        tier_key = tier.value  # "simple" or "complex"
        options = [
            (fname, info)
            for fname, info in CANDIDATE_REGISTRY.items()
            if tier_key in info["tiers"]
        ]
        models_dir = ModelRouter.MODELS_DIR.resolve()
        print()
        print(f"  Select {tier_key.upper()} tier model:")
        print()
        for i, (fname, info) in enumerate(options, 1):
            exists = (models_dir / fname).exists()
            tag = "  [available locally]" if exists else f"  ({info['size_gb']} GB — will download if chosen)"
            print(f"    [{i}]  {info['desc']}{tag}")
        print(f"    [0]  Keep current / cancel")
        print()
        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if raw == "0" or raw.lower() in ("q", "cancel", ""):
                return None
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]  # return just the filename
            except ValueError:
                pass
            print(f"    Enter a number between 1 and {len(options)}, or 0 to cancel.")

    def switch_context(self, from_tier: TaskTier, to_tier: TaskTier) -> None:
        """
        Optional: Transfer conversation context between models.
        Useful if a conversation starts simple but turns complex.
        """
        # This would require serializing the KV cache state
        # and reloading into the other model. Advanced feature.
        logger.info(f"[Router] Context switch {from_tier} -> {to_tier} requested")
        # TODO: Implement if needed
