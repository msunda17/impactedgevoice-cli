"""
document_processor.py — Batch document summarization using 9B model.

Separate from the voice orchestrator. Designed for:
  * Long documents (10K-100K tokens)
  * Single-shot or multi-chunk summarization
  * Text output (no TTS)

Usage:
    from whisperloop.document_processor import DocumentProcessor
    
    proc = DocumentProcessor()
    summary = proc.summarize_file("paper.pdf", method="map_reduce")
    print(summary)
"""

import logging
import re
from pathlib import Path
from typing import Optional

from whisperloop.kv_cache import KVCacheManager
from whisperloop.model_router import ModelRouter, TaskTier

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Batch document processing using the COMPLEX (7B) model tier.
    Optimized for throughput, not latency.
    """

    # Chunk size in tokens. Sized at runtime as a fraction of the model's
    # actual n_ctx so we don't OOM the KV cache mid-summarization.
    # The default below is the upper bound; the effective value is computed
    # in `_effective_chunk_size()` based on the loaded model's n_ctx.
    CHUNK_SIZE = 6000      # Default ceiling (used when n_ctx is unknown)
    CHUNK_OVERLAP = 200    # Overlap to maintain coherence between chunks
    # Reserve this fraction of n_ctx for system prompt + per-chunk summary +
    # safety margin. Effective chunk size becomes n_ctx * (1 - this).
    CTX_RESERVE_FRACTION = 0.45

    def __init__(self, model_router: Optional[ModelRouter] = None):
        self.router = model_router or ModelRouter()
        # Always use the complex model for documents
        self.kv: Optional[KVCacheManager] = None

    def _ensure_loaded(self) -> KVCacheManager:
        """Lazy-load the complex model."""
        if self.kv is None:
            self.kv = self.router.select_model(
                query="document summarization",
                task_hint="summarize",
                force_tier=TaskTier.COMPLEX
            )
        return self.kv

    def _effective_chunk_size(self) -> int:
        """Chunk size in tokens, scaled to the loaded model's n_ctx."""
        try:
            kv = self._ensure_loaded()
            n_ctx = getattr(kv, "n_ctx", None) or self.CHUNK_SIZE
            safe = int(n_ctx * (1.0 - self.CTX_RESERVE_FRACTION))
            return max(512, min(self.CHUNK_SIZE, safe))
        except Exception:
            return self.CHUNK_SIZE

    def size_limits(self) -> dict:
        """
        Return human-friendly bounds for the CLI to display.
        Approximate — uses 4 chars/token and 2500 chars/page heuristics.
        """
        chunk_tokens = self._effective_chunk_size()
        # Map-reduce supports many chunks (KV resets between them) but the
        # final reduce step must fit combined chunk summaries. With ~256-token
        # chunk summaries we can comfortably reduce ~12 chunks before recursion.
        max_chunks_one_pass = 12
        max_tokens = chunk_tokens * max_chunks_one_pass
        max_chars = max_tokens * 4
        max_pages_est = max_chars // 2500
        return {
            "chunk_tokens": chunk_tokens,
            "chunk_chars": chunk_tokens * 4,
            "max_tokens": max_tokens,
            "max_chars": max_chars,
            "max_pages_est": max_pages_est,
        }

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
        """
        Split text into overlapping chunks that fit in the model's context.
        
        Uses paragraph-aware splitting where possible.
        """
        chunk_size = chunk_size or self._effective_chunk_size()
        overlap = overlap or self.CHUNK_OVERLAP

        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text.strip())

        chunks = []
        words = text.split()
        current_chunk = []
        current_len = 0

        # Rough token estimate: 1 token ≈ 0.75 words for English
        words_per_chunk = int(chunk_size * 0.75)
        overlap_words = int(overlap * 0.75)

        for word in words:
            current_chunk.append(word)
            current_len += 1

            if current_len >= words_per_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap for next chunk
                current_chunk = current_chunk[-overlap_words:]
                current_len = len(current_chunk)

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"[DocProc] Split into {len(chunks)} chunks")
        return chunks

    def summarize_chunk(self, chunk: str, max_summary_tokens: int = 256) -> str:
        """Summarize a single chunk using the complex-tier model."""
        kv = self._ensure_loaded()

        # Reset KV to the system checkpoint before each chunk. Without this,
        # chunk N+1's prefill compounds on top of chunk N and rapidly exceeds
        # n_ctx ("failed to find a memory slot for batch of size 512").
        try:
            kv.restore_checkpoint("system")
        except Exception as e:
            logger.warning("[DocProc] KV reset before chunk failed: %s", e)

        prompt = (
            "Summarize the following text concisely, capturing the main points:\n\n"
            f"{chunk}\n\n"
            "Summary:"
        )

        tokens = []
        for tok in kv.generate(prompt, max_tokens=max_summary_tokens):
            tokens.append(tok)

        summary = ''.join(tokens).strip()
        logger.info(f"[DocProc] Chunk summary: {len(summary)} chars")
        return summary

    def map_reduce_summarize(
        self,
        text: str,
        max_final_tokens: int = 512,
        chunk_summary_tokens: int = 256
    ) -> str:
        """
        Map-reduce summarization: summarize chunks, then combine.
        
        Best for very long documents where no single prompt fits.
        """
        chunks = self.chunk_text(text)

        # MAP: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"[DocProc] Summarizing chunk {i+1}/{len(chunks)}")
            summary = self.summarize_chunk(chunk, chunk_summary_tokens)
            chunk_summaries.append(summary)

        # REDUCE: Combine summaries into final summary
        combined = "\n\n".join(chunk_summaries)

        if len(combined.split()) < 500:  # Fits in one prompt
            final_prompt = (
                "Combine these section summaries into a coherent overall summary:\n\n"
                f"{combined}\n\n"
                "Overall Summary:"
            )
        else:
            # Still too long — recursive summarize
            return self.map_reduce_summarize(
                combined,
                max_final_tokens=max_final_tokens,
                chunk_summary_tokens=chunk_summary_tokens
            )

        kv = self._ensure_loaded()
        tokens = []
        for tok in kv.generate(final_prompt, max_tokens=max_final_tokens):
            tokens.append(tok)

        return ''.join(tokens).strip()

    def single_pass_summarize(self, text: str, max_tokens: int = 512) -> str:
        """
        Single-pass summarization for documents that fit in context.
        Higher quality than map-reduce (sees full context).
        """
        # Check if text fits
        chunk_tokens = self._effective_chunk_size()
        est_tokens = len(text.split()) / 0.75
        if est_tokens > chunk_tokens:
            logger.warning(f"[DocProc] Text too long for single-pass ({est_tokens:.0f} tokens), using map-reduce")
            return self.map_reduce_summarize(text, max_final_tokens=max_tokens)

        # Reset KV to the system checkpoint so we don't compound across calls.
        try:
            kv = self._ensure_loaded()
            kv.restore_checkpoint("system")
        except Exception:
            pass

        kv = self._ensure_loaded()

        prompt = (
            "Provide a comprehensive summary of the following document. "
            "Include key points, conclusions, and important details:\n\n"
            f"{text}\n\n"
            "Summary:"
        )

        tokens = []
        for tok in kv.generate(prompt, max_tokens=max_tokens):
            tokens.append(tok)

        return ''.join(tokens).strip()

    def summarize_file(self, path: str, method: str = "auto", max_tokens: int = 512) -> str:
        """
        Summarize a file. Auto-detects format and method.
        
        Args:
            path: Path to document (txt, md, pdf if PyPDF2 installed)
            method: 'single_pass', 'map_reduce', or 'auto' (chooses based on length)
            max_tokens: Maximum output length
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        # Extract text based on file type
        if path.suffix.lower() == '.pdf':
            text = self._extract_pdf(path)
        elif path.suffix.lower() in ('.txt', '.md', '.rst'):
            text = path.read_text(encoding='utf-8')
        else:
            # Try as text
            text = path.read_text(encoding='utf-8')

        logger.info(f"[DocProc] Loaded document: {len(text)} chars")

        # Pre-flight size check. We CAN handle bigger via recursive map-reduce
        # but the user should know they're past the comfortable limit.
        limits = self.size_limits()
        if len(text) > limits["max_chars"]:
            logger.warning(
                "[DocProc] Document is %d chars (~%d pages); soft limit is "
                "%d chars (~%d pages). Falling back to recursive map-reduce "
                "(slower, lower fidelity).",
                len(text), len(text) // 2500,
                limits["max_chars"], limits["max_pages_est"],
            )

        # Choose method
        chunk_tokens = self._effective_chunk_size()
        if method == "auto":
            est_tokens = len(text.split()) / 0.75
            if est_tokens < chunk_tokens:
                method = "single_pass"
            else:
                method = "map_reduce"
            logger.info(f"[DocProc] Auto-selected method: {method}")

        if method == "single_pass":
            return self.single_pass_summarize(text, max_tokens)
        else:
            return self.map_reduce_summarize(text, max_final_tokens=max_tokens)

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF using pypdf (or PyPDF2 as fallback)."""
        try:
            import pypdf as _pdf_lib
        except ImportError:
            try:
                import PyPDF2 as _pdf_lib  # type: ignore[no-redef]
            except ImportError:
                raise ImportError(
                    "No PDF library found. Install with: pip install pypdf"
                )
        text = ""
        with open(path, 'rb') as f:
            reader = _pdf_lib.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    def answer_question(self, document: str, question: str, max_tokens: int = 256) -> str:
        """
        RAG-style: Answer a question about a document.
        For long docs, uses chunking + retrieval (simplified version).
        """
        kv = self._ensure_loaded()

        # Simple approach: if doc fits, include it all
        est_tokens = len(document.split()) / 0.75 + len(question.split()) / 0.75

        if est_tokens < self.CHUNK_SIZE:
            prompt = (
                "Based on the following document, answer the question.\n\n"
                f"Document:\n{document}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
        else:
            # Find most relevant chunk (simple keyword match)
            chunks = self.chunk_text(document)
            best_chunk = max(chunks, key=lambda c: self._relevance_score(c, question))

            prompt = (
                "Based on the following excerpt from the document, answer the question.\n\n"
                f"Excerpt:\n{best_chunk}\n\n"
                f"Question: {question}\n\n"
                "Answer (note: answer only based on the excerpt provided):"
            )

        tokens = []
        for tok in kv.generate(prompt, max_tokens=max_tokens):
            tokens.append(tok)

        return ''.join(tokens).strip()

    def _relevance_score(self, chunk: str, question: str) -> float:
        """Simple keyword overlap scoring for chunk selection."""
        chunk_words = set(chunk.lower().split())
        question_words = set(question.lower().split())
        return len(chunk_words & question_words)
