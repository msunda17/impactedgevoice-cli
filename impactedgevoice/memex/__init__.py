"""
Memex — compact, queryable memory of past conversations and documents.

Public API:
    from impactedgevoice.memex import Memex

    memex = Memex()
    memex.store(user_query, response, modality="voice")
    memories = memex.recall(query, k=5)
    thread = memex.get_thread(mem_id)   # cross-session linked chain
    memex.summarize_session()
"""

from impactedgevoice.memex.manager import Memex, Memory
from impactedgevoice.memex.linker import MemoryLinker

__all__ = ["Memex", "Memory", "MemoryLinker"]
