"""
Memex — compact, queryable memory of past conversations and documents.

Public API:
    from whisperloop.memex import Memex

    memex = Memex()
    memex.store(user_query, response, modality="voice")
    memories = memex.recall(query, k=5)
    thread = memex.get_thread(mem_id)   # cross-session linked chain
    memex.summarize_session()
"""

from whisperloop.memex.manager import Memex, Memory
from whisperloop.memex.linker import MemoryLinker

__all__ = ["Memex", "Memory", "MemoryLinker"]
