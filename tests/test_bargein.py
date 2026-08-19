"""
test_bargein.py — Barge-in state machine + KV cache truncation tests.

These tests verify:
  1. State transitions follow the documented FSM exactly.
  2. should_interrupt() debouncing rejects brief noise and accepts
     sustained loud audio.
  3. interrupt() truncates the KV cache back to the pre-assistant
     checkpoint and clears the TTS queue.
  4. The cancel_event reaches the kv.generate() decode loop and stops
     it within a few tokens (we don't require zero-latency cancel,
     but we do require it to happen before max_tokens).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL_PATH = str(Path(__file__).parent.parent / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf")


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def kv_manager():
    from impactedgevoice.kv_cache import KVCacheManager
    mgr = KVCacheManager(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1)
    mgr.load()
    mgr.warm_up()
    return mgr


@pytest.fixture
def controller(kv_manager):
    from impactedgevoice.bargein import BargeInController
    q: asyncio.Queue = asyncio.Queue()
    return BargeInController(kv_manager, q)


# ---------------------------------------------------------------------- #
# 1. State transitions                                                    #
# ---------------------------------------------------------------------- #

def test_initial_state_is_idle(controller):
    from impactedgevoice.bargein import State
    assert controller.state == State.IDLE


def test_idle_to_user_speaking_on_speech_start(controller):
    from impactedgevoice.bargein import State
    ok = controller.on_user_speech_start()
    assert ok is True
    assert controller.state == State.USER_SPEAKING


def test_duplicate_speech_start_is_noop(controller):
    from impactedgevoice.bargein import State
    controller.on_user_speech_start()
    ok = controller.on_user_speech_start()
    assert ok is False
    assert controller.state == State.USER_SPEAKING


def test_user_speaking_to_thinking_on_speech_end(controller):
    from impactedgevoice.bargein import State
    controller.on_user_speech_start()
    controller.on_user_speech_end()
    assert controller.state == State.THINKING


def test_thinking_to_speaking_on_first_audio(controller):
    from impactedgevoice.bargein import State
    controller.on_user_speech_start()
    controller.on_user_speech_end()
    controller.on_first_tts_audio()
    assert controller.state == State.SPEAKING


def test_speaking_to_idle_on_turn_complete(controller):
    from impactedgevoice.bargein import State
    controller.on_user_speech_start()
    controller.on_user_speech_end()
    controller.on_first_tts_audio()
    controller.on_turn_complete()
    assert controller.state == State.IDLE


# ---------------------------------------------------------------------- #
# 2. Barge-in detection (debouncing)                                      #
# ---------------------------------------------------------------------- #

def _into_speaking(controller):
    controller.on_user_speech_start()
    controller.on_user_speech_end()
    controller.on_first_tts_audio()


def test_should_interrupt_rejects_quiet_audio(controller):
    _into_speaking(controller)
    quiet = np.zeros(512, dtype=np.float32)  # silence
    vad = {"start": 0}
    assert controller.should_interrupt(quiet, vad) is False


def test_should_interrupt_rejects_brief_loud_burst(controller):
    """One single loud chunk (~32ms) is below the 200ms debounce floor."""
    _into_speaking(controller)
    loud = (np.random.randn(512).astype(np.float32) * 0.5)  # large RMS
    vad = {"start": 0}
    # Single chunk — should NOT trigger
    assert controller.should_interrupt(loud, vad) is False


def test_should_interrupt_accepts_sustained_loud_audio(controller):
    _into_speaking(controller)
    loud = (np.random.randn(512).astype(np.float32) * 0.5)
    vad = {"start": 0}
    triggered = False
    # Feed 10 consecutive loud chunks (~320ms), well above the 200ms floor
    for _ in range(10):
        if controller.should_interrupt(loud, vad):
            triggered = True
            break
    assert triggered, "Sustained loud audio should trigger barge-in"


def test_should_interrupt_ignored_when_not_speaking(controller):
    """Even loud audio should be ignored if state != SPEAKING."""
    # We're still in IDLE
    loud = (np.random.randn(512).astype(np.float32) * 0.5)
    vad = {"start": 0}
    for _ in range(10):
        assert controller.should_interrupt(loud, vad) is False


# ---------------------------------------------------------------------- #
# 3. Interrupt sequence: KV truncation + queue drain                      #
# ---------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_interrupt_drains_tts_queue_and_truncates_kv(kv_manager, controller):
    from impactedgevoice.bargein import State

    # Move into SPEAKING
    _into_speaking(controller)

    # Snapshot KV length, save as the "pre_assistant" checkpoint
    kv_pre = kv_manager.kv_len
    controller.pre_assistant_checkpoint = kv_manager.save_checkpoint("test_bargein_pre")

    # Simulate the KV cache growing during decode (push some real tokens in)
    # We don't actually run generate(); we just append junk to grow the cache.
    junk = kv_manager._encode("blah blah blah extra tokens", add_bos=False)
    kv_manager._eval(junk)
    assert kv_manager.kv_len > kv_pre

    # Fill the TTS queue with pretend pending sentences
    for s in ("Sentence one.", "Sentence two.", "Sentence three."):
        await controller.tts_queue.put(s)
    assert controller.tts_queue.qsize() == 3

    # Trigger interrupt
    await controller.interrupt()

    # KV should be back to pre_assistant checkpoint
    assert kv_manager.kv_len == kv_pre, (
        f"KV not restored: got {kv_manager.kv_len}, expected {kv_pre}"
    )

    # TTS queue should be drained
    assert controller.tts_queue.empty()

    # Post-interrupt we should be USER_SPEAKING
    assert controller.state == State.USER_SPEAKING

    # cancel_event must be cleared so the next turn isn't auto-cancelled
    assert not controller.cancel_event.is_set()


# ---------------------------------------------------------------------- #
# 4. cancel_event reaches kv.generate()                                   #
# ---------------------------------------------------------------------- #

def test_cancel_event_stops_decode_loop(kv_manager):
    """
    Set the cancel_event after a short delay; verify generate() returns
    far fewer tokens than max_tokens.
    """
    cancel = threading.Event()

    # Trip the event 50ms in — well before a 150-token decode finishes
    def trip_after_delay():
        time.sleep(0.05)
        cancel.set()
    threading.Thread(target=trip_after_delay, daemon=True).start()

    kv_pre = kv_manager.kv_len
    tokens_produced = []
    for tok in kv_manager.generate(
        "Tell me a very long story about dragons.",
        max_tokens=150,
        cancel_event=cancel,
    ):
        tokens_produced.append(tok)

    # We allow up to ~30 tokens before cancellation lands (decode is fast)
    assert len(tokens_produced) < 150, (
        f"Decode did not honor cancel_event (got {len(tokens_produced)} tokens)"
    )
    print(f"\n[TEST] Decode stopped after {len(tokens_produced)} tokens")

    # KV cache grew partially but did NOT get the trailing EOT
    kv_post = kv_manager.kv_len
    assert kv_post > kv_pre
    # Clean up so other tests aren't affected
    kv_manager.truncate_to(kv_pre)
