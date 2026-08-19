import sounddevice as sd
import numpy as np
import asyncio

class AudioStreamer:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = None
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        # Push to queue thread-safely (callback runs on a separate C thread!)
        chunk = indata.copy().flatten()
        if hasattr(self, 'loop') and self.loop:
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, chunk)

    def start_recording(self):
        # Capture the event loop here (after asyncio.run has started)
        self.loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

async def play_audio_async(audio: np.ndarray, sample_rate: int = 22050):
    """
    Plays audio on a background thread using sd.wait() to ensure Windows
    audio drivers correctly flush the buffer, without blocking the asyncio loop.
    """
    def _play():
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        
    await asyncio.to_thread(_play)

