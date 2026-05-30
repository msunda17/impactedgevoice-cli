import numpy as np
from silero_vad import load_silero_vad, VADIterator

class VAD:
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        print("Loading Silero VAD model...")
        self.model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(self.model, threshold=threshold, sampling_rate=sampling_rate)
        print("VAD model loaded.")

    def process_chunk(self, chunk: np.ndarray):
        """Returns dict with 'start' or 'end' trigger, or None"""
        # Silero VAD takes chunk sizes of 512, 1024, or 1536
        return self.vad_iterator(chunk)

    def reset(self):
        self.vad_iterator.reset_states()
