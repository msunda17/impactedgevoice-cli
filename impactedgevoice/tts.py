import wave
import io
import numpy as np
from piper import PiperVoice

class TTS:
    def __init__(self, model_path: str):
        print(f"Loading TTS model from {model_path}...")
        self.voice = PiperVoice.load(model_path)
        print("TTS model loaded.")

    def synthesize(self, text: str) -> np.ndarray:
        # Piper synthesize() returns a generator of AudioChunk objects.
        # We must iterate it to actually generate the audio!
        chunks = []
        for audio_chunk in self.voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)
            
        if not chunks:
            return np.zeros(0, dtype=np.float32)
            
        return np.concatenate(chunks)
