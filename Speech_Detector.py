from typing import Callable
from silero_vad import load_silero_vad
import numpy as np
import torch
from numpy.typing import NDArray



def load_speech_detect_model(threshold=0.8, sample_rate: int=16000)->Callable[[NDArray[np.float32]],bool]:
    model = load_silero_vad()
    def detect_speech(audio:NDArray[np.float32]) -> bool:
        """Runs Silero VAD on the current audio buffer to find speech segments."""
        audio_tensor = torch.from_numpy(audio)     
        speech_prob = model(audio_tensor, sample_rate).item()
        return speech_prob>= threshold
    return detect_speech
