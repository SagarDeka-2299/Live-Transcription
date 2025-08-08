from faster_whisper import WhisperModel
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Callable
    

def load_stt_model()->Callable[[NDArray[np.float32]],str]:
    model = WhisperModel(
        "base", 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        compute_type= "float16" if torch.cuda.is_available() else "int8"
    )
    def inference(audio_np: NDArray[np.float32])->str:
        segments, _ = model.transcribe(
            audio_np,
            beam_size=5,
            word_timestamps=False,
            task="transcribe",
            language="en"
        )
        full_text = "".join(segment.text for segment in segments).strip()
        return full_text
    return inference