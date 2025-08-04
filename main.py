import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
import asyncio
from typing import List, Dict, Optional
import os
import json

# --- Configuration & Constants ---

# Whisper Model Configuration
MODEL_SIZE = "base"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# Print device info
print(f"🖥️  PyTorch using device: {DEVICE_TYPE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Control Parameters (all times in milliseconds)
SAMPLING_RATE = 16000
VAD_CHUNK_DURATION_MS = 1000
VAD_CHUNK_SIZE = (VAD_CHUNK_DURATION_MS * SAMPLING_RATE) // 1000
MIN_SPEECH_DURATION_MS = 400
VAD_MIN_SILENCE_DURATION_MS = 500  # Silence duration for VAD model (smaller chunks)
ASR_MIN_SILENCE_DURATION_MS = 1200  # Silence duration for aggregation (manual check)
MAX_ASR_DURATION_MS = 30000
OVERLAP_DURATION_MS = 2000  # 2 seconds overlap for long segments

# --- Model Loading ---
from faster_whisper import WhisperModel

# Global variables to hold the loaded models
silero_model = None
silero_utils = None
whisper_model = None

def load_models():
    """Loads Silero VAD and Whisper models into global variables."""
    global silero_model, silero_utils, whisper_model
    
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )
    # # Move Silero VAD to GPU if available
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     print("✅ Silero VAD model loaded on GPU")
    # else:
    #     print("✅ Silero VAD model loaded on CPU")
    
    silero_model = model
    silero_utils = utils

    print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE_TYPE}...")
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    print(f"✅ Whisper model loaded successfully on {DEVICE_TYPE}")

# --- Core Transcription Logic ---

class AudioProcessor:
    """
    Manages audio buffering and speech detection for a single client.
    """
    def __init__(self, vad_model, vad_utils, websocket: WebSocket):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.vad_model = vad_model
        self.vad_utils = vad_utils
        self.websocket = websocket
        self.last_segment_words = []  # Store words from previous segment for overlap comparison
        self.current_transcription = ""
        print("AudioProcessor initialized for a new connection.")

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """Converts raw audio bytes (16-bit PCM) to a float32 numpy array."""
        raw_data = np.frombuffer(audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0

    def add_chunk(self, audio_bytes: bytes):
        """Adds a new chunk of audio to the internal buffer."""
        audio_float32 = self._bytes_to_float32(audio_bytes)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float32])

    def _get_speech_timestamps(self) -> List[Dict[str, int]]:
        """Runs Silero VAD on the current audio buffer to find speech segments."""
        (get_speech_timestamps, _, _, *_) = self.vad_utils
        
        # Convert to tensor and move to GPU if available
        audio_tensor = torch.from_numpy(self.audio_buffer)
        
        # Get timestamps - using VAD-specific silence duration
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=SAMPLING_RATE,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS  # Use VAD-specific silence duration
        )
        
        return timestamps

    def process_audio(self) -> Optional[tuple[np.ndarray, bool]]:
        """
        Processes the audio buffer to find a segment for transcription.
        Returns: (audio_segment, has_overlap) or None
        """
        if len(self.audio_buffer) < VAD_CHUNK_SIZE:
            return None

        speech_timestamps = self._get_speech_timestamps()

        if not speech_timestamps:
            # No speech detected, keep a small tail of the buffer
            # keep_samples = (VAD_MIN_SILENCE_DURATION_MS * SAMPLING_RATE) // 1000  # Keep last 2000ms
            # if len(self.audio_buffer) > keep_samples:
            #     self.audio_buffer = self.audio_buffer[-keep_samples:]
            self.audio_buffer = np.array([], dtype=np.float32)  # Clear buffer if no speech
            return None

        # Get last speech endpoint
        last_speech_end = speech_timestamps[-1]['end']
        buffer_duration_ms = (len(self.audio_buffer) * 1000) // SAMPLING_RATE
        
        # Check if we've reached max duration
        max_duration_reached = buffer_duration_ms >= MAX_ASR_DURATION_MS
        
        # Manual silence detection for aggregation
        # Check if we have enough silence after the last speech segment
        buffer_end_samples = len(self.audio_buffer)
        silence_duration_samples = buffer_end_samples - last_speech_end
        silence_duration_ms = (silence_duration_samples * 1000) // SAMPLING_RATE
        
        # Use aggregation-specific silence duration for manual check
        silence_detected = silence_duration_ms >= ASR_MIN_SILENCE_DURATION_MS
        
        if silence_detected or max_duration_reached:
            if max_duration_reached and not silence_detected:
                # Split at max duration with overlap
                overlap_samples = (OVERLAP_DURATION_MS * SAMPLING_RATE) // 1000
                segment_to_transcribe = self.audio_buffer[:last_speech_end]
                # Keep overlap in buffer for next segment
                self.audio_buffer = self.audio_buffer[last_speech_end - overlap_samples:]
                
                segment_duration_ms = (len(segment_to_transcribe) * 1000) // SAMPLING_RATE
                print(f"✅ Max duration reached. Splitting with {OVERLAP_DURATION_MS}ms overlap. Duration: {segment_duration_ms}ms")
                return segment_to_transcribe, True  # True indicates overlap
            else:
                # Normal split at silence - transcribe everything up to last speech
                segment_to_transcribe = self.audio_buffer[:last_speech_end]
                # Keep remaining audio after speech for next processing
                self.audio_buffer = self.audio_buffer[last_speech_end:]
                
                segment_duration_ms = (len(segment_to_transcribe) * 1000) // SAMPLING_RATE
                print(f"✅ Segment ready for ASR (silence detected: {silence_duration_ms}ms). Duration: {segment_duration_ms}ms")
                return segment_to_transcribe, False  # False indicates no overlap

        return None

    async def send_transcription(self, text: str, is_final: bool = False):
        """Send transcription text to the client."""
        try:
            if is_final:
                # Send empty message to signal end of sentence
                await self.websocket.send_text("")
            else:
                # Send the transcribed text
                await self.websocket.send_text(text)
        except Exception as e:
            print(f"Error sending transcription: {e}")

async def transcribe_chunk(audio_np: np.ndarray, model: WhisperModel, audio_processor: AudioProcessor, has_overlap: bool):
    """
    Transcribes an audio segment using Whisper.
    Only uses word timestamps when there's overlap.
    """
    # print(f"🚀 Starting transcription... (overlap: {has_overlap})")
    
    if has_overlap and audio_processor.last_segment_words:
        # Use word timestamps for overlap handling
        segments, info = model.transcribe(
            audio_np,
            beam_size=5,
            word_timestamps=True,
            language="en"
        )
        
        # Collect all words with timestamps and confidence scores
        all_words = []
        for segment in segments:
            if not segment.words:
                continue
            for word in segment.words:
                all_words.append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
        
        # Calculate overlap boundary (2 seconds from start of current segment)
        overlap_boundary = OVERLAP_DURATION_MS / 1000.0
        
        # Find overlapping words
        previous_overlap_words = [w for w in audio_processor.last_segment_words if w['end'] > (audio_processor.last_segment_words[-1]['end'] - overlap_boundary)]
        current_overlap_words = [w for w in all_words if w['start'] < overlap_boundary]
        
        if previous_overlap_words and current_overlap_words:
            # print(f"📋 Overlap detected. Previous: {len(previous_overlap_words)} words, Current: {len(current_overlap_words)} words")
            
            # Merge the transcriptions
            merged_words = []
            
            # Add all words from previous segment EXCEPT the overlapping ones
            for word in audio_processor.last_segment_words:
                if word not in previous_overlap_words:
                    merged_words.append(word)
            
            # Process overlapping region - choose best words based on confidence
            used_current_indices = set()
            
            for prev_word in previous_overlap_words:
                best_match = None
                best_match_idx = -1
                
                # Find matching word in current segment
                for idx, curr_word in enumerate(current_overlap_words):
                    time_diff = abs(prev_word['start'] - curr_word['start'])
                    if time_diff < 0.1:  # Words at same position (within 100ms)
                        if best_match is None or curr_word['probability'] > best_match['probability']:
                            best_match = curr_word
                            best_match_idx = idx
                
                # Add the word with higher confidence
                if best_match and best_match['probability'] > prev_word['probability']:
                    merged_words.append(best_match)
                    used_current_indices.add(best_match_idx)
                    # print(f"   Replaced '{prev_word['word']}' (conf: {prev_word['probability']:.2f}) with '{best_match['word']}' (conf: {best_match['probability']:.2f})")
                else:
                    merged_words.append(prev_word)
            
            # Add any current overlap words that didn't match previous words
            for idx, curr_word in enumerate(current_overlap_words):
                if idx not in used_current_indices:
                    merged_words.append(curr_word)
                    # print(f"   Added new word '{curr_word['word']}' from current segment")
            
            # Add all remaining words from current segment (after overlap region)
            for word in all_words:
                if word not in current_overlap_words:
                    merged_words.append(word)
            
            # Sort by timestamp to ensure correct order
            merged_words.sort(key=lambda w: w['start'])
            
            # Build final text from merged words
            full_text = " ".join(word['word'] for word in merged_words).strip()
            
            # Update stored words to just the current segment for next overlap
            audio_processor.last_segment_words = all_words
            
            # print(f"📝 Merged transcription: {len(merged_words)} total words")
            
        else:
            # No overlap to process, use current words as-is
            full_text = " ".join(word['word'] for word in all_words).strip()
            audio_processor.last_segment_words = all_words
        
    else:
        # Simple transcription without word timestamps
        segments, _ = model.transcribe(
            audio_np,
            beam_size=5,
            word_timestamps=False,
            language="en"
        )
        
        full_text = "".join(segment.text for segment in segments).strip()
        
        # Clear last segment words since no overlap
        audio_processor.last_segment_words = []
    
    if full_text:
        # print("\n" + "="*50)
        # print(f"🎤 TRANSCRIPTION: {full_text}")
        # print("="*50 + "\n")
        
        # Send transcription to client
        await audio_processor.send_transcription(full_text)
        # Send empty message to signal end of sentence
        await audio_processor.send_transcription("", is_final=True)
    # else:
        # print("ℹ️ Transcription resulted in empty text.")

# --- FastAPI Server ---

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Load models when the server starts."""
    load_models()

@app.get("/")
async def get():
    """Serves the HTML page."""
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for audio streaming and transcription."""
    await websocket.accept()
    print("WebSocket connection established.")

    audio_processor = AudioProcessor(silero_model, silero_utils, websocket)

    if not whisper_model:
        print("❌ Whisper model not loaded. Cannot process audio.")
        await websocket.close(code=1001, reason="Whisper model not loaded")
        exit()

    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()
            
            # Add PCM data to processor
            audio_processor.add_chunk(data)
            
            # Check if we have a segment ready for transcription
            result = audio_processor.process_audio()

            if result is not None:
                transcription_segment, has_overlap = result
                # Run transcription in background task
                asyncio.create_task(transcribe_chunk(transcription_segment, whisper_model, audio_processor, has_overlap))

    except WebSocketDisconnect:
        print("WebSocket connection closed by client.")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)