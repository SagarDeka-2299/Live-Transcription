
# main.py
# Backend for the Live Transcription App using FastAPI and Faster-Whisper

# Installation:
# 1. Install Python 3.9+
# 2. Install PyTorch with CUDA support:
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 3. Install other dependencies:
#    pip install "faster-whisper" fastapi uvicorn websockets python-multipart numpy aiofiles "pyannote.audio"

import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# --- Configuration ---
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- IMPORTANT ---
# You need a Hugging Face token to use the pyannote.audio VAD model.
# 1. Visit https://huggingface.co/pyannote/voice-activity-detection and accept the user agreement.
# 2. Visit https://huggingface.co/settings/tokens to create an access token.
# 3. Paste your token here.
HUGGING_FACE_TOKEN = os.getenv("HF_KEY") # Replace with your actual token
    
# Whisper Model Configuration
MODEL_SIZE = "small"
DEVICE_TYPE = "cpu"
COMPUTE_TYPE = "float32"
if torch.cuda.is_available():
    MODEL_SIZE = "large-v3"
    DEVICE_TYPE = "cuda"
    COMPUTE_TYPE = "float16"
# VAD & Audio Configuration
SAMPLE_RATE = 16000
# We will process audio in chunks of this size. 2 seconds is a good balance of latency and VAD accuracy.
VAD_CHUNK_DURATION_S = 2.0
VAD_CHUNK_SIZE_BYTES = int(SAMPLE_RATE * VAD_CHUNK_DURATION_S * 2) # 2 bytes per sample (int16)

# Silence-based segmentation configuration
MIN_SILENCE_DURATION = 1.0  # Minimum silence duration in seconds to create a segment break
MAX_SILENCE_DURATION = 1.5  # Maximum silence duration to wait before forced segmentation
SILENCE_THRESHOLD = 0.01    # RMS threshold below which audio is considered silence
MIN_SEGMENT_DURATION = 0.5  # Minimum segment duration in seconds


# --- Initialization ---
app = FastAPI()

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)


# Load Whisper Model
logger.info(f"Loading Whisper model '{MODEL_SIZE}'...")
try:
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    exit()

# Load pyannote.audio VAD Pipeline
logger.info("Loading pyannote.audio VAD pipeline...")
try:
    vad_pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=HUGGING_FACE_TOKEN
    )
    vad_pipeline.to(torch.device(DEVICE_TYPE))
    logger.info(f"pyannote.audio VAD pipeline loaded successfully to '{DEVICE_TYPE}'.")
except Exception as e:
    logger.error(f"Error loading pyannote.audio pipeline: {e}")
    vad_pipeline = None
    exit()


# Mount the 'static' directory to serve CSS and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")


def find_silence_breaks(audio_np: np.ndarray, sample_rate: int) -> list:
    """
    Find silence breaks in audio based on RMS energy threshold.
    Returns list of break points (in samples) where silence exceeds the threshold duration.
    """
    # Calculate RMS energy in small windows
    window_size = int(0.1 * sample_rate)  # 100ms windows
    hop_size = int(0.05 * sample_rate)    # 50ms hop
    
    rms_values = []
    positions = []
    
    for i in range(0, len(audio_np) - window_size, hop_size):
        window = audio_np[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
        positions.append(i)
    
    # Find silence regions
    silence_mask = np.array(rms_values) < SILENCE_THRESHOLD
    
    # Find continuous silence regions
    silence_regions = []
    start_silence = None
    
    for i, is_silent in enumerate(silence_mask):
        if is_silent and start_silence is None:
            start_silence = i
        elif not is_silent and start_silence is not None:
            end_silence = i
            silence_duration = (positions[end_silence] - positions[start_silence]) / sample_rate
            if silence_duration >= MIN_SILENCE_DURATION:
                # Use the middle of the silence region as break point
                break_point = positions[start_silence] + (positions[end_silence] - positions[start_silence]) // 2
                silence_regions.append((break_point, silence_duration))
            start_silence = None
    
    # Handle case where audio ends with silence
    if start_silence is not None:
        end_silence = len(silence_mask)
        silence_duration = (len(audio_np) - positions[start_silence]) / sample_rate
        if silence_duration >= MIN_SILENCE_DURATION:
            break_point = positions[start_silence] + (len(audio_np) - positions[start_silence]) // 2
            silence_regions.append((break_point, silence_duration))
    
    return [region[0] for region in silence_regions]


def segment_audio_by_silence(audio_np: np.ndarray, sample_rate: int) -> list:
    """
    Segment audio based on silence breaks and VAD results.
    Returns list of audio segments as numpy arrays.
    """
    # First, check if there's any speech in the audio using VAD
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
    input_data = {"waveform": audio_tensor, "sample_rate": sample_rate}
    
    try:
        vad_result = vad_pipeline(input_data)
        has_speech = len(list(vad_result.itersegments())) > 0
        
        if not has_speech:
            return []  # No speech detected, return empty list
    except Exception as e:
        logger.error(f"VAD error: {e}")
        return []
    
    # Find silence-based break points
    break_points = find_silence_breaks(audio_np, sample_rate)
    
    # If no significant silence breaks found, return the entire audio if it contains speech
    if not break_points:
        min_samples = int(MIN_SEGMENT_DURATION * sample_rate)
        if len(audio_np) >= min_samples:
            return [audio_np]
        else:
            return []
    
    # Create segments based on break points
    segments = []
    start_idx = 0
    
    for break_point in break_points:
        if break_point > start_idx:
            segment = audio_np[start_idx:break_point]
            # Only include segments that meet minimum duration and contain speech
            if len(segment) >= int(MIN_SEGMENT_DURATION * sample_rate):
                # Quick check if segment has speech using simple energy threshold
                segment_rms = np.sqrt(np.mean(segment ** 2))
                if segment_rms > SILENCE_THRESHOLD:
                    segments.append(segment)
            start_idx = break_point
    
    # Add the final segment if it exists
    if start_idx < len(audio_np):
        final_segment = audio_np[start_idx:]
        if len(final_segment) >= int(MIN_SEGMENT_DURATION * sample_rate):
            segment_rms = np.sqrt(np.mean(final_segment ** 2))
            if segment_rms > SILENCE_THRESHOLD:
                segments.append(final_segment)
    
    return segments


# --- WebSocket Handling ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected.")
    
    if not vad_pipeline:
        await websocket.close(code=1011, reason="VAD pipeline not available.")
        return

    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # Process audio in fixed chunks
            while len(audio_buffer) >= VAD_CHUNK_SIZE_BYTES:
                # Extract a chunk for processing
                chunk_to_process = audio_buffer[:VAD_CHUNK_SIZE_BYTES]
                # Remove the processed chunk from the buffer
                audio_buffer = audio_buffer[VAD_CHUNK_SIZE_BYTES:]
                
                # Convert byte chunk to numpy array
                audio_np = np.frombuffer(chunk_to_process, dtype=np.int16).astype(np.float32) / 32768.0

                # Create a task to process this chunk in the background
                asyncio.create_task(process_and_transcribe(websocket, audio_np))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
        await websocket.close(code=1011, reason=f"An error occurred: {e}")


async def process_and_transcribe(websocket: WebSocket, audio_np: np.ndarray):
    """
    Segments audio based on silence breaks and transcribes each segment.
    """
    try:
        # Segment the audio based on silence breaks
        segments = segment_audio_by_silence(audio_np, SAMPLE_RATE)
        
        # Transcribe each segment
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)} (duration: {len(segment)/SAMPLE_RATE:.2f}s)")
            await transcribe_and_send(websocket, segment)

    except Exception as e:
        logger.error(f"Error in processing/transcription task: {e}")


async def transcribe_and_send(websocket: WebSocket, audio_np: np.ndarray):
    """
    Transcribes an audio segment using Whisper and sends the words to the client.
    """
    try:
        segments, _ = whisper_model.transcribe(audio_np, beam_size=5, word_timestamps=True, language="en")
        
        full_sentence = []
        for segment in segments:
            for word in segment.words:
                await websocket.send_text(word.word + " ")
                full_sentence.append(word.word)
                await asyncio.sleep(0.01) # Prevents blocking and gives a "real-time" feel
        
        # Send a blank message to signal the end of the sentence on the frontend
        if full_sentence: 
             await websocket.send_text("") 
        logger.info(f"Transcribed sentence: {''.join(full_sentence).strip()}")

    except Exception as e:
        logger.error(f"Transcription Error: {e}")


# --- Serve the Frontend ---
@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = "index.html"
    if not os.path.exists(index_path):
        return HTMLResponse("<html><body><h1>Error</h1><p>index.html not found. Please create it.</p></body></html>", status_code=404)
    return FileResponse(index_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)