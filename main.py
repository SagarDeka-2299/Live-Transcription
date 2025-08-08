import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from STT_Model import load_stt_model
stt=load_stt_model()

from Speech_Detector import load_speech_detect_model
detect_speech = load_speech_detect_model()

from Timebomb import ResettableTimer

waveform_np=np.array([], dtype=np.float32)  # Buffer for audio data


def get_transcription_handler(socket:WebSocket):
    async def transcribe()->None:
        global waveform_np

        """Called when timeout happens - transcribe accumulated audio"""
        if len(waveform_np) > 0:
            txt = stt(waveform_np)
            print(f"Transcription: {txt}")
            print("sending to client...")
            await socket.send_text(txt)
            waveform_np = np.array([], dtype=np.float32)  # Reset the buffer
        else:
            print("\n⏰ Timeout - no audio to transcribe")
    return transcribe



# --- FastAPI Server ---

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get():
    """Serves the HTML page."""
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for audio streaming and transcription."""
    global waveform_np
    await websocket.accept()
    print("WebSocket connection established.")

    try:
        timer = ResettableTimer(0.7, get_transcription_handler(websocket))
        while True:
            # Receive binary audio data 512 bytes chunk size
            audio_chunk_bytes = await websocket.receive_bytes()
            audio_chunk_np_i16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            audio_chunk_np_f32 = audio_chunk_np_i16.astype(np.float32) / 32768.0
            # Accumulate waveform
            waveform_np = np.concatenate([waveform_np, audio_chunk_np_f32])
            
            # VAD detection
            is_speech = detect_speech(audio_chunk_np_f32)
            if is_speech:
                # print("SPEECH DETECTED!", end='\r')
                timer.reset()
            # else:
            #     print("Silence...", end='\r')

    except WebSocketDisconnect:
        print("WebSocket connection closed by client.")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)