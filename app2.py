import os
import uuid
import asyncio
import platform
import numpy as np
import torch
import logging
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from espnet2.bin.tts_inference import Text2Speech

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning with MP3, WAV, and ULAW output formats.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (TTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

logging.info("📥 Loading espnet/fastspeech2_conformer_hifigan model...")

def load_model():
    try:
        model = Text2Speech.from_pretrained("espnet/fastspeech2_conformer_hifigan")
        logging.info("✅ TTS Model ready for voice cloning!")
        return model
    except Exception as e:
        logging.error(f"❌ Error initializing TTS model: {e}")
        return None

# Load the model
tts_model = load_model()

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def generate_tts(text, speaker_wav):
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")
    wav = tts_model(text)["wav"].numpy()
    return wav

# =============================================================================
# Voice Cloning Endpoints (TTS)
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    wav_array = generate_tts(request.text, speaker_wav)
    
    audio = AudioSegment(
        data=(np.array(wav_array) * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=22050,
        channels=1
    )
    
    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    audio.export(output_path, format=request.output_format)
    
    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"audio/{request.output_format}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
