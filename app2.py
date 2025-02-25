import os
import uuid
import asyncio
import logging
import numpy as np
import torch
import string
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Import FastSpeech2_MFA & HiFi-GAN from Coqui TTS
from TTS.api import TTS  

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI
app = FastAPI(title="Hindi & English Voice Cloning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global Variables
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = asyncio.Lock()  # Async Lock for thread safety

# =============================================================================
# Load TTS Models
# =============================================================================
def load_tts_model():
    """Loads FastSpeech2_MFA & HiFi-GAN model from Coqui TTS."""
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/fastspeech2")  # Multilingual model
        logging.info("✅ Loaded FastSpeech2_MFA with HiFi-GAN!")
        return tts
    except Exception as e:
        logging.error(f"❌ Error initializing TTS model: {e}")
        return None

# Load model
tts_model = load_tts_model()

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", description="Language: 'en' or 'hi' (Hindi)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensures audio is at least a minimum length."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def remove_punctuation(text: str) -> str:
    """Removes punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

# =============================================================================
# Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads and processes an audio file for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Convert and preprocess audio
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Store voice data
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"✅ Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generates cloned speech from text using FastSpeech2_MFA and HiFi-GAN."""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize.")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    text_without_punctuation = remove_punctuation(request.text)

    # Generate speech
    async with tts_lock:
        output_wav = tts_model.tts(text=text_without_punctuation, speaker_wav=speaker_wav, language=request.language)

    # Convert to desired format
    final_audio = AudioSegment(
        data=(np.array(output_wav) * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=22050,
        channels=1
    )

    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    final_audio.export(output_path, format=request.output_format)

    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"audio/{request.output_format}")

# =============================================================================
# Run FastAPI Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
