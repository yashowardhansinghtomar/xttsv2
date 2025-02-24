import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="Hindi & English Voice Cloning API",
    description="API for voice cloning using FastSpeech2 model, supporting Hindi & English input.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix Windows event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =============================================================================
# Initialize FastSpeech2 Model
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
model_lock = asyncio.Lock()

print("📥 Loading FastSpeech2 model...")

# Download and load the TTS model if not present locally
tts_model_path = "tts_models/en/ljspeech/fast_pitch"
tts = TTS(tts_model_path, gpu=torch.cuda.is_available())
print("✅ FastSpeech2 Model ready!")

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    text: str
    language: str = Field(default="en", regex="^(en|hi)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/generate_speech/")
async def generate_speech(request: GenerateClonedSpeechRequest):
    """Generate speech from input text."""
    logging.info(f"Processing text: {request.text} in {request.language} language")
    temp_output_files = []

    try:
        # Generate speech using FastSpeech2
        async with model_lock:
            wav = tts.tts(text=request.text)
            sample_rate = 22050
            wav = np.array(wav, dtype=np.float32)

        if len(wav) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")

        # Convert to audio format
        audio = AudioSegment(
            wav.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )

        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_speech_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        # Export based on output format
        if request.output_format.lower() == "mp3":
            audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            with open(output_path, "rb") as audio_file:
                return Response(audio_file.read(), media_type="audio/mpeg")

        elif request.output_format.lower() == "wav":
            audio.export(output_path, format="wav")
            with open(output_path, "rb") as wav_file:
                return Response(wav_file.read(), media_type="audio/wav")

        elif request.output_format.lower() == "ulaw":
            wav_path = output_path.replace(".ulaw", ".wav")
            audio.export(wav_path, format="wav")
            temp_output_files.append(wav_path)

            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", output_path],
                check=True,
            )
            with open(output_path, "rb") as ulaw_file:
                return Response(ulaw_file.read(), media_type="audio/mulaw")

        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

    finally:
        # Clean up temporary files
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# =============================================================================
# Run the FastAPI App
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
