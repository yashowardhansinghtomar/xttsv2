import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from threading import Lock
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
    description="FastSpeech2-based API for cloning voices in Hindi and English.",
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
# Load & Cache the Model Locally
# =============================================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "tts_models/multilingual/multi-dataset/vits"  # Ensure correct model
MODEL_PATH = os.path.join(MODEL_DIR, "fastspeech2_model.pth")

print("📥 Loading or Downloading Multilingual VITS model...")

if not os.path.exists(MODEL_PATH):
    tts = TTS(MODEL_NAME, gpu=torch.cuda.is_available())
    tts.model.save(MODEL_PATH)
else:
    tts = TTS(MODEL_NAME, gpu=torch.cuda.is_available(), progress_bar=False)

print("✅ Model Loaded & Cached!")

# Lock to prevent concurrent model execution issues
model_lock = Lock()

# Store uploaded voices
voice_registry = {}

# =============================================================================
# Request Model
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", pattern="^(en|hi)$")  # ✅ Fixed Pydantic v2 issue
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        os.makedirs("uploads", exist_ok=True)

        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Convert to WAV format
        audio = AudioSegment.from_file(upload_path)
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
    """Generate voice cloned speech."""
    logging.info(f"Received request: {request}")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        # Generate speech using the model
        with model_lock:
            audio_output = tts.tts(
                text=request.text,
                speaker_wav=speaker_wav,
                language=request.language,
                speed=request.speed,
            )

        if isinstance(audio_output, tuple):
            wav, sample_rate = audio_output
        else:
            raise HTTPException(status_code=500, detail="Unexpected TTS output format")

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
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
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
