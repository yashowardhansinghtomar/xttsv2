import os
import uuid
import torch
import logging
import subprocess
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.api import TTS
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="Hindi-English Voice Cloning API",
    description="API for voice cloning using FastSpeech2 supporting Hindi and English.",
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

# Model Setup
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
model_lock = Lock()

print("📥 Loading FastSpeech2 model...")
tts = TTS("tts_models/en/ljspeech/fast_pitch", gpu=torch.cuda.is_available())
print("✅ FastSpeech2 Model ready!")

# Request Models
class GenerateClonedSpeechRequest(BaseModel):
    text: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, ulaw")

# Voice Cloning Endpoint
@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate speech from text."""
    logging.info(f"Received request: {request}")
    
    try:
        with model_lock:
            audio_output = tts.tts(text=request.text, speed=request.speed)
            sample_rate = 22050  # FastSpeech2 default
            audio_data = np.array(audio_output, dtype=np.float32)

        if len(audio_data) == 0:
            raise HTTPException(status_code=500, detail="Generated empty audio")

        audio = AudioSegment(
            audio_data.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )

        output_path = f"output_{uuid.uuid4()}.{request.output_format}"

        if request.output_format.lower() == "mp3":
            audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
        elif request.output_format.lower() == "wav":
            audio.export(output_path, format="wav")
        elif request.output_format.lower() == "ulaw":
            temp_wav = output_path.replace(".ulaw", ".wav")
            audio.export(temp_wav, format="wav")
            subprocess.run(["ffmpeg", "-y", "-i", temp_wav, "-ar", "8000", "-ac", "1", "-f", "mulaw", output_path], check=True)
            os.remove(temp_wav)
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

        with open(output_path, "rb") as audio_file:
            response = Response(audio_file.read(), media_type=f"audio/{request.output_format}")
        os.remove(output_path)
        return response
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
