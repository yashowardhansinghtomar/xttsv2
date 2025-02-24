import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
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
    description="API for voice cloning using FastSpeech2 with Hindi and English support.",
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

# Initialize Model and Cache Directory
MODEL_NAME = "tts_models/en/ljspeech/fast_pitch"
voice_registry = {}
model_lock = Lock()

def get_tts_model():
    logging.info(f"Loading model: {MODEL_NAME}")
    tts = TTS(MODEL_NAME)
    tts.to("cuda" if torch.cuda.is_available() else "cpu")
    return tts

# Load TTS Model
tts = get_tts_model()
logging.info("✅ FastPitch Model ready!")

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", pattern="^(en|hi)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        voice_id = str(uuid.uuid4())
        upload_path = os.path.join("uploads", f"{voice_id}_{file.filename}")
        os.makedirs("uploads", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        audio = AudioSegment.from_file(upload_path)
        preprocessed_path = os.path.join("uploads", f"{voice_id}_preprocessed.wav")
        audio.export(preprocessed_path, format="wav")
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    logging.info(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    output_path = None  # ✅ Initialize output_path before the try block

    try:
        with model_lock:
            wav = tts.tts(
                text=request.text, speaker_wav=speaker_wav, language=request.language, speed=request.speed
            )
        
        audio = AudioSegment(
            np.array(wav, dtype=np.float32).tobytes(), sample_width=2, frame_rate=22050, channels=1
        )
        
        output_path = f"temp_output.{request.output_format}"  # ✅ Assign value here
        
        if request.output_format == "mp3":
            audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            return Response(open(output_path, "rb").read(), media_type="audio/mpeg")
        elif request.output_format == "wav":
            audio.export(output_path, format="wav")
            return Response(open(output_path, "rb").read(), media_type="audio/wav")
        elif request.output_format == "ulaw":
            wav_path = output_path.replace(".ulaw", ".wav")
            audio.export(wav_path, format="wav")
            subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", output_path], check=True)
            return Response(open(output_path, "rb").read(), media_type="audio/mulaw")
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

    finally:
        if output_path and os.path.exists(output_path):  # ✅ Check if output_path is assigned
            os.remove(output_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
