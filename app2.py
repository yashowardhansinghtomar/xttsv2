import os
import uuid
import asyncio
import numpy as np
import torch
import logging
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(title="Voice Cloning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],)

# Ensure 'uploads' directory exists
os.makedirs("uploads", exist_ok=True)

# Load AI4Bharat TTS Model
tts_pipeline = pipeline("text-to-speech", model="ai4bharat/vits_rasa_13")

# Request Model
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", description="Language choice: 'en' or 'hi'")
    output_format: str = Field(default="mp3", description="Output format: 'mp3', 'wav', or 'ulaw'")

# Voice Registry
voice_registry = {}

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and returns a voice ID for cloning."""
    voice_id = str(uuid.uuid4())
    upload_path = f"uploads/{voice_id}_{file.filename}"
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    audio = AudioSegment.from_file(upload_path)
    preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
    audio.export(preprocessed_path, format="wav")

    voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
    return {"voice_id": voice_id}

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates cloned speech from text input."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    
    if request.language not in ["en", "hi"]:
        raise HTTPException(status_code=400, detail="Invalid language. Choose 'en' or 'hi'")
    
    try:
        speech = tts_pipeline(request.text)
        audio_data = np.array(speech["audio"]) * 32767
        audio_segment = AudioSegment(audio_data.astype(np.int16).tobytes(), sample_width=2, frame_rate=22050, channels=1)
        
        output_path = f"output_{request.voice_id}.{request.output_format}"
        audio_segment.export(output_path, format=request.output_format)
        
        with open(output_path, "rb") as f:
            return Response(f.read(), media_type=f"audio/{request.output_format}")
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Generation Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
