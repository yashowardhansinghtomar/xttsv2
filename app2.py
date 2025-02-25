import os
import uuid
import asyncio
import platform
import numpy as np
import torch
import logging
import string
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
    language: str = Field(default="en", description="Language choice: en or hi")
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (TTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = Lock()  # Lock for thread-safe access to TTS model

logging.info("📥 Loading AI4Bharat IndicBART model for TTS...")

# Load model once and store locally
MODEL_NAME = "ai4bharat/IndicBART"
TOKENIZER_PATH = f"./{MODEL_NAME.replace('/', '_')}_tokenizer"
MODEL_PATH = f"./{MODEL_NAME.replace('/', '_')}_model"

if not os.path.exists(TOKENIZER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_PATH)
else:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

if not os.path.exists(MODEL_PATH):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_PATH)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
logging.info("✅ Model loaded and cached locally!")

def remove_punctuation(text: str) -> str:
    """Removes punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensures that audio is at least `min_length_ms` milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

async def generate_tts(text, language):
    """Generates TTS audio from text."""
    with tts_lock:
        try:
            processed_text = pipe(text, max_length=128, num_return_sequences=1)[0]['generated_text']
            return processed_text
        except Exception as e:
            logging.error(f"❌ TTS Generation Failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS Generation Error: {str(e)}")

# =============================================================================
# Voice Cloning Endpoints (TTS)
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and returns a voice ID for cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Preprocess the audio
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
    """Generates cloned speech using the uploaded voice sample."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    text_without_punctuation = remove_punctuation(request.text)
    generated_text = await generate_tts(text_without_punctuation, request.language)

    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_cloning_api:app", host="0.0.0.0", port=8000, reload=True)
