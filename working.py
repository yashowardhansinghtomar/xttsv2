import os
import uuid
import asyncio
import platform
import logging
import torch
import numpy as np
import string
from io import BytesIO
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer
from TTS.api import TTS


# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI App & CORS
app = FastAPI(title="Optimized Voice Cloning API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

# Windows event loop fix
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Request Model
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3")

# Voice Cloning Setup
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = Lock()

logging.info("📥 Loading XTTS model...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts_model = torch.jit.script(tts_model)  # Optimize model inference
logging.info("✅ XTTS Model loaded!")

# Tokenizer for efficient text chunking
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_text(text: str, max_tokens: int = 400) -> list:
    """Efficiently splits text into chunks based on token count."""
    tokens = tokenizer.tokenize(text)
    token_chunks, chunk = [], []
    for token in tokens:
        if len(chunk) + 1 > max_tokens:
            token_chunks.append(tokenizer.convert_tokens_to_string(chunk))
            chunk = [token]
        else:
            chunk.append(token)
    if chunk:
        token_chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return token_chunks

def generate_tts(text, speaker_wav, language):
    """Thread-safe XTTS inference."""
    with tts_lock:
        return tts_model.tts(text=text, speaker_wav=speaker_wav, language=language)

def wav_array_to_audio(wav_array, sample_rate: int) -> AudioSegment:
    """Converts numpy waveform array to pydub AudioSegment."""
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads and preprocesses reference audio."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        audio = AudioSegment.from_file(upload_path).set_channels(1).set_frame_rate(24000)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generates cloned speech efficiently."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    text_chunks = chunk_text(request.text, max_tokens=400)
    
    with ThreadPoolExecutor(max_workers=min(4, cpu_count())) as executor:
        futures = [executor.submit(generate_tts, chunk, speaker_wav, request.language) for chunk in text_chunks]
        final_audio = sum((wav_array_to_audio(future.result(), 24000) for future in as_completed(futures)), AudioSegment.empty())
    
    buffer = BytesIO()
    final_audio.export(buffer, format=request.output_format.lower())
    buffer.seek(0)
    return Response(buffer.getvalue(), media_type=f"audio/{request.output_format.lower()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
