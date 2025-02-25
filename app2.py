import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer

# --- Safe globals for TTS model deserialization ---
from parlertts.utils.manage import ModelManager  # Change this import to Parler
from parlertts.utils.synthesizer import Synthesizer  # Change this import to Parler

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
    language: str = "en"  # Default to English
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (TTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = Lock()  # Lock for thread-safe access to TTS model

logging.info("📥 Loading TTS model for voice cloning...")
model_manager = ModelManager()

try:
    model_path, config_path, model_item = model_manager.download_model("parler_models/en/your_model_name")  # Change this line to Parler model
    vocoder_path, vocoder_config_path, vocoder_item = model_manager.download_model("vocoder_models/en/your_vocoder_name")  # Change this line to Parler vocoder

    # Debugging: Check if paths are correctly set
    logging.info(f"Model path: {model_path}")
    logging.info(f"Config path: {config_path}")
    logging.info(f"Vocoder path: {vocoder_path}")
    logging.info(f"Vocoder config path: {vocoder_config_path}")

    if not model_path or not vocoder_path:
        raise ValueError("❌ Model or vocoder path is missing!")

    # Initialize the synthesizer
    synthesizer = Synthesizer(
        model_path=model_path,
        config_path=config_path,
        vocoder_path=vocoder_path,
        vocoder_config_path=vocoder_config_path,
        use_cuda=torch.cuda.is_available()
    )

    logging.info("✅ TTS Model ready for voice cloning!")

except Exception as e:
    logging.error(f"❌ Error initializing TTS model: {e}")
    synthesizer = None

# Load a tokenizer to split text into chunks based on token count.
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

LANGUAGE_CODES = {
    "en": "english",
    "hi": "hindi",
}

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)  # Smooth transition
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_length + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_tts(text, speaker_wav, language):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")

    with tts_lock:
        language_code = LANGUAGE_CODES.get(language, "english")
        return synthesizer.tts(text, speaker_wav=speaker_wav, language=language_code)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    current_dbfs = audio.dBFS
    if current_dbfs < target_dbfs:
        change_in_dbfs = target_dbfs - current_dbfs
        return audio.apply_gain(change_in_dbfs)
    return audio

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
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    text_without_punctuation = remove_punctuation(request.text)
    text_chunks = chunk_text_by_sentences(text_without_punctuation, max_tokens=400)

    final_audio = AudioSegment.empty()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_tts, chunk, speaker_wav, request.language) for chunk in text_chunks]

        for future in as_completed(futures):
            wav_array = future.result()
            chunk_audio = AudioSegment(
                data=(np.array(wav_array) * 32767).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=22050,
                channels=1
            )
            final_audio += chunk_audio

    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    final_audio.export(output_path, format=request.output_format)

    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"audio/{request.output_format}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
