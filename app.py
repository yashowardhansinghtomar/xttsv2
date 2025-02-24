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
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

print("📥 Loading TTS model for voice cloning...")
model_manager = ModelManager()

# Load the TTS model and vocoder
model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
vocoder_path, vocoder_config_path, _ = model_manager.download_model("vocoder_models/en/ljspeech/waveglow")

synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, use_cuda=torch.cuda.is_available())

print("✅ TTS Model ready for voice cloning!")

# Load a tokenizer to split text into chunks based on token count.
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

# Add language code mapping
LANGUAGE_CODES = {
    "en": "english",
    "hi": "hindi",
}

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)  # Smooth transition
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split the input text into chunks based on sentence boundaries and token count."""
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

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert a numpy waveform array to a pydub AudioSegment."""
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=1
    )

def generate_tts(text, speaker_wav, language):
    """Handles calling the TTS model properly."""
    with tts_lock:  # Ensure thread-safe access to the TTS model
        language_code = LANGUAGE_CODES.get(language, "en")  # Default to English if language code is not found
        wav = synthesizer.tts(text, speaker_wav=speaker_wav, language=language_code)
        return wav

def remove_punctuation(text: str) -> str:
    """Remove all punctuation from the input text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize the audio to the target dBFS level, avoiding over-normalization."""
    current_dbfs = audio.dBFS
    if current_dbfs < target_dbfs:
        cha
