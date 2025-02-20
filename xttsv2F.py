import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
import re
import multiprocessing

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning (XTTS) with MP3, WAV, and ULAW output formats.",
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
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("voices", exist_ok=True)  # Base directory for storing voice data
voice_registry = {}

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    tts_model = torch.nn.DataParallel(tts_model)

print("✅ XTTS Model ready for voice cloning!")

# Load a tokenizer to split text into chunks based on token count.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)  # Smooth transition
    return audio

def clean_text(text: str) -> str:
    """Remove punctuation marks from the text."""
    return re.sub(r'[^\w\s]', '', text)

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split the input text into chunks based on sentence boundaries and token count."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Improved regex-based sentence splitting
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
    """Handles calling the TTS model properly, whether DataParallel is used or not."""
    model = tts_model.module if isinstance(tts_model, torch.nn.DataParallel) else tts_model
    return model.tts(text=text, speaker_wav=speaker_wav, language=language)

def process_chunk(args):
    """Generate TTS for a text chunk in parallel."""
    chunk, speaker_wav, language = args
    wav_array = generate_tts(chunk, speaker_wav, language)
    return wav_array_to_audio_segment(wav_array, sample_rate=24000)

# =============================================================================
# Voice Cloning Endpoints (XTTS)
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess reference audio for voice cloning. Returns a unique voice_id."""
    try:
        voice_id = str(uuid.uuid4())
        voice_dir = os.path.join("voices", voice_id)
        os.makedirs(voice_dir, exist_ok=True)

        upload_path = os.path.join(voice_dir, file.filename)
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)

        preprocessed_path = os.path.join(voice_dir, "preprocessed.wav")
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"voice_dir": voice_dir, "preprocessed_file": preprocessed_path}
        logging.info(f"Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate voice cloned speech using the XTTS model."""
    logging.info(f"Received request: {request}")

    if request.voice_id not in voice_registry:
        logging.error("Voice ID not found")
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    voice_dir = voice_registry[request.voice_id]["voice_dir"]

    try:
        cleaned_text = clean_text(request.text)  # Remove punctuation
        text_chunks = chunk_text_by_sentences(cleaned_text, max_tokens=400)
        logging.info(f"Text split into {len(text_chunks)} chunks.")

        with multiprocessing.Pool(processes=4) as pool:
            chunk_audio_list = pool.map(process_chunk, [(chunk, speaker_wav, request.language) for chunk in text_chunks])

        final_audio = sum(chunk_audio_list, AudioSegment.silent(duration=200))

        output_path = os.path.join(voice_dir, f"cloned_speech.{request.output_format}")

        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
        elif request.output_format.lower() == "ulaw":
            final_audio.export(output_path, format="wav")
            subprocess.run(['ffmpeg', '-y', '-i', output_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path], check=True)
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

        with open(output_path, "rb") as audio_file:
            return Response(audio_file.read(), media_type=f"audio/{request.output_format}")

    except Exception as e:
        logging.error(f"Error generating speech: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating speech.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
