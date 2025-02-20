import os
import uuid
import asyncio
import platform
import json
import logging
import numpy as np
import torch
import nltk

from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from TTS.api import TTS

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
# Download NLTK tokenizer
nltk.download('punkt')

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
# Load and Configure XTTS Model
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load voice registry from file
def load_voice_registry():
    global voice_registry
    if os.path.exists("voice_registry.json"):
        with open("voice_registry.json", "r") as f:
            voice_registry = json.load(f)

def save_voice_registry():
    with open("voice_registry.json", "w") as f:
        json.dump(voice_registry, f)

load_voice_registry()

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

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
# Helper Functions
# =============================================================================
def remove_punctuation(text):
    """Remove punctuation from text."""
    return ''.join(c for c in text if c.isalnum() or c.isspace())

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split text efficiently using NLTK sentence tokenization."""
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
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
    """Convert numpy waveform array to a pydub AudioSegment."""
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )

async def generate_audio_chunk(chunk, speaker_wav, language):
    """Generate TTS for a single chunk asynchronously."""
    wav_array = tts_model.tts(text=chunk, speaker_wav=speaker_wav, language=language)
    return wav_array_to_audio_segment(wav_array, sample_rate=24000)

# =============================================================================
# Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess reference audio for voice cloning."""
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
        save_voice_registry()

        logging.info(f"Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generate voice-cloned speech using the XTTS model."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    # Remove punctuation before processing
    request.text = remove_punctuation(request.text)

    text_chunks = chunk_text_by_sentences(request.text, max_tokens=400)
    logging.info(f"Text split into {len(text_chunks)} chunks.")

    # Process all chunks in parallel
    tasks = [generate_audio_chunk(chunk, speaker_wav, request.language) for chunk in text_chunks]
    audio_segments = await asyncio.gather(*tasks)

    final_audio = sum(audio_segments, AudioSegment.silent(duration=200))

    # Generate unique output filename
    output_path = f"uploads/{request.voice_id}_cloned.{request.output_format}"

    if request.output_format.lower() == "mp3":
        final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
        return FileResponse(output_path, media_type="audio/mpeg", filename="cloned.mp3")

    elif request.output_format.lower() == "wav":
        final_audio.export(output_path, format="wav")
        return FileResponse(output_path, media_type="audio/wav", filename="cloned.wav")

    elif request.output_format.lower() == "ulaw":
        wav_path = output_path.replace('.ulaw', '.wav')
        final_audio.export(wav_path, format='wav')

        # Convert to uLaw using FFmpeg
        ulaw_path = output_path
        os.system(f'ffmpeg -y -i {wav_path} -ar 8000 -ac 1 -f mulaw {ulaw_path}')

        return FileResponse(ulaw_path, media_type="audio/mulaw", filename="cloned.ulaw")

    else:
        raise HTTPException(status_code=400, detail="Invalid output format.")


# =============================================================================
# Run API
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
