import os
import uuid
import asyncio
from concurrent.futures import as_completed

import platform
import logging
import numpy as np
import torch
import string
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Hindi & English Voice Cloning API",
    description="FastAPI-based TTS API using AI4Bharat VITS & HiFi-GAN, supporting MP3, WAV, and ULAW output formats.",
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
    language: str = Field(default="en", description="Language code (e.g., en for English, hi for Hindi)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (TTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = asyncio.Lock()  # Lock for thread-safe access to TTS model

logging.info("📥 Loading AI4Bharat VITS model for voice cloning...")

# Function to download and load the AI4Bharat model
def load_model():
    try:
        model_name = "ai4bharat/vits_rasa_13"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        logging.info("✅ AI4Bharat VITS Model ready for voice cloning!")
        return model, processor
    except Exception as e:
        logging.error(f"❌ Error initializing AI4Bharat TTS model: {e}")
        return None, None

# Load the model and processor
model, processor = load_model()

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure the audio has a minimum length by adding silence if needed."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)  # Smooth transition
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split text into smaller chunks to fit AI4Bharat VITS model constraints."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = sentence.split()
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

async def generate_tts(text, speaker_wav, language):
    """Generate speech using AI4Bharat VITS and HiFi-GAN vocoder."""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")

    async with tts_lock:
        inputs = processor(text=text, return_tensors="pt", language=language)
        speaker_embedding = processor(audio=speaker_wav, return_tensors="pt").input_values
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)
        return speech

def remove_punctuation(text: str) -> str:
    """Remove punctuation from input text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize audio volume to a target dBFS level."""
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
    """Upload and preprocess speaker audio for voice cloning."""
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
    """Generate cloned speech using AI4Bharat VITS."""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize. Try restarting the server.")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    text_without_punctuation = remove_punctuation(request.text)
    text_chunks = chunk_text_by_sentences(text_without_punctuation, max_tokens=400)

    final_audio = AudioSegment.empty()

    # Run tasks asynchronously
    for chunk in text_chunks:
        wav_array = await generate_tts(chunk, speaker_wav, request.language)  # Corrected

        # Convert numpy array to AudioSegment
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
