import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer

from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Hindi Voice Cloning API",
    description="API for voice cloning using Tacotron2 + HiFi-GAN.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "नमस्ते, यह एक परीक्षण है।"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
model_lock = Lock()

print("📥 Loading Tacotron2 + HiFi-GAN model for Hindi...")

# Initialize Tacotron2 + HiFi-GAN model
tts = TTS("tts_models/hi/coqui/tacotron2-DDC", gpu=torch.cuda.is_available())

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts.to(device)

print("✅ Tacotron2 + HiFi-GAN ready for voice cloning!")

# Load tokenizer for text chunking
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split the input text into chunks based on sentence boundaries and token count."""
    sentences = text.split('।')  # Hindi sentence separator
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

def generate_speech(text: str) -> np.ndarray:
    """Generate speech using Tacotron2 + HiFi-GAN."""
    with model_lock:
        wav = tts.tts(text=text)
        wav = np.array(wav, dtype=np.float32)
        if len(wav) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")

        sample_rate = tts.synthesizer.output_sample_rate or 24000
        return wav, sample_rate

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize the audio to the target dBFS level, avoiding over-normalization."""
    current_dbfs = audio.dBFS
    if current_dbfs < target_dbfs:
        change_in_dbfs = target_dbfs - current_dbfs
        return audio.apply_gain(change_in_dbfs)
    return audio

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Process audio
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")
        voice_registry[voice_id] = {
            "preprocessed_file": preprocessed_path
        }
        logging.info(f"Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate voice cloned speech."""
    logging.info(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        logging.error("Voice ID not found")
        raise HTTPException(status_code=404, detail="Voice ID not found")

    temp_output_files = []

    try:
        # Process text and generate speech
        text_chunks = chunk_text_by_sentences(request.text, max_tokens=400)
        logging.info(f"Text split into {len(text_chunks)} chunks.")

        final_audio = AudioSegment.empty()

        # Process chunks in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_speech, chunk) for chunk in text_chunks]

            for future in as_completed(futures):
                wav_array, sample_rate = future.result()
                chunk_audio = AudioSegment(
                    wav_array.tobytes(),
                    sample_width=2,
                    frame_rate=sample_rate,
                    channels=1
                )
                chunk_audio = normalize_audio(chunk_audio)

                if final_audio:
                    final_audio = final_audio.append(chunk_audio, crossfade=50)
                else:
                    final_audio = chunk_audio

        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        # Export the generated audio in the requested format
        final_audio.export(output_path, format=request.output_format)

        with open(output_path, "rb") as audio_file:
            return Response(audio_file.read(), media_type=f"audio/{request.output_format}")
    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
