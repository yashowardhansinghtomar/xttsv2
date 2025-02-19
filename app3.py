import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --- SQLAlchemy Setup for Shared Voice Registrations ---
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./voice_registry.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VoiceRegistration(Base):
    __tablename__ = "voice_registrations"
    voice_id = Column(String, primary_key=True, index=True)
    preprocessed_file = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

# --- XTTS Model Setup ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

# ✅ Fix for PyTorch 2.6+ Safe Loading
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="Multi GPU Voice Cloning API with optimized quality and speed control.",
    version="2.0.0"
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
    text: str
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)

# --- Multi-GPU Setup ---
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
tts_models = []
model_locks = []

print(f"📥 Loading XTTS model for voice cloning on {num_gpus} device(s)...")
for i in range(num_gpus):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    # We instantiate the model without using gpu=True now since we will move it manually.
    model_instance = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    model_instance.to(device)
    tts_models.append(model_instance)
    model_locks.append(asyncio.Lock())
print("✅ Multi GPU TTS Model pool ready for voice cloning!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def chunk_text(text: str, max_length: int = 150) -> list:
    return textwrap.wrap(text, width=max_length) if len(text) > max_length else [text]

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

async def process_chunk(chunk: str, speaker_wav: str, language: str, speed: float, model: TTS, lock: asyncio.Lock) -> AudioSegment:
    loop = asyncio.get_running_loop()
    async with lock:
        # Run the blocking TTS call in an executor
        wav_array = await loop.run_in_executor(None, model.tts, chunk, speaker_wav, language, speed)
    return wav_array_to_audio_segment(wav_array, 24000)

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Preprocess: convert to WAV with 24000Hz and mono, ensure minimum length.
        audio = AudioSegment.from_file(upload_path).set_frame_rate(24000).set_channels(1)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Save registration to the shared database.
        with SessionLocal() as db:
            registration = VoiceRegistration(voice_id=voice_id, preprocessed_file=preprocessed_path)
            db.add(registration)
            db.commit()
            print(f"✅ Registered voice ID: {voice_id} with file: {preprocessed_path}")

        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    print(f"🎤 Generating speech for voice_id: {request.voice_id}")

    # Retrieve the voice registration from the shared database.
    with SessionLocal() as db:
        registration = db.query(VoiceRegistration).filter(VoiceRegistration.voice_id == request.voice_id).first()
        if registration is None:
            print(f"❌ Voice ID {request.voice_id} not found in database.")
            raise HTTPException(status_code=404, detail="Voice ID not found")
        speaker_wav = registration.preprocessed_file
        print(f"✅ Found voice registration: {registration.voice_id} -> {speaker_wav}")

    # Split text into chunks and schedule parallel inference.
    text_chunks = chunk_text(request.text, max_length=150)
    print(f"📖 Text split into {len(text_chunks)} chunks.")

    tasks = []
    # Distribute chunks in round-robin fashion across available models.
    for idx, chunk in enumerate(text_chunks):
        model_index = idx % len(tts_models)
        tasks.append(
            process_chunk(chunk, speaker_wav, request.language, request.speed,
                          tts_models[model_index], model_locks[model_index])
        )
    chunk_audios = await asyncio.gather(*tasks)

    # Concatenate all audio chunks.
    final_audio = AudioSegment.empty()
    for segment in chunk_audios:
        final_audio += segment

    # Prepare a temporary output file.
    unique_hash = abs(hash(request.text + str(asyncio.get_running_loop().time())))
    output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"

    # Export final audio in the requested format.
    if request.output_format.lower() == "mp3":
        final_audio.export(output_path, format="mp3")
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(output_path)
        return Response(audio_bytes, media_type="audio/mpeg")
    elif request.output_format.lower() == "wav":
        final_audio.export(output_path, format="wav")
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(output_path)
        return Response(audio_bytes, media_type="audio/wav")
    elif request.output_format.lower() == "ulaw":
        wav_path = output_path.replace('.ulaw', '.wav')
        final_audio.export(wav_path, format="wav")
        subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path], check=True)
        with open(output_path, 'rb') as f:
            audio_bytes = f.read()
        os.remove(wav_path)
        os.remove(output_path)
        return Response(audio_bytes, media_type="audio/mulaw", headers={"X-Sample-Rate": "8000"})
    else:
        raise HTTPException(status_code=400, detail="Invalid output format specified.")

# =============================================================================
# Run FastAPI Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app3:app", host="0.0.0.0", port=8000, reload=True)
