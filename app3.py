import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap
from io import BytesIO
import aiofiles

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --- Safe globals for XTTS model deserialization ---
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
    description="API for voice cloning (XTTS) with optimized quality and speed control.",
    version="2.0.0"
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
    text: str
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)  # Allow users to set speed
    output_format: str = Field(default="mp3", description="Format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

# ---------------------------------------------------------------------
# Load a pool of TTS models on available GPUs (or CPU if no GPU found)
# ---------------------------------------------------------------------
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
tts_models = []
model_locks = []  # One lock per model to ensure single-threaded use per device

print(f"📥 Loading XTTS model for voice cloning on {num_gpus} device(s)...")
for i in range(num_gpus):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    model_instance = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    model_instance.to(device)
    tts_models.append(model_instance)
    model_locks.append(asyncio.Lock())
print("✅ XTTS Model pool ready for voice cloning!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def chunk_text(text: str, max_length: int = 150) -> list:
    """Split long text into smaller chunks while maintaining word integrity."""
    return textwrap.wrap(text, width=max_length) if len(text) > max_length else [text]

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert numpy waveform array to pydub AudioSegment."""
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

# ---------------------------------------------------------------------
# Asynchronous helper: process a single text chunk using a given model
# ---------------------------------------------------------------------
async def process_chunk(chunk: str, speaker_wav: str, language: str, speed: float, model: TTS, lock: asyncio.Lock) -> AudioSegment:
    loop = asyncio.get_running_loop()
    # Ensure only one inference call per model at a time
    async with lock:
        wav_array = await loop.run_in_executor(None, model.tts, chunk, speaker_wav, language, speed)
    chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate=24000)
    return chunk_audio

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Convert audio to WAV if necessary
        audio = AudioSegment.from_file(upload_path).set_frame_rate(24000).set_channels(1)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")
        
        # Register the speaker embedding in each model's speaker_manager.
        # We now access the synthesizer's TTS model to compute and register the embedding.
        for model in tts_models:
            try:
                embedding = model.synthesizer.tts_model.speaker_manager.compute_embedding(preprocessed_path)
                model.synthesizer.tts_model.speaker_manager.speakers[preprocessed_path] = {
                    "gpt_cond_latent": embedding,
                    "speaker_embedding": embedding
                }
            except Exception as e:
                print(f"Error registering speaker for model: {e}")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        print(f"✅ Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate high-quality cloned speech from user-provided voice and text."""
    print(f"🎤 Generating speech for voice_id: {request.voice_id}")
    
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    text_chunks = chunk_text(request.text, max_length=150)
    print(f"📖 Text split into {len(text_chunks)} chunks.")

    # Schedule asynchronous inference for each chunk, assigning models in round-robin fashion.
    tasks = []
    for i, chunk in enumerate(text_chunks):
        model_index = i % len(tts_models)
        model = tts_models[model_index]
        lock = model_locks[model_index]
        tasks.append(process_chunk(chunk, speaker_wav, request.language, request.speed, model, lock))
    
    # Run all inference tasks concurrently
    chunk_audios = await asyncio.gather(*tasks)

    # Concatenate all audio segments into one final AudioSegment
    final_audio = AudioSegment.empty()
    for audio_seg in chunk_audios:
        final_audio += audio_seg

    # Export final audio using in-memory BytesIO where possible
    if request.output_format.lower() in ["mp3", "wav"]:
        output_buffer = BytesIO()
        final_audio.export(output_buffer, format=request.output_format.lower())
        output_buffer.seek(0)
        media_type = "audio/mpeg" if request.output_format.lower() == "mp3" else "audio/wav"
        return Response(content=output_buffer.getvalue(), media_type=media_type)
    
    elif request.output_format.lower() == "ulaw":
        # For ulaw, export WAV to a temporary file then convert with ffmpeg asynchronously.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_temp:
            final_audio.export(wav_temp.name, format="wav")
            wav_path = wav_temp.name

        ulaw_path = wav_path.replace('.wav', '.ulaw')
        # Run ffmpeg asynchronously
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', ulaw_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        # Read the resulting ulaw file asynchronously
        async with aiofiles.open(ulaw_path, mode='rb') as f:
            ulaw_bytes = await f.read()
        
        # Clean up temporary files
        os.remove(wav_path)
        os.remove(ulaw_path)
        
        return Response(content=ulaw_bytes, media_type="audio/mulaw", headers={"X-Sample-Rate": "8000"})
    
    else:
        raise HTTPException(status_code=400, detail="Invalid output format specified.")

# =============================================================================
# Run FastAPI Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app3:app", host="0.0.0.0", port=8000, reload=True)
