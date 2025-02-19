import os
import uuid
import asyncio
import platform
import audioop
import struct
import numpy as np
import torch
import warnings
import textwrap

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment, effects

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Optimized Voice Cloning API",
    description="Voice cloning using XTTS with GPU, model optimization, and in-memory conversions for real-time performance.",
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
# Request Model for Voice Cloning
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
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure the audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def chunk_text(text: str, max_length: int = 500) -> list:
    """Split long text into larger chunks while maintaining word integrity."""
    return textwrap.wrap(text, width=max_length)

def smooth_transition(audio1: AudioSegment, audio2: AudioSegment, transition_ms: int = 100) -> AudioSegment:
    """Add a brief silence between chunks to smooth transitions."""
    silence = AudioSegment.silent(duration=transition_ms)
    return audio1 + silence + audio2

# =============================================================================
# Voice Cloning Storage
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Optimized Voice Cloning API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/upload_audio": "Upload reference audio for voice cloning",
            "/generate_cloned_speech": "Generate voice cloned speech (XTTS)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload and preprocess reference audio for voice cloning.
    Returns a unique voice_id.
    """
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
        print(f"✅ Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# =============================================================================
# Load the XTTS Model for Voice Cloning with GPU & Optimization
# =============================================================================
print("📥 Loading XTTS model for voice cloning...")

from TTS.api import TTS

# Enable GPU if available
use_gpu = torch.cuda.is_available()
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

# If GPU is used, try to cast the model to half precision for faster inference.
if use_gpu:
    try:
        # Access the underlying model through the synthesizer, if available.
        if hasattr(tts_model.synthesizer, "model"):
            tts_model.synthesizer.model = tts_model.synthesizer.model.half()
            print("✅ Model cast to half precision.")
    except Exception as e:
        print(f"Warning: Could not cast model to half precision: {e}")

# Attempt to patch the configuration to differentiate pad from eos tokens.
try:
    if hasattr(tts_model.synthesizer, "model"):
        if tts_model.synthesizer.model.config.pad_token_id == tts_model.synthesizer.model.config.eos_token_id:
            tts_model.synthesizer.model.config.pad_token_id = 0
except Exception as e:
    print(f"Warning: Could not patch pad_token_id due to: {e}")

print("✅ XTTS Model ready for voice cloning!")

# =============================================================================
# Optimized Voice Cloning Endpoint
# =============================================================================
@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """
    Generate voice cloned speech using the XTTS model.
    The output format can be raw μ-law bytes, MP3, or WAV.
    All operations are performed in memory to minimize file I/O.
    """
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    # Split text into larger chunks for better audio quality
    text_chunks = chunk_text(request.text, max_length=500)

    # Process text chunks in parallel using multiprocessing and GPUs
    num_gpus = torch.cuda.device_count()
    args = [(chunk, speaker_wav, request.language, i % num_gpus) for i, chunk in enumerate(text_chunks)]

    with Pool(processes=num_gpus) as pool:
        results = pool.map(process_chunk_on_gpu, args)

    # Combine audio segments with smooth transitions
    final_audio = results[0]
    for i in range(1, len(results)):
        final_audio = smooth_transition(final_audio, results[i])

    # Create a unique temporary output path.
    unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
    output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"

    # Convert audio to the desired output format.
    if request.output_format.lower() == "mp3":
        mp3_data = final_audio.export(format="mp3").read()
        return Response(
            content=mp3_data,
            media_type="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg"
            }
        )
    elif request.output_format.lower() == "wav":
        wav_data = final_audio.export(format="wav").read()
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav"
            }
        )
    elif request.output_format.lower() == "ulaw":
        # Resample audio to telephony standard: 8000 Hz, mono.
        audio = audio.set_channels(1).set_frame_rate(8000)
        
        # Get the final PCM data directly from the AudioSegment.
        final_pcm = audio.raw_data

        # Convert the PCM data to raw μ-law encoded bytes.
        mu_law_data = audioop.lin2ulaw(final_pcm, 2)  # 2 bytes per sample (16-bit)
        
        return Response(
            content=mu_law_data,
            media_type="audio/mulaw",
            headers={
                "Content-Type": "audio/mulaw",
                "X-Sample-Rate": "8000"
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid output format specified.")

# =============================================================================
# Run the Application
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
