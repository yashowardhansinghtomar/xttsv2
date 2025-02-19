import os
import uuid
import asyncio
import platform
import warnings
from multiprocessing import Pool
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
import numpy as np
import torch
import textwrap

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

def chunk_text(text: str, max_length: int = 1000) -> list:
    """Split long text into larger chunks while maintaining word integrity."""
    return textwrap.wrap(text, width=max_length)

def smooth_transition(audio1: AudioSegment, audio2: AudioSegment, transition_ms: int = 200) -> AudioSegment:
    """Add a brief silence between chunks to smooth transitions."""
    overlap = AudioSegment.silent(duration=transition_ms)
    return audio1.append(audio2, crossfade=transition_ms)

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert numpy waveform array to pydub AudioSegment."""
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

def process_chunk_on_gpu(args):
    """Process a single text chunk using TTS model on a specific GPU and return audio segment."""
    chunk, speaker_wav, language, gpu_id = args
    device = f"cuda:{gpu_id}"
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    tts_model.to(device)
    wav_array = tts_model.tts(
        text=chunk,
        speaker_wav=speaker_wav,
        language=language
    )
    wav_array = np.array(wav_array, dtype=np.float32)
    if len(wav_array) == 0:
        raise HTTPException(status_code=500, detail="TTS model generated empty audio")

    return wav_array_to_audio_segment(wav_array, sample_rate=24000)

# =============================================================================
# Voice Cloning Storage
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

# =============================================================================
# API Endpoints
# =============================================================================

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
    print(f"Warning: Could not patch configuration: {e}")

# =============================================================================
# Generate Cloned Speech Endpoint
# =============================================================================

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """
    Generate cloned speech using the uploaded voice and specified text.
    """
    try:
        voice_id = request.voice_id
        text = request.text
        language = request.language
        speed = request.speed
        output_format = request.output_format

        if voice_id not in voice_registry:
            raise HTTPException(status_code=404, detail="Voice ID not found")

        preprocessed_file = voice_registry[voice_id]["preprocessed_file"]
        speaker_wav = AudioSegment.from_wav(preprocessed_file)

        # Split text into chunks
        text_chunks = chunk_text(text)

        # Process each chunk on GPU
        with Pool(processes=torch.cuda.device_count()) as pool:
            audio_segments = pool.map(process_chunk_on_gpu, [(chunk, speaker_wav, language, i) for i, chunk in enumerate(text_chunks)])

        # Combine audio segments with smooth transitions
        final_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            final_audio = smooth_transition(final_audio, segment)

        # Adjust speed
        final_audio = final_audio.speedup(playback_speed=speed)

        # Export final audio in the desired format
        output_path = f"uploads/{voice_id}_cloned_speech.{output_format}"
        final_audio.export(output_path, format=output_format)

        # Return the audio file
        with open(output_path, "rb") as f:
            return Response(content=f.read(), media_type=f"audio/{output_format}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# =============================================================================
# Run the Application
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
