#dividing data into chunka working fine but taking longer time
import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap

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

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model ready for voice cloning!")

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
    temp_output_files = []

    try:
        text_chunks = chunk_text(request.text, max_length=150)
        print(f"📖 Text split into {len(text_chunks)} chunks.")

        final_audio = AudioSegment.empty()
        sample_rate = 24000  # Standard XTTS output sample rate

        # Process each chunk
        for idx, chunk in enumerate(text_chunks):
            print(f"🔹 Processing chunk {idx+1}/{len(text_chunks)}")
            
            wav_array = tts_model.tts(
                text=chunk,
                speaker_wav=speaker_wav,
                language=request.language,
                speed=request.speed  # Apply user-defined speed
            )

            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate)
            final_audio += chunk_audio  # Concatenate chunks

        # Create unique output file
        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        # Export in requested format
        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3")
            with open(output_path, "rb") as f:
                return Response(f.read(), media_type="audio/mpeg")
        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
            with open(output_path, "rb") as f:
                return Response(f.read(), media_type="audio/wav")
        elif request.output_format.lower() == "ulaw":
            wav_path = output_path.replace('.ulaw', '.wav')
            final_audio.export(wav_path, format='wav')
            temp_output_files.append(wav_path)

            ulaw_path = output_path
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', ulaw_path], check=True)
            with open(ulaw_path, 'rb') as f:
                return Response(f.read(), media_type="audio/mulaw", headers={"X-Sample-Rate": "8000"})
        else:
            raise HTTPException(status_code=400, detail="Invalid output format specified.")
    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# =============================================================================
# Run FastAPI Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
