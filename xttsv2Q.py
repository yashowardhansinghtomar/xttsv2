import os
import uuid
import asyncio
import platform
import wave
import audioop
import struct
import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for generating voice cloned speech (XTTS) with real-time considerations and proper µ-law WAV output.",
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

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure the audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def create_wav_from_mulaw(data: bytes, sample_rate: int = 8000, channels: int = 1, bits_per_sample: int = 8) -> bytes:
    """
    Create a WAV file (as bytes) with a proper header for µ-law encoded data.
    This is an in-memory solution that avoids external calls (like ffmpeg).
    """
    chunk_size = 36 + len(data)
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    header = struct.pack("<4sI4s4sIHHIIHH4sI",
                         b"RIFF",
                         chunk_size,
                         b"WAVE",
                         b"fmt ",
                         16,            # Subchunk1Size for PCM
                         7,             # AudioFormat: 7 for μ-law (G.711)
                         channels,
                         sample_rate,
                         byte_rate,
                         block_align,
                         bits_per_sample,
                         b"data",
                         len(data)
                        )
    return header + data

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
        "message": "Voice Cloning API",
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
# Load the XTTS Model for Voice Cloning
# =============================================================================
print("📥 Loading XTTS model for voice cloning...")
from TTS.api import TTS
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Attempt to patch the model's configuration to differentiate pad from eos tokens.
try:
    # Some versions expose the underlying model on tts_model.synthesizer.model.
    if hasattr(tts_model.synthesizer, "model"):
        if tts_model.synthesizer.model.config.pad_token_id == tts_model.synthesizer.model.config.eos_token_id:
            tts_model.synthesizer.model.config.pad_token_id = 0
except Exception as e:
    print(f"Warning: Could not patch pad_token_id due to: {e}")

print("✅ XTTS Model ready for voice cloning!")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """
    Generate voice cloned speech using the XTTS model.
    The output is returned as a WAV file with µ-law encoding (wrapped with a proper header).
    """
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_mp3_path = f"temp_cloned_{request.voice_id}_{abs(hash(request.text + str(asyncio.get_event_loop().time())))}.mp3"
    temp_wav_path = temp_mp3_path.replace('.mp3', '.wav')
    
    try:
        # Generate speech using the XTTS voice cloning model.
        wav_array = tts_model.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")
        
        # Determine sample rate (default to 24000 if not set).
        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        
        # Convert the float32 waveform to int16 PCM bytes.
        pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
        
        # Create an AudioSegment from the PCM data.
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=2,  # 16-bit audio
            frame_rate=sample_rate,
            channels=1
        )
        
        # Export the generated audio as an MP3.
        audio.export(temp_mp3_path, format="mp3")
        
        # Reload the MP3 file.
        audio = AudioSegment.from_mp3(temp_mp3_path)
        
        # Apply speed adjustment if needed.
        if request.speed != 1.0:
            original_frame_rate = audio.frame_rate
            new_frame_rate = int(original_frame_rate * request.speed)
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(original_frame_rate)
        
        # Set audio to mono and adjust to telephony sample rate (8000 Hz).
        audio = audio.set_channels(1).set_frame_rate(8000)
        
        # Export to a temporary WAV file.
        audio.export(temp_wav_path, format='wav')
        
        # Read PCM data from the temporary WAV file.
        with wave.open(temp_wav_path, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
        
        # Convert the PCM data to µ-law encoded bytes.
        mu_law_data = audioop.lin2ulaw(pcm_data, 2)  # 2 bytes per sample (16-bit)
        
        # Wrap the raw µ-law data with a proper WAV header.
        wav_with_header = create_wav_from_mulaw(mu_law_data, sample_rate=8000, channels=1, bits_per_sample=8)
        
        return Response(
            content=wav_with_header,
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "8000"
            }
        )
    finally:
        # Clean up temporary files.
        for temp_file in [temp_mp3_path, temp_wav_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# =============================================================================
# Run the Application
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xttsv2Q:app", host="0.0.0.0", port=8000, reload=True)
