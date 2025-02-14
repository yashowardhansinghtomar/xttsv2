import os
import uuid
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from pydub import AudioSegment
import subprocess

# Allowlist required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# --- Setup configurations ---
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
os.makedirs("uploads", exist_ok=True)

# Voice registry for storing preprocessed audio files
voice_registry = {}

# Load TTS model (CPU only)
print("📥 Loading XTTS model...")
tts = TTS(model_name=MODEL_NAME, gpu=False)
print("✅ Model ready!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure minimum audio length"""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def add_silence(audio: np.ndarray, duration_ms: int = 1000, sample_rate: int = 24000) -> np.ndarray:
    """Add silence at the end of the audio to ensure proper length"""
    silence = np.zeros(int(duration_ms * sample_rate / 1000))
    return np.concatenate((audio, silence))

def wav_to_ulaw(wav_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert WAV NumPy array to u-law encoded bytes using ffmpeg"""
    try:
        # Ensure the data is in int16 format
        wav_int16 = (wav_data * 32767).astype(np.int16)

        # Create FFmpeg process
        process = subprocess.Popen(
            ['ffmpeg',
             '-f', 's16le',  # Input format: signed 16-bit little-endian
             '-ar', str(sample_rate),  # Input sample rate
             '-ac', '1',  # Input channels (mono)
             '-i', 'pipe:0',  # Read from stdin
             '-f', 'mulaw',  # Output format: u-law
             '-ar', '8000',  # Output sample rate (set to 8000 Hz)
             '-ac', '1',  # Output channels (mono)
             'pipe:1'  # Write to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Send PCM data to ffmpeg and get u-law encoded output
        ulaw_data, stderr = process.communicate(wav_int16.tobytes())

        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()}")

        return ulaw_data

    except Exception as e:
        print(f"Error converting to u-law: {str(e)}")
        raise

class GenerateSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess reference audio"""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"

        # Save uploaded file
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Preprocess audio
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Store in registry
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}

        print(f"✅ Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateSpeechRequest):
    """Generate cloned speech and return raw u-law encoded audio bytes"""
    try:
        if request.voice_id not in voice_registry:
            raise HTTPException(status_code=404, detail="Voice ID not found")

        # Get preprocessed reference audio
        speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
        print(f"🔊 Using speaker WAV: {speaker_wav}")

        # Generate speech using the TTS model
        wav = tts.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )

        # Ensure output is a NumPy array
        wav = np.array(wav, dtype=np.float32)
        if len(wav) == 0:
            raise Exception("TTS model generated empty audio.")

        print(f"✅ Generated waveform: {wav[:10]}... (first 10 samples)")

        # Get sample rate
        sample_rate = tts.synthesizer.output_sample_rate
        if sample_rate is None:
            sample_rate = 24000  # Default to 24kHz

        print(f"🎚️ Sample Rate: {sample_rate}")

        # Add silence to ensure proper length
        wav = add_silence(wav, duration_ms=1000, sample_rate=sample_rate)

        # Convert to µ-law format
        ulaw_bytes = wav_to_ulaw(wav, sample_rate)

        # Return µ-law encoded audio buffer in response body
        headers = {
            "Content-Disposition": "inline; filename=\"audio.ulaw\"",
            "Content-Type": "audio/ulaw; rate=8000",
        }
        return Response(content=ulaw_bytes, headers=headers)

    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
