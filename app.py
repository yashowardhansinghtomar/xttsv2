import os
import uuid
import torch
import logging
import numpy as np
import asyncio
import subprocess
from threading import Lock
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.utils.synthesizer import Synthesizer
from transformers import AutoTokenizer

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# FastAPI Setup
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for Hindi & English voice cloning with FastSpeech2.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Setup - Load from Local Storage
# =============================================================================
MODEL_DIR = "models"
TTS_MODEL_PATH = os.path.join(MODEL_DIR, "tts_model.pth")
TTS_CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
VOCODER_MODEL_PATH = os.path.join(MODEL_DIR, "vocoder_model.pth")
VOCODER_CONFIG_PATH = os.path.join(MODEL_DIR, "vocoder_config.json")

if not all(os.path.exists(p) for p in [TTS_MODEL_PATH, TTS_CONFIG_PATH, VOCODER_MODEL_PATH, VOCODER_CONFIG_PATH]):
    raise RuntimeError("❌ Models are missing! Please run the download script first.")

# Load the model
tts_lock = Lock()
synthesizer = Synthesizer(
    model_path=TTS_MODEL_PATH,
    config_path=TTS_CONFIG_PATH,
    vocoder_path=VOCODER_MODEL_PATH,
    vocoder_config_path=VOCODER_CONFIG_PATH,
    use_cuda=torch.cuda.is_available(),
)

logging.info("✅ TTS Model loaded successfully from local storage!")

# Tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

LANGUAGE_CODES = {"en": "english", "hi": "hindi"}

voice_registry = {}  # Stores voice references

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, ulaw")

# =============================================================================
# Audio Processing Helpers
# =============================================================================
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensures that audio is at least min_length_ms long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert numpy waveform array to pydub AudioSegment."""
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize the audio to a target dBFS level."""
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads a reference voice sample and returns a unique voice_id."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        os.makedirs("uploads", exist_ok=True)

        with open(upload_path, "wb") as f:
            f.write(await file.read())

        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        return {"voice_id": voice_id}

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates voice-cloned speech from text."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        with tts_lock:
            wav_array = synthesizer.tts(request.text, speaker_wav=speaker_wav, language=LANGUAGE_CODES.get(request.language, "english"))
        
        final_audio = wav_array_to_audio_segment(wav_array, sample_rate=22050)
        final_audio = normalize_audio(final_audio)

        output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
        temp_output_files.append(output_path)

        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            return Response(open(output_path, "rb").read(), media_type="audio/mpeg")

        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
            return Response(open(output_path, "rb").read(), media_type="audio/wav")

        elif request.output_format.lower() == "ulaw":
            final_audio.export("temp.wav", format="wav")
            subprocess.run(["ffmpeg", "-y", "-i", "temp.wav", "-ar", "8000", "-ac", "1", "-f", "mulaw", output_path], check=True)
            return Response(open(output_path, "rb").read(), media_type="audio/mulaw")

        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
