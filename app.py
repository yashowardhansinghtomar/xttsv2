import os
import uuid
import torch
import logging
import requests
import numpy as np
import subprocess
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Mozilla TTS imports
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.speakers import SpeakerManager

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# FastAPI Setup
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for Hindi & English voice cloning with Mozilla TTS.",
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
# Model Download & Setup
# =============================================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Mozilla TTS model URLs and paths
MODEL_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/tts_model.pth.tar"
CONFIG_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/config.json"
VOCODER_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/vocoder_model.pth.tar"
VOCODER_CONFIG_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/vocoder_config.json"
SPEAKER_ENCODER_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/model_se.pth.tar"
SPEAKER_ENCODER_CONFIG_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.10/config_se.json"

# Local paths for models
MODEL_PATH = os.path.join(MODEL_DIR, "tts_model.pth.tar")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
VOCODER_PATH = os.path.join(MODEL_DIR, "vocoder_model.pth.tar")
VOCODER_CONFIG_PATH = os.path.join(MODEL_DIR, "vocoder_config.json")
SPEAKER_ENCODER_PATH = os.path.join(MODEL_DIR, "model_se.pth.tar")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(MODEL_DIR, "config_se.json")

# Download function for models
def download_model(url, path):
    if not os.path.exists(path):
        logging.info(f"Downloading {url} to {path}")
        response = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded {path}")
    else:
        logging.info(f"File already exists: {path}")

# Download models
download_model(MODEL_URL, MODEL_PATH)
download_model(CONFIG_URL, CONFIG_PATH)
download_model(VOCODER_URL, VOCODER_PATH)
download_model(VOCODER_CONFIG_URL, VOCODER_CONFIG_PATH)
download_model(SPEAKER_ENCODER_URL, SPEAKER_ENCODER_PATH)
download_model(SPEAKER_ENCODER_CONFIG_URL, SPEAKER_ENCODER_CONFIG_PATH)

# Initialize the Mozilla TTS synthesizer
device = "cuda" if torch.cuda.is_available() else "cpu"
synthesizer = Synthesizer(
    tts_checkpoint=MODEL_PATH,
    tts_config_path=CONFIG_PATH,
    vocoder_checkpoint=VOCODER_PATH,
    vocoder_config=VOCODER_CONFIG_PATH,
    encoder_checkpoint=SPEAKER_ENCODER_PATH,
    encoder_config=SPEAKER_ENCODER_CONFIG_PATH,
    use_cuda=torch.cuda.is_available()
)

# Initialize speaker manager for voice embedding management
speaker_manager = SpeakerManager(encoder_model_path=SPEAKER_ENCODER_PATH, 
                                encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
                                use_cuda=torch.cuda.is_available())

LANGUAGE_CODES = {"en": "english", "hi": "hindi"}

voice_registry = {}  # Stores voice references and embeddings

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

        # Process the audio file
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Compute the speaker embedding
        speaker_embedding = speaker_manager.compute_embedding_from_wav(preprocessed_path)
        
        # Store the voice reference
        voice_registry[voice_id] = {
            "preprocessed_file": preprocessed_path,
            "embedding": speaker_embedding
        }
        
        return {"voice_id": voice_id}

    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {e}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates voice-cloned speech from text."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_embedding = voice_registry[request.voice_id]["embedding"]
    temp_output_files = []

    try:
        # Generate Speech with Mozilla TTS
        output_path = f"temp_cloned_{request.voice_id}.wav"
        temp_output_files.append(output_path)

        # Generate the waveform
        wav = synthesizer.tts(
            text=request.text,
            speaker_embedding=speaker_embedding,
            language_id=LANGUAGE_CODES.get(request.language, "english")
        )

        # Save the waveform
        synthesizer.save_wav(wav, output_path)

        # Load and normalize audio
        final_audio = AudioSegment.from_file(output_path)
        final_audio = normalize_audio(final_audio)

        output_format_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
        temp_output_files.append(output_format_path)

        if request.output_format.lower() == "mp3":
            final_audio.export(output_format_path, format="mp3", parameters=["-q:a", "0"])
            return Response(open(output_format_path, "rb").read(), media_type="audio/mpeg")

        elif request.output_format.lower() == "wav":
            final_audio.export(output_format_path, format="wav")
            return Response(open(output_format_path, "rb").read(), media_type="audio/wav")

        elif request.output_format.lower() == "ulaw":
            final_audio.export("temp.wav", format="wav")
            subprocess.run(["ffmpeg", "-y", "-i", "temp.wav", "-ar", "8000", "-ac", "1", "-f", "mulaw", output_format_path], check=True)
            return Response(open(output_format_path, "rb").read(), media_type="audio/mulaw")

        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")

    except Exception as e:
        logging.error(f"Voice cloning error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning error: {e}")

    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "Mozilla TTS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
