# --- Begin: Safe global registration (place at the very top) ---
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # Import XttsArgs here
from TTS.config.shared_configs import BaseDatasetConfig  # Already needed

# Allowlist all required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
# --- End: Safe global registration ---

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
import os
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from TTS.api import TTS
import numpy as np
import audioop
import json

# --------------------------
# Application Configuration
# --------------------------
app = FastAPI(
    title="Voice Cloning API",
    description="API for generating voice-cloned speech using Coqui TTS's xTTS_v2 model.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories for audio samples and model storage (adjust paths as needed)
BASE_DIR = "/home/arindam/tts/xttsv2"
AUDIO_DIR = os.path.join(BASE_DIR, "audio1")   # Directory for raw/processed voice samples
MODEL_DIR = os.path.join(BASE_DIR, "models")     # Directory for saved TTS models
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Path for persistent voice ID mapping
PERSISTENT_MAP_FILE = os.path.join(BASE_DIR, "voice_id_map.json")

# --------------------------
# Model Configuration and Loading
# --------------------------
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_PATH = os.path.join(MODEL_DIR, "xtts_v2")

if os.path.exists(MODEL_PATH):
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Loading model from {MODEL_PATH}...")
        tts = TTS(MODEL_PATH, gpu=True)
else:
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Downloading model to {MODEL_PATH}...")
        tts = TTS(model_name=MODEL_NAME, gpu=True)
        logger.info("Model downloaded and ready for use!")

# --------------------------
# Persistent Voice ID Mapping
# --------------------------
# Global mapping for voice sample IDs.
# Each key is a generated voice_id and the value is the full path to the processed WAV file.
voice_id_map = {}

def load_voice_id_map():
    global voice_id_map
    if os.path.exists(PERSISTENT_MAP_FILE):
        try:
            with open(PERSISTENT_MAP_FILE, 'r') as f:
                voice_id_map = json.load(f)
            logger.info("Loaded voice_id_map from disk.")
        except Exception as e:
            logger.error(f"Error loading voice_id_map from {PERSISTENT_MAP_FILE}: {e}")
            voice_id_map = {}
    else:
        voice_id_map = {}

def save_voice_id_map():
    try:
        with open(PERSISTENT_MAP_FILE, 'w') as f:
            json.dump(voice_id_map, f)
        logger.info("Saved voice_id_map to disk.")
    except Exception as e:
        logger.error(f"Error saving voice_id_map to {PERSISTENT_MAP_FILE}: {e}")

# Load the voice mapping on startup
load_voice_id_map()

# --------------------------
# Pydantic Models
# --------------------------
class ClonedTTSRequest(BaseModel):
    text: str = Field(..., description="Text to be synthesized")
    language: str = Field("hi", description="Language code (e.g., 'hi' for Hindi)")
    speaker_id: str = Field(
        None, description="Processed voice ID (obtained from /upload_voice) to select a voice sample"
    )

class ClonedTTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: str = None  # (Optional) if you prefer returning base64 audio instead of raw bytes

# --------------------------
# Helper Functions
# --------------------------
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio.set_frame_rate(8000).set_channels(1)

def process_voice_sample_from_file(file_path: str, original_filename: str) -> str:
    try:
        voice_id = str(uuid.uuid4())
        processed_path = os.path.join(AUDIO_DIR, f"{voice_id}.wav")
        audio = AudioSegment.from_wav(file_path)
        audio = ensure_min_length(audio)
        audio.export(processed_path, format="wav", bitrate="192k")
        logger.info(f"Processed voice sample: {original_filename} -> ID: {voice_id}")
        voice_id_map[voice_id] = processed_path
        save_voice_id_map()  # Save mapping after update
        return voice_id
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {e}")
        return None

# --------------------------
# API Endpoints
# --------------------------
@app.get("/")
async def root():
    return {
        "message": "Voice Cloning API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/upload_voice": "Upload one or more voice samples and get their voice IDs",
            "/generate/cloned": "Generate voice-cloned speech using a stored voice sample"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload_voice")
async def upload_voice(files: List[UploadFile] = File(...)):
    try:
        voice_ids = {}
        for file in files:
            if not file.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="Only WAV files are accepted.")
            temp_filename = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
            with open(temp_filename, "wb") as f:
                content = await file.read()
                f.write(content)
            voice_id = process_voice_sample_from_file(temp_filename, file.filename)
            os.remove(temp_filename)
            if voice_id:
                voice_ids[file.filename] = voice_id
            else:
                voice_ids[file.filename] = "Processing failed"
        return {"voice_ids": voice_ids, "message": "Voice file(s) uploaded and processed successfully."}
    except Exception as e:
        logger.error(f"Error uploading voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/cloned")
async def generate_cloned_speech(request: ClonedTTSRequest):
    try:
        if request.speaker_id:
            processed_file = voice_id_map.get(request.speaker_id)
            if not processed_file or not os.path.exists(processed_file):
                raise HTTPException(status_code=400, detail="Speaker sample with the given ID not found.")
        else:
            if not voice_id_map:
                raise HTTPException(status_code=400, detail="No processed voice sample available. Please upload a voice sample first.")
            first_voice_id = next(iter(voice_id_map))
            processed_file = voice_id_map[first_voice_id]

        logger.info(f"Generating cloned speech using sample: {processed_file}")
        audio_list = tts.tts(text=request.text, speaker_wav=processed_file, language=request.language)
        audio_array = np.array(audio_list)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        raw_pcm_bytes = audio_int16.tobytes()
        mu_law_data = audioop.lin2ulaw(raw_pcm_bytes, 2)
        return Response(
            content=mu_law_data,
            media_type="audio/mulaw",
            headers={
                "Content-Type": "audio/mulaw",
                "X-Sample-Rate": "8000"
            }
        )
    except Exception as e:
        logger.error(f"Error generating cloned speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing voice cloning: {str(e)}")

# --------------------------
# Run the Application
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("xttsv2A:app", host="0.0.0.0", port=8000, reload=True)
