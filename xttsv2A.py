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

# Model configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_PATH = os.path.join(MODEL_DIR, "xtts_v2")

# --- Option A: Wrap TTS instantiation with the safe_globals context manager ---
if os.path.exists(MODEL_PATH):
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Loading model from {MODEL_PATH}...")
        tts = TTS(MODEL_PATH, gpu=True)
else:
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Downloading model to {MODEL_PATH}...")
        tts = TTS(model_name=MODEL_NAME, gpu=True)
        logger.info("Model downloaded and ready for use!")

# Global mapping for voice sample IDs.
# Each key is a generated voice_id and the value is the full path to the processed WAV file.
voice_id_map = {}

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
    """
    Ensures the audio is at least `min_length_ms` milliseconds long by appending silence if needed.
    Also resamples the audio to 8000 Hz and converts it to mono.
    """
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio.set_frame_rate(8000).set_channels(1)

def process_voice_sample_from_file(file_path: str, original_filename: str) -> str:
    """
    Processes an uploaded voice sample (WAV) by:
      - Generating a unique voice ID.
      - Resampling and converting to a standard format.
      - Saving the processed file as <voice_id>.wav in AUDIO_DIR.
    Returns the generated voice_id.
    """
    try:
        voice_id = str(uuid.uuid4())
        processed_path = os.path.join(AUDIO_DIR, f"{voice_id}.wav")
        audio = AudioSegment.from_wav(file_path)
        audio = ensure_min_length(audio)
        audio.export(processed_path, format="wav", bitrate="192k")
        logger.info(f"Processed voice sample: {original_filename} -> ID: {voice_id}")
        # Update the global mapping with the processed file's path.
        voice_id_map[voice_id] = processed_path
        return voice_id
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {e}")
        return None

# --------------------------
# API Endpoints
# --------------------------
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
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
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.post("/upload_voice")
async def upload_voice(files: List[UploadFile] = File(...)):
    """
    Endpoint for uploading one or more voice samples.
    Each file is processed and stored, and a list of generated voice IDs is returned.
    """
    try:
        voice_ids = {}
        for file in files:
            if not file.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="Only WAV files are accepted.")
            # Save the uploaded file temporarily.
            temp_filename = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
            with open(temp_filename, "wb") as f:
                content = await file.read()
                f.write(content)
            # Process the uploaded file to generate a voice_id.
            voice_id = process_voice_sample_from_file(temp_filename, file.filename)
            # Remove the temporary file.
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
    """
    Generates cloned speech using a stored voice sample.
    The user must provide a valid speaker_id obtained from the /upload_voice endpoint.
    If no speaker_id is provided, the first available processed sample is used.
    """
    try:
        # Determine which processed file to use.
        if request.speaker_id:
            processed_file = voice_id_map.get(request.speaker_id)
            if not processed_file or not os.path.exists(processed_file):
                raise HTTPException(status_code=400, detail="Speaker sample with the given ID not found.")
        else:
            if not voice_id_map:
                raise HTTPException(status_code=400, detail="No processed voice sample available. Please upload a voice sample first.")
            # Use the first available voice sample.
            first_voice_id = next(iter(voice_id_map))
            processed_file = voice_id_map[first_voice_id]

        logger.info(f"Generating cloned speech using sample: {processed_file}")

        # Generate the cloned speech audio (a list of floats in the range [-1, 1])
        audio_list = tts.tts(text=request.text, speaker_wav=processed_file, language=request.language)

        # Convert the list to a NumPy array and then to 16-bit PCM.
        audio_array = np.array(audio_list)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        raw_pcm_bytes = audio_int16.tobytes()

        # Convert the PCM bytes to μ-law (using 2 bytes per sample).
        mu_law_data = audioop.lin2ulaw(raw_pcm_bytes, 2)

        # Do NOT delete the voice sample file or its mapping.

        # Return the synthesized audio as raw binary data (μ-law) with headers.
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
