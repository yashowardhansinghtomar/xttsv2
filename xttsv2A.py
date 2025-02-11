# --- Begin: Safe global registration (place at the very top) ---
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # Import XttsArgs here
from TTS.config.shared_configs import BaseDatasetConfig  # Already needed

# Allowlist all required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
# --- End: Safe global registration ---

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
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
        
# --- Option B: If Option A does not work, try removing the context manager and relying solely on the top-of-file registration ---
# if os.path.exists(MODEL_PATH):
#     logger.info(f"Loading model from {MODEL_PATH}...")
#     tts = TTS(MODEL_PATH, gpu=True)
# else:
#     logger.info(f"Downloading model to {MODEL_PATH}...")
#     tts = TTS(model_name=MODEL_NAME, gpu=True)
#     logger.info("Model downloaded and ready for use!")

# Global mapping for voice sample IDs
voice_id_map = {}

# --------------------------
# Pydantic Models
# --------------------------
class ClonedTTSRequest(BaseModel):
    text: str = Field(..., description="Text to be synthesized")
    language: str = Field("hi", description="Language code (e.g., 'hi' for Hindi)")
    speaker_id: str = Field(None, description="Optional processed speaker ID to select a voice sample")

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

def process_voice_sample(filename: str):
    """
    Processes a raw voice sample (WAV) by:
      - Generating a unique voice ID.
      - Resampling and converting to a standard format.
      - Saving the processed file as <voice_id>.wav in AUDIO_DIR.
    Returns a tuple: (original filename, voice_id)
    """
    try:
        if filename not in voice_id_map:
            voice_id = str(uuid.uuid4())
            voice_id_map[filename] = voice_id
        else:
            voice_id = voice_id_map[filename]

        raw_sample_path = os.path.join(AUDIO_DIR, filename)
        processed_path = os.path.join(AUDIO_DIR, f"{voice_id}.wav")

        if not os.path.exists(processed_path):
            audio = AudioSegment.from_wav(raw_sample_path)
            audio = ensure_min_length(audio)
            audio.export(processed_path, format="wav", bitrate="192k")
            logger.info(f"Processed voice sample: {filename} -> ID: {voice_id}")
        return filename, voice_id
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return filename, None

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
            "/assign_voice_ids": "Process raw voice samples and assign voice IDs",
            "/generate/cloned": "Generate voice-cloned speech"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/assign_voice_ids")
async def assign_voice_ids():
    """
    Scans AUDIO_DIR for raw .wav files, processes them, and assigns a unique voice ID to each.
    Returns a mapping of original file names to assigned voice IDs.
    """
    try:
        wav_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav") and not len(f[:-4]) == 36]
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_voice_sample, wav_files)
        return {"voice_ids": dict(results)}
    except Exception as e:
        logger.error(f"Error assigning voice IDs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/cloned")
async def generate_cloned_speech(request: ClonedTTSRequest):
    """
    Generates cloned speech using a processed voice sample.
    If a speaker_id is provided in the request, that sample is used.
    Otherwise, the first available processed sample is selected.
    The generated audio is converted to 16-bit PCM, then to μ-law format,
    and returned as a binary response with appropriate headers.
    """
    try:
        # Find processed voice sample files (filenames that are valid UUIDs ending with .wav)
        processed_files = [f for f in os.listdir(AUDIO_DIR)
                           if f.lower().endswith(".wav") and len(f[:-4]) == 36]

        # Select sample based on provided speaker_id (if any)
        if request.speaker_id:
            filename = f"{request.speaker_id}.wav"
            if filename not in processed_files:
                raise HTTPException(status_code=400, detail="Speaker sample with given ID not found.")
        else:
            if not processed_files:
                raise HTTPException(status_code=400, detail="No processed voice sample available. Please assign a voice ID first.")
            filename = processed_files[0]

        speaker_wav_path = os.path.join(AUDIO_DIR, filename)
        logger.info(f"Generating cloned speech using sample: {filename}")

        # Generate the cloned speech audio (a list of floats in the range [-1, 1])
        audio_list = tts.tts(text=request.text, speaker_wav=speaker_wav_path, language=request.language)

        # Convert the list to a NumPy array and then to 16-bit PCM
        audio_array = np.array(audio_list)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        raw_pcm_bytes = audio_int16.tobytes()

        # Convert the PCM bytes to μ-law (using 2 bytes per sample)
        mu_law_data = audioop.lin2ulaw(raw_pcm_bytes, 2)

        # Cleanup: delete the used processed file and remove its mapping
        try:
            os.remove(speaker_wav_path)
            logger.info(f"Deleted used voice sample file: {speaker_wav_path}")
        except Exception as e:
            logger.error(f"Error deleting file {speaker_wav_path}: {e}")

        used_voice_id = filename[:-4]
        keys_to_delete = [k for k, v in voice_id_map.items() if v == used_voice_id]
        for k in keys_to_delete:
            del voice_id_map[k]
            logger.info(f"Deleted voice ID mapping for file: {k}")

        # Optionally, remove any additional processed files to avoid buildup
        for other_file in processed_files:
            if other_file != filename:
                other_path = os.path.join(AUDIO_DIR, other_file)
                try:
                    os.remove(other_path)
                    logger.info(f"Deleted old processed voice sample file: {other_file}")
                except Exception as e:
                    logger.error(f"Error deleting file {other_file}: {e}")
                old_voice_id = other_file[:-4]
                keys_to_delete = [k for k, v in voice_id_map.items() if v == old_voice_id]
                for k in keys_to_delete:
                    del voice_id_map[k]
                    logger.info(f"Deleted voice ID mapping for old file: {k}")

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
