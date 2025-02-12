# --- Begin: Safe global registration (place at the very top) ---
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # Import XttsArgs here
from TTS.config.shared_configs import BaseDatasetConfig  # Already needed

# Allowlist all required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
# --- End: Safe global registration ---

# --------------------------
# Standard Imports & Setup
# --------------------------
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import platform
import os
import uuid
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.effects import speedup  # For speeding up audio
import wave
import numpy as np
import io
import edge_tts  # For predefined TTS voices

# Import Coqui TTS API for voice cloning
from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Multilingual TTS API with Voice Cloning",
    description="API for generating TTS using predefined voices (via Edge TTS) and voice cloning (via Coqui xTTS_v2).",
    version="1.1.0"
)

# Enable CORS (adjust origins as needed)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For Windows event loop compatibility
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ----------------------------------------
# Directories, Persistent Mapping, & Model
# (For Coqui xTTS_v2 voice cloning)
# ----------------------------------------
# Use your specified base directory.
BASE_DIR = "/home/arindam/tts/xttsv2"
AUDIO_DIR = os.path.join(BASE_DIR, "audio1")           # For raw/processed voice samples
MODEL_DIR = os.path.join(BASE_DIR, "models")             # For saved TTS models
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_outputs") # For saving generated files

# Create directories if they do not exist.
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Persistent mapping file for voice sample IDs to processed file paths
PERSISTENT_MAP_FILE = os.path.join(BASE_DIR, "voice_id_map.json")

# Global in-memory mapping (loaded from disk if available)
voice_id_map = {}

def load_voice_id_map():
    global voice_id_map
    if os.path.exists(PERSISTENT_MAP_FILE):
        try:
            with open(PERSISTENT_MAP_FILE, 'r') as f:
                voice_id_map = json.load(f)
            logger.info("Loaded voice_id_map from disk.")
        except Exception as e:
            logger.error(f"Error loading voice_id_map: {e}")
            voice_id_map = {}
    else:
        voice_id_map = {}

def save_voice_id_map():
    try:
        with open(PERSISTENT_MAP_FILE, 'w') as f:
            json.dump(voice_id_map, f)
        logger.info("Saved voice_id_map to disk.")
    except Exception as e:
        logger.error(f"Error saving voice_id_map: {e}")

load_voice_id_map()

# Initialize Coqui TTS model (xTTS_v2) for voice cloning
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_PATH = os.path.join(MODEL_DIR, "xtts_v2")
# If you have a CUDA-enabled PyTorch installation, set gpu=True. Otherwise, use gpu=False.
if os.path.exists(MODEL_PATH):
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Loading model from {MODEL_PATH}...")
        tts_voice_clone = TTS(MODEL_PATH, gpu=True)
else:
    with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
        logger.info(f"Downloading model to {MODEL_PATH}...")
        tts_voice_clone = TTS(model_name=MODEL_NAME, gpu=True)
        logger.info("Model downloaded and ready for use!")

# -----------------------------------
# Endpoints for Predefined TTS (Edge TTS)
# -----------------------------------

# Pydantic models for predefined TTS endpoints
class TTSRequest(BaseModel):
    text: str
    language_code: str
    voice: str = "female"  # "male" or "female"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: Optional[str] = None
    detected_language: Optional[str] = None

# Configuration for predefined voices (adjust as needed)
LANGUAGE_MODELS = {
    # Indian Languages
    'hi': {"name": "Hindi", "male_voice": "hi-IN-MadhurNeural", "female_voice": "hi-IN-SwaraNeural"},
    'te': {"name": "Telugu", "male_voice": "te-IN-MohanNeural", "female_voice": "te-IN-ShrutiNeural"},
    'ta': {"name": "Tamil", "male_voice": "ta-IN-ValluvarNeural", "female_voice": "ta-IN-PallaviNeural"},
    'gu': {"name": "Gujarati", "male_voice": "gu-IN-NiranjanNeural", "female_voice": "gu-IN-DhwaniNeural"},
    'mr': {"name": "Marathi", "male_voice": "mr-IN-ManoharNeural", "female_voice": "mr-IN-AarohiNeural"},
    'bn': {"name": "Bengali", "male_voice": "bn-IN-BashkarNeural", "female_voice": "bn-IN-TanishaaNeural"},
    'ml': {"name": "Malayalam", "male_voice": "ml-IN-MidhunNeural", "female_voice": "ml-IN-SobhanaNeural"},
    'kn': {"name": "Kannada", "male_voice": "kn-IN-GaganNeural", "female_voice": "kn-IN-SapnaNeural"},
    'pa': {"name": "Punjabi", "male_voice": "pa-IN-GuruNeural", "female_voice": "pa-IN-SargunNeural"},
    # International Languages
    'en': {"name": "English", "male_voice": "en-US-ChristopherNeural", "female_voice": "en-US-JennyNeural"},
    'es': {"name": "Spanish", "male_voice": "es-ES-AlvaroNeural", "female_voice": "es-ES-ElviraNeural"},
    'fr': {"name": "French", "male_voice": "fr-FR-HenriNeural", "female_voice": "fr-FR-DeniseNeural"},
    'de': {"name": "German", "male_voice": "de-DE-ConradNeural", "female_voice": "de-DE-KatjaNeural"},
    'zh': {"name": "Chinese", "male_voice": "zh-CN-YunxiNeural", "female_voice": "zh-CN-XiaoxiaoNeural"},
    'ja': {"name": "Japanese", "male_voice": "ja-JP-KeitaNeural", "female_voice": "ja-JP-NanamiNeural"},
    'ko': {"name": "Korean", "male_voice": "ko-KR-InJoonNeural", "female_voice": "ko-KR-SunHiNeural"},
    'ru': {"name": "Russian", "male_voice": "ru-RU-DmitryNeural", "female_voice": "ru-RU-SvetlanaNeural"},
    'ar': {"name": "Arabic", "male_voice": "ar-SA-HamedNeural", "female_voice": "ar-SA-ZariyahNeural"},
    'tr': {"name": "Turkish", "male_voice": "tr-TR-AhmetNeural", "female_voice": "tr-TR-EmelNeural"}
}

def detect_language(text: str) -> str:
    from langdetect import detect
    try:
        detected = detect(text)
        return detected if detected in LANGUAGE_MODELS else 'en'
    except Exception:
        return 'en'

async def generate_edge_tts_voice(text: str, output_path: str, voice: str) -> bool:
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error generating Edge TTS voice: {e}")
        return False

@app.get("/")
async def root():
    return {
        "message": "Multilingual TTS API with Voice Cloning",
        "version": "1.1.0",
        "endpoints": {
            "/languages": "List supported languages",
            "/generate": "Generate TTS audio using predefined voices",
            "/generate/stream": "Generate TTS audio with speed control (streaming)",
            "/upload_voice": "Upload voice samples for cloning",
            "/generate/cloned": "Generate voice-cloned speech using a stored voice sample",
            "/health": "Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/languages")
async def get_languages():
    return {
        "languages": [
            {"code": code, "name": info["name"], "voices": ["male", "female"]}
            for code, info in LANGUAGE_MODELS.items()
        ]
    }

@app.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    try:
        if request.language_code not in LANGUAGE_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported language code: {request.language_code}")
        output_path = f"temp_{request.language_code}_{hash(request.text)}.mp3"
        try:
            voice = LANGUAGE_MODELS[request.language_code]["male_voice"] if request.voice.lower() == "male" else LANGUAGE_MODELS[request.language_code]["female_voice"]
            success = await generate_edge_tts_voice(request.text, output_path, voice)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to generate speech")
            with open(output_path, "rb") as audio_file:
                raw_audio = audio_file.read()
            return Response(raw_audio, media_type="audio/mpeg")
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_speech_stream(request: TTSRequest):
    try:
        if request.language_code not in LANGUAGE_MODELS:
            raise HTTPException(status_code=400, detail=f"Unsupported language code: {request.language_code}")
        output_path = f"temp_{request.language_code}_{abs(hash(request.text + str(asyncio.get_event_loop().time())))}.mp3"
        wav_path = output_path.replace('.mp3', '.wav')
        try:
            voice = LANGUAGE_MODELS[request.language_code]["male_voice"] if request.voice.lower() == "male" else LANGUAGE_MODELS[request.language_code]["female_voice"]
            success = await generate_edge_tts_voice(request.text, output_path, voice)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to generate speech")
            audio = AudioSegment.from_mp3(output_path)
            if request.speed != 1.0:
                original_frame_rate = audio.frame_rate
                new_frame_rate = int(original_frame_rate * request.speed)
                audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
                audio = audio.set_frame_rate(original_frame_rate)
            audio = audio.set_channels(1).set_frame_rate(8000)
            audio.export(wav_path, format='wav')
            with wave.open(wav_path, 'rb') as wav_file:
                pcm_data = wav_file.readframes(wav_file.getnframes())
                # For streaming, we're still returning μ-law here.
                import audioop
                mu_law_data = audioop.lin2ulaw(pcm_data, 2)
            return Response(
                content=mu_law_data,
                media_type="audio/mulaw",
                headers={"Content-Type": "audio/mulaw", "X-Sample-Rate": "8000"}
            )
        finally:
            for temp_file in [output_path, wav_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# ----------------------------------------------------
# Endpoints for Voice Cloning using Coqui xTTS_v2 Model
# ----------------------------------------------------

# Helper: Ensure the uploaded voice sample meets a minimum length
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio.set_frame_rate(8000).set_channels(1)

# Helper: Process an uploaded voice sample and store it
def process_voice_sample_from_file(file_path: str, original_filename: str) -> Optional[str]:
    try:
        voice_id = str(uuid.uuid4())
        processed_path = os.path.join(AUDIO_DIR, f"{voice_id}.wav")
        audio = AudioSegment.from_wav(file_path)
        audio = ensure_min_length(audio)
        audio.export(processed_path, format="wav", bitrate="192k")
        logger.info(f"Processed voice sample: {original_filename} -> ID: {voice_id}")
        voice_id_map[voice_id] = processed_path
        save_voice_id_map()  # Persist the updated mapping
        return voice_id
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {e}")
        return None

# Endpoint to upload one or more voice samples (WAV only)
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

# Pydantic model for a cloned voice TTS request
class ClonedTTSRequest(BaseModel):
    text: str = Field(..., description="Text to be synthesized")
    language: str = Field("hi", description="Language code (e.g., 'hi' for Hindi)")
    speaker_id: Optional[str] = Field(None, description="Processed voice ID (obtained from /upload_voice) to select a voice sample")

# Endpoint to generate speech using a cloned voice sample (output in μ-law format at 8000 Hz and 1.25x speed)
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
            # If no speaker_id is provided, use the first available sample.
            first_voice_id = next(iter(voice_id_map))
            processed_file = voice_id_map[first_voice_id]

        logger.info(f"Generating cloned speech using sample: {processed_file}")
        # Synthesize speech using the Coqui xTTS_v2 model
        audio_list = tts_voice_clone.tts(text=request.text, speaker_wav=processed_file, language=request.language)
        audio_array = np.array(audio_list)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create an AudioSegment from the synthesized audio (assumed original sample rate: 22050 Hz)
        audio_segment = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,
            frame_rate=22050,
            channels=1
        )
        # Speed up the audio to 1.25x
        audio_segment = speedup(audio_segment, playback_speed=1.25)
        # Resample the sped-up audio to 8000 Hz.
        audio_segment = audio_segment.set_frame_rate(8000)
        
        # Convert the raw PCM data of the AudioSegment to μ-law format.
        import audioop
        pcm_data = audio_segment.raw_data
        mu_law_data = audioop.lin2ulaw(pcm_data, audio_segment.sample_width)
        
        # Optionally, save the μ-law file to disk for inspection.
        output_filename = os.path.join(OUTPUT_DIR, f"generated_{uuid.uuid4().hex}.mulaw")
        with open(output_filename, "wb") as f:
            f.write(mu_law_data)
        logger.info(f"Generated μ-law file saved at: {output_filename}")

        return Response(
            content=mu_law_data,
            media_type="audio/mulaw",
            headers={"Content-Type": "audio/mulaw", "X-Sample-Rate": "8000"}
        )
    except Exception as e:
        logger.error(f"Error generating cloned speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing voice cloning: {str(e)}")

# --------------------------
# Run the Application
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
