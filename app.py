import os
import uuid
import torch
import logging
import requests
import numpy as np
import subprocess
import json
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
os.makedirs("uploads", exist_ok=True)

# We'll use a specific Mozilla TTS model that's known to work well for voice cloning
# These are from the Mozilla TTS repository for Tacotron2 with multi-speaker support
CHECKPOINT_URL = "https://drive.google.com/uc?id=1sgEjHt0lJGK--S8_U0bUCxo4XSCY0NEN"
CONFIG_URL = "https://drive.google.com/uc?id=1mDNvtcS1nqO2BgGNz12g-HGQCmZL58N1"
VOCODER_URL = "https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu"
VOCODER_CONFIG_URL = "https://drive.google.com/uc?id=1OjXI3d4k7HQdOJcuLmLIT0mVUjUdNmjI"
SPEAKER_ENCODER_URL = "https://github.com/mozilla/TTS/releases/download/v0.0.17/speaker-encoder.pt"

# Use gdown to download from Google Drive
def download_from_gdrive(url, output_path):
    if not os.path.exists(output_path):
        try:
            logger.info(f"Downloading {url} to {output_path}")
            # For direct downloads
            if "github.com" in url:
                response = requests.get(url)
                with open(output_path, "wb") as f:
                    f.write(response.content)
            # For Google Drive links
            else:
                subprocess.run(["gdown", "--fuzzy", url, "-O", output_path], check=True)
            logger.info(f"Successfully downloaded to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download: {str(e)}")
            return False
    else:
        logger.info(f"File already exists: {output_path}")
        return True

# Local paths
TTS_CHECKPOINT = os.path.join(MODEL_DIR, "tts_model.pth.tar")
TTS_CONFIG = os.path.join(MODEL_DIR, "tts_config.json")
VOCODER_CHECKPOINT = os.path.join(MODEL_DIR, "vocoder_model.pth.tar")
VOCODER_CONFIG = os.path.join(MODEL_DIR, "vocoder_config.json")
SPEAKER_ENCODER_CHECKPOINT = os.path.join(MODEL_DIR, "speaker_encoder.pt")

# Create dummy configs if download fails to avoid JSON decode errors
def create_dummy_config(path, config_type):
    if config_type == "tts":
        config = {
            "model": "tacotron2",
            "num_chars": 126,
            "num_speakers": 10,
            "r": 2,
            "out_channels": 80
        }
    elif config_type == "vocoder":
        config = {
            "model": "hifigan",
            "sampling_rate": 22050,
            "num_mels": 80
        }
    
    with open(path, "w") as f:
        json.dump(config, f)

# Download all required files
logger.info("Setting up Mozilla TTS model files...")
tts_model_ok = download_from_gdrive(CHECKPOINT_URL, TTS_CHECKPOINT)
tts_config_ok = download_from_gdrive(CONFIG_URL, TTS_CONFIG)
vocoder_model_ok = download_from_gdrive(VOCODER_URL, VOCODER_CHECKPOINT)
vocoder_config_ok = download_from_gdrive(VOCODER_CONFIG_URL, VOCODER_CONFIG)
encoder_ok = download_from_gdrive(SPEAKER_ENCODER_URL, SPEAKER_ENCODER_CHECKPOINT)

# Create dummy configs if necessary
if not tts_config_ok:
    create_dummy_config(TTS_CONFIG, "tts")
if not vocoder_config_ok:
    create_dummy_config(VOCODER_CONFIG, "vocoder")

# =============================================================================
# Mozilla TTS Setup
# =============================================================================
try:
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.models import setup_model
    from TTS.tts.utils.synthesis import synthesis
    from TTS.utils.io import load_config
    
    # Load configs
    tts_config = load_config(TTS_CONFIG)
    vocoder_config = load_config(VOCODER_CONFIG)
    
    # Setup models
    # We'll initialize these lazily when needed to avoid errors on startup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Initialize audio processor for vocoder
    ap = AudioProcessor(**vocoder_config.audio)
    
    logger.info(f"Models loaded successfully. Using device: {device}")
    
except Exception as e:
    logger.error(f"Error initializing TTS models: {str(e)}")
    # We'll try to continue and load models when needed

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

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize the audio to a target dBFS level."""
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs)

# =============================================================================
# Lazy Loading Functions for TTS Components
# =============================================================================
def get_speaker_manager():
    try:
        from TTS.tts.utils.speakers import SpeakerManager
        speaker_manager = SpeakerManager()
        speaker_manager.load_encoder_model(SPEAKER_ENCODER_CHECKPOINT, device)
        return speaker_manager
    except Exception as e:
        logger.error(f"Failed to load speaker manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize speaker encoder")

def get_tts_model():
    try:
        from TTS.tts.models import setup_model
        model = setup_model(tts_config)
        model.load_checkpoint(tts_config, TTS_CHECKPOINT, eval=True)
        model.to(device)
        return model
    except Exception as e:
        logger.error(f"Failed to load TTS model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize TTS model")

def get_vocoder_model():
    try:
        from TTS.vocoder.models import setup_model as setup_vocoder_model
        vocoder = setup_vocoder_model(vocoder_config)
        vocoder.load_checkpoint(vocoder_config, VOCODER_CHECKPOINT, eval=True)
        vocoder.to(device)
        return vocoder
    except Exception as e:
        logger.error(f"Failed to load vocoder: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize vocoder")

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads a reference voice sample and returns a unique voice_id."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"

        # Save uploaded file
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Process the audio file
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Store the voice reference
        voice_registry[voice_id] = {
            "preprocessed_file": preprocessed_path,
        }
        
        logger.info(f"Voice uploaded and processed. Voice ID: {voice_id}")
        
        return {"voice_id": voice_id}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates voice-cloned speech from text."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    reference_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        # Lazy load models when needed
        speaker_manager = get_speaker_manager()
        tts_model = get_tts_model()
        vocoder = get_vocoder_model()
        
        # Extract speaker embedding
        speaker_embedding = speaker_manager.compute_embedding_from_wav(reference_wav)
        
        # Generate speech with Mozilla TTS
        output_path = f"temp_cloned_{request.voice_id}.wav"
        temp_output_files.append(output_path)

        # Synthesize with the target voice
        outputs = synthesis(
            model=tts_model,
            text=request.text,
            ap=ap,
            speaker_id=None,
            style_wav=None,
            speaker_embeddings=speaker_embedding.reshape(1, -1),
            use_griffin_lim=False,
            d_vector_dim=speaker_embedding.shape[0]
        )
        
        # Convert to waveform using vocoder
        waveform = vocoder.inference(outputs["outputs"][0].unsqueeze(0))
        ap.save_wav(waveform, output_path)
        
        # Process the generated audio
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
        logger.error(f"Voice cloning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning error: {str(e)}")

    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model": "Mozilla TTS", 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "registered_voices": len(voice_registry)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
