import os
import uuid
import torch
import logging
import requests
import numpy as np
import subprocess
import json
import shutil
import sys  # Added this import at the top level
from pathlib import Path
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
    description="API for voice cloning with Mozilla TTS.",
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
# Constants and Setup
# =============================================================================
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Use a simpler Mozilla TTS model that's more reliable
# We'll use YourTTS which is a multi-speaker, multi-lingual model with voice cloning capability
YOURTTTS_REPO = "https://github.com/Edresson/YourTTS"
YOURTTTS_DIR = os.path.join(MODEL_DIR, "YourTTS")

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
# Model Setup Functions
# =============================================================================
def setup_yourtts():
    """Setup YourTTS model for voice cloning."""
    try:
        # Check if YourTTS is already cloned
        if not os.path.exists(YOURTTTS_DIR):
            logger.info("Cloning YourTTS repository...")
            subprocess.run(
                ["git", "clone", YOURTTTS_REPO, YOURTTTS_DIR], 
                check=True
            )
            
            # Install YourTTS requirements
            requirements_file = os.path.join(YOURTTTS_DIR, "requirements.txt")
            logger.info("Installing YourTTS requirements...")
            subprocess.run(
                ["pip", "install", "-r", requirements_file],
                check=True
            )
            
            # Install TTS package from the cloned repo
            logger.info("Installing YourTTS package...")
            subprocess.run(
                ["pip", "install", "-e", YOURTTTS_DIR],
                check=True
            )
            
            # Verify TTS package installation
            logger.info("Verifying TTS installation...")
            try:
                subprocess.run(
                    ["python", "-c", "import TTS; print('TTS package imported successfully')"],
                    check=True
                )
                logger.info("TTS package verification successful")
            except subprocess.CalledProcessError:
                logger.error("TTS package verification failed")
                return False
            
        # Download pre-trained model if not exists
        yourttts_model_dir = os.path.join(MODEL_DIR, "yourttts_model")
        os.makedirs(yourttts_model_dir, exist_ok=True)
        
        # Download checkpoint file - check if exists first
        checkpoint_file = os.path.join(yourttts_model_dir, "model.pth.tar")
        if not os.path.exists(checkpoint_file):
            logger.info("Downloading YourTTS pre-trained model...")
            download_url = "https://drive.google.com/uc?id=1Je16cZLA02H3l-70BJi9M5rkFcjgTNKA"
            subprocess.run(
                ["gdown", "--fuzzy", download_url, "-O", checkpoint_file],
                check=True
            )
            
        # Download config file
        config_file = os.path.join(yourttts_model_dir, "config.json")
        if not os.path.exists(config_file):
            logger.info("Downloading YourTTS config...")
            config_url = "https://drive.google.com/uc?id=1OZF5uVUes4g3pCpFl5ZYlC6FfCYIjP6e"
            subprocess.run(
                ["gdown", "--fuzzy", config_url, "-O", config_file],
                check=True
            )
            
        logger.info("YourTTS setup complete!")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up YourTTS: {str(e)}")
        return False

# Setup YourTTS on startup
setup_success = setup_yourtts()

# =============================================================================
# TTS Synthesis Functions
# =============================================================================
def load_yourtts_model():
    """Load the YourTTS model for inference."""
    try:
        # Debug available modules
        logger.info(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
        logger.info(f"Current sys.path: {sys.path}")
        
        # Try multiple import approaches
        try:
            # First try: standard import (if pip install was successful)
            from TTS.tts.models import YourTTS
            from TTS.utils.audio import AudioProcessor
            from TTS.tts.utils.speakers import SpeakerManager
            from TTS.config import load_config
            logger.info("Successfully imported TTS modules using standard import")
        except ImportError as e1:
            logger.warning(f"Standard import failed: {e1}")
            
            # Second try: Import with explicit path
            if YOURTTTS_DIR not in sys.path:
                sys.path.insert(0, YOURTTTS_DIR)
            
            try:
                from TTS.tts.models.YourTTS import YourTTS
                from TTS.utils.audio import AudioProcessor
                from TTS.tts.utils.speakers import SpeakerManager
                from TTS.config import load_config
                logger.info("Successfully imported TTS modules after adding to sys.path")
            except ImportError as e2:
                logger.warning(f"Import with explicit path failed: {e2}")
                
                # Third try: Import from YourTTS subdirectory
                try:
                    from YourTTS.TTS.tts.models.YourTTS import YourTTS
                    from YourTTS.TTS.utils.audio import AudioProcessor
                    from YourTTS.TTS.tts.utils.speakers import SpeakerManager
                    from YourTTS.TTS.config import load_config
                    logger.info("Successfully imported TTS modules from YourTTS subdirectory")
                except ImportError as e3:
                    logger.error(f"All import attempts failed. Last error: {e3}")
                    
                    # Check module structure
                    logger.info("Checking TTS module structure...")
                    try:
                        import pkgutil
                        for pkg in pkgutil.iter_modules():
                            if 'tts' in pkg.name.lower():
                                logger.info(f"Found module: {pkg.name}")
                        
                        # Check if TTS is an importable package
                        import importlib
                        if importlib.util.find_spec("TTS") is not None:
                            logger.info("TTS package exists")
                            tts_pkg = importlib.import_module("TTS")
                            logger.info(f"TTS package path: {tts_pkg.__path__}")
                            
                            # Print the directory structure
                            tts_path = tts_pkg.__path__[0]
                            for root, dirs, files in os.walk(tts_path):
                                logger.info(f"Directory: {root}")
                                for file in files:
                                    if file.endswith('.py'):
                                        logger.info(f"  Python file: {file}")
                    except Exception as e:
                        logger.error(f"Error inspecting modules: {e}")
                    
                    raise HTTPException(status_code=500, detail=f"Failed to import TTS modules after multiple attempts. Please check the installation and file structure.")
        
        # Load config
        yourttts_model_dir = os.path.join(MODEL_DIR, "yourttts_model")
        config_path = os.path.join(yourttts_model_dir, "config.json")
        config = load_config(config_path)
        
        # Initialize audio processor
        ap = AudioProcessor(**config.audio)
        
        # Initialize speaker manager for embeddings
        speaker_manager = SpeakerManager()
        
        # Load YourTTS model
        model = YourTTS(config, ap)
        
        # Load checkpoint
        checkpoint_path = os.path.join(yourttts_model_dir, "model.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return model, speaker_manager, ap, device
        
    except Exception as e:
        logger.error(f"Error loading YourTTS model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")

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
        upload_path = os.path.join(UPLOAD_DIR, f"{voice_id}_{file.filename}")
        
        # Save uploaded file
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        
        # Process the audio file
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        
        # Convert to 16-bit PCM WAV at 22050 Hz (YourTTS requirement)
        preprocessed_path = os.path.join(UPLOAD_DIR, f"{voice_id}_preprocessed.wav")
        audio = audio.set_frame_rate(22050)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit
        audio.export(preprocessed_path, format="wav")
        
        # Store the voice reference
        voice_registry[voice_id] = {
            "preprocessed_file": preprocessed_path,
        }
        
        logger.info(f"Voice uploaded and processed. Voice ID: {voice_id}")
        
        return {"voice_id": voice_id}
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates voice-cloned speech from text."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    
    reference_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []
    
    try:
        # Load YourTTS model
        model, speaker_manager, ap, device = load_yourtts_model()
        
        # Compute speaker embedding (d-vector) from the reference audio
        speaker_embedding = speaker_manager.compute_d_vector_from_clip(reference_wav)
        
        # Generate speech
        output_path = os.path.join(UPLOAD_DIR, f"temp_cloned_{request.voice_id}.wav")
        temp_output_files.append(output_path)
        
        # Map language code
        lang_id = "en" if request.language not in ["en", "fr", "pt", "es"] else request.language
        
        # Generate waveform
        with torch.no_grad():
            # Get model output
            outputs = model.inference(
                text=request.text,
                d_vector=speaker_embedding.unsqueeze(0),
                language_id=lang_id
            )
            
            # Convert to waveform
            waveform = outputs["waveform"]
            
            # Save as WAV
            ap.save_wav(waveform, output_path)
        
        # Process the generated audio
        final_audio = AudioSegment.from_file(output_path)
        final_audio = normalize_audio(final_audio)
        
        # Convert to requested format
        output_format_path = os.path.join(UPLOAD_DIR, f"temp_cloned_{request.voice_id}.{request.output_format}")
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
        # Clean up temporary files
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

@app.get("/voices")
async def list_voices():
    """Returns the list of registered voices."""
    return {
        "voices": [
            {
                "voice_id": voice_id,
                "file": os.path.basename(data["preprocessed_file"])
            } 
            for voice_id, data in voice_registry.items()
        ],
        "count": len(voice_registry)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model": "YourTTS (Mozilla TTS)", 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "registered_voices": len(voice_registry)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
