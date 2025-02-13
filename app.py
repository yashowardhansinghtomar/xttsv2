import os
import uuid
import torch
import base64
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment
import subprocess

# Allowlist required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# --- Setup configurations ---
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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

def generate_cloned_speech(text: str, output_path: str, language: str, speaker_wav: str):
    """Generate cloned speech using reference audio"""
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=output_path,
        language=language
    )
    print(f"✅ Generated cloned speech: {output_path}")

def get_audio_buffer(file_path: str) -> str:
    """Read audio file and return as base64 encoded string"""
    try:
        with open(file_path, 'rb') as audio_file:
            audio_content = audio_file.read()
            return base64.b64encode(audio_content).decode('utf-8')
    except Exception as e:
        print(f"Error reading audio file: {str(e)}")
        raise

class CloneAudioRequest(BaseModel):
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

@app.post("/clone_audio/")
async def clone_audio(request: CloneAudioRequest):
    """Generate cloned speech and return audio buffer"""
    try:
        if request.voice_id not in voice_registry:
            raise HTTPException(status_code=404, detail="Voice ID not found")
        
        # Get preprocessed reference audio
        speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
        output_path = f"outputs/{request.voice_id}_cloned.wav"
        
        # Generate cloned speech
        generate_cloned_speech(
            text=request.text,
            output_path=output_path,
            language=request.language,
            speaker_wav=speaker_wav
        )
        
        # Get audio buffer as base64
        base64_audio = get_audio_buffer(output_path)
        
        # Cleanup temporary file
        os.remove(output_path)
        
        # Return the base64 encoded audio
        return JSONResponse(content={"audio_buffer": base64_audio})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloning error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
