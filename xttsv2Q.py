import os
import uuid
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
import warnings

warnings.filterwarnings("ignore")

# TTS imports with safe globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI(
    title="High Quality Voice Cloning API",
    description="Fast and high-quality voice cloning with XTTS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3")

# Initialize TTS with optimizations
print("📥 Loading XTTS model...")

def setup_optimized_model():
    model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Move model to GPU and optimize
        model.to("cuda")
        model.model = torch.jit.script(model.model) if hasattr(model, 'model') else model
        
        # Set model to inference mode
        if hasattr(model, 'model'):
            model.model.eval()
            
        # Enable TensorFloat-32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    return model

tts_model = setup_optimized_model()
print("✅ XTTS Model loaded with optimizations!")

os.makedirs("uploads", exist_ok=True)
voice_registry = {}

def process_audio(audio: AudioSegment) -> AudioSegment:
    """Process audio for optimal quality"""
    # Convert to high quality settings
    audio = audio.set_frame_rate(44100)  # High sample rate
    audio = audio.set_channels(1)  # Mono for better processing
    
    # Normalize audio levels
    target_db = -20.0
    change_in_db = target_db - audio.dBFS
    normalized = audio.apply_gain(change_in_db)
    
    # Ensure minimum length
    if len(normalized) < 2000:
        silence = AudioSegment.silent(duration=2000 - len(normalized))
        normalized += silence
    
    return normalized

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process reference audio"""
    try:
        voice_id = str(uuid.uuid4())
        temp_path = f"uploads/temp_{voice_id}_{file.filename}"
        
        # Save uploaded file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process audio
        audio = AudioSegment.from_file(temp_path)
        processed_audio = process_audio(audio)
        
        # Save processed audio
        processed_path = f"uploads/{voice_id}_processed.wav"
        processed_audio.export(
            processed_path,
            format="wav",
            parameters=["-ar", "44100", "-ac", "1", "-sample_fmt", "s16"]
        )
        
        voice_registry[voice_id] = {"file": processed_path}
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {"voice_id": voice_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generate high-quality cloned speech"""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
        
    try:
        with torch.inference_mode():
            # Generate speech with optimized settings
            wav = tts_model.tts(
                text=request.text,
                speaker_wav=voice_registry[request.voice_id]["file"],
                language=request.language,
                speaker_wav_format="wav"
            )
            
            if len(wav) == 0:
                raise HTTPException(status_code=500, detail="Empty audio generated")
                
            # Convert to high-quality audio
            audio = AudioSegment(
                wav.tobytes(),
                frame_rate=24000,
                sample_width=2,
                channels=1
            )
            
            # Apply speed adjustment if needed
            if request.speed != 1.0:
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * request.speed)
                })
            
            # Process for quality
            audio = process_audio(audio)
            
            # Export in requested format
            if request.output_format.lower() == "mp3":
                output = audio.export(
                    format="mp3",
                    parameters=["-q:a", "0", "-ar", "44100"]  # Highest quality MP3
                ).read()
                return Response(output, media_type="audio/mpeg")
                
            elif request.output_format.lower() == "wav":
                output = audio.export(
                    format="wav",
                    parameters=["-ar", "44100", "-sample_fmt", "s16"]
                ).read()
                return Response(output, media_type="audio/wav")
                
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
