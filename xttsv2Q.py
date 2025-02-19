import os
import uuid
import torch
import numpy as np
import audioop
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
    title="Voice Cloning API",
    description="Voice cloning with multiple output formats",
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
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

# Initialize model
print("📥 Loading XTTS model...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Enable CUDA optimizations if available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    tts_model.to("cuda")

print("✅ XTTS Model loaded!")

os.makedirs("uploads", exist_ok=True)
voice_registry = {}

def process_audio(audio_data: np.ndarray, sample_rate: int = 24000) -> AudioSegment:
    """Convert numpy array to AudioSegment safely"""
    # Ensure the audio data is float32
    audio_data = np.array(audio_data, dtype=np.float32)
    
    # Clip to prevent overflow
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create AudioSegment
    return AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

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
        
        # Load and process audio
        audio = AudioSegment.from_file(temp_path)
        
        # Ensure minimum length
        if len(audio) < 2000:
            silence = AudioSegment.silent(duration=2000 - len(audio))
            audio += silence
        
        # Save processed audio
        processed_path = f"uploads/{voice_id}_processed.wav"
        audio.export(processed_path, format="wav")
        
        voice_registry[voice_id] = {"file": processed_path}
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {"voice_id": voice_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generate cloned speech in requested format"""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
        
    try:
        # Generate speech using TTS model
        with torch.inference_mode():
            wav = tts_model.tts(
                text=request.text,
                speaker_wav=voice_registry[request.voice_id]["file"],
                language=request.language
            )
            
        if len(wav) == 0:
            raise HTTPException(status_code=500, detail="Empty audio generated")
            
        # Convert to AudioSegment
        audio = process_audio(wav)
        
        # Apply speed adjustment if needed
        if request.speed != 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * request.speed)
            })
        
        # Handle different output formats
        if request.output_format.lower() == "mp3":
            # Export as high-quality MP3
            output = audio.export(
                format="mp3",
                parameters=["-q:a", "0"]  # Highest quality
            ).read()
            return Response(
                content=output,
                media_type="audio/mpeg",
                headers={"Content-Type": "audio/mpeg"}
            )
            
        elif request.output_format.lower() == "wav":
            # Export as WAV
            output = audio.export(format="wav").read()
            return Response(
                content=output,
                media_type="audio/wav",
                headers={"Content-Type": "audio/wav"}
            )
            
        elif request.output_format.lower() == "ulaw":
            # Convert to 8kHz mono for μ-law encoding
            audio = audio.set_channels(1).set_frame_rate(8000)
            
            # Get raw PCM data
            pcm_data = audio.raw_data
            
            # Convert to μ-law
            ulaw_data = audioop.lin2ulaw(pcm_data, audio.sample_width)
            
            return Response(
                content=ulaw_data,
                media_type="audio/mulaw",
                headers={
                    "Content-Type": "audio/mulaw",
                    "X-Sample-Rate": "8000"
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid output format. Use mp3, wav, or ulaw")
                
    except RecursionError:
        raise HTTPException(status_code=500, detail="Text too long, please use shorter text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
