import os
import uuid
import asyncio
import platform
import audioop
import numpy as np
import torch
import warnings
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Initialize FastAPI App
app = FastAPI(
    title="High-Quality Voice Cloning API",
    description="Optimized voice cloning with enhanced audio quality and performance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")
    quality: str = Field(default="high", description="Quality preset: high, medium, or low")

# Global settings for quality presets
QUALITY_SETTINGS = {
    "high": {
        "sample_rate": 48000,
        "mp3_bitrate": "320k",
        "chunk_size": 1000  # Larger chunks for better coherence
    },
    "medium": {
        "sample_rate": 24000,
        "mp3_bitrate": "192k",
        "chunk_size": 500
    },
    "low": {
        "sample_rate": 16000,
        "mp3_bitrate": "128k",
        "chunk_size": 250
    }
}

# Initialize model with optimizations
print("📥 Loading XTTS model with optimizations...")

def initialize_model():
    model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Cast model to half precision for faster inference
        if hasattr(model, "model"):
            model.model = model.model.half()
        
        # Move model to GPU
        device = torch.device("cuda")
        if hasattr(model, "model"):
            model.model = model.model.to(device)
    
    return model

tts_model = initialize_model()
print("✅ XTTS Model loaded with optimizations!")

# Voice registry with memory cache
voice_registry = {}
voice_cache = {}

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure minimum audio length with silence padding."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def optimize_audio_quality(audio: AudioSegment, quality_settings: dict) -> AudioSegment:
    """Apply quality optimizations to audio segment."""
    # Resample to target sample rate
    audio = audio.set_frame_rate(quality_settings["sample_rate"])
    
    # Normalize audio levels
    normalized_audio = AudioSegment.normalize(audio)
    
    # Apply subtle compression for better clarity
    threshold = -20.0
    ratio = 4.0
    attack = 5.0
    release = 50.0
    
    def compress_sample(sample):
        if abs(sample) > threshold:
            factor = (abs(sample) - threshold) / ratio
            return np.sign(sample) * (threshold + factor)
        return sample
    
    compressed_audio = normalized_audio._spawn(
        np.array([compress_sample(s) for s in normalized_audio.get_array_of_samples()], 
                dtype=normalized_audio.array_type)
    )
    
    return compressed_audio

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process audio with high quality settings
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        audio = optimize_audio_quality(audio, QUALITY_SETTINGS["high"])
        
        # Save processed audio
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")
        
        # Store in registry and cache
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        voice_cache[voice_id] = audio
        
        print(f"✅ Processed high-quality audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
    finally:
        # Cleanup temporary upload file
        if os.path.exists(upload_path):
            os.remove(upload_path)

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate high-quality cloned speech with optimized processing."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    quality_settings = QUALITY_SETTINGS[request.quality]
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    try:
        # Generate speech with optimized settings
        wav_array = tts_model.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )

        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")

        # Convert to audio segment and apply quality optimizations
        audio = wav_array_to_audio_segment(wav_array, quality_settings["sample_rate"])
        audio = optimize_audio_quality(audio, quality_settings)

        # Apply speed adjustment if needed
        if request.speed != 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * request.speed)
            })

        # Generate output in requested format
        if request.output_format.lower() == "mp3":
            output = audio.export(format="mp3", bitrate=quality_settings["mp3_bitrate"]).read()
            return Response(output, media_type="audio/mpeg")
            
        elif request.output_format.lower() == "wav":
            output = audio.export(format="wav").read()
            return Response(output, media_type="audio/wav")
            
        elif request.output_format.lower() == "ulaw":
            # Convert to 8kHz mono for μ-law encoding
            audio = audio.set_channels(1).set_frame_rate(8000)
            raw_data = audio.raw_data
            mu_law_data = audioop.lin2ulaw(raw_data, audio.sample_width)
            return Response(
                mu_law_data,
                media_type="audio/mulaw",
                headers={"X-Sample-Rate": "8000"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid output format specified")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech generation error: {str(e)}")

def wav_array_to_audio_segment(wav_array: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert numpy array to AudioSegment with optimized processing."""
    wav_array = np.array(wav_array, dtype=np.float32)
    
    # Clip values to prevent distortion
    wav_array = np.clip(wav_array, -1.0, 1.0)
    
    # Convert to 16-bit PCM
    pcm_data = (wav_array * 32767).astype(np.int16)
    
    return AudioSegment(
        pcm_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
