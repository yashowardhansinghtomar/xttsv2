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
from torch import cuda
from typing import List
import re

warnings.filterwarnings("ignore")

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI(title="Fast Voice Cloning API")

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

# Optimize CUDA settings
if cuda.is_available():
    cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Initialize model with optimizations
print("📥 Loading XTTS model...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

if cuda.is_available():
    tts_model.to('cuda')
    if hasattr(tts_model, 'model'):
        tts_model.model = tts_model.model.half()  # Use half precision
        tts_model.model.eval()  # Set to evaluation mode

print("✅ XTTS Model loaded!")

os.makedirs("uploads", exist_ok=True)
voice_registry = {}

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences intelligently."""
    # Split on sentence endings and other natural breaks
    sentences = re.split('([.!?]+[\s]+)', text)
    # Recombine sentence endings with their sentences
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    return [s.strip() for s in result if s.strip()]

def preprocess_audio(audio: AudioSegment) -> AudioSegment:
    """Optimize audio for TTS input."""
    # Normalize to standard parameters
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(24000)
    
    # Normalize volume
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)
    
    # Ensure minimum length
    if len(audio) < 2000:
        silence = AudioSegment.silent(duration=2000 - len(audio))
        audio = audio + silence
    
    return audio

def process_batch(sentences: List[str], speaker_wav: str, language: str) -> np.ndarray:
    """Process a batch of sentences with optimized settings."""
    with torch.inference_mode():
        audio_segments = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Generate audio for sentence
            wav = tts_model.tts(
                text=sentence,
                speaker_wav=speaker_wav,
                language=language
            )
            
            if len(wav) > 0:
                audio_segments.append(wav)
        
        # Combine all segments
        if audio_segments:
            return np.concatenate(audio_segments)
        return np.array([])

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and optimize reference audio."""
    try:
        voice_id = str(uuid.uuid4())
        temp_path = f"uploads/temp_{voice_id}_{file.filename}"
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Load and optimize audio
        audio = AudioSegment.from_file(temp_path)
        audio = preprocess_audio(audio)
        
        # Save optimized audio
        processed_path = f"uploads/{voice_id}_processed.wav"
        audio.export(processed_path, format="wav", parameters=["-ar", "24000", "-ac", "1"])
        
        voice_registry[voice_id] = {"file": processed_path}
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {"voice_id": voice_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generate high-quality cloned speech with batched processing."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
        
    try:
        # Split text into sentences for better processing
        sentences = split_into_sentences(request.text)
        
        # Process all sentences
        wav_array = process_batch(
            sentences,
            voice_registry[request.voice_id]["file"],
            request.language
        )
        
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
        # Convert to audio segment
        audio = AudioSegment(
            wav_array.tobytes(),
            frame_rate=24000,
            sample_width=2,
            channels=1
        )
        
        # Apply speed adjustment
        if request.speed != 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * request.speed)
            })
        
        # Handle different output formats
        if request.output_format.lower() == "mp3":
            output = audio.export(
                format="mp3",
                parameters=["-q:a", "0", "-ar", "44100"]  # High quality MP3
            ).read()
            return Response(content=output, media_type="audio/mpeg")
            
        elif request.output_format.lower() == "wav":
            output = audio.export(
                format="wav",
                parameters=["-ar", "44100", "-sample_fmt", "s16"]  # High quality WAV
            ).read()
            return Response(content=output, media_type="audio/wav")
            
        elif request.output_format.lower() == "ulaw":
            # Convert to 8kHz for μ-law
            audio_8k = audio.set_frame_rate(8000).set_channels(1)
            ulaw_data = audioop.lin2ulaw(audio_8k.raw_data, audio_8k.sample_width)
            return Response(
                content=ulaw_data,
                media_type="audio/mulaw",
                headers={"X-Sample-Rate": "8000"}
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
