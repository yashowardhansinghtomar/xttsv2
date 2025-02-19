import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from langdetect import detect

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Initialize FastAPI App & CORS
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning (XTTS) with optional output formats (mp3, wav, ulaw).",
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
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# Setup
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model ready for voice cloning!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert a numpy waveform array to a pydub AudioSegment."""
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and preprocess reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        print(f"✅ Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate voice cloned speech using the XTTS model."""
    print(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        # Process the entire text in one go
        wav_array = tts_model.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )
        
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")

        # Convert to audio segment
        final_audio = wav_array_to_audio_segment(wav_array, sample_rate=24000)

        # Adjust speed if needed
        if request.speed != 1.0:
            final_audio = final_audio._spawn(final_audio.raw_data, overrides={
                "frame_rate": int(final_audio.frame_rate * request.speed)
            })

        # Create output path
        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        # Handle different output formats
        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            with open(output_path, "rb") as audio_file:
                return Response(audio_file.read(), media_type="audio/mpeg")
                
        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
            with open(output_path, "rb") as wav_file:
                return Response(wav_file.read(), media_type="audio/wav")
                
        elif request.output_format.lower() == "ulaw":
            wav_path = output_path.replace('.ulaw', '.wav')
            final_audio.export(wav_path, format='wav')
            temp_output_files.append(wav_path)
            
            command = [
                'ffmpeg',
                '-y',
                '-i', wav_path,
                '-ar', '8000',
                '-ac', '1',
                '-f', 'mulaw',
                output_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            with open(output_path, 'rb') as f:
                return Response(
                    f.read(),
                    media_type="audio/mulaw",
                    headers={"Content-Type": "audio/mulaw", "X-Sample-Rate": "8000"}
                )
        else:
            raise HTTPException(status_code=400, detail="Invalid output format specified.")
            
    finally:
        # Clean up temporary files
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
