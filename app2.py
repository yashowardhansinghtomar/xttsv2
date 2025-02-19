import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.api import TTS

# Initialize FastAPI
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

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning Setup
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model ready!")

# =============================================================================
# Helper Functions
# =============================================================================
def chunk_text(text: str, max_length: int = 250) -> list:
    return textwrap.wrap(text, width=max_length)

def wav_array_to_audio_segment(wav_array, sample_rate: int, speed: float) -> AudioSegment:
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    audio = AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)
    
    if speed != 1.0:
        new_frame_rate = int(sample_rate * speed)
        audio = audio.set_frame_rate(new_frame_rate)
    
    return audio

# =============================================================================
# Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        AudioSegment.from_file(upload_path).set_channels(1).export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        text_chunks = chunk_text(request.text, max_length=250)
        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        final_audio = AudioSegment.empty()

        for chunk in text_chunks:
            wav_array = tts_model.tts(
                text=chunk,
                reference_wav=speaker_wav,
                language=request.language,
                speaker="default"  # Fix: Specify a default speaker
            )

            if len(wav_array) == 0:
                raise HTTPException(status_code=500, detail="TTS model generated empty audio.")

            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate, request.speed)
            final_audio += chunk_audio

        output_path = f"output_{request.voice_id}.wav"
        temp_output_files.append(output_path)
        final_audio.export(output_path, format="wav")

        if request.output_format == "mp3":
            mp3_path = output_path.replace(".wav", ".mp3")
            final_audio.export(mp3_path, format="mp3")
            with open(mp3_path, "rb") as audio_file:
                return Response(audio_file.read(), media_type="audio/mpeg")
        elif request.output_format == "ulaw":
            ulaw_path = output_path.replace(".wav", ".ulaw")
            command = ["ffmpeg", "-y", "-i", output_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", ulaw_path]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(ulaw_path, "rb") as f:
                return Response(f.read(), media_type="audio/mulaw")

        with open(output_path, "rb") as wav_file:
            return Response(wav_file.read(), media_type="audio/wav")

    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
