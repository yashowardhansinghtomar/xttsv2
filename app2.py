import os
import nltk
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap
import re
nltk.download('punkt_tab')

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for high-quality voice cloning with speed control.",
    version="1.2.0"
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
    text: str
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, ulaw")

# =============================================================================
# Voice Cloning Setup
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

print("📥 Loading XTTS model...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model Ready!")

def split_text_sentences(text: str, max_length: int = 200) -> list:
    """Split text into sentences while keeping chunks within the max length limit."""
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    chunks, temp_chunk = [], ""

    for sentence in sentences:
        if len(temp_chunk) + len(sentence) < max_length:
            temp_chunk += " " + sentence
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = sentence

    if temp_chunk:
        chunks.append(temp_chunk.strip())

    return chunks

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert numpy array to a pydub AudioSegment."""
    pcm_bytes = (np.array(wav_array, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

def change_speed(audio: AudioSegment, speed: float) -> AudioSegment:
    """Change audio speed without affecting pitch using FFT resampling."""
    new_frame_rate = int(audio.frame_rate * speed)
    return audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate}).set_frame_rate(audio.frame_rate)

# =============================================================================
# API Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        AudioSegment.from_file(upload_path).export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        print(f"✅ Processed voice ID: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate cloned speech while maintaining high quality."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_files = []

    try:
        # Split text into properly structured sentences
        text_chunks = split_text_sentences(request.text, max_length=200)
        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        final_audio = AudioSegment.silent(duration=500)

        # Generate audio for each chunk
        for idx, chunk in enumerate(text_chunks):
            print(f"Processing chunk {idx+1}: {chunk}")
            wav_array = tts_model.tts(
                text=chunk,
                speaker_wav=speaker_wav,
                language=request.language
            )

            if len(wav_array) == 0:
                raise HTTPException(status_code=500, detail="TTS model generated empty audio.")

            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate)
            final_audio += chunk_audio + AudioSegment.silent(duration=200)  # Add slight pause

        # Adjust speed while preserving quality
        final_audio = change_speed(final_audio, request.speed)

        # Output filename
        output_path = f"output_{request.voice_id}.{request.output_format}"
        temp_files.append(output_path)

        # Export final file in requested format
        if request.output_format == "mp3":
            final_audio.export(output_path, format="mp3")
            return Response(open(output_path, "rb").read(), media_type="audio/mpeg")
        elif request.output_format == "wav":
            return Response(open(output_path, "rb").read(), media_type="audio/wav")
        elif request.output_format == "ulaw":
            ulaw_path = output_path.replace('.wav', '.ulaw')
            subprocess.run(["ffmpeg", "-y", "-i", output_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", ulaw_path], check=True)
            return Response(open(ulaw_path, "rb").read(), media_type="audio/mulaw")
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
