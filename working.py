import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
from functools import lru_cache
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoProcessor
import io

# [Previous imports remain the same]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Voice Cloning API",
    description="Optimized API for voice cloning (XTTS) with MP3, WAV, and ULAW output formats.",
    version="1.1.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# Optimization 1: Faster audio processing functions
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long using efficient processing."""
    if len(audio) < min_length_ms:
        # Create silence more efficiently
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        # Use in-memory concatenation
        return audio + silence
    return audio

def process_audio_chunk(chunk: bytes) -> AudioSegment:
    """Process audio chunk efficiently using in-memory operations."""
    return AudioSegment.from_file(io.BytesIO(chunk), format="wav")

# Optimization 2: Efficient voice sample processing
class VoiceProcessor:
    def __init__(self):
        self.cache = {}
        self.lock = Lock()

    async def process_voice_sample(self, file_content: bytes, filename: str) -> tuple[str, str]:
        """Process voice sample efficiently using in-memory operations."""
        voice_id = str(uuid.uuid4())
        
        # Process audio in memory
        audio = AudioSegment.from_file(io.BytesIO(file_content))
        
        # Normalize and ensure minimum length
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(24000)  # Set standard frame rate
        audio = ensure_min_length(audio)
        
        # Export to WAV in memory
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        
        # Save processed file
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        os.makedirs("uploads", exist_ok=True)
        
        # Write file asynchronously
        await asyncio.to_thread(
            lambda: open(preprocessed_path, "wb").write(wav_io.getvalue())
        )
        
        return voice_id, preprocessed_path

voice_processor = VoiceProcessor()

# Optimization 3: Faster voice ID creation endpoint
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Optimized audio upload endpoint with faster processing."""
    try:
        # Read file content into memory
        content = await file.read()
        
        # Process voice sample efficiently
        voice_id, preprocessed_path = await voice_processor.process_voice_sample(
            content, 
            file.filename
        )
        
        # Store in registry
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# [Rest of the previous optimized code remains the same]

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(
    request: GenerateClonedSpeechRequest,
    background_tasks: BackgroundTasks
):
    """Optimized speech generation endpoint."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        # Optimized text processing
        text_without_punctuation = remove_punctuation(request.text)
        text_chunks = chunk_text_by_sentences(text_without_punctuation, max_tokens=400)
        
        # Generate audio asynchronously
        final_audio = await generate_audio_async(text_chunks, speaker_wav, request.language)
        
        # Export in requested format
        output_path = f"temp_cloned_{request.voice_id}_{uuid.uuid4()}.{request.output_format}"
        temp_output_files.append(output_path)

        if request.output_format.lower() == "mp3":
            await asyncio.to_thread(
                lambda: final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            )
            content_type = "audio/mpeg"
        elif request.output_format.lower() == "wav":
            await asyncio.to_thread(lambda: final_audio.export(output_path, format="wav"))
            content_type = "audio/wav"
        elif request.output_format.lower() == "ulaw":
            wav_path = output_path.replace('.ulaw', '.wav')
            temp_output_files.append(wav_path)
            await asyncio.to_thread(lambda: final_audio.export(wav_path, format='wav'))
            await asyncio.to_thread(
                lambda: subprocess.run(
                    ['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path],
                    check=True
                )
            )
            content_type = "audio/mulaw"
        else:
            raise HTTPException(status_code=400, detail="Invalid output format")

        # Read and return the file
        content = await asyncio.to_thread(lambda: open(output_path, "rb").read())
        
        # Clean up in background
        background_tasks.add_task(lambda: [os.remove(f) for f in temp_output_files if os.path.exists(f)])
        
        return Response(content, media_type=content_type)
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    optimize_model()  # Initialize optimized model
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
