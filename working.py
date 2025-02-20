import os
import io
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
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoProcessor

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize voice registry and create uploads directory
voice_registry: Dict[str, Dict[str, str]] = {}
os.makedirs("uploads", exist_ok=True)

app = FastAPI(
    title="Voice Cloning API",
    description="Optimized API for voice cloning (XTTS) with MP3, WAV, and ULAW output formats.",
    version="1.1.0"
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

# Initialize TTS model
print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Optimize model if multiple GPUs are available
if torch.cuda.device_count() > 1:
    tts_model = torch.nn.DataParallel(tts_model)

print("✅ XTTS Model ready for voice cloning!")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tts_lock = Lock()

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long using efficient processing."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        return audio + silence
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> List[str]:
    """Split text into chunks based on token count and sentence boundaries."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_length + len(tokens) > max_tokens:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_length = len(tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(tokens)

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text efficiently."""
    return text.translate(str.maketrans("", "", string.punctuation))

class VoiceProcessor:
    def __init__(self):
        self.cache = {}
        self.lock = Lock()

    async def process_voice_sample(self, file_content: bytes, filename: str) -> Tuple[str, str]:
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
        
        # Write file asynchronously
        await asyncio.to_thread(
            lambda: open(preprocessed_path, "wb").write(wav_io.getvalue())
        )
        
        return voice_id, preprocessed_path

class AudioProcessor:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
    
    async def process_audio_chunk(self, wav_array: np.ndarray, sample_rate: int) -> AudioSegment:
        """Process audio chunks in parallel."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_chunk,
            wav_array,
            sample_rate
        )
    
    @staticmethod
    def _process_chunk(wav_array: np.ndarray, sample_rate: int) -> AudioSegment:
        wav_array = np.array(wav_array, dtype=np.float32)
        pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
        return AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )

# Initialize processors
voice_processor = VoiceProcessor()
audio_processor = AudioProcessor()

async def generate_audio_async(text_chunks: List[str], speaker_wav: str, language: str) -> AudioSegment:
    """Generate audio asynchronously with batched processing."""
    final_audio = AudioSegment.empty()
    
    for chunk in text_chunks:
        with tts_lock:
            wav_array = tts_model.tts(text=chunk, speaker_wav=speaker_wav, language=language)
            chunk_audio = await audio_processor.process_audio_chunk(wav_array, 24000)
            
            if final_audio:
                final_audio = final_audio.append(chunk_audio, crossfade=25)
            else:
                final_audio = chunk_audio
    
    return final_audio

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
