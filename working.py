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

# Optimization 1: Cache frequently used voice samples
@lru_cache(maxsize=100)
def load_voice_sample(voice_id: str) -> np.ndarray:
    """Cache and load preprocessed voice samples."""
    file_path = voice_registry[voice_id]["preprocessed_file"]
    audio = AudioSegment.from_file(file_path)
    return np.array(audio.get_array_of_samples())

# Optimization 2: Batch processing for text chunks
class TextChunkProcessor:
    def __init__(self, batch_size: int = 3):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def create_batches(self, chunks: List[str]) -> List[List[str]]:
        """Create optimally sized batches of text chunks."""
        return [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

# Optimization 3: Parallel audio processing
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

# Optimization 4: Model optimization
def optimize_model():
    """Optimize the TTS model for inference."""
    global tts_model
    if torch.cuda.is_available():
        tts_model = tts_model.cuda()
        tts_model = torch.jit.script(tts_model)  # JIT compilation
        tts_model = torch.jit.optimize_for_inference(tts_model)
    return tts_model

# Initialize optimized components
text_processor = TextChunkProcessor()
audio_processor = AudioProcessor()
voice_registry: Dict[str, dict] = {}
tts_lock = Lock()

# Optimization 5: Asynchronous audio generation
async def generate_audio_async(text_chunks: List[str], speaker_wav: str, language: str) -> AudioSegment:
    """Generate audio asynchronously with batched processing."""
    batches = text_processor.create_batches(text_chunks)
    final_audio = AudioSegment.empty()
    
    for batch in batches:
        audio_futures = []
        for chunk in batch:
            with tts_lock:
                wav_array = tts_model.tts(text=chunk, speaker_wav=speaker_wav, language=language)
                audio_futures.append(audio_processor.process_audio_chunk(wav_array, 24000))
        
        batch_results = await asyncio.gather(*audio_futures)
        for chunk_audio in batch_results:
            if final_audio:
                final_audio = final_audio.append(chunk_audio, crossfade=25)
            else:
                final_audio = chunk_audio
    
    return final_audio

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Optimized audio upload endpoint."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        
        # Async file writing
        content = await file.read()
        await asyncio.to_thread(lambda: open(upload_path, "wb").write(content))
        
        # Process audio in background
        audio = await asyncio.to_thread(AudioSegment.from_file, upload_path)
        audio = await asyncio.to_thread(lambda: ensure_min_length(audio))
        
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        await asyncio.to_thread(lambda: audio.export(preprocessed_path, format="wav"))
        
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
    optimize_model()  # Initialize optimized model
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
