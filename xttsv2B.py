import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
import string
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

# --- Safe globals for XTTS model deserialization ---
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# Initialize FastAPI App & CORS
# =============================================================================
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning (XTTS) with MP3, WAV, and ULAW output formats.",
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
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

def setup(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)  # Smooth transition
    return audio

def chunk_text_by_characters(text: str, char_limit: int = 250) -> list:
    """Split the input text into chunks based on character limit, respecting word boundaries."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > char_limit:  # +1 for space
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert a numpy waveform array to a pydub AudioSegment."""
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=1
    )

def generate_tts(text, speaker_wav, language, model):
    """Handles calling the TTS model properly."""
    return model.tts(text=text, speaker_wav=speaker_wav, language=language)

def remove_punctuation(text: str) -> str:
    """Remove all punctuation from the input text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def process_chunk(chunk, speaker_wav, language, model):
    """Process a single chunk of text to generate speech."""
    try:
        wav_array = generate_tts(chunk, speaker_wav, language, model)
        return wav_array_to_audio_segment(wav_array, sample_rate=24000)
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise

def train_model_on_directory(directory_path):
    """Train the model on multiple voices from a directory."""
    # Placeholder for training logic
    # Implement the training process using the voice samples from the directory
    logging.info(f"Training model on voices from directory: {directory_path}")

def main(rank, world_size):
    setup(rank, world_size)

    # Load the model and wrap it with DDP
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    tts_model = DDP(tts_model.to(rank), device_ids=[rank])

    # Train the model on multiple voices from a directory
    train_model_on_directory("path/to/voice/samples")

    # Load a tokenizer to split text into chunks based on token count.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # =============================================================================
    # Voice Cloning Endpoints (XTTS)
    # =============================================================================
    @app.post("/upload_audio/")
    async def upload_audio(file: UploadFile = File(...)):
        """Upload and preprocess reference audio for voice cloning. Returns a unique voice_id."""
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
            logging.info(f"Processed audio for voice_id: {voice_id}")
            return {"voice_id": voice_id}
        except Exception as e:
            logging.error(f"Upload error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

    @app.post("/generate_cloned_speech/")
    async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
        """Generate voice cloned speech using the XTTS model."""
        logging.info(f"Received request: {request}")
        if request.voice_id not in voice_registry:
            logging.error("Voice ID not found")
            raise HTTPException(status_code=404, detail="Voice ID not found")

        speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
        temp_output_files = []

        try:
            # Remove punctuation from the input text
            text_without_punctuation = remove_punctuation(request.text)
            text_chunks = chunk_text_by_characters(text_without_punctuation, char_limit=250)
            logging.info(f"Text split into {len(text_chunks)} chunks.")
            final_audio = AudioSegment.empty()

            # Process text chunks in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_chunk, chunk, speaker_wav, request.language, tts_model)
                    for chunk in text_chunks
                ]
                for idx, future in enumerate(futures):
                    chunk_audio = future.result()
                    final_audio += chunk_audio
                    if idx < len(text_chunks) - 1:
                        final_audio += AudioSegment.silent(duration=200)

            unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
            output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
            temp_output_files.append(output_path)

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
                subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path], check=True)
                with open(output_path, "rb") as ulaw_file:
                    return Response(ulaw_file.read(), media_type="audio/mulaw")
            else:
                raise HTTPException(status_code=400, detail="Invalid output format.")
        finally:
            for temp_file in temp_output_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    cleanup()

def run_ddp():
    world_size = 4  # Number of GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_ddp()
