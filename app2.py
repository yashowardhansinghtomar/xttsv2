import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from TTS.tts.utils.tokenizer import TTSTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.api import TTS
from TTS.config import get_from_config_or_model_args_with_default

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="Hindi & English Voice Cloning API",
    description="API for voice cloning using FastSpeech2 with Hindi and English support.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix Windows event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize Model and Cache Directory
MODEL_NAME = "tts_models/en/ljspeech/fast_pitch"
voice_registry = {}
model_lock = Lock()

def get_tts_model():
    logging.info(f"Loading model: {MODEL_NAME}")
    tts = TTS(MODEL_NAME)
    tts.to("cuda" if torch.cuda.is_available() else "cpu")
    return tts

# Load TTS Model
tts = get_tts_model()
logging.info("✅ FastPitch Model ready!")

# Initialize Tokenizer
config_path = "path/to/model_config.json"  # Replace with your model config path
config = get_from_config_or_model_args_with_default(config_path)
tokenizer = TTSTokenizer(config=config)

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", pattern="^(en|hi)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split the input text into chunks based on sentence boundaries and token count."""
    sentences = text.split('।')  # Hindi sentence separator
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)  # Use the defined tokenizer
        if current_length + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

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

def generate_speech(text: str, speaker_wav: str) -> np.ndarray:
    """Generate speech using TTS model."""
    with model_lock:
        wav_array = tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language="hi"
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")

        sample_rate = tts.synthesizer.output_sample_rate or 24000
        return wav_array, sample_rate

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalize the audio to the target dBFS level, avoiding over-normalization."""
    current_dbfs = audio.dBFS
    if current_dbfs < target_dbfs:
        change_in_dbfs = target_dbfs - current_dbfs
        return audio.apply_gain(change_in_dbfs)
    return audio

# =============================================================================
# Voice Cloning Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and process reference audio for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = os.path.join("uploads", f"{voice_id}_{file.filename}")
        os.makedirs("uploads", exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = os.path.join("uploads", f"{voice_id}_preprocessed.wav")
        audio.export(preprocessed_path, format="wav")
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"Processed audio for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generate voice cloned speech."""
    logging.info(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    output_path = None  # ✅ Initialize output_path before the try block

    try:
        text_chunks = chunk_text_by_sentences(request.text, max_tokens=400)
        logging.info(f"Text split into {len(text_chunks)} chunks.")

        final_audio = AudioSegment.empty()

        # Process chunks in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_speech, chunk, speaker_wav) for chunk in text_chunks]

            for future in as_completed(futures):
                wav_array, sample_rate = future.result()
                chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate)
                chunk_audio = normalize_audio(chunk_audio)

                if final_audio:
                    final_audio = final_audio.append(chunk_audio, crossfade=50)
                else:
                    final_audio = chunk_audio

        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"

        if request.output_format == "mp3":
            final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            with open(output_path, "rb") as audio_file:
                return Response(audio_file.read(), media_type="audio/mpeg")
        elif request.output_format == "wav":
            final_audio.export(output_path, format="wav")
            with open(output_path, "rb") as wav_file:
                return Response(wav_file.read(), media_type="audio/wav")
        elif request.output_format == "ulaw":
            wav_path = output_path.replace('.ulaw', '.wav')
            final_audio.export(wav_path, format='wav')
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path], check=True)
            with open(output_path, "rb") as ulaw_file:
                return Response(ulaw_file.read(), media_type="audio/mulaw")
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")
    finally:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
