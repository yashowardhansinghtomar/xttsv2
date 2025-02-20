import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
import aiofiles

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI App & CORS
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

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = Lock()

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

if torch.cuda.device_count() > 1:
    tts_model = torch.nn.DataParallel(tts_model)

print("✅ XTTS Model ready for voice cloning!")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
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
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )

async def generate_tts(text, speaker_wav, language):
    loop = asyncio.get_event_loop()
    with tts_lock:
        model = tts_model.module if isinstance(tts_model, torch.nn.DataParallel) else tts_model
        return await loop.run_in_executor(None, model.tts, text, speaker_wav, language)

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    current_dbfs = audio.dBFS
    if current_dbfs < target_dbfs:
        change_in_dbfs = target_dbfs - current_dbfs
        return audio.apply_gain(change_in_dbfs)
    return audio

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        async with aiofiles.open(upload_path, "wb") as f:
            await f.write(await file.read())
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
    logging.info(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        logging.error("Voice ID not found")
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        text_without_punctuation = remove_punctuation(request.text)
        text_chunks = chunk_text_by_sentences(text_without_punctuation, max_tokens=400)
        logging.info(f"Text split into {len(text_chunks)} chunks.")

        futures = [generate_tts(chunk, speaker_wav, request.language) for chunk in text_chunks]
        results = await asyncio.gather(*futures)

        final_audio = AudioSegment.empty()
        for wav_array in results:
            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate=24000)
            chunk_audio = normalize_audio(chunk_audio)

            if final_audio:
                final_audio = final_audio.append(chunk_audio, crossfade=50)
            else:
                final_audio = chunk_audio

        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3", parameters=["-q:a", "0"])
            async with aiofiles.open(output_path, "rb") as audio_file:
                return Response(await audio_file.read(), media_type="audio/mpeg")
        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
            async with aiofiles.open(output_path, "rb") as wav_file:
                return Response(await wav_file.read(), media_type="audio/wav")
        elif request.output_format.lower() == "ulaw":
            wav_path = output_path.replace('.ulaw', '.wav')
            final_audio.export(wav_path, format='wav')
            temp_output_files.append(wav_path)
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', output_path], check=True)
            async with aiofiles.open(output_path, "rb") as ulaw_file:
                return Response(await ulaw_file.read(), media_type="audio/mulaw")
        else:
            raise HTTPException(status_code=400, detail="Invalid output format.")
    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, r
