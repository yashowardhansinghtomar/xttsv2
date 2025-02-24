import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import logging
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="Multilingual Voice Cloning API",
    description="API for voice cloning using FastSpeech2 with multiple language support.",
    version="2.0.0",
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

# Define supported models for different languages
MODEL_MAPPING = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "hi": "tts_models/hi/cmu-arctic/tacotron2-DDC",
    "fr": "tts_models/fr/css10/vits",
    "de": "tts_models/de/thorsten/tacotron2-DCA",
    "es": "tts_models/es/mai/tacotron2-DDC",
    "zh": "tts_models/zh-CN/baker/tacotron2-DDC",
}

voice_registry = {}
model_lock = Lock()

def get_tts_model(language: str):
    if language not in MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    logging.info(f"Loading model: {MODEL_MAPPING[language]}")
    tts = TTS(MODEL_MAPPING[language])
    tts.to("cuda" if torch.cuda.is_available() else "cpu")
    return tts

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str
    language: str = Field(default="en", description="Language of the speech")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="mp3, wav, or ulaw")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    sentences = text.split(".")
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

def generate_speech(text: str, speaker_wav: str, language: str) -> np.ndarray:
    with model_lock:
        tts = get_tts_model(language)
        wav_array = tts.tts(text=text, speaker_wav=speaker_wav)
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")
        sample_rate = tts.synthesizer.output_sample_rate or 24000
        return np.array(wav_array, dtype=np.float32), sample_rate

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
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
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    text_chunks = chunk_text_by_sentences(request.text)
    final_audio = AudioSegment.empty()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_speech, chunk, speaker_wav, request.language) for chunk in text_chunks]
        for future in as_completed(futures):
            wav_array, sample_rate = future.result()
            chunk_audio = AudioSegment(
                data=(wav_array * 32767).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1
            )
            final_audio = final_audio.append(chunk_audio, crossfade=50) if final_audio else chunk_audio
    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    final_audio.export(output_path, format=request.output_format)
    with open(output_path, "rb") as audio_file:
        return Response(audio_file.read(), media_type=f"audio/{request.output_format}")

if _name_ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
