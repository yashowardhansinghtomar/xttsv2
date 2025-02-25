import os
import uuid
import asyncio
import platform
import numpy as np
import torch
import logging
import string
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Voice Cloning API", version="1.0.0")

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
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, ulaw")

os.makedirs("uploads", exist_ok=True)
voice_registry = {}
logging.info("Loading TTS model for voice cloning...")

def load_model():
    try:
        model_name = "ai4bharat/vits_rasa_13"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, trust_remote_code=True)
        logging.info("TTS Model ready!")
        return model, processor
    except Exception as e:
        logging.error(f"Error initializing TTS model: {e}")
        return None, None

model, processor = load_model()

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def generate_tts(text, speaker_wav):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="TTS model failed to initialize.")
    try:
        inputs = processor(text=text, return_tensors="pt")
        speaker_embedding = processor(audio=speaker_wav, return_tensors="pt").input_values
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)
        return speech
    except Exception as e:
        logging.error(f"TTS Generation Failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Generation Error: {str(e)}")

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
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
        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    loop = asyncio.get_event_loop()
    tts_future = loop.run_in_executor(None, generate_tts, request.text, speaker_wav)
    wav_array = await tts_future
    final_audio = AudioSegment(
        data=(np.array(wav_array) * 32767).astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=22050,
        channels=1
    )
    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    final_audio.export(output_path, format=request.output_format)
    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"audio/{request.output_format}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
