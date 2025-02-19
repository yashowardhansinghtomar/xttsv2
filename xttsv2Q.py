import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

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
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

# =============================================================================
# Voice Cloning Endpoints (XTTS)
# =============================================================================
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
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    output_path = f"temp_cloned_{request.voice_id}_{uuid.uuid4()}.mp3"
    
    try:
        wav_array = tts_model.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        wav_array = wav_array / np.max(np.abs(wav_array))  # Normalize
        pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()

        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        audio.export(output_path, format="mp3")

        if request.output_format.lower() == "mp3":
            with open(output_path, "rb") as audio_file:
                raw_audio = audio_file.read()
            return Response(raw_audio, media_type="audio/mpeg")
        else:
            audio = AudioSegment.from_mp3(output_path)
            if request.speed != 1.0:
                audio = audio.set_frame_rate(int(audio.frame_rate * request.speed))
            audio = audio.set_channels(1).set_frame_rate(8000)
            wav_path = output_path.replace('.mp3', '.wav')
            audio.export(wav_path, format='wav')

            if request.output_format.lower() == "wav":
                with open(wav_path, "rb") as wav_file:
                    return Response(wav_file.read(), media_type="audio/wav")
            elif request.output_format.lower() == "ulaw":
                ulaw_path = wav_path.replace('.wav', '.ulaw')
                command = ['ffmpeg', '-y', '-i', wav_path, '-ar', '8000', '-ac', '1', '-f', 'mulaw', ulaw_path]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(ulaw_path, 'rb') as f:
                    return Response(f.read(), media_type="audio/mulaw")
            else:
                raise HTTPException(status_code=400, detail="Invalid output format specified.")
    finally:
        for temp_file in [output_path, output_path.replace('.mp3', '.wav'), output_path.replace('.mp3', '.ulaw')]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
