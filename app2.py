import os
import uuid
import asyncio
import platform
import subprocess
import numpy as np
import torch
import textwrap

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
    speed: float = Field(default=1.0, ge=0.5, le=2.0)  # User-defined speed
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# =============================================================================
# Voice Cloning Setup
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model ready for voice cloning!")

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def chunk_text(text: str, max_length: int = 250) -> list:
    """
    Split the input text into smaller chunks. This uses textwrap to avoid breaking words.
    A longer chunk size ensures better flow and reduces voice degradation.
    """
    return textwrap.wrap(text, width=max_length)

def wav_array_to_audio_segment(wav_array, sample_rate: int, speed: float) -> AudioSegment:
    """
    Convert a numpy waveform array to a pydub AudioSegment with speed control.
    """
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    audio = AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=1
    )

    # Adjust playback speed without making it sound unnatural
    if speed != 1.0:
        new_frame_rate = int(sample_rate * speed)
        audio = audio.set_frame_rate(new_frame_rate)

    return audio

# =============================================================================
# Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload and preprocess reference audio for voice cloning.
    Returns a unique voice_id.
    """
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Convert to WAV if necessary
        audio = AudioSegment.from_file(upload_path).set_channels(1)
        audio = ensure_min_length(audio)  # Ensure minimum length

        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        print(f"✅ Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """
    Generate voice-cloned speech using the XTTS model.
    Maintains good audio quality for long text and allows natural speed control.
    """
    print(f"Received request: {request}")
    
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []

    try:
        # Split text into chunks to prevent quality degradation
        text_chunks = chunk_text(request.text, max_length=250)
        print(f"Text split into {len(text_chunks)} chunk(s).")

        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        final_audio = AudioSegment.empty()

        for idx, chunk in enumerate(text_chunks):
            print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk}")
            
            # Generate speech
            wav_array = tts_model.tts(
                text=chunk,
                reference_wav=speaker_wav,
                language=request.language,
                speaker="default"
            )
            
            if len(wav_array) == 0:
                raise HTTPException(status_code=500, detail="TTS model generated empty audio.")

            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate, request.speed)
            final_audio += chunk_audio  # Stitch chunks together

        # Create a unique output file
        output_path = f"output_{request.voice_id}.wav"
        temp_output_files.append(output_path)

        final_audio.export(output_path, format="wav")
        
        # Convert to requested format
        if request.output_format == "mp3":
            mp3_path = output_path.replace(".wav", ".mp3")
            final_audio.export(mp3_path, format="mp3")
            with open(mp3_path, "rb") as audio_file:
                return Response(audio_file.read(), media_type="audio/mpeg")
        elif request.output_format == "ulaw":
            ulaw_path = output_path.replace(".wav", ".ulaw")
            command = ["ffmpeg", "-y", "-i", output_path, "-ar", "8000", "-ac", "1", "-f", "mulaw", ulaw_path]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(ulaw_path, "rb") as f:
                return Response(f.read(), media_type="audio/mulaw")

        # Default WAV response
        with open(output_path, "rb") as wav_file:
            return Response(wav_file.read(), media_type="audio/wav")

    finally:
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
