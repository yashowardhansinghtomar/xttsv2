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
from langdetect import detect

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

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✅ XTTS Model ready for voice cloning!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def chunk_text(text: str, max_length: int = 150) -> list:
    """
    Split the input text into smaller chunks. This uses textwrap to avoid breaking words.
    Adjust the max_length parameter based on your model's optimal input size.
    """
    # If the text is already short enough, return it as a single chunk.
    if len(text) <= max_length:
        return [text]
    # Otherwise, split the text using textwrap.
    return textwrap.wrap(text, width=max_length)

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """
    Convert a numpy waveform array to a pydub AudioSegment.
    """
    wav_array = np.array(wav_array, dtype=np.float32)
    pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
    return AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit audio
        frame_rate=sample_rate,
        channels=1
    )

# =============================================================================
# Voice Cloning Endpoints (XTTS)
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
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
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
    Generate voice cloned speech using the XTTS model.
    This endpoint generates an audio file from the XTTS model output in the requested format (mp3, wav, or ulaw).
    """
    print(f"Received request: {request}")
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    temp_output_files = []  # Keep track of temporary files to delete later

    try:
        # Chunk the input text if it is too long.
        text_chunks = chunk_text(request.text, max_length=150)
        print(f"Text split into {len(text_chunks)} chunks.")
        final_audio = AudioSegment.empty()

        # Process each chunk separately.
        for idx, chunk in enumerate(text_chunks):
            print(f"Processing chunk {idx+1}: {chunk}")
            wav_array = tts_model.tts(
                text=chunk,
                speaker_wav=speaker_wav,
                language=request.language
            )
            wav_array = np.array(wav_array, dtype=np.float32)
            if len(wav_array) == 0:
                raise HTTPException(status_code=500, detail="TTS model generated empty audio for a chunk")

            chunk_audio = wav_array_to_audio_segment(wav_array, sample_rate=24000)
            final_audio += chunk_audio  # Stitch the chunk together

        # Create a unique temporary output path.
        unique_hash = abs(hash(request.text + str(asyncio.get_event_loop().time())))
        output_path = f"temp_cloned_{request.voice_id}_{unique_hash}.{request.output_format}"
        temp_output_files.append(output_path)

        # Export the generated audio in the requested format.
        if request.output_format.lower() == "mp3":
            final_audio.export(output_path, format="mp3")
            with open(output_path, "rb") as audio_file:
                raw_audio = audio_file.read()
            return Response(raw_audio, media_type="audio/mpeg")
        elif request.output_format.lower() == "wav":
            final_audio.export(output_path, format="wav")
            with open(output_path, "rb") as wav_file:
                wav_bytes = wav_file.read()
            return Response(wav_bytes, media_type="audio/wav")
        elif request.output_format.lower() == "ulaw":
            # Export to WAV first.
            wav_path = output_path.replace('.ulaw', '.wav')
            final_audio.export(wav_path, format='wav')
            temp_output_files.append(wav_path)
            # Convert the WAV file to μ-law using FFmpeg.
            ulaw_path = output_path
            command = [
                'ffmpeg',
                '-y',
                '-i', wav_path,
                '-ar', '8000',
                '-ac', '1',
                '-f', 'mulaw',
                ulaw_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(ulaw_path, 'rb') as f:
                ulaw_bytes = f.read()
            return Response(
                ulaw_bytes,
                media_type="audio/mulaw",
                headers={"Content-Type": "audio/mulaw", "X-Sample-Rate": "8000"}
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid output format specified.")
    finally:
        # Clean up temporary files.
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.post("/convert_ulaw_to_wav/")
async def convert_ulaw_to_wav(file: UploadFile = File(...)):
    """
    Convert ulaw encoded audio back to WAV format.
    """
    try:
        ulaw_path = f"temp_{uuid.uuid4()}.ulaw"
        with open(ulaw_path, "wb") as f:
            f.write(await file.read())

        wav_path = ulaw_path.replace('.ulaw', '.wav')
        command = [
            'ffmpeg',
            '-y',
            '-f', 'mulaw',
            '-ar', '8000',
            '-ac', '1',
            '-i', ulaw_path,
            wav_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(wav_path, "rb") as wav_file:
            wav_bytes = wav_file.read()

        return Response(wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
    finally:
        if os.path.exists(ulaw_path):
            os.remove(ulaw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
