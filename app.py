import os
import uuid
import asyncio
import platform
import wave
import audioop
import subprocess
import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment
from langdetect import detect
import edge_tts

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
    title="Multilingual TTS and Voice Cloning API",
    description="API for generating text-to-speech (Edge TTS) and for voice cloning (XTTS) with speed adjustments and µ-law conversion.",
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
class TTSRequest(BaseModel):
    text: str
    language_code: str
    voice: str = "female"  # "male" or "female"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)  # Speed multiplier between 0.5x and 2.0x

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_base64: str = None
    detected_language: str = None

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

# =============================================================================
# Multilingual TTS (Edge TTS) Configuration & Helpers
# =============================================================================
LANGUAGE_MODELS = {
    # Indian Languages
    'hi': {"name": "Hindi", "male_voice": "hi-IN-MadhurNeural", "female_voice": "hi-IN-SwaraNeural"},
    'te': {"name": "Telugu", "male_voice": "te-IN-MohanNeural", "female_voice": "te-IN-ShrutiNeural"},
    'ta': {"name": "Tamil", "male_voice": "ta-IN-ValluvarNeural", "female_voice": "ta-IN-PallaviNeural"},
    'gu': {"name": "Gujarati", "male_voice": "gu-IN-NiranjanNeural", "female_voice": "gu-IN-DhwaniNeural"},
    'mr': {"name": "Marathi", "male_voice": "mr-IN-ManoharNeural", "female_voice": "mr-IN-AarohiNeural"},
    'bn': {"name": "Bengali", "male_voice": "bn-IN-BashkarNeural", "female_voice": "bn-IN-TanishaaNeural"},
    'ml': {"name": "Malayalam", "male_voice": "ml-IN-MidhunNeural", "female_voice": "ml-IN-SobhanaNeural"},
    'kn': {"name": "Kannada", "male_voice": "kn-IN-GaganNeural", "female_voice": "kn-IN-SapnaNeural"},
    'pa': {"name": "Punjabi", "male_voice": "pa-IN-GuruNeural", "female_voice": "pa-IN-SargunNeural"},
    # International Languages
    'en': {"name": "English", "male_voice": "en-US-ChristopherNeural", "female_voice": "en-US-JennyNeural"},
    'es': {"name": "Spanish", "male_voice": "es-ES-AlvaroNeural", "female_voice": "es-ES-ElviraNeural"},
    'fr': {"name": "French", "male_voice": "fr-FR-HenriNeural", "female_voice": "fr-FR-DeniseNeural"},
    'de': {"name": "German", "male_voice": "de-DE-ConradNeural", "female_voice": "de-DE-KatjaNeural"},
    'zh': {"name": "Chinese", "male_voice": "zh-CN-YunxiNeural", "female_voice": "zh-CN-XiaoxiaoNeural"},
    'ja': {"name": "Japanese", "male_voice": "ja-JP-KeitaNeural", "female_voice": "ja-JP-NanamiNeural"},
    'ko': {"name": "Korean", "male_voice": "ko-KR-InJoonNeural", "female_voice": "ko-KR-SunHiNeural"},
    'ru': {"name": "Russian", "male_voice": "ru-RU-DmitryNeural", "female_voice": "ru-RU-SvetlanaNeural"},
    'ar': {"name": "Arabic", "male_voice": "ar-SA-HamedNeural", "female_voice": "ar-SA-ZariyahNeural"},
    'tr': {"name": "Turkish", "male_voice": "tr-TR-AhmetNeural", "female_voice": "tr-TR-EmelNeural"}
}

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        detected = detect(text)
        return detected if detected in LANGUAGE_MODELS else 'en'
    except Exception:
        return 'en'

async def generate_edge_tts_voice(text: str, output_path: str, voice: str) -> bool:
    """Generate voice using Microsoft Edge TTS service."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating Edge TTS voice: {e}")
        return False

# =============================================================================
# Multilingual TTS Endpoints
# =============================================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multilingual TTS and Voice Cloning API",
        "version": "1.0.0",
        "endpoints": {
            "/languages": "Get list of supported languages for Edge TTS",
            "/generate": "Generate TTS audio (Edge TTS)",
            "/generate/stream": "Generate TTS audio with speed control (Edge TTS)",
            "/upload_audio": "Upload reference audio for voice cloning",
            "/generate_cloned_speech": "Generate voice cloned speech (XTTS)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/languages")
async def get_languages():
    """Get list of supported languages for Edge TTS."""
    return {
        "languages": [
            {"code": code, "name": info["name"], "voices": ["male", "female"]}
            for code, info in LANGUAGE_MODELS.items()
        ]
    }

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """
    Generate speech using Edge TTS and return MP3 audio.
    """
    if request.language_code not in LANGUAGE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported language code: {request.language_code}")
    
    output_path = f"temp_{request.language_code}_{hash(request.text)}.mp3"
    try:
        voice = (LANGUAGE_MODELS[request.language_code]["male_voice"]
                 if request.voice.lower() == "male"
                 else LANGUAGE_MODELS[request.language_code]["female_voice"])
        success = await generate_edge_tts_voice(request.text, output_path, voice)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        with open(output_path, "rb") as audio_file:
            raw_audio = audio_file.read()
            return Response(raw_audio, media_type="audio/mpeg")
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)

@app.post("/generate/stream")
async def generate_speech_stream(request: TTSRequest):
    """
    Generate speech with speed control using Edge TTS and return µ-law encoded audio.
    """
    if request.language_code not in LANGUAGE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported language code: {request.language_code}")
    
    output_path = f"temp_{request.language_code}_{abs(hash(request.text + str(asyncio.get_event_loop().time())))}.mp3"
    wav_path = output_path.replace('.mp3', '.wav')
    
    try:
        voice = (LANGUAGE_MODELS[request.language_code]["male_voice"]
                 if request.voice.lower() == "male"
                 else LANGUAGE_MODELS[request.language_code]["female_voice"])
        success = await generate_edge_tts_voice(request.text, output_path, voice)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        audio = AudioSegment.from_mp3(output_path)
        if request.speed != 1.0:
            original_frame_rate = audio.frame_rate
            new_frame_rate = int(original_frame_rate * request.speed)
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(original_frame_rate)
        audio = audio.set_channels(1).set_frame_rate(8000)
        audio.export(wav_path, format='wav')
        with wave.open(wav_path, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
            mu_law_data = audioop.lin2ulaw(pcm_data, 2)
            return Response(
                content=mu_law_data,
                media_type="audio/mulaw",
                headers={
                    "Content-Type": "audio/mulaw",
                    "X-Sample-Rate": "8000"
                }
            )
    finally:
        for temp_file in [output_path, wav_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# =============================================================================
# Voice Cloning (XTTS) Setup & Helpers
# =============================================================================
os.makedirs("uploads", exist_ok=True)
voice_registry = {}

print("📥 Loading XTTS model for voice cloning...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("✅ XTTS Model ready for voice cloning!")

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensure audio is at least min_length_ms milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def add_silence(audio: np.ndarray, duration_ms: int = 1000, sample_rate: int = 24000) -> np.ndarray:
    """Add silence at the end of the audio."""
    silence = np.zeros(int(duration_ms * sample_rate / 1000))
    return np.concatenate((audio, silence))

def wav_to_ulaw(wav_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert a NumPy array (float32) representing WAV data to µ-law encoded bytes using ffmpeg."""
    try:
        wav_int16 = (wav_data * 32767).astype(np.int16)
        process = subprocess.Popen(
            [
                'ffmpeg',
                '-f', 's16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-f', 'mulaw',
                '-ar', '8000',
                '-ac', '1',
                'pipe:1'
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ulaw_data, stderr = process.communicate(wav_int16.tobytes())
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {stderr.decode()}")
        return ulaw_data
    except Exception as e:
        print(f"Error converting to µ-law: {str(e)}")
        raise

# =============================================================================
# Voice Cloning Endpoints
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
    This endpoint generates an MP3 from the XTTS model output, applies speed control,
    converts the MP3 to a WAV file, and then converts it to µ-law encoded audio.
    """
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")
    
    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    output_path = f"temp_cloned_{request.voice_id}_{abs(hash(request.text + str(asyncio.get_event_loop().time())))}.mp3"
    wav_path = output_path.replace('.mp3', '.wav')
    
    try:
        # Generate speech using the XTTS voice cloning model.
        wav_array = tts_model.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )
        wav_array = np.array(wav_array, dtype=np.float32)
        if len(wav_array) == 0:
            raise HTTPException(status_code=500, detail="TTS model generated empty audio")
        
        sample_rate = tts_model.synthesizer.output_sample_rate or 24000
        # Convert float32 waveform to int16 PCM bytes.
        pcm_bytes = (wav_array * 32767).astype(np.int16).tobytes()
        
        # Create an AudioSegment from the PCM data.
        audio = AudioSegment(
            data=pcm_bytes,
            sample_width=2,  # 16-bit audio
            frame_rate=sample_rate,
            channels=1
        )
        # Export the generated audio as an MP3.
        audio.export(output_path, format="mp3")
        
        # Reload the MP3 file.
        audio = AudioSegment.from_mp3(output_path)
        # Apply speed adjustment if needed.
        if request.speed != 1.0:
            original_frame_rate = audio.frame_rate
            new_frame_rate = int(original_frame_rate * request.speed)
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(original_frame_rate)
        audio = audio.set_channels(1).set_frame_rate(8000)
        audio.export(wav_path, format='wav')
        
        with wave.open(wav_path, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
            mu_law_data = audioop.lin2ulaw(pcm_data, 2)
            return Response(
                content=mu_law_data,
                media_type="audio/mulaw",
                headers={
                    "Content-Type": "audio/mulaw",
                    "X-Sample-Rate": "8000"
                }
            )
    finally:
        # Clean up temporary files.
        for temp_file in [output_path, wav_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

# =============================================================================
# Run the Application
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
