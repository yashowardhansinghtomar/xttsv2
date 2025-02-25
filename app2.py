import os
import uuid
import asyncio
import logging
import numpy as np
import torch
import string
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydub import AudioSegment

# Import FastSpeech2_MFA and HiFi-GAN
from some_tts_library import FastSpeech2_MFA  # Replace with actual library
from some_vocoder_library import HiFiGAN  # Replace with actual HiFi-GAN import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI
app = FastAPI(title="Hindi & English Voice Cloning API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global Variables
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tts_lock = asyncio.Lock()  # Async Lock for thread safety

# =============================================================================
# Load TTS Models
# =============================================================================
def load_model(language="english"):
    """Loads the appropriate FastSpeech2_MFA & HiFi-GAN model for the given language."""
    try:
        if language == "hindi":
            model_path = "models/hindi_fastspeech2_mfa.pth"
            vocoder_path = "models/hifi_gan_hindi.pth"
        else:
            model_path = "models/english_fastspeech2_mfa.pth"
            vocoder_path = "models/hifi_gan_english.pth"

        # Load TTS and vocoder models
        tts_model = FastSpeech2_MFA.load(model_path)
        vocoder = HiFiGAN.load(vocoder_path)

        logging.info(f"✅ Loaded FastSpeech2_MFA & HiFi-GAN for {language}!")
        return tts_model, vocoder
    except Exception as e:
        logging.error(f"❌ Error initializing {language} TTS model: {e}")
        return None, None

# Load both models
tts_english, vocoder_english = load_model("english")
tts_hindi, vocoder_hindi = load_model("hindi")

# =============================================================================
# Request Models
# =============================================================================
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="english", description="Language: 'hindi' or 'english'")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Output format: mp3, wav, or ulaw")

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000) -> AudioSegment:
    """Ensures audio is at least a minimum length."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio = audio.append(silence, crossfade=50)
    return audio

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Splits text into smaller chunks."""
    sentences = text.split('. ')
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        tokens = sentence.split()
        if current_length + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], len(tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def remove_punctuation(text: str) -> str:
    """Removes punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def normalize_audio(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """Normalizes audio volume."""
    change_in_dbfs = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dbfs) if change_in_dbfs > 0 else audio

# =============================================================================
# Endpoints
# =============================================================================
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads and processes an audio file for voice cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Convert and preprocess audio
        audio = AudioSegment.from_file(upload_path)
        audio = ensure_min_length(audio)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        # Store voice data
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"✅ Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech_endpoint(request: GenerateClonedSpeechRequest):
    """Generates cloned speech from text using FastSpeech2_MFA and HiFi-GAN."""
    if request.language == "hindi":
        tts, vocoder = tts_hindi, vocoder_hindi
    else:
        tts, vocoder = tts_english, vocoder_english

    if tts is None or vocoder is None:
        raise HTTPException(status_code=500, detail=f"TTS model for {request.language} failed to initialize.")

    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]
    text_without_punctuation = remove_punctuation(request.text)
    text_chunks = chunk_text_by_sentences(text_without_punctuation, max_tokens=400)

    final_audio = AudioSegment.empty()

    async with tts_lock:
        for chunk in text_chunks:
            mel_spectrogram = tts.synthesize(chunk, speaker_wav)
            wav = vocoder.vocode(mel_spectrogram)

            chunk_audio = AudioSegment(
                data=(np.array(wav) * 32767).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=22050,
                channels=1
            )
            final_audio += chunk_audio

    output_path = f"temp_cloned_{request.voice_id}.{request.output_format}"
    final_audio.export(output_path, format=request.output_format)

    with open(output_path, "rb") as f:
        return Response(f.read(), media_type=f"audio/{request.output_format}")

# =============================================================================
# Run FastAPI Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
