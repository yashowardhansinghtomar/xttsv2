import os
import uuid
import asyncio
import re
import logging
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment
from transformers import AutoTokenizer
from TTS.api import TTS


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# FastAPI Setup
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning with MP3, WAV, and uLaw output formats.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load XTTS Model
print("📥 Loading XTTS model...")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
if torch.cuda.device_count() > 1:
    tts_model = torch.nn.DataParallel(tts_model)
print("✅ XTTS Model loaded!")

# Initialize directories and tokenizer
os.makedirs("uploads", exist_ok=True)
voice_registry = {}
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ========================= DATA MODEL =========================

class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = "en"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, ulaw, ulaw8000")

# ========================= HELPERS =========================

def remove_punctuation(text: str) -> str:
    """Remove punctuation from the input text."""
    return re.sub(r'[^\w\s]', '', text)

def chunk_text_by_sentences(text: str, max_tokens: int = 400) -> list:
    """Split text into chunks for processing."""
    sentences = text.split('. ')
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_length + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], len(tokens)
        else:
            current_chunk.append(sentence)
            current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def wav_array_to_audio_segment(wav_array, sample_rate: int) -> AudioSegment:
    """Convert numpy waveform array to AudioSegment."""
    pcm_bytes = (wav_array * 32767).astype('int16').tobytes()
    return AudioSegment(data=pcm_bytes, sample_width=2, frame_rate=sample_rate, channels=1)

async def generate_audio_chunk(text, speaker_wav, language):
    """Generate a single chunk of TTS audio."""
    model = tts_model.module if isinstance(tts_model, torch.nn.DataParallel) else tts_model
    wav_array = model.tts(text=text, speaker_wav=speaker_wav, language=language)
    return wav_array_to_audio_segment(wav_array, sample_rate=48000)

# ========================= ENDPOINTS =========================

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload a reference voice sample."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Process the audio file
        audio = AudioSegment.from_file(upload_path).set_frame_rate(24000)
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        return {"voice_id": voice_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generate high-quality cloned speech."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    # Remove punctuation
    request.text = remove_punctuation(request.text)

    # Process text in chunks
    text_chunks = chunk_text_by_sentences(request.text, max_tokens=400)
    audio_segments = await asyncio.gather(*[generate_audio_chunk(chunk, speaker_wav, request.language) for chunk in text_chunks])
    final_audio = sum(audio_segments, AudioSegment.silent(duration=100)).set_frame_rate(48000)

    # Output file setup
    output_filename = f"{request.voice_id}_cloned.{request.output_format}"
    output_path = f"uploads/{output_filename}"

    if request.output_format.lower() == "mp3":
        final_audio.export(output_path, format="mp3", parameters=["-b:a", "192k"])  # High-bitrate MP3
        return FileResponse(output_path, media_type="audio/mpeg", filename=output_filename)

    elif request.output_format.lower() == "wav":
        final_audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)

    elif request.output_format.lower() == "ulaw":
        wav_path = output_path.replace('.ulaw', '.wav')
        final_audio.export(wav_path, format='wav', parameters=["-acodec", "pcm_s16le"])

        # Convert WAV to uLaw 8000
        ulaw_path = output_path
        os.system(f'ffmpeg -y -i {wav_path} -ar 8000 -ac 1 -codec pcm_mulaw {ulaw_path}')
        return FileResponse(ulaw_path, media_type="audio/mulaw", filename=output_filename)

    elif request.output_format.lower() == "ulaw8000":
        wav_path = output_path.replace('.ulaw8000', '.wav')
        final_audio.export(wav_path, format='wav', parameters=["-acodec", "pcm_s16le"])

        # Convert to uLaw 8000 with higher precision
        ulaw8000_path = output_path
        os.system(f'ffmpeg -y -i {wav_path} -ar 8000 -ac 1 -codec pcm_mulaw {ulaw8000_path}')
        return FileResponse(ulaw8000_path, media_type="audio/mulaw", filename=output_filename)

    else:
        raise HTTPException(status_code=400, detail="Invalid output format.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
