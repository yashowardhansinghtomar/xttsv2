import os
import uuid
import torch
import struct
import audioop
import io
import base64
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment

# Allowlist required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# --- Constants ---
MIN_LENGTH_MS = 2000

# Target for μ-law conversion: 8000 Hz, mono, 8-bit (μ-law)
TARGET_SAMPLE_RATE = 8000
TARGET_CHANNELS = 1

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# In-memory registry mapping voice_id to the preprocessed audio file.
voice_registry = {}

def ensure_min_length(audio: AudioSegment, min_length_ms: int = MIN_LENGTH_MS) -> AudioSegment:
    """Ensure the audio is at least `min_length_ms` milliseconds long."""
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio

def postprocess_audio(input_path: str, output_path: str):
    """
    Postprocess the generated audio by ensuring a minimum length.
    (This writes an intermediate WAV file.)
    """
    audio = AudioSegment.from_file(input_path)
    audio = ensure_min_length(audio)
    audio.export(output_path, format="wav")
    print(f"✅ Postprocessed audio saved at: {output_path}")

def generate_cloned_speech(text: str, output_path: str, language: str, speaker_wav: str):
    """
    Generate cloned speech using the provided reference audio (speaker_wav).
    """
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=output_path, language=language)
    print(f"✅ Cloned speech generated at: {output_path}")

def convert_to_ulaw_raw_buffer(input_path: str) -> io.BytesIO:
    """
    Convert the input WAV file (from the cloning pipeline) to raw μ-law encoded bytes in-memory.
    Target format: 8000 Hz, mono, 8-bit μ-law.
    No WAV header is added—the output is raw μ-law data.
    """
    # Load audio using pydub and convert to the target format.
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
    
    # Convert the raw PCM data (assumed to be 16-bit) to μ-law encoded bytes.
    ulaw_data = audioop.lin2ulaw(audio.raw_data, audio.sample_width)
    
    buffer = io.BytesIO(ulaw_data)
    buffer.seek(0)
    print("✅ Raw μ-law encoded buffer generated in-memory")
    return buffer

# --- Load the TTS model with GPU enabled ---
print("📥 Downloading or loading XTTS model...")
tts = TTS(model_name=MODEL_NAME, gpu=True)
print("✅ Model ready for use!")

# --- Pydantic model for clone_audio request (raw JSON input) ---
class CloneAudioRequest(BaseModel):
    voice_id: str
    text: str = "आपके सभी कार्यों को सरल और सुविधाजनक बना देगा"
    language: str = "hi"

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file, ensure it meets a minimum length,
    and cache the processed file.
    """
    try:
        voice_id = str(uuid.uuid4())
        file_path = f"uploads/{voice_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process the audio to ensure a minimum length.
        audio = AudioSegment.from_file(file_path)
        audio = ensure_min_length(audio)
        preprocessed_audio_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_audio_path, format="wav")
        
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_audio_path}
        print(f"✅ Audio uploaded and processed for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

@app.post("/clone_audio/")
async def clone_audio(request: CloneAudioRequest):
    """
    Generate a cloned voice using the cached reference audio,
    postprocess the output, convert it to raw μ-law encoded bytes,
    base64-encode the μ-law data, and return it in a JSON response.
    """
    try:
        voice_id = request.voice_id
        text = request.text
        language = request.language
        
        if voice_id not in voice_registry:
            raise HTTPException(status_code=404, detail="Voice ID not found")
        
        speaker_wav = voice_registry[voice_id]["preprocessed_file"]
        cloned_temp_path = f"outputs/{voice_id}_cloned_temp.wav"
        cloned_output_path = f"outputs/{voice_id}_cloned.wav"
        
        generate_cloned_speech(text, cloned_temp_path, language, speaker_wav)
        postprocess_audio(cloned_temp_path, cloned_output_path)
        
        # Convert the processed audio to raw μ-law encoded data.
        ulaw_buffer = convert_to_ulaw_raw_buffer(cloned_output_path)
        ulaw_bytes = ulaw_buffer.getvalue()
        
        # Base64-encode the μ-law data.
        encoded_ulaw = base64.b64encode(ulaw_bytes).decode("utf-8")
        
        # Return the base64-encoded data in a JSON response.
        return JSONResponse(content={"audio_base64": encoded_ulaw})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cloned voice: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
