import os
import uuid
import torch
import struct
import audioop
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydub import AudioSegment

# Allowlist required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# Set directories and ensure they exist
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# In-memory registry mapping voice_id to the preprocessed audio file.
voice_registry = {}

# -------------------------------
# Utility functions
# -------------------------------

def ensure_min_length(audio: AudioSegment, min_length_ms: int = 2000, frame_rate: int = 22050, channels: int = 1) -> AudioSegment:
    """
    Ensure the audio is at least `min_length_ms` milliseconds long and set its frame rate and channel count.
    """
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=(min_length_ms - len(audio)))
        audio += silence
    return audio.set_frame_rate(frame_rate).set_channels(channels)

def postprocess_audio(input_path: str, output_path: str, min_length_ms: int = 2000, frame_rate: int = 22050, channels: int = 1):
    """
    Postprocess the generated audio to enforce minimum length, frame rate, and channel settings.
    """
    audio = AudioSegment.from_file(input_path)
    audio = ensure_min_length(audio, min_length_ms, frame_rate, channels)
    audio.export(output_path, format="wav")
    print(f"✅ Postprocessed audio saved at: {output_path}")

def generate_cloned_speech(text: str, output_path: str, language: str, speaker_wav: str):
    """
    Generate cloned speech using the provided reference audio file as speaker_wav.
    """
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=output_path, language=language)
    print(f"✅ Cloned speech generated at: {output_path}")

def convert_to_ulaw(input_path: str, output_path: str):
    """
    Convert the input WAV file to a μ-law encoded WAV file with:
      - 8000 Hz sample rate,
      - Mono (1 channel),
      - 8-bit samples (after μ-law conversion).

    This function writes a valid WAV header manually with the μ-law format (format code 7).
    """
    # Load audio using pydub and resample to 8000 Hz, mono.
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(8000).set_channels(1)
    
    # Convert the raw PCM data to μ-law encoded data.
    ulaw_data = audioop.lin2ulaw(audio.raw_data, audio.sample_width)
    
    # Calculate sizes for WAV header.
    data_size = len(ulaw_data)
    # RIFF file size = 4 + (8 + fmt_chunk_size) + (8 + data_size) = 36 + data_size.
    file_size = 36 + data_size
    
    with open(output_path, 'wb') as f:
        # Write RIFF header.
        f.write(b'RIFF')
        f.write(struct.pack('<I', file_size))
        f.write(b'WAVE')
        # Write fmt chunk.
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))       # Subchunk1 size for PCM.
        f.write(struct.pack('<H', 7))        # Audio format 7 indicates μ-law.
        f.write(struct.pack('<H', 1))        # Number of channels.
        f.write(struct.pack('<I', 8000))     # Sample rate.
        f.write(struct.pack('<I', 8000))     # Byte rate = sample_rate * block_align (block_align=1).
        f.write(struct.pack('<H', 1))        # Block align (channels * bytes per sample).
        f.write(struct.pack('<H', 8))        # Bits per sample.
        # Write data chunk header.
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        # Write the μ-law encoded data.
        f.write(ulaw_data)
    print(f"✅ μ-law encoded audio saved at: {output_path}")

# -------------------------------
# Load the TTS model with GPU enabled
# -------------------------------
print("📥 Downloading or loading XTTS model...")
tts = TTS(model_name=MODEL_NAME, gpu=True)
print("✅ Model ready for use!")

# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/upload_audio/")
async def upload_audio(
    file: UploadFile = File(...),
    min_length_ms: int = Form(2000),
    frame_rate: int = Form(22050),
    channels: int = Form(1)
):
    """
    Upload an audio file, process it with desired settings, and cache the preprocessed file.
    """
    try:
        # Generate a unique voice_id.
        voice_id = str(uuid.uuid4())
        file_path = f"uploads/{voice_id}_{file.filename}"
        
        # Save the uploaded file.
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Load and process the audio.
        audio = AudioSegment.from_file(file_path)
        processed_audio = ensure_min_length(audio, min_length_ms, frame_rate, channels)
        preprocessed_audio_path = f"uploads/{voice_id}_preprocessed.wav"
        processed_audio.export(preprocessed_audio_path, format="wav")
        
        # Cache the preprocessed file for later use.
        voice_registry[voice_id] = {"preprocessed_file": preprocessed_audio_path}
        
        print(f"✅ Audio uploaded and processed for voice_id: {voice_id}")
        return {"voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

@app.post("/clone_audio/")
async def clone_audio(
    voice_id: str = Form(...),
    text: str = Form("आपके सभी कार्यों को सरल और सुविधाजनक बना देगा"),
    language: str = Form("hi"),
    min_length_ms: int = Form(2000),
    frame_rate: int = Form(22050),
    channels: int = Form(1)
):
    """
    Generate a cloned voice using the cached reference audio, postprocess the output, and convert it to μ-law encoding.
    """
    try:
        if voice_id not in voice_registry:
            raise HTTPException(status_code=404, detail="Voice ID not found")
        
        # Retrieve the preprocessed reference audio file.
        speaker_wav = voice_registry[voice_id]["preprocessed_file"]
        
        # Temporary path for the raw cloned audio.
        cloned_temp_path = f"outputs/{voice_id}_cloned_temp.wav"
        # Final output path after postprocessing.
        cloned_output_path = f"outputs/{voice_id}_cloned.wav"
        # Path for the μ-law encoded output.
        ulaw_output_path = f"outputs/{voice_id}_cloned_ulaw.wav"
        
        # Generate cloned speech using the cached reference audio.
        generate_cloned_speech(text, cloned_temp_path, language, speaker_wav)
        
        # Postprocess the generated audio.
        postprocess_audio(cloned_temp_path, cloned_output_path, min_length_ms, frame_rate, channels)
        
        # Convert the processed audio to μ-law encoding.
        convert_to_ulaw(cloned_output_path, ulaw_output_path)
        
        return {
            "message": "Voice cloning completed successfully!",
            "audio_file": ulaw_output_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cloned voice: {e}")

# -------------------------------
# Run the FastAPI app using Uvicorn
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
