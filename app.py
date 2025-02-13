import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.api import TTS
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydub import AudioSegment
import ffmpeg

# Allowlist all required globals for safe deserialization
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

app = FastAPI()

# Set directories
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Load model directly without saving
print("📥 Downloading or loading XTTS model...")
tts = TTS(model_name=MODEL_NAME, gpu=False)
print("✅ Model ready for use!")

# Function to ensure audio has minimum length and correct format
def ensure_min_length(audio, min_length_ms=2000):
    if len(audio) < min_length_ms:
        silence = AudioSegment.silent(duration=min_length_ms - len(audio))
        audio += silence
    return audio.set_frame_rate(22050).set_channels(1)

# Function to generate cloned speech
def generate_cloned_speech(reference_audio_path, text, output_path, language='hi'):
    # Load and preprocess the reference audio
    audio = AudioSegment.from_wav(reference_audio_path)
    audio = ensure_min_length(audio)

    # Save the preprocessed reference audio
    preprocessed_audio_path = "preprocessed_reference.wav"
    audio.export(preprocessed_audio_path, format="wav")

    # Generate speech using the preprocessed reference audio
    tts.tts_to_file(text=text, speaker_wav=preprocessed_audio_path, file_path=output_path, language=language)
    print(f"✅ Cloned speech saved at: {output_path}")

# Convert the generated audio to u-law encoding while maintaining good quality
def convert_to_ulaw(input_path, output_path):
    # Resample the audio to 16000 Hz with a high bitrate for better quality
    intermediate_path = "intermediate_resampled.wav"
    (
        ffmpeg
        .input(input_path)
        .output(intermediate_path, ar=16000, ac=1, audio_bitrate="256k")
        .run(overwrite_output=True)
    )

    # Convert to 8000 Hz u-law encoding with a high bitrate for better quality
    (
        ffmpeg
        .input(intermediate_path)
        .output(output_path, ar=8000, ac=1, acodec='pcm_mulaw', audio_bitrate="256k")
        .run(overwrite_output=True)
    )
    print(f"✅ u-law encoded audio saved at: {output_path}")

# FastAPI endpoint to upload audio and generate cloned speech
@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...), text: str = "आपके सभी कार्यों को सरल और सुविधाजनक बना देगा"):
    try:
        # Save the uploaded audio file
        reference_audio_path = f"temp_{file.filename}"
        with open(reference_audio_path, "wb") as f:
            f.write(file.file.read())

        # Paths for output files
        cloned_output_path = "cloned_output2.wav"
        ulaw_output_path = "cloned_output_ulaw.wav"

        # Generate cloned speech
        generate_cloned_speech(reference_audio_path, text, cloned_output_path)

        # Convert the cloned speech to u-law encoding while maintaining quality
        convert_to_ulaw(cloned_output_path, ulaw_output_path)

        return {"message": "Voice cloning and u-law conversion completed successfully!", "audio_file": ulaw_output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

