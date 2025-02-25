import os
import uuid
import asyncio
import platform
import numpy as np
import torch
import logging
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI App
app = FastAPI(
    title="Voice Cloning API",
    description="API for voice cloning with MP3, WAV, and ULAW output formats.",
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

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Request Model
class GenerateClonedSpeechRequest(BaseModel):
    voice_id: str
    text: str = "Hello, this is a test."
    language: str = Field(default="en", description="Language code: en or hi")
    output_format: str = Field(default="mp3", description="Desired output format: mp3, wav, or ulaw")

# Load AI4Bharat VITS model for voice cloning
logging.info("Loading AI4Bharat's VITS voice cloning model...")
model_name = "ai4bharat/vits_rasa_13"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, trust_remote_code=True)
logging.info("TTS Model ready!")

# Voice sample storage
voice_registry = {}

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Uploads an audio file and returns a voice ID for cloning."""
    try:
        voice_id = str(uuid.uuid4())
        upload_path = f"uploads/{voice_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        # Preprocess the audio
        audio = AudioSegment.from_file(upload_path)
        audio = audio.set_channels(1).set_frame_rate(22050)  # Ensure compatibility
        preprocessed_path = f"uploads/{voice_id}_preprocessed.wav"
        audio.export(preprocessed_path, format="wav")

        voice_registry[voice_id] = {"preprocessed_file": preprocessed_path}
        logging.info(f"Processed audio for voice_id: {voice_id}")

        return {"voice_id": voice_id}
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/generate_cloned_speech/")
async def generate_cloned_speech(request: GenerateClonedSpeechRequest):
    """Generates cloned speech using the uploaded voice sample."""
    if request.voice_id not in voice_registry:
        raise HTTPException(status_code=404, detail="Voice ID not found")

    speaker_wav = voice_registry[request.voice_id]["preprocessed_file"]

    try:
        # Prepare inputs for the model
        inputs = processor(text=request.text, return_tensors="pt", padding=True)
        speaker_embedding = processor(audio=speaker_wav, return_tensors="pt").input_values

        # Generate speech
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)

        # Convert numpy array to an AudioSegment
        audio = AudioSegment(
            data=(np.array(speech) * 32767).astype(np.int16).tobytes(),
            sample_width=2,
            frame_rate=22050,
            channels=1
        )

        # Save output in the desired format
        output_filename = f"cloned_speech_{uuid.uuid4()}.{request.output_format}"
        output_path = os.path.join("uploads", output_filename)
        audio.export(output_path, format=request.output_format)

        # Return the audio file
        with open(output_path, "rb") as f:
            return Response(f.read(), media_type=f"audio/{request.output_format}")

    except Exception as e:
        logging.error(f"Error generating cloned speech: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
