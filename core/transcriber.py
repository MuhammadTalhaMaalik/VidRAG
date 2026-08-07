from unittest import result
from xml.parsers.expat import model

import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # Default to "small" if not set in environment variables

_model = None

 #Download and load the Whisper model only once to avoid repeated loading
def load_whisper_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model loaded successfully.")
    return _model

def transcribe_chunk(chunk_path: str , translate: bool = False) -> str: #Transcribe a single audio chunk using the Whisper model.
    "Transcribe a single audio chunk using the Whisper model."
    model = load_whisper_model()
    task="translate" if translate else "transcribe" 
    result = model.transcribe(chunk_path, task=task)

    return result["text"]

def transcribe_audio_chunks(chunks: list, translate: bool = False) -> str: #Transcribe multiple audio chunks and combine the results.
    "Transcribe multiple audio chunks and combine the results."
    full_transcription = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}")
        text = transcribe_chunk(chunk, translate=translate)
        full_transcription += text + " "

    print("Trascription completed.")

    return full_transcription
