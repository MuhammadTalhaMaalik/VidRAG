from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks

source = ""

chunks = process_input(source)

print(transcribe_audio_chunks(chunks))

