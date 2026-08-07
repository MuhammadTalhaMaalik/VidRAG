import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

        filename = (
            filename.replace(".webm", ".wav")
                    .replace(".m4a", ".wav")
        )

        return filename




def convert_audio_to_wav(input_path: str) -> str:
    "Convert any audio file to WAV format using pydub."
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate to 16kHz
    audio.export(output_path, format="wav")

    return output_path



def chunk_audio(wave_path: str, chunk_minute: int = 10) -> list:
    "Chunk the audio file into smaller segments."
    audio = AudioSegment.from_wav(wave_path)
    chunk_ms = chunk_minute * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wave_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    "Process the input source (YouTube URL or local audio file) and return a list of audio chunks."
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wave_path = download_youtube_audio(source)
    else:
        print("Detected local audio file. Converting to WAV format...")
        wave_path = convert_audio_to_wav(source)

    print("Chunking audio into smaller segments...")
    chunks = chunk_audio(wave_path)
    print(f"Audio processing complete. Generated {len(chunks)} chunks created.")
    return chunks