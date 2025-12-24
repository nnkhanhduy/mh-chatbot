import whisper
from gtts import gTTS
import uuid
import os

whisper_model = whisper.load_model("tiny")

def speech_to_text(audio_path: str) -> str:
    if audio_path is None or not os.path.exists(audio_path):
        raise ValueError("Invalid audio file")

    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()

    if text == "":
        raise ValueError("No speech detected")

    return text

def text_to_speech(text: str) -> str:
    if not text:
        raise ValueError("Empty text for TTS")

    filename = f"tts_{uuid.uuid4().hex}.wav"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)

    return filename
