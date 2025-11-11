from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def record_audio(duration=5, samplerate=44100):
    """
    Record audio from the default microphone for a given duration (in seconds).
    """
    print(f"🎙️ Speak now... (recording {duration}s)")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, samplerate, audio)
    return temp_wav.name


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe speech from audio file to text using Whisper.
    """
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )
    return transcript.text


def chat_with_gpt(prompt: str) -> str:
    """
    Get GPT-4o-mini text response using new API.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise and helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def speak_text(text: str):
    """
    Generate and play TTS audio using GPT-4o-mini-tts.
    """
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            response.stream_to_file(tmp.name)
            tmp_path = tmp.name

    audio = AudioSegment.from_file(tmp_path, format="mp3")
    play(audio)
    os.remove(tmp_path)


def main():
    print("=== GPT-4o-mini CLI Chat (Voice Input + Voice Output) ===")
    print("Type 'exit' to quit.\n")

    while True:
        mode = input("Press [Enter] to speak or type your message: ").strip()

        # Handle exit
        if mode.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if mode == "":
            # Record audio input
            audio_path = record_audio(duration=6)
            user_input = transcribe_audio(audio_path)
            os.remove(audio_path)
            print(f"You said: {user_input}")
        else:
            user_input = mode

        print("Thinking...")
        reply = chat_with_gpt(user_input)

        print("\nAssistant:", reply)
        speak_text(reply)
        print()


if __name__ == "__main__":
    main()
