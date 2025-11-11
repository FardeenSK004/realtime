from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio(duration=6, samplerate=44100):
    print(" Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp.name, samplerate, audio)
    return tmp.name

def transcribe_audio(path):
    with open(path, "rb") as f:
        text = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        ).text
    return text

def chat_with_gpt(prompt):
    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise and helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return reply.choices[0].message.content

def speak_text(text):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    ) as resp:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            resp.stream_to_file(tmp.name)
            path = tmp.name
    audio = AudioSegment.from_file(path, format="mp3")
    play(audio)
    os.remove(path)

def main():
    print("=== GPT-4o-mini Continuous Voice Chat ===")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            audio_path = record_audio(duration=6)
            text = transcribe_audio(audio_path)
            os.remove(audio_path)

            if not text.strip():
                print("(No speech detected, retrying...)")
                continue

            print(f"\n You said: {text}")
            print("Thinking...")
            reply = chat_with_gpt(text)
            print(f"Assistant: {reply}\n")
            speak_text(reply)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n Exiting cleanly. Goodbye!")

if __name__ == "__main__":
    main()
