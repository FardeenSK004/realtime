from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # Play audio via pydub
    audio = AudioSegment.from_file(tmp_path, format="mp3")
    play(audio)
    os.remove(tmp_path)

def main():
    print("=== GPT-4o-mini CLI Chat (Fedora Compatible, v1 API) ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print("Thinking...")
        reply = chat_with_gpt(user_input)

        print("\nAssistant:", reply)
        speak_text(reply)
        print()

if __name__ == "__main__":
    main()