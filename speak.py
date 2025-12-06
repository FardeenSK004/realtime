import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import simpleaudio as sa

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def main():
    print("OpenAI Chatbot (type 'exit' to quit)\n")
    history = []
    counter = 1

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye")
            break

        history.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {"role": "system", "content": "you are NOA, an AI assistant that responds with both text and audio."
                
                },
                *history
            ]
        )

        message = completion.choices[0].message
        text = message.content[0].text
        print(f"NOA: {text}\n")

        # save and play audio
        audio_data = message.audio.data
        wav_bytes = base64.b64decode(audio_data)
        filename = f"reply_{counter}.wav"
        with open(filename, "wb") as f:
            f.write(wav_bytes)

        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        counter += 1
        history.append({"role": "assistant", "content": text})

if __name__ == "__main__":
    main()
