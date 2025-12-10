import asyncio
import websockets
import json
import base64
import sounddevice as sd
import time
import numpy as np
import requests
import queue
import threading

SERVER_URL = "http://localhost:8000/session"

# Global queue for audio playback
playback_queue = queue.Queue()

def audio_playback_thread(samplerate):
    """Separate thread for smooth audio playback"""
    stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16', blocksize=4096)
    stream.start()

    try:
        while True:
            audio_data = playback_queue.get()
            if audio_data is None:  # Poison pill to stop thread
                break
            stream.write(audio_data)
    finally:
        stream.stop()
        stream.close()

async def sender(ws, queue):
    """Sends audio from the queue to the WebSocket with minimal latency."""
    assistant_is_speaking = False
    
    try:
        while True:
            try:
                # Send audio immediately without buffering
                audio_b64, is_silent = await asyncio.wait_for(queue.get(), timeout=0.01)
                
                # Don't send audio while assistant is speaking (prevent feedback)
                if not assistant_is_speaking:
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }))

            except asyncio.TimeoutError:
                pass

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"⚠ Sender error: {e}")

async def receiver(ws, sender_control):
    """Receives messages and handles both user and assistant transcription."""
    assistant_text_buffer = ""
    is_assistant_responding = False

    try:
        async for msg in ws:
            data = json.loads(msg)
            msg_type = data.get("type")
            now = time.time()

            # Track when assistant starts/stops speaking
            if msg_type == "response.audio.delta":
                sender_control['assistant_speaking'] = True
                audio_b64 = data.get("delta")
                if audio_b64:
                    audio = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                    playback_queue.put(audio)
            
            elif msg_type == "response.audio.done":
                sender_control['assistant_speaking'] = False

            # ✅ Assistant text output
            elif msg_type == "response.text.delta":
                delta = data.get("delta", "")
                if delta:
                    if not is_assistant_responding:
                        print(f"\n🤖 [Assistant]: ", end="", flush=True)
                        is_assistant_responding = True
                    print(f"{delta}", end="", flush=True)
                    assistant_text_buffer += delta
            
            # ✅ Assistant audio transcript
            elif msg_type == "response.audio_transcript.delta":
                delta = data.get("delta", "")
                if delta:
                    if not is_assistant_responding:
                        print(f"\n🤖 [Assistant]: ", end="", flush=True)
                        is_assistant_responding = True
                    print(f"{delta}", end="", flush=True)
                    assistant_text_buffer += delta
            
            elif msg_type == "response.audio_transcript.done":
                transcript = data.get("transcript", "").strip()
                if transcript:
                    if not is_assistant_responding and not assistant_text_buffer:
                        print(f"\n🤖 [Assistant]: {transcript}")
                    # Mark as done
                    pass

            # ✅ User speech transcription (from File 1)
            elif msg_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "").strip()
                if transcript:
                    print(f"\n{'-' * 60}")
                    print(f"- YOU: {transcript}")
                    print(f"{'-' * 60}")

            # Debug: Print all message types to see what we're receiving

            # Status events
            elif msg_type == "input_audio_buffer.speech_started":
                print("\n\n👂 Listening...", flush=True)

            elif msg_type == "input_audio_buffer.speech_stopped":
                print("⚙️  Processing...", flush=True)

            elif msg_type == "session.created":
                print("\n" + "=" * 60)
                print("✅ Session established")

            elif msg_type == "session.updated":
                print("✅ Transcription enabled")
                print("=" * 60)
                print("\n🎙️ Start speaking now...\n")

            elif msg_type == "response.done":
                if is_assistant_responding:
                    print("\n")  # Add spacing after response
                    is_assistant_responding = False
                    assistant_text_buffer = ""

            elif msg_type == "error":
                error_msg = data.get("error", {}).get("message", "")
                if "buffer" not in error_msg.lower():
                    print(f"\n❌ Error: {error_msg}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"⚠ Receiver error: {e}")


async def audio_stream():
    # Fetch ephemeral token
    try:
        resp = requests.post(SERVER_URL)
        resp.raise_for_status()
        session_info = resp.json()
        print("✅ Session token received")
    except requests.exceptions.RequestException:
        print(f"❌ Error connecting to the server at {SERVER_URL}.")
        print("Please make sure the server is running with 'uvicorn server:app --reload'")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    client_secret_obj = session_info.get('client_secret', {})
    token = client_secret_obj.get('value')

    if not token:
        print("❌ Error: Could not find 'client_secret' in server response.")
        print("Full response:", json.dumps(session_info, indent=2))
        return

    realtime_ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {token}",
        "OpenAI-Beta": "realtime=v1"
    }

    print("🔌 Connecting to WebSocket...")

    samplerate = 16000
    playback_thread = threading.Thread(target=audio_playback_thread, args=(samplerate,), daemon=True)
    playback_thread.start()

    try:
        async with websockets.connect(
            realtime_ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20
        ) as ws:
            print("✅ Connected. Listening for audio...\n")

            # ✅ Enable input AND output audio transcription with FASTER VAD
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": "You are a sweet calm and friendly therapist listening to my conversations and answering my issues only and only in English language",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,              # Lower = more sensitive, faster response
                        "prefix_padding_ms": 200,      # Reduced from 300ms
                        "silence_duration_ms": 400     # Reduced from 800ms - triggers faster
                    },
                    "modalities": ["text", "audio"]
                }
            }))

            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            
            # Shared control for preventing feedback
            sender_control = {'assistant_speaking': False}

            energy_threshold = 300  # Lower threshold for faster detection

            def callback(indata, frames, time_info, status):
                # Don't process input while assistant is speaking
                if sender_control['assistant_speaking']:
                    return

                # Flatten the audio buffer and ensure correct dtype
                samples = indata[:, 0].astype(np.int16)

                # Compute RMS safely (avoid int16 overflow)
                energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                is_silent = energy < energy_threshold

                # Serialize audio to bytes for sending
                audio_bytes = samples.tobytes()
                b64 = base64.b64encode(audio_bytes).decode("utf-8")

                loop.call_soon_threadsafe(queue.put_nowait, (b64, is_silent))

            input_samplerate = 24000  # Match output samplerate for better performance
            blocksize = 512  # Smaller blocks = lower latency
            print(f"🎙️ Recording at {input_samplerate}Hz. Speak clearly when ready!\n")

            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=input_samplerate,
                dtype='int16',
                blocksize=blocksize
            ):
                sender_task = asyncio.create_task(sender(ws, queue))
                receiver_task = asyncio.create_task(receiver(ws, sender_control))

                try:
                    await asyncio.wait(
                        [sender_task, receiver_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except KeyboardInterrupt:
                    print("\n\n🛑 Stopping...")
                finally:
                    for task in [sender_task, receiver_task]:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ WebSocket connection failed with status {e.status_code}")
    except Exception as e:
        print(f"⚠ WebSocket error: {type(e).__name__}: {e}")
    finally:
        playback_queue.put(None)
        playback_thread.join(timeout=1)

if __name__ == "__main__":
    try:
        asyncio.run(audio_stream())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")