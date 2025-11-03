import asyncio
import audioop
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
    """Sends audio from the queue to the WebSocket."""
    is_speaking = False
    silence_frames = 0
    # 🌟 INCREASED: Need more silence frames (was 3). 
    # At 16000Hz, blocksize 1024, 1 frame is 64ms.
    # 15 frames * 64ms ≈ 960ms (close to server's 800ms)
    silence_threshold = 15  

    try:
        buffer = []
        last_commit_time = time.time()
        # 🌟 DECREASED: Commit more frequently while speaking for better latency.
        commit_interval = 0.2  

        while True:
            try:
                # Get audio with timeout to allow periodic commits
                # The timeout is key to processing the periodic commit
                audio_b64, is_silent = await asyncio.wait_for(queue.get(), timeout=0.05) # Lower timeout
                
                # ... (rest of the VAD logic remains the same)
                if not is_silent:
                    is_speaking = True
                    silence_frames = 0
                    buffer.append(audio_b64)
                elif is_speaking:
                    silence_frames += 1
                    if silence_frames >= silence_threshold:
                        is_speaking = False

            except asyncio.TimeoutError:
                pass

            current_time = time.time()
            
            # Commit logic:
            commit_ready = (current_time - last_commit_time) >= commit_interval
            commit_due_to_silence = buffer and not is_speaking
            
            if buffer and (commit_ready or commit_due_to_silence):
                # Send the entire buffer of audio chunks
                for audio_chunk in buffer:
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_chunk
                    }))
                
                # 📢 The key difference: Only commit if speech has stopped!
                # Otherwise, just append the data and wait for the next interval/turn.
                if commit_due_to_silence:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    # Reset is_speaking to ensure we don't commit again until new speech
                    is_speaking = False 
                    print(" [Input committed]", flush=True) # Debug/confirmation print
                
                buffer = []
                last_commit_time = current_time

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Sender error: {e}")

async def receiver(ws):
    """Receives messages from the WebSocket and processes them."""
    try:
        async for msg in ws:
            data = json.loads(msg)
            msg_type = data.get("type")

            if msg_type == "response.text.delta":
                print(data.get("delta", ""), end="", flush=True)
            elif msg_type == "response.audio.delta":
                audio_b64 = data.get("delta")
                if audio_b64:
                    audio = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                    playback_queue.put(audio)
            elif msg_type == "response.audio_transcript.delta":
                print(data.get("delta", ""), end="", flush=True)
            elif msg_type == "input_audio_buffer.speech_started":
                print("\n [Speech detected...]", flush=True)
            elif msg_type == "input_audio_buffer.speech_stopped":
                print(" [Processing...]", flush=True)
            elif msg_type == "error":
                error_msg = data.get('error', {}).get('message', '')
                if 'buffer' not in error_msg.lower():
                    print(f"\n❌ Error: {error_msg}")
            elif msg_type == "session.created":
                print(" Session established")
            elif msg_type == "session.updated":
                print(" Session configured with adjusted VAD")
            elif msg_type == "response.done":
                print("\n")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ Receiver error: {e}")

async def audio_stream():
    # 1. Fetch ephemeral token just before connecting
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

    # Extract token
    client_secret_obj = session_info.get('client_secret', {})
    token = client_secret_obj.get('value')

    if not token:
        print("❌ Error: Could not find 'client_secret' in server response.")
        print("Full response:", json.dumps(session_info, indent=2))
        return

    # Construct WebSocket URL
    realtime_ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = {
        "Authorization": f"Bearer {token}",
        "OpenAI-Beta": "realtime=v1"
    }

    print("🔌 Connecting to WebSocket...")

    samplerate = 24000
    playback_thread = threading.Thread(target=audio_playback_thread, args=(samplerate,), daemon=True)
    playback_thread.start()

    try:
        async with websockets.connect(
            realtime_ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20
        ) as ws:
            print("✅ Connected. Listening for audio...")

            # Configure session with LESS sensitive VAD settings
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": "You are a sweet calm and friendly therapist listening to my converstaions and answering my issues only and only  in English language",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.85,           # Even less sensitive (higher = less triggers)
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800   # Wait a bit longer before ending speech
                    }
                }
            }))

            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            # Client-side VAD parameters
            energy_threshold = 400  # Increased threshold to reduce sensitivity to background noise

            def callback(indata, frames, time_info, status):
                energy = audioop.rms(indata, 2)  # 2 is for 16-bit audio
                is_silent = energy < energy_threshold

                b64 = base64.b64encode(indata.tobytes()).decode("utf-8")
                loop.call_soon_threadsafe(queue.put_nowait, (b64, is_silent))

            input_samplerate = 16000
            blocksize = 1024
            print(f"🎤 Recording at {input_samplerate}Hz. Speak clearly when ready!\n")

            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=input_samplerate,
                dtype='int16',
                blocksize=blocksize
            ):
                sender_task = asyncio.create_task(sender(ws, queue))
                receiver_task = asyncio.create_task(receiver(ws))

                try:
                    await asyncio.wait(
                        [sender_task, receiver_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except KeyboardInterrupt:
                    print("\n\n Stopping...")
                finally:
                    for task in [sender_task, receiver_task]:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    except websockets.exceptions.InvalidStatusCode as e:
        print(f" WebSocket connection failed with status {e.status_code}")
    except Exception as e:
        print(f" WebSocket error: {type(e).__name__}: {e}")
    finally:
        # Stop playback thread
        playback_queue.put(None)
        playback_thread.join(timeout=1)

if __name__ == "__main__":
    try:
        asyncio.run(audio_stream())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
