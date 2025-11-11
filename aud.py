import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
import time

# Load the model (use "tiny" or "small" for faster results)
model = WhisperModel("small", device="cpu", compute_type="int8")

sample_rate = 16000
frame_duration = 1.0  # seconds of audio per frame
max_buffer = 10       # seconds of rolling audio
running = True

# Audio queue for cross-thread communication
audio_q = queue.Queue()

def record_audio():
    """Continuously record audio and add frames to queue."""
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        print("Listening... press Ctrl+C to stop.")
        while running:
            data, _ = stream.read(int(sample_rate * frame_duration))
            audio_q.put(np.copy(data))

def transcribe_audio():
    """Continuously process audio from queue and print live transcription."""
    buffer = np.zeros(0, dtype=np.float32)
    last_text = ""

    while running:
        if not audio_q.empty():
            frame = audio_q.get()
            frame = np.squeeze(frame)
            buffer = np.concatenate((buffer, frame))

            # Keep only recent audio (rolling window)
            if len(buffer) > sample_rate * max_buffer:
                buffer = buffer[-sample_rate * max_buffer:]

            # Transcribe buffer
            segments, _ = model.transcribe(buffer, beam_size=1)
            text = "".join([segment.text for segment in segments]).strip()

            if text and text != last_text:
                print("\r" + text, end="", flush=True)
                last_text = text

        else:
            time.sleep(0.1)

# Start recording and transcribing in parallel threads
record_thread = threading.Thread(target=record_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)

record_thread.start()
transcribe_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False
    record_thread.join()
    transcribe_thread.join()
    print("\nStopped.")
