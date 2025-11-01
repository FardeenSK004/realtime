# Real-time AI Voice Conversation

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This project enables real-time, two-way voice conversations with OpenAI's `gpt-4o-realtime-preview` model using Python.

## Description

This application captures audio from your microphone, streams it to the OpenAI Real-time API, and plays back the AI's audio response, creating a low-latency conversational experience. It features a client-server architecture:

*   **Server (`server.py`):** A lightweight FastAPI server that securely handles your OpenAI API key. Its sole purpose is to "mint" short-lived session tokens for the client, preventing the need to expose your API key on the client-side.
*   **Client (`client.py`):** An asynchronous Python script that connects to the OpenAI WebSocket API using the session token. It handles audio recording, client-side Voice Activity Detection (VAD), streaming audio to the API, and receiving/playing back the AI's audio and text responses.

## Features

*   🗣️ **Real-time Voice Interaction:** Engage in fluid, low-latency conversations with an AI.
*   🎙️ **Client-Side VAD:** Uses a simple energy-based Voice Activity Detection to only stream audio when you are speaking, saving bandwidth.
*   ⚙️ **Configurable VAD:** Both client-side and server-side VAD parameters can be tuned for different environments and speaking styles.
*   ⚡ **Asynchronous Architecture:** Built with `asyncio`, `websockets`, and `threading` to handle I/O operations (recording, network, playback) concurrently without blocking.
*   📝 **Live Transcription:** See the AI's transcription of your speech and its text response printed to the console in real-time.
*   🔐 **Secure Token Authentication:** The server-side token minting process keeps your OpenAI API key safe.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.11+
*   An OpenAI API key.
*   Git for cloning the repository.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FardeenSK004/realtime.git
    cd realtime
    ```


2.  **Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment. The client script is configured to use a virtual environment named `rlt`.
    ```bash
    # Create the virtual environment
    python -m venv venv
    
    # Activate it
    # On macOS and Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    A `requirements.txt` file is needed to install the necessary Python packages. If one isn't present, you can create it with the command below after installing the packages manually (`pip install fastapi uvicorn python-dotenv requests websockets sounddevice numpy audioop`).
    ```bash
    pip install -r requirements.txt
    ```


## Usage

1.  **Start the Server:**
    Open a terminal, navigate to the project directory, and run the FastAPI server. It will act as the token provider for the client.
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    ```
    You should see output indicating the server is running.

2.  **Run the Client:**
    Open a *second* terminal, navigate to the same project directory, and run the client script.
    ```bash
    python client.py
    ```

3.  **Start Talking:**
    Once the client connects successfully, you will see "✅ Connected. Listening for audio...". You can now start speaking. The console will display:
    *   A `🎤 [Speech detected...]` message when you start talking.
    *   A live transcription of your speech.
    *   The AI's text response as it's being generated.
    *   The AI's audio response will be played through your speakers.

    To stop the client, press `Ctrl+C`.

## Customization

You can customize the AI's behavior and voice by modifying the `session.update` payload in `client.py`:

```python
# In client.py, inside the audio_stream() function

await ws.send(json.dumps({
    "type": "session.update",
    "session": {
        "instructions": "You are a helpful assistant.", # Change the system prompt
        "voice": "alloy",  # Other voices: echo, fable, onyx, nova, shimmer
        # ... other VAD settings
    }
}))
```

