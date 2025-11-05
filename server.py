# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# uvicorn server:app --reload --host 0.0.0.0 --port 8000


import os
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/session")
def create_session():
    """Mint a short-lived Realtime session token"""
    if not OPENAI_API_KEY:
        return JSONResponse(
            {"error": "OPENAI_API_KEY not set"},
            status_code=500
        )
    
    resp = requests.post(
        "https://api.openai.com/v1/realtime/sessions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-realtime-preview",
            "voice": "alloy",
            "modalities": ["audio", "text"],
        },
    )
    resp.raise_for_status()
    return JSONResponse(resp.json())

@app.get("/")
def read_root():
    return {"message": "Server is running"}