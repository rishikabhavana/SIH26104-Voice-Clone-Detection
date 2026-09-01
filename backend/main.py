from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import APP_NAME, APP_VERSION


app = FastAPI(
    title=APP_NAME,
    description=(
        "Multilingual AI-powered caller intelligence "
        "and cyber-fraud detection platform for India."
    ),
    version=APP_VERSION
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "message": "VoiceGuard AI backend is running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": APP_NAME
    }