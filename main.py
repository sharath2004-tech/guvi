from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import httpx
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import io
from typing import Optional
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Voice Detection API",
    version="1.0.0",
    description="API for detecting AI-generated vs human-generated voice samples"
)

# Add CORS middleware for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Configuration
API_KEY = os.getenv("API_KEY", "sk_live_abc123xyz789_secure_key_2024")

# Security scheme
security = HTTPBearer()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Completed in {process_time:.2f}s with status {response.status_code}")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Load wav2vec2 model for feature extraction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

class VoiceRequest(BaseModel):
    audio_url: HttpUrl
    message: str

class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    explanation: str

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

async def download_audio(url: str) -> bytes:
    """Download audio file from URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Skip content-type check for Google Drive files
            return response.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

def extract_audio_features(audio_bytes: bytes):
    """Convert MP3 to waveform and extract features"""
    try:
        # Load audio using librosa
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        # Extract pitch variance
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_variance = np.var(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Wav2Vec2 embeddings
        inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state
        
        return {
            "audio_data": audio_data,
            "sr": sr,
            "mfcc": mfcc,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zero_crossing_rate,
            "pitch_variance": pitch_variance,
            "embeddings": embeddings
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

def detect_ai_voice(features: dict) -> tuple[str, float, str]:
    """
    Detect if voice is AI-generated based on audio features
    
    AI-generated voices typically show:
    - Lower pitch variance (more monotone)
    - Smoother spectral characteristics
    - Less natural zero-crossing patterns
    - More uniform MFCC patterns
    """
    
    # Calculate detection metrics
    mfcc_std = np.std(features["mfcc"])
    spectral_centroid_mean = np.mean(features["spectral_centroid"])
    spectral_rolloff_std = np.std(features["spectral_rolloff"])
    zcr_variance = np.var(features["zero_crossing_rate"])
    pitch_variance = features["pitch_variance"]
    
    # Scoring system (heuristic-based for open-source approach)
    ai_score = 0.0
    reasons = []
    
    # Check pitch variance (AI voices tend to have lower variance)
    if pitch_variance < 1000:
        ai_score += 0.25
        reasons.append("low pitch variance indicating synthetic monotone")
    else:
        reasons.append("natural pitch variation detected")
    
    # Check MFCC uniformity (AI voices have more uniform patterns)
    if mfcc_std < 15:
        ai_score += 0.25
        reasons.append("uniform MFCC patterns typical of synthesis")
    else:
        reasons.append("varied MFCC patterns suggesting human speech")
    
    # Check spectral characteristics
    if spectral_rolloff_std < 500:
        ai_score += 0.2
        reasons.append("smooth spectral rolloff lacking natural artifacts")
    else:
        reasons.append("natural spectral variations present")
    
    # Check zero-crossing rate variance
    if zcr_variance < 0.001:
        ai_score += 0.15
        reasons.append("consistent zero-crossing rate typical of AI")
    else:
        reasons.append("natural zero-crossing variations")
    
    # Check spectral centroid (AI voices often have specific frequency focus)
    if 1000 < spectral_centroid_mean < 3000:
        ai_score += 0.15
        reasons.append("spectral centroid in typical AI synthesis range")
    else:
        reasons.append("spectral centroid shows natural distribution")
    
    # Determine classification
    confidence = ai_score if ai_score > 0.5 else (1 - ai_score)
    classification = "AI-generated" if ai_score > 0.5 else "Human-generated"
    explanation = f"Analysis based on: {', '.join(reasons)}. Metrics: pitch_var={pitch_variance:.2f}, mfcc_std={mfcc_std:.2f}, spectral_rolloff_std={spectral_rolloff_std:.2f}"
    
    return classification, round(confidence, 4), explanation

@app.post("/test-voice")
async def test_voice_no_auth(request: VoiceRequest):
    """
    Test endpoint without authorization - for easy testing
    """
    try:
        # Download audio
        audio_bytes = await download_audio(str(request.audio_url))
        
        # Extract features
        features = extract_audio_features(audio_bytes)
        
        # Detect AI voice
        classification, confidence, explanation = detect_ai_voice(features)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "explanation": explanation,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

@app.post("/detect-voice", response_model=VoiceResponse)
async def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect if a voice sample is AI-generated or human-generated
    
    Requires:
    - Authorization header: Bearer <API_KEY>
    - JSON body with audio_url and message
    
    Returns:
    - classification: "AI-generated" or "Human-generated"
    - confidence: float between 0 and 1
    - explanation: detailed analysis
    """
    try:
        logger.info(f"Processing request for audio: {request.audio_url}")
        
        # Download audio
        audio_bytes = await download_audio(str(request.audio_url))
        logger.info(f"Downloaded {len(audio_bytes)} bytes")
        
        # Extract features
        features = extract_audio_features(audio_bytes)
        logger.info("Features extracted successfully")
        
        # Detect AI voice
        classification, confidence, explanation = detect_ai_voice(features)
        logger.info(f"Classification: {classification} (confidence: {confidence})")
        
        return VoiceResponse(
            classification=classification,
            confidence=confidence,
            explanation=explanation
        )
    except HTTPException as he:
        logger.error(f"HTTP error: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "service": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "primary": {
                "path": "/detect-voice",
                "method": "POST",
                "auth": "Required - Bearer token in Authorization header",
                "description": "Detect if voice is AI-generated or human"
            },
            "test": {
                "path": "/test-voice",
                "method": "POST",
                "auth": "Not required",
                "description": "Test endpoint without authentication"
            },
            "health": {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            }
        },
        "documentation": "/docs",
        "api_key_format": "Authorization: Bearer <YOUR_API_KEY>"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model is not None
    }
