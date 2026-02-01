from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import httpx
import librosa
import numpy as np
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
    version="2.0.0",
    description="API for detecting AI-generated vs human-generated voice samples (Hugging Face Deployment)"
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

# Use FULL model on Hugging Face (16GB RAM available!)
USE_LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"

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

# Lazy loading for models
processor = None
model = None

def get_model():
    """Lazy load Wav2Vec2 model"""
    global processor, model
    if not USE_LIGHTWEIGHT_MODE and model is None:
        try:
            logger.info("Loading Wav2Vec2 model...")
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            model.eval()
            logger.info("Model loaded successfully")
        except ImportError as e:
            logger.warning(f"torch/transformers not available: {e}")
            return None, None
    return processor, model

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
            return response.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

def extract_audio_features(audio_bytes: bytes):
    """Convert audio to features"""
    try:
        # Load audio
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=30.0)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_variance = np.var(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Use Wav2Vec2 embeddings if available
        embeddings = None
        if not USE_LIGHTWEIGHT_MODE:
            proc, mdl = get_model()
            if proc is not None and mdl is not None:
                try:
                    import torch
                    inputs = proc(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        embeddings = mdl(**inputs).last_hidden_state
                        logger.info("ML embeddings extracted successfully")
                except Exception as e:
                    logger.warning(f"Failed to extract embeddings: {e}")
        
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
    """Enhanced AI voice detection with ML embeddings support"""
    
    # Calculate metrics
    mfcc_std = np.std(features["mfcc"])
    spectral_centroid_std = np.std(features["spectral_centroid"])
    spectral_rolloff_std = np.std(features["spectral_rolloff"])
    zcr_variance = np.var(features["zero_crossing_rate"])
    pitch_variance = features["pitch_variance"]
    
    # Enhanced scoring
    ai_score = 0.0
    confidence_factors = []
    reasons = []
    
    # Pitch analysis (30% weight)
    if pitch_variance < 500:
        ai_score += 0.30
        confidence_factors.append(0.95)
        reasons.append("very low pitch variance (highly synthetic)")
    elif pitch_variance < 2000:
        ai_score += 0.20
        confidence_factors.append(0.70)
        reasons.append("low pitch variance (possibly synthetic)")
    elif pitch_variance > 100000:
        confidence_factors.append(0.90)
        reasons.append("high natural pitch variation (human)")
    else:
        confidence_factors.append(0.75)
        reasons.append("moderate pitch variation")
    
    # MFCC analysis (25% weight)
    if mfcc_std < 10:
        ai_score += 0.25
        confidence_factors.append(0.85)
        reasons.append("uniform MFCC patterns (AI)")
    elif mfcc_std < 30:
        ai_score += 0.15
        confidence_factors.append(0.70)
        reasons.append("somewhat uniform MFCC")
    else:
        confidence_factors.append(0.85)
        reasons.append("diverse MFCC patterns (natural)")
    
    # Spectral analysis (20% weight)
    if spectral_rolloff_std < 300:
        ai_score += 0.20
        confidence_factors.append(0.80)
        reasons.append("smooth spectral (AI)")
    else:
        confidence_factors.append(0.80)
        reasons.append("natural spectral variations")
    
    # ZCR analysis (15% weight)
    if zcr_variance < 0.0005:
        ai_score += 0.15
        confidence_factors.append(0.75)
        reasons.append("consistent ZCR (AI)")
    else:
        confidence_factors.append(0.75)
        reasons.append("natural ZCR variations")
    
    # ML embeddings analysis (10% weight) - BONUS if available
    if features.get("embeddings") is not None:
        try:
            embeddings = features["embeddings"]
            embedding_variance = float(embeddings.var())
            
            if embedding_variance < 0.1:
                ai_score += 0.10
                confidence_factors.append(0.95)
                reasons.append("low embedding variance (AI)")
            else:
                confidence_factors.append(0.95)
                reasons.append("natural embedding patterns (human)")
        except Exception as e:
            logger.warning(f"Embedding analysis failed: {e}")
    
    # Calculate confidence
    avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
    
    if ai_score > 0.6:
        classification = "AI-generated"
        confidence = min(0.99, avg_confidence * (ai_score / 0.6))
    elif ai_score < 0.3:
        classification = "Human-generated"
        confidence = min(0.99, avg_confidence * ((1 - ai_score) / 0.7))
    else:
        classification = "AI-generated" if ai_score >= 0.45 else "Human-generated"
        confidence = max(0.50, avg_confidence * 0.7)
    
    model_status = "with ML model" if features.get("embeddings") is not None else "heuristic-only"
    explanation = f"{model_status.upper()}: {', '.join(reasons[:3])}. Metrics: pitch={pitch_variance:.1f}, mfcc_std={mfcc_std:.2f}, spectral_std={spectral_rolloff_std:.2f}"
    
    return classification, round(confidence, 4), explanation

@app.post("/test-voice")
async def test_voice_no_auth(request: VoiceRequest):
    """Test endpoint without authentication"""
    try:
        audio_bytes = await download_audio(str(request.audio_url))
        features = extract_audio_features(audio_bytes)
        classification, confidence, explanation = detect_ai_voice(features)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "explanation": explanation,
            "model_used": "Wav2Vec2" if features.get("embeddings") is not None else "Heuristic",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/detect-voice", response_model=VoiceResponse)
async def detect_voice(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    """Main detection endpoint with API key"""
    try:
        logger.info(f"Processing: {request.audio_url}")
        audio_bytes = await download_audio(str(request.audio_url))
        features = extract_audio_features(audio_bytes)
        classification, confidence, explanation = detect_ai_voice(features)
        
        return VoiceResponse(
            classification=classification,
            confidence=confidence,
            explanation=explanation
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "AI Voice Detection API",
        "version": "2.0.0",
        "platform": "Hugging Face Spaces",
        "model": "Wav2Vec2 (16GB RAM)" if not USE_LIGHTWEIGHT_MODE else "Lightweight",
        "endpoints": {
            "test": "/test-voice (POST, no auth)",
            "main": "/detect-voice (POST, requires API key)",
            "health": "/health (GET)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "platform": "Hugging Face",
        "lightweight_mode": USE_LIGHTWEIGHT_MODE,
        "model_loaded": model is not None,
        "ml_enabled": not USE_LIGHTWEIGHT_MODE
    }
