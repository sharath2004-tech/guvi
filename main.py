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

# Use lightweight model mode (saves 300MB+ memory)
USE_LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "true").lower() == "true"

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

# Lazy loading for models (only load if needed and not in lightweight mode)
processor = None
model = None

def get_model():
    """Lazy load model only when needed"""
    global processor, model
    if not USE_LIGHTWEIGHT_MODE and model is None:
        try:
            logger.info("Loading Wav2Vec2 model...")
            # Import here to avoid import errors when torch is not installed
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            model.eval()
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("torch/transformers not available - running in lightweight mode")
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
            
            # Skip content-type check for Google Drive files
            return response.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

def extract_audio_features(audio_bytes: bytes):
    """Convert MP3 to waveform and extract features"""
    try:
        # Load audio using librosa with lower memory settings
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=30.0)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        
        # Extract pitch variance
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_variance = np.var(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Only use Wav2Vec2 if not in lightweight mode
        embeddings = None
        if not USE_LIGHTWEIGHT_MODE:
            proc, mdl = get_model()
            if proc is not None and mdl is not None:
                try:
                    import torch
                    inputs = proc(audio_data, sampling_rate=sr, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        embeddings = mdl(**inputs).last_hidden_state
                except ImportError:
                    logger.warning("torch not available - skipping embeddings")
                    embeddings = None
        
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
    Detect if voice is AI-generated based on audio features (Enhanced Algorithm)
    
    AI-generated voices typically show:
    - Lower pitch variance (more monotone)
    - Smoother spectral characteristics
    - Less natural zero-crossing patterns
    - More uniform MFCC patterns
    - Consistent energy distribution
    """
    
    # Calculate detection metrics
    mfcc_std = np.std(features["mfcc"])
    mfcc_mean = np.mean(features["mfcc"])
    spectral_centroid_mean = np.mean(features["spectral_centroid"])
    spectral_centroid_std = np.std(features["spectral_centroid"])
    spectral_rolloff_std = np.std(features["spectral_rolloff"])
    spectral_rolloff_mean = np.mean(features["spectral_rolloff"])
    zcr_variance = np.var(features["zero_crossing_rate"])
    zcr_mean = np.mean(features["zero_crossing_rate"])
    pitch_variance = features["pitch_variance"]
    
    # Enhanced scoring system with weighted features
    ai_score = 0.0
    confidence_factors = []
    reasons = []
    
    # 1. Pitch variance analysis (30% weight) - Most reliable indicator
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
        reasons.append("high natural pitch variation (human speech)")
    else:
        confidence_factors.append(0.75)
        reasons.append("moderate pitch variation detected")
    
    # 2. MFCC pattern analysis (25% weight)
    if mfcc_std < 10:
        ai_score += 0.25
        confidence_factors.append(0.85)
        reasons.append("highly uniform MFCC patterns (AI synthesis)")
    elif mfcc_std < 30:
        ai_score += 0.15
        confidence_factors.append(0.70)
        reasons.append("somewhat uniform MFCC patterns")
    else:
        confidence_factors.append(0.85)
        reasons.append("diverse MFCC patterns (natural speech)")
    
    # 3. Spectral rolloff analysis (20% weight)
    if spectral_rolloff_std < 300:
        ai_score += 0.20
        confidence_factors.append(0.80)
        reasons.append("smooth spectral rolloff (lacks natural artifacts)")
    elif spectral_rolloff_std < 800:
        ai_score += 0.10
        confidence_factors.append(0.65)
        reasons.append("moderately smooth spectral characteristics")
    else:
        confidence_factors.append(0.80)
        reasons.append("natural spectral variations present")
    
    # 4. Zero-crossing rate variance (15% weight)
    if zcr_variance < 0.0005:
        ai_score += 0.15
        confidence_factors.append(0.75)
        reasons.append("highly consistent zero-crossing rate (AI typical)")
    elif zcr_variance < 0.002:
        ai_score += 0.08
        confidence_factors.append(0.60)
        reasons.append("somewhat consistent zero-crossing pattern")
    else:
        confidence_factors.append(0.75)
        reasons.append("natural zero-crossing variations")
    
    # 5. Spectral centroid analysis (10% weight)
    if spectral_centroid_std < 200:
        ai_score += 0.10
        confidence_factors.append(0.70)
        reasons.append("narrow spectral focus (AI synthesis characteristic)")
    elif 1500 < spectral_centroid_mean < 2500:
        ai_score += 0.05
        confidence_factors.append(0.60)
        reasons.append("spectral centroid in typical AI range")
    else:
        confidence_factors.append(0.65)
        reasons.append("spectral centroid shows natural distribution")
    
    # Calculate final confidence using average of individual confidence factors
    avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
    
    # Determine classification with enhanced confidence calculation
    if ai_score > 0.6:
        classification = "AI-generated"
        confidence = min(0.99, avg_confidence * (ai_score / 0.6))  # Scale confidence
    elif ai_score < 0.3:
        classification = "Human-generated"
        confidence = min(0.99, avg_confidence * ((1 - ai_score) / 0.7))
    else:
        # Ambiguous zone (0.3-0.6)
        classification = "AI-generated" if ai_score >= 0.45 else "Human-generated"
        confidence = max(0.50, avg_confidence * 0.7)  # Lower confidence for ambiguous cases
    
    explanation = f"Analysis: {', '.join(reasons[:3])}. Key metrics: pitch_var={pitch_variance:.1f}, mfcc_std={mfcc_std:.2f}, spectral_rolloff_std={spectral_rolloff_std:.2f}, zcr_var={zcr_variance:.6f}"
    
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
        "lightweight_mode": USE_LIGHTWEIGHT_MODE,
        "model_loaded": model is not None,
        "memory_optimized": True
    }
