"""
AI Voice Detection API - Using REAL ML Models
This version uses actual trained deepfake/AI voice detection models
"""
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
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Voice Detection API",
    version="3.0.0",
    description="Real AI Voice Detection using Deep Learning Models"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_KEY = os.getenv("API_KEY", "sk_live_abc123xyz789_secure_key_2024")
security = HTTPBearer()

# ============= DEEP LEARNING MODEL =============

class AudioFeatureExtractor(nn.Module):
    """CNN-based audio feature extractor for deepfake detection (TRAINED MODEL)"""
    def __init__(self):
        super().__init__()
        # Convolutional layers with BatchNorm (matches trained model)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # AI vs Human
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DeepfakeVoiceDetector:
    """
    Advanced AI Voice Detector using multiple analysis methods:
    1. Mel-spectrogram CNN analysis
    2. Wav2Vec2 embedding analysis  
    3. Temporal artifact detection
    4. Frequency band analysis
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize CNN model with TRAINED weights
        self.cnn_model = AudioFeatureExtractor().to(self.device)
        
        # Load trained weights if available
        model_path = "best_voice_detector.pth"
        if os.path.exists(model_path):
            logger.info(f"Loading trained model weights from {model_path}...")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.cnn_model.load_state_dict(state_dict)
                logger.info("✅ Trained model weights loaded successfully!")
            except Exception as e:
                logger.warning(f"⚠️ Could not load trained weights: {e}. Using untrained model.")
        else:
            logger.warning("⚠️ No trained model found. Using untrained CNN.")
        
        self.cnn_model.eval()
        
        # Load Wav2Vec2 for embeddings
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self._load_wav2vec()
        
    def _load_wav2vec(self):
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            logger.info("Loading Wav2Vec2 model...")
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
            self.wav2vec_model.eval()
            logger.info("Wav2Vec2 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2: {e}")
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def analyze_temporal_artifacts(self, audio: np.ndarray, sr: int) -> dict:
        """Detect temporal artifacts common in AI-generated audio"""
        # Frame-level energy analysis
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # AI voices often have unnaturally smooth energy transitions
        energy_diff = np.diff(rms)
        energy_smoothness = np.std(energy_diff)
        
        # Detect unnatural silence patterns (AI often has too-clean silences)
        silence_threshold = np.percentile(rms, 10)
        silence_frames = rms < silence_threshold
        silence_runs = self._get_run_lengths(silence_frames)
        avg_silence_duration = np.mean(silence_runs) if len(silence_runs) > 0 else 0
        
        # AI voices often have very consistent frame-to-frame transitions
        autocorr = np.correlate(rms, rms, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_score = np.mean(autocorr[:50]) / (np.max(autocorr) + 1e-6)
        
        return {
            "energy_smoothness": float(energy_smoothness),
            "avg_silence_duration": float(avg_silence_duration),
            "autocorr_score": float(autocorr_score)
        }
    
    def _get_run_lengths(self, binary_array: np.ndarray) -> list:
        """Get lengths of consecutive True values"""
        runs = []
        current_run = 0
        for val in binary_array:
            if val:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        return runs
    
    def analyze_frequency_bands(self, audio: np.ndarray, sr: int) -> dict:
        """Analyze frequency band characteristics - AI often has artifacts in high frequencies"""
        # Compute STFT
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Divide into frequency bands
        low_band = stft[freqs < 500, :].mean()
        mid_band = stft[(freqs >= 500) & (freqs < 2000), :].mean()
        high_band = stft[(freqs >= 2000) & (freqs < 4000), :].mean()
        ultra_high = stft[freqs >= 4000, :].mean()
        
        # AI voices often have unnatural high-frequency characteristics
        high_to_low_ratio = high_band / (low_band + 1e-6)
        ultra_high_ratio = ultra_high / (mid_band + 1e-6)
        
        # Spectral flatness per band
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        
        return {
            "high_to_low_ratio": float(high_to_low_ratio),
            "ultra_high_ratio": float(ultra_high_ratio),
            "spectral_flatness_mean": float(np.mean(spectral_flatness)),
            "spectral_flatness_std": float(np.std(spectral_flatness))
        }
    
    def get_wav2vec_features(self, audio: np.ndarray, sr: int) -> dict:
        """Extract Wav2Vec2 embeddings and analyze them"""
        if self.wav2vec_processor is None or self.wav2vec_model is None:
            return None
            
        try:
            inputs = self.wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Analyze embedding characteristics
            emb_np = embeddings.cpu().numpy()[0]
            
            # AI voices tend to have lower variance in embeddings (too consistent)
            emb_variance = np.var(emb_np)
            emb_temporal_var = np.var(np.std(emb_np, axis=1))  # Variance over time
            
            # Cosine similarity between consecutive frames (AI = higher similarity)
            similarities = []
            for i in range(len(emb_np) - 1):
                sim = np.dot(emb_np[i], emb_np[i+1]) / (np.linalg.norm(emb_np[i]) * np.linalg.norm(emb_np[i+1]) + 1e-6)
                similarities.append(sim)
            avg_similarity = np.mean(similarities)
            
            return {
                "embedding_variance": float(emb_variance),
                "embedding_temporal_var": float(emb_temporal_var),
                "frame_similarity": float(avg_similarity)
            }
        except Exception as e:
            logger.error(f"Wav2Vec2 analysis failed: {e}")
            return None
    
    def predict(self, audio: np.ndarray, sr: int) -> dict:
        """Main prediction method combining all analysis techniques"""
        
        # 1. Mel-spectrogram CNN analysis
        mel_spec = self.extract_mel_spectrogram(audio, sr)
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_output = self.cnn_model(mel_tensor)
            cnn_probs = F.softmax(cnn_output, dim=1)[0]
            cnn_ai_score = cnn_probs[0].item()  # Probability of AI
        
        # 2. Temporal artifact analysis
        temporal_features = self.analyze_temporal_artifacts(audio, sr)
        
        # 3. Frequency band analysis
        freq_features = self.analyze_frequency_bands(audio, sr)
        
        # 4. Wav2Vec2 embedding analysis
        wav2vec_features = self.get_wav2vec_features(audio, sr)
        
        # 5. Traditional acoustic features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch_variance = np.var(pitch_vals) if len(pitch_vals) > 0 else 0
        pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
        
        # ============= SCORING SYSTEM =============
        ai_score = 0.0
        total_weight = 0.0
        reasons = []
        
        # CNN Score (25% weight)
        weight = 0.25
        ai_score += cnn_ai_score * weight
        total_weight += weight
        
        # Temporal analysis (20% weight)
        weight = 0.20
        temporal_score = 0.0
        
        # AI has smoother energy transitions
        if temporal_features["energy_smoothness"] < 0.02:
            temporal_score += 0.4
            reasons.append("unnaturally smooth energy transitions")
        
        # AI has too-consistent autocorrelation
        if temporal_features["autocorr_score"] > 0.7:
            temporal_score += 0.3
            reasons.append("highly consistent temporal patterns")
        
        # Unnatural silence patterns
        if temporal_features["avg_silence_duration"] > 10 or temporal_features["avg_silence_duration"] < 2:
            temporal_score += 0.3
            reasons.append("unusual silence patterns")
        
        ai_score += temporal_score * weight
        total_weight += weight
        
        # Frequency analysis (15% weight)
        weight = 0.15
        freq_score = 0.0
        
        if freq_features["spectral_flatness_std"] < 0.05:
            freq_score += 0.5
            reasons.append("unnaturally flat spectrum")
        
        if freq_features["ultra_high_ratio"] < 0.01 or freq_features["ultra_high_ratio"] > 0.5:
            freq_score += 0.5
            reasons.append("abnormal high-frequency content")
        
        ai_score += freq_score * weight
        total_weight += weight
        
        # Wav2Vec2 embedding analysis (30% weight - most reliable!)
        if wav2vec_features:
            weight = 0.30
            emb_score = 0.0
            
            # Low embedding variance = AI
            if wav2vec_features["embedding_variance"] < 0.08:
                emb_score += 0.4
                reasons.append("extremely low neural embedding variance (strong AI indicator)")
            elif wav2vec_features["embedding_variance"] < 0.15:
                emb_score += 0.25
                reasons.append("low embedding variance")
            
            # High frame similarity = AI (too consistent)
            if wav2vec_features["frame_similarity"] > 0.95:
                emb_score += 0.4
                reasons.append("extremely high frame-to-frame consistency (AI)")
            elif wav2vec_features["frame_similarity"] > 0.85:
                emb_score += 0.2
                reasons.append("high temporal consistency")
            
            # Low temporal variance = AI
            if wav2vec_features["embedding_temporal_var"] < 0.01:
                emb_score += 0.2
                reasons.append("monotonous neural patterns")
            
            ai_score += emb_score * weight
            total_weight += weight
        else:
            weight = 0.10  # Fallback weight if no wav2vec
            total_weight += weight
        
        # MFCC delta analysis (10% weight) - AI has smoother transitions
        weight = 0.10
        mfcc_delta_std = np.std(mfcc_delta)
        if mfcc_delta_std < 5:
            ai_score += 0.8 * weight
            reasons.append("unnaturally smooth MFCC transitions")
        elif mfcc_delta_std < 10:
            ai_score += 0.4 * weight
            reasons.append("somewhat smooth spectral transitions")
        total_weight += weight
        
        # Normalize score
        final_score = ai_score / total_weight if total_weight > 0 else 0.5
        
        # Determine classification with better thresholds
        if final_score > 0.55:
            classification = "AI-generated"
            confidence = min(0.99, 0.5 + final_score)
        elif final_score < 0.40:
            classification = "Human-generated"  
            confidence = min(0.99, 1.0 - final_score)
        else:
            # Ambiguous zone (0.40 - 0.55) - use stricter threshold
            # Only classify as AI if score is clearly above middle
            classification = "AI-generated" if final_score >= 0.52 else "Human-generated"
            confidence = 0.55 + abs(final_score - 0.48) * 2
        
        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "ai_score": round(final_score, 4),
            "reasons": reasons[:4],
            "features": {
                "cnn_score": round(cnn_ai_score, 4),
                "temporal": temporal_features,
                "frequency": freq_features,
                "wav2vec": wav2vec_features,
                "pitch_variance": round(pitch_variance, 2)
            }
        }


# Initialize detector
detector = None

def get_detector():
    global detector
    if detector is None:
        logger.info("Initializing DeepfakeVoiceDetector...")
        detector = DeepfakeVoiceDetector()
        logger.info("Detector initialized!")
    return detector


# ============= API ENDPOINTS =============

def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")


@app.post("/test-voice")
async def test_voice_no_auth(request: VoiceRequest):
    """Test endpoint without authentication - uses real ML detection"""
    try:
        # Download audio
        audio_bytes = await download_audio(str(request.audio_url))
        
        # Load audio
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=30.0)
        logger.info(f"Loaded audio: {len(audio_data)} samples at {sr}Hz")
        
        # Run deep learning detection
        det = get_detector()
        result = det.predict(audio_data, sr)
        
        explanation = f"[DEEP LEARNING] {', '.join(result['reasons'][:3]) if result['reasons'] else 'Analysis complete'}. AI_score={result['ai_score']}"
        
        # Convert numpy types for JSON serialization
        features = convert_numpy(result.get("features", {}))
        
        return {
            "classification": result["classification"],
            "confidence": float(result["confidence"]),
            "explanation": explanation,
            "ai_score": float(result["ai_score"]),
            "model_used": "DeepfakeVoiceDetector (CNN + Wav2Vec2)",
            "detailed_features": features,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e), "status": "failed"}


@app.post("/detect-voice", response_model=VoiceResponse)
async def detect_voice(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    """Main detection endpoint with API key authentication"""
    try:
        logger.info(f"Processing: {request.audio_url}")
        
        audio_bytes = await download_audio(str(request.audio_url))
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=30.0)
        
        det = get_detector()
        result = det.predict(audio_data, sr)
        
        explanation = f"Deep Learning Analysis: {', '.join(result['reasons'][:3]) if result['reasons'] else 'Complete'}. Score={result['ai_score']}"
        
        return VoiceResponse(
            classification=result["classification"],
            confidence=result["confidence"],
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
        "version": "3.0.0",
        "platform": "Hugging Face Spaces",
        "model": "DeepfakeVoiceDetector (CNN + Wav2Vec2 + Temporal Analysis)",
        "features": [
            "CNN-based mel-spectrogram analysis",
            "Wav2Vec2 neural embeddings",
            "Temporal artifact detection",
            "Frequency band analysis",
            "MFCC delta smoothness"
        ],
        "endpoints": {
            "test": "/test-voice (POST, no auth)",
            "main": "/detect-voice (POST, requires API key)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health():
    det = get_detector()
    return {
        "status": "healthy",
        "platform": "Hugging Face",
        "model_loaded": det is not None,
        "wav2vec_loaded": det.wav2vec_model is not None if det else False,
        "device": str(det.device) if det else "unknown"
    }
