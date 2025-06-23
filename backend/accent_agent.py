import os
import re
import tempfile
import subprocess
import requests
import numpy as np
import librosa
import joblib
import logging
import hashlib
import shutil 
import time
import random
import uuid
import yt_dlp as youtube_dl

from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
from groq import Groq

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from prometheus_client import Counter, Histogram, start_http_server
from langgraph.graph import StateGraph, END


#Configuration
class Settings:
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
        self.MODEL_PATH = os.path.join(os.path.dirname(__file__), "accent_model_v2.joblib")
        self.MIN_AUDIO_SEC = 3.0
        self.MAX_AUDIO_SEC = 600.0
        self.CONFIDENCE_THRESHOLD = 0.7
        self.PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8001))
        self.SAMPLE_RATE = 16000
        self.MAX_RETRIES = 3
        self.REQUEST_TIMEOUT = 180
        self.WORKERS = int(os.getenv("WORKERS", 2))
        self.REFLECTION_MODEL = "deepseek-r1-distill-llama-70b"
        self.PLANNING_MODEL = "deepseek-r1-distill-llama-70b"
        self.DEFAULT_GOAL = "Evaluate candidate's accent from video and provide hiring assessment"
        self.NOISE_PROFILE = os.path.join(os.path.dirname(__file__), "noise_profile.prof")
        self.GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AccentAgent")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "accent_agent_requests_total", 
    "Total accent detection requests", 
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "accent_agent_request_latency_seconds", 
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)
ERROR_COUNT = Counter(
    "accent_agent_errors_total",
    "Total processing errors",
    ["stage"]
)
RETRY_COUNT = Counter(
    "accent_agent_retries_total",
    "Total step retries",
    ["step"]
)
AUGMENTATION_COUNT = Counter(
    "accent_agent_augmentations_total",
    "Total augmentations applied",
    ["type"]
)

start_http_server(settings.PROMETHEUS_PORT)

def load_or_train_model(path: str) -> Optional[Any]:
    try:
        if os.path.exists(path):
            logger.info("Loading accent model from %s", path)
            return joblib.load(path)
        
        logger.warning("Training new accent model (placeholder)")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier

        # Create realistic accent patterns
        np.random.seed(42)
        n_samples = 5000
        n_features = 42
        
        accent_patterns = {
            "american": np.array([0.2, 0.8, 0.5, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6, 0.2]),
            "british": np.array([0.7, 0.3, 0.9, 0.5, 0.2, 0.8, 0.1, 0.6, 0.4, 0.7]),
            "australian": np.array([0.5, 0.6, 0.3, 0.8, 0.4, 0.5, 0.7, 0.2, 0.9, 0.1]),
            "non_english": np.array([0.9, 0.1, 0.7, 0.2, 0.5, 0.9, 0.3, 0.8, 0.1, 0.6])
        }
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        accent_keys = list(accent_patterns.keys())
        
        for i in range(n_samples):
            accent_idx = i % len(accent_keys)
            base_pattern = accent_patterns[accent_keys[accent_idx]]
            features = np.concatenate([
                base_pattern + np.random.normal(0, 0.2, len(base_pattern)),
                np.random.rand(n_features - len(base_pattern))
            ])
            X[i] = features
            y[i] = accent_idx

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=7,
                subsample=0.8, 
                random_state=42
            ))
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, path)
        logger.info("Saved new model to %s", path)
        return pipeline
        
    except Exception as e:
        logger.exception("Model initialization failed")
        ERROR_COUNT.labels(stage="model_init").inc()
        return None

model_pipeline = load_or_train_model(settings.MODEL_PATH)

# Groq LLM Client (HTTP Implementation)
class GroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt: str, model: str = "deepseek-r1-distill-llama-70b", temperature: float = 0.2) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 500
        }
        
        for attempt in range(settings.MAX_RETRIES):
            try:
                response = requests.post(
                    settings.GROQ_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt < settings.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning("Groq API error, retrying in %ds: %s", wait, str(e))
                    time.sleep(wait)
                else:
                    logger.error("Groq API failed after %d attempts: %s", settings.MAX_RETRIES, str(e))
                    return f"LLM generation failed: {str(e)}"
        return "LLM generation failed after retries"

groq_client = GroqClient(settings.GROQ_API_KEY)

# State Definition
class AgentState(TypedDict):
    video_url: str
    temp_files: List[str]
    audio_features: List[float]
    classification: Dict[str, Any]
    needs_coaching: bool
    summary: str
    error: Optional[str]
    request_id: str
    start_time: float
    goal: str
    plan: str
    retry_count: Dict[str, int]
    last_step: str
    reflection_decision: str

# Node Implementations
def track_step(step_name: str):
    """Decorator to track last executed step and log execution"""
    def decorator(func):
        def wrapper(state: AgentState) -> AgentState:
            if state.get("error"):
                return state
                
            state["last_step"] = step_name
            logger.info(f"[{state['request_id']}] Executing step: {step_name}")
            start_time = time.time()
            try:
                result = func(state)
                duration = time.time() - start_time
                logger.info(f"[{state['request_id']}] Step {step_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                state["error"] = f"{step_name} failed: {str(e)}"
                ERROR_COUNT.labels(stage=step_name).inc()
                logger.exception(f"[{state['request_id']}] Step {step_name} failed")
                return state
        return wrapper
    return decorator

def plan_workflow(state: AgentState) -> AgentState:
    """Generate step-by-step plan using LLM"""
    if state.get("error"):
        return state
        
    try:
        prompt = (
            "You are an expert on planning. Create a detailed step-by-step plan to achieve the following goal:\n"
            f"GOAL: {state['goal']}\n\n"
            "STRUCTURE YOUR RESPONSE:\n"
            "1. Break down the goal into sequential steps\n"
            "2. Consider audio processing, AI classification, and reporting needs\n"
            "3. Include validation and error handling considerations\n"
            "4. Specify required tools and techniques\n"
            "5. Format as a numbered list\n\n"
            "OUTPUT ONLY THE PLAN, NO ADDITIONAL TEXT"
        )
        
        state["plan"] = groq_client.generate_text(prompt, model=settings.PLANNING_MODEL)
        logger.info(f"[{state['request_id']}] Generated plan:\n{state['plan']}")
        return state
        
    except Exception as e:
        state["error"] = f"Planning failed: {str(e)}"
        ERROR_COUNT.labels(stage="planning").inc()
    return state

@track_step("validate_and_download")
def validate_and_download(state: AgentState) -> AgentState:
    """Validate URL and download video with platform-specific handling"""
    try:
        if not is_valid_video_url(state["video_url"]):
            raise ValueError("Invalid video URL format")

        output_path = generate_temp_path(".mp4")
        state["temp_files"].append(output_path)
        handler = get_platform_handler(state["video_url"])
        handler(state, output_path)
        verify_video_file(output_path, min_size_kb=100)
        
    except Exception as e:
        state["error"] = f"Download failed: {str(e)}"
        ERROR_COUNT.labels(stage="download").inc()
    return state

def generate_temp_path(suffix: str) -> str:
    """Generate unique temporary path without creating file"""
    return os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")

def is_valid_video_url(url: str) -> bool:
    """Check if URL matches supported video patterns"""
    patterns = [
        r"^https?://.*\.(mp4|mov|webm|mkv|avi|flv)",
        r"^https?://(www\.)?youtube\.com/watch\?v=",
        r"^https?://youtu\.be/",
        r"^https?://(www\.)?youtube\.com/shorts/",
        r"^https?://(www\.)?loom\.com/share/",
        r"^https?://vimeo\.com/"
    ]
    return any(re.match(p, url) for p in patterns)

def get_platform_handler(url: str):
    """Select appropriate download handler based on URL"""
    if "youtube.com" in url or "youtu.be" in url:
        return download_youtube_video
    if "loom.com" in url:
        return download_loom_video
    if "vimeo.com" in url:
        return download_vimeo_video
    return download_direct_video

def verify_video_file(path: str, min_size_kb: int) -> None:
    """Verify video file meets size requirements"""
    time.sleep(0.5)  
    if not os.path.exists(path):
        raise RuntimeError(f"Video file not found: {path}")
    
    size = os.path.getsize(path)
    if size < min_size_kb * 1024:
        raise RuntimeError(f"Video file too small: {size/1024:.1f}KB < {min_size_kb}KB")

def download_youtube_video(state: AgentState, output_path: str) -> None:
    """
    Download YouTube videos using yt-dlp with robust error handling
    - Uses unique temp paths to prevent "already downloaded" issues
    - Ensures files are only created when data is written
    """
    temp_download_path = None
    try:
        temp_download_path = generate_temp_path(".mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': temp_download_path,  
            'quiet': True,
            'no_warnings': False,
            'ignoreerrors': False,
            'retries': 5,
            'fragment-retries': 10,
            'socket-timeout': 60,
            'merge_output_format': 'mp4',
            'concurrent-fragment-downloads': 6,
            'http-chunk-size': '15M',
            'verbose': False,
            'cookiefile': 'cookies.txt',
            'nocheckcertificate': True,
            'fixup': 'detect_or_warn',
            'continuedl': False,  
            'nopart': True,  
        }
        
        logger.info(f"Downloading YouTube video: {state['video_url']}")
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(state["video_url"], download=False)
            if not info_dict:
                raise RuntimeError("Failed to get video info")
                
            duration = info_dict.get('duration', 0)
            logger.info(f"Video title: {info_dict.get('title', 'unknown')}, Duration: {duration}s")

            ydl.download([state["video_url"]])
        
        verify_video_file(temp_download_path, min_size_kb=100)
        
        if duration > settings.MAX_AUDIO_SEC:
            logger.info(f"Trimming {duration}s video to {settings.MAX_AUDIO_SEC}s")
            try:
                trim_video(temp_download_path, output_path, settings.MAX_AUDIO_SEC)
            except Exception as e:
                logger.error(f"Trimming failed: {e}, using original video")
                shutil.move(temp_download_path, output_path)
            finally:
                # Cleanup temp file if still exists
                if temp_download_path and os.path.exists(temp_download_path):
                    os.unlink(temp_download_path)
        else:
            # Move directly to final path
            shutil.move(temp_download_path, output_path)
        
        # Final verification
        verify_video_file(output_path, min_size_kb=100)
        logger.info(f"YouTube download successful: {os.path.getsize(output_path)/1024/1024:.2f}MB")
            
    except youtube_dl.utils.DownloadError as e:
        if temp_download_path and os.path.exists(temp_download_path):
            os.unlink(temp_download_path)
        raise RuntimeError(f"YouTube download failed: {str(e)}")
    except Exception as e:
        if temp_download_path and os.path.exists(temp_download_path):
            os.unlink(temp_download_path)
        raise RuntimeError(f"YouTube download error: {str(e)}")

def download_direct_video(state: AgentState, output_path: str) -> None:
    """Handle direct video downloads with retry logic"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "video/webm,video/mp4,video/*;q=0.9",
        "Accept-Encoding": "identity;q=1, *;q=0",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Range": "bytes=0-",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "video",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site"
    }
    
    for attempt in range(settings.MAX_RETRIES):
        try:
            logger.info(f"Downloading direct video (attempt {attempt+1}): {state['video_url']}")
            with requests.get(state["video_url"], headers=headers, 
                            stream=True, timeout=60) as r:
                r.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(1024*1024):  
                        f.write(chunk)
            
            verify_video_file(output_path, min_size_kb=100)
            logger.info(f"Direct download successful: {os.path.getsize(output_path)/1024/1024:.2f}MB")
            return
            
        except Exception as e:
            if os.path.exists(output_path):
                os.unlink(output_path)
                
            if attempt == settings.MAX_RETRIES - 1:
                raise RuntimeError(f"Direct download failed after {settings.MAX_RETRIES} attempts: {str(e)}")
            
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning("Download attempt %d failed, retrying in %.1fs: %s", 
                          attempt+1, wait, str(e))
            time.sleep(wait)

def download_loom_video(state: AgentState, output_path: str) -> None:
    """Loom video download handler (uses YouTube method)"""
    download_youtube_video(state, output_path)

def download_vimeo_video(state: AgentState, output_path: str) -> None:
    """Vimeo video download handler (uses YouTube method)"""
    download_youtube_video(state, output_path)

def trim_video(input_path: str, output_path: str, max_seconds: float) -> None:
    """Trim video to specified duration using efficient methods"""
    try:
        run_ffmpeg([
            "-y", 
            "-ss", "0", 
            "-i", input_path,
            "-t", str(max_seconds), 
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path
        ], "Stream copy")
    except Exception as e:
        logger.warning(f"Stream copy trimming failed: {str(e)}, trying re-encoding")
        run_ffmpeg([
            "-y", 
            "-ss", "0", 
            "-i", input_path,
            "-t", str(max_seconds),
            "-c:v", "libx264", 
            "-preset", "veryfast", 
            "-crf", "23",
            "-c:a", "aac", 
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ], "Re-encode")

def run_ffmpeg(args: List[str], method_name: str) -> None:
    """Execute FFmpeg command with standardized error handling"""
    cmd = ["ffmpeg"] + args
    logger.info(f"Running FFmpeg ({method_name}): {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=True,
            timeout=300 
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"{method_name} failed ({e.returncode}): {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except subprocess.TimeoutExpired:
        error_msg = f"{method_name} timed out after 5 minutes"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not os.path.exists(cmd[-1]) or os.path.getsize(cmd[-1]) < 1024:
        raise RuntimeError(f"{method_name} created invalid output")
    
    logger.info(f"FFmpeg {method_name} completed successfully")


@track_step("extract_audio")
def extract_audio(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
        
    try:
        in_vid = state["temp_files"][-1]  
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_wav = tmp.name
        state["temp_files"].append(out_wav)

        result = subprocess.run(
            [
                "ffmpeg", "-y", 
                "-i", in_vid,
                "-ac", "1",
                "-ar", str(settings.SAMPLE_RATE),
                "-vn", 
                "-loglevel", "error", 
                out_wav
            ],
            capture_output=True,
            check=True
        )
        return state
    except subprocess.CalledProcessError as e:
        state["error"] = f"Audio extraction failed: {e.stderr.decode('utf-8')}"
        ERROR_COUNT.labels(stage="audio_extract").inc()
    except Exception as e:
        state["error"] = f"Audio extraction error: {str(e)}"
        ERROR_COUNT.labels(stage="audio_extract").inc()
    return state

@track_step("preprocess_audio")
def preprocess_audio(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
        
    try:
        audio_path = state["temp_files"][-1]  
        y, sr = librosa.load(audio_path, sr=settings.SAMPLE_RATE, mono=True)
        dur = len(y) / sr
        
        if dur < settings.MIN_AUDIO_SEC:
            raise ValueError(f"Audio too short ({dur:.1f}s < {settings.MIN_AUDIO_SEC}s)")
        if dur > settings.MAX_AUDIO_SEC:
            raise ValueError(f"Audio too long ({dur:.1f}s > {settings.MAX_AUDIO_SEC}s)")
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        features = np.concatenate([
            np.mean(mfcc, axis=1),         
            np.std(mfcc, axis=1),          
            [np.mean(spectral_centroid)],  
            [np.mean(spectral_bandwidth)], 
            [np.mean(spectral_rolloff)],   
            [np.mean(zero_crossing)],      
            np.mean(chroma, axis=1),       
            [np.mean(rms)],                
            np.mean(tonnetz, axis=1)       
        ])
        
        if len(features) > 42:
            features = features[:42]
        elif len(features) < 42:
            features = np.pad(features, (0, 42 - len(features)), 'constant')
        
        state["audio_features"] = features.tolist()
        return state
        
    except Exception as e:
        state["error"] = f"Audio processing failed: {str(e)}"
        ERROR_COUNT.labels(stage="audio_processing").inc()
    return state

@track_step("classify_accent")
def classify_accent(state: AgentState) -> AgentState:
    if state.get("error") or not model_pipeline:
        if not model_pipeline:
            state["error"] = "Accent model not available"
            ERROR_COUNT.labels(stage="classification").inc()
        return state
        
    try:
        arr = np.array(state["audio_features"]).reshape(1, -1)
        proba = model_pipeline.predict_proba(arr)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
        
        labels = {
            0: "American English",
            1: "British English",
            2: "Australian English",
            3: "Non-English"
        }
        accent = labels.get(idx, "Unknown")
        
        if conf < 0.5:
            accent = f"Likely {accent}"
        elif conf < settings.CONFIDENCE_THRESHOLD:
            accent = f"Probably {accent}"
        
        english_score = 100 * conf if idx < 3 else 0.0
        
        state["classification"] = {
            "accent": accent,
            "confidence": conf,
            "english_score": english_score
        }
        state["needs_coaching"] = english_score < (settings.CONFIDENCE_THRESHOLD * 100)
        return state
        
    except Exception as e:
        state["error"] = f"Classification failed: {str(e)}"
        ERROR_COUNT.labels(stage="classification").inc()
    return state

@track_step("augment_audio")
def augment_audio(state: AgentState) -> AgentState:
    """Apply audio enhancement techniques if SoX is available"""
    if state.get("error"):
        return state
        
    try:

        sox_installed = subprocess.run(
            ["sox", "--version"],
            capture_output=True,
            text=True
        ).returncode == 0
        
        if not sox_installed:
            logger.warning(f"[{state['request_id']}] SoX not installed, skipping audio enhancement")
            return state
            
        audio_path = state["temp_files"][-1]
        logger.info(f"[{state['request_id']}] Applying audio enhancement to {audio_path}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            enhanced_path = tmp.name
        state["temp_files"].append(enhanced_path)
        
        subprocess.run(
            [
                "sox", audio_path, enhanced_path,
                "noisered", settings.NOISE_PROFILE, "0.21",
                "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2"
            ],
            check=True,
            capture_output=True
        )
        
        state["temp_files"].remove(audio_path)
        os.unlink(audio_path)
        
        AUGMENTATION_COUNT.labels(type="audio").inc()
        logger.info(f"[{state['request_id']}] Audio enhancement complete")
        return state
        
    except Exception as e:
        if "No such file or directory: 'sox'" in str(e):
            logger.warning(f"[{state['request_id']}] SoX not installed, skipping audio enhancement")
            return state
        state["error"] = f"Audio enhancement failed: {str(e)}"
        ERROR_COUNT.labels(stage="audio_augmentation").inc()
    return state

@track_step("augment_classification")
def augment_classification(state: AgentState) -> AgentState:
    """Enhance classification with feature augmentation"""
    if state.get("error") or not model_pipeline:
        return state
        
    try:
        features = np.array(state["audio_features"])
        
        augmented = []
        for _ in range(3):  
            noise = np.random.normal(0, 0.05, features.shape)
            scaled = features * (0.95 + np.random.random() * 0.1)
            augmented.append(features + noise)
            augmented.append(scaled)
        
        predictions = []
        confidences = []
        for aug in augmented:
            arr = aug.reshape(1, -1)
            proba = model_pipeline.predict_proba(arr)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            predictions.append(idx)
            confidences.append(conf)
        
        final_idx = max(set(predictions), key=predictions.count)
        avg_conf = np.mean(confidences)
        
        labels = {
            0: "American English",
            1: "British English",
            2: "Australian English",
            3: "Non-English"
        }
        accent = labels.get(final_idx, "Unknown")
        
        if avg_conf < 0.5:
            accent = f"Likely {accent}"
        elif avg_conf < settings.CONFIDENCE_THRESHOLD:
            accent = f"Probably {accent}"
        
        english_score = 100 * avg_conf if final_idx < 3 else 0.0
        
        state["classification"] = {
            "accent": accent,
            "confidence": avg_conf,
            "english_score": english_score
        }
        state["needs_coaching"] = english_score < (settings.CONFIDENCE_THRESHOLD * 100)
        
        AUGMENTATION_COUNT.labels(type="classification").inc()
        logger.info(f"[{state['request_id']}] Classification augmented: {accent} ({avg_conf:.2f})")
        return state
        
    except Exception as e:
        state["error"] = f"Classification augmentation failed: {str(e)}"
        ERROR_COUNT.labels(stage="classification_augmentation").inc()
    return state

@track_step("generate_summary")
def generate_summary(state: AgentState) -> AgentState:
    if state.get("error") or "classification" not in state:
        return state
        
    try:
        c = state["classification"]
        prompt = (
            "**Accent Evaluation Report**\n"
            "Generate a professional hiring evaluation summary based on:\n"
            f"- Detected accent: {c['accent']}\n"
            f"- Confidence: {c['confidence']:.0%}\n"
            f"- English proficiency score: {c['english_score']}/100\n\n"
            "**Structure your response:**\n"
            "1. Accent classification summary\n"
            "2. English proficiency assessment\n"
            "3. Key pronunciation observations\n"
            "4. Hiring suitability recommendation\n"
            "5. Coaching suggestions (if needed)\n\n"
            "**Guidelines:**\n"
            "- Be objective and professional\n"
            "- Focus on communication clarity\n"
            "- Consider accent diversity as a strength\n"
            "- Provide actionable feedback\n"
            "- Limit to 200 words\n"
        )
        state["summary"] = groq_client.generate_text(prompt)
        return state
        
    except Exception as e:
        state["error"] = f"Summary generation failed: {str(e)}"
        ERROR_COUNT.labels(stage="summary_generation").inc()
    return state

@track_step("generate_coaching")
def generate_coaching(state: AgentState) -> AgentState:
    if state.get("error") or "classification" not in state:
        return state
        
    try:
        c = state["classification"]
        prompt = (
            "**Pronunciation Coaching Recommendations**\n"
            "Generate targeted coaching suggestions based on:\n"
            f"- Detected accent: {c['accent']}\n"
            f"- Confidence: {c['confidence']:.0%}\n"
            f"- English proficiency score: {c['english_score']}/100\n\n"
            "**Structure your response:**\n"
            "1. Key pronunciation challenges\n"
            "2. Specific sound exercises\n"
            "3. Intonation practice\n"
            "4. Recommended resources\n"
            "5. Practice schedule\n\n"
            "**Guidelines:**\n"
            "- Be constructive and supportive\n"
            "- Focus on 3-5 key areas\n"
            "- Provide concrete examples\n"
            "- Limit to 150 words\n"
        )
        coaching = groq_client.generate_text(prompt)
        state["summary"] += f"\n\n**Coaching Recommendations:**\n{coaching}"
        return state
        
    except Exception as e:
        state["error"] = f"Coaching generation failed: {str(e)}"
        ERROR_COUNT.labels(stage="coaching_generation").inc()
    return state

@track_step("cleanup")
def cleanup_resources(state: AgentState) -> AgentState:
    """Clean up temporary files"""
    for path in state.get("temp_files", []):
        try:
            if os.path.exists(path):
                os.unlink(path)
                logger.debug(f"Cleaned up: {path}")
        except Exception as e:
            logger.warning(f"[{state['request_id']}] Failed to delete {path}: {str(e)}")
    return state

def reflect_on_step(state: AgentState) -> AgentState:
    """Evaluate step results and decide next action with state awareness"""
    if state.get("error"):
        return state
    
    last_step = state.get("last_step", "")
    retry_count = state["retry_count"].get(last_step, 0)
    
    prompt = (
        "You are an expert on quality control. Evaluate the results of the last processing step:\n"
        f"LAST STEP: {last_step}\n"
        f"RETRY COUNT: {retry_count}/{settings.MAX_RETRIES}\n\n"
    )
    
    if last_step == "validate_and_download":
        file_status = "File missing"
        if state.get("temp_files"):
            video_path = state["temp_files"][-1]
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                file_status = f"File exists ({file_size/1024:.1f}KB)"
        
        prompt += (
            "Check if the video was successfully downloaded. "
            f"Current file status: {file_status}\n"
            "Consider file size (should be >100KB), format validity, and network errors. "
            "Should we retry downloading or proceed?\n"
        )
    
    elif last_step == "extract_audio":
        audio_status = "Audio file missing"
        if state.get("temp_files") and len(state["temp_files"]) > 1:
            audio_path = state["temp_files"][-1]
            if os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
                audio_status = f"Audio file exists ({audio_size/1024:.1f}KB)"
        
        prompt += (
            "Check audio extraction quality. "
            f"Current status: {audio_status}\n"
            "Consider duration, sample rate, and waveform characteristics. "
            "Is the audio clear enough for analysis?\n"
        )
    
    elif last_step == "preprocess_audio":
        feature_status = "Features missing"
        if state.get("audio_features"):
            feature_count = len(state["audio_features"])
            feature_status = f"Features extracted ({feature_count} dimensions)"
        
        prompt += (
            "Evaluate audio preprocessing. "
            f"Current status: {feature_status}\n"
            "Consider signal-to-noise ratio, feature extraction quality, "
            "and any artifacts. Should we apply noise reduction?\n"
        )
    
    elif last_step == "classify_accent":
        c = state.get("classification", {})
        prompt += (
            f"Evaluate classification results:\n"
            f"- Accent: {c.get('accent', 'N/A')}\n"
            f"- Confidence: {c.get('confidence', 0.0):.0%}\n"
            f"- English Score: {c.get('english_score', 0.0)}/100\n\n"
            "Is the confidence sufficient? If not, should we:\n"
            "1. Retry classification with augmented features\n"
            "2. Adjust model parameters\n"
            "3. Proceed with low-confidence results\n"
        )
    
    prompt += (
        "\nOPTIONS:\n"
        "A) Proceed to next step\n"
        "B) Retry current step\n"
        "C) Apply data augmentation\n"
        "D) Abort processing\n\n"
        "Provide your decision and a brief reason. Use format: [LETTER] Reason"
    )
    
    try:
        reflection = groq_client.generate_text(prompt, model=settings.REFLECTION_MODEL)
        logger.info(f"[{state['request_id']}] Reflection for {last_step}: {reflection}")
        decision_match = re.search(r"\[([A-D])\]", reflection)
        decision = decision_match.group(1) if decision_match else "A"
        state["reflection_decision"] = decision
        
        if decision == "B":  
            if retry_count < settings.MAX_RETRIES:
                state["retry_count"][last_step] = retry_count + 1
                RETRY_COUNT.labels(step=last_step).inc()
                logger.warning(f"[{state['request_id']}] Retrying {last_step} (attempt {retry_count+1})")
            else:
                logger.error(f"[{state['request_id']}] Max retries exceeded for {last_step}")
                state["error"] = f"Max retries exceeded for {last_step}"
                ERROR_COUNT.labels(stage="retry_limit").inc()
        
        elif decision == "C":  
            logger.info(f"[{state['request_id']}] Augmentation requested for {last_step}")
        
        return state
        
    except Exception as e:
        state["error"] = f"Reflection failed: {str(e)}"
        ERROR_COUNT.labels(stage="reflection").inc()
        return state

def create_reflection_node(step_name: str):
    def node(state: AgentState) -> AgentState:
        state["last_step"] = step_name
        return reflect_on_step(state)
    return node

workflow = StateGraph(AgentState)

workflow.add_node("plan_workflow", plan_workflow)
workflow.add_node("validate_and_download", validate_and_download)
workflow.add_node("extract_audio", extract_audio)
workflow.add_node("preprocess_audio", preprocess_audio)
workflow.add_node("classify_accent", classify_accent)
workflow.add_node("augment_audio", augment_audio)
workflow.add_node("augment_classification", augment_classification)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("generate_coaching", generate_coaching)
workflow.add_node("cleanup", cleanup_resources)
workflow.add_node("reflect_download", create_reflection_node("download"))
workflow.add_node("reflect_audio_extract", create_reflection_node("audio_extract"))
workflow.add_node("reflect_preprocessing", create_reflection_node("preprocessing"))
workflow.add_node("reflect_classification", create_reflection_node("classification"))


workflow.set_entry_point("plan_workflow")
workflow.add_edge("plan_workflow", "validate_and_download")
workflow.add_edge("validate_and_download", "reflect_download")

def after_download_reflection(state: AgentState) -> str:
    if state.get("error"):
        return "cleanup"
    
    decision = state.get("reflection_decision", "A")
    step_name = "validate_and_download"
    retry_count = state["retry_count"].get(step_name, 0)
    
    download_success = False
    if state.get("temp_files"):
        video_path = state["temp_files"][-1]
        if os.path.exists(video_path) and os.path.getsize(video_path) > 1024 * 100:  # 100KB min
            download_success = True
    
    if download_success:
        if decision == "B":
            logger.info(f"[{state['request_id']}] Overriding reflection retry: Download was successful")
        return "extract_audio"  
    
    if decision == "B" and retry_count < settings.MAX_RETRIES:
        return "validate_and_download"
    elif decision == "C" or decision == "D":
        state["error"] = f"Reflection decision: {decision}"
        return "cleanup"
    return "extract_audio" 

workflow.add_conditional_edges(
    "reflect_download",
    after_download_reflection,
    {
        "validate_and_download": "validate_and_download",
        "extract_audio": "extract_audio",
        "cleanup": "cleanup"
    }
)
workflow.add_edge("extract_audio", "reflect_audio_extract")

def after_audio_extract_reflection(state: AgentState) -> str:
    if state.get("error"):
        return "cleanup"
    
    decision = state.get("reflection_decision", "A")
    step_name = "extract_audio"
    retry_count = state["retry_count"].get(step_name, 0)
    
    if decision == "B" and retry_count < settings.MAX_RETRIES:
        return "extract_audio"
    elif decision == "C":
        return "augment_audio"
    elif decision == "D":
        state["error"] = f"Reflection decision: {decision}"
        return "cleanup"
    return "preprocess_audio"

workflow.add_conditional_edges(
    "reflect_audio_extract",
    after_audio_extract_reflection,
    {
        "extract_audio": "extract_audio",
        "augment_audio": "augment_audio",
        "preprocess_audio": "preprocess_audio",
        "cleanup": "cleanup"
    }
)

workflow.add_edge("augment_audio", "preprocess_audio")
workflow.add_edge("preprocess_audio", "reflect_preprocessing")
def after_preprocessing_reflection(state: AgentState) -> str:
    if state.get("error"):
        return "cleanup"
    
    decision = state.get("reflection_decision", "A")
    step_name = "preprocess_audio"
    retry_count = state["retry_count"].get(step_name, 0)
    
    if decision == "B" and retry_count < settings.MAX_RETRIES:
        return "preprocess_audio"
    elif decision == "C":
        return "augment_audio"
    elif decision == "D":
        state["error"] = f"Reflection decision: {decision}"
        return "cleanup"
    return "classify_accent"

workflow.add_conditional_edges(
    "reflect_preprocessing",
    after_preprocessing_reflection,
    {
        "preprocess_audio": "preprocess_audio",
        "augment_audio": "augment_audio",
        "classify_accent": "classify_accent",
        "cleanup": "cleanup"
    }
)
workflow.add_edge("classify_accent", "reflect_classification")

def after_classification_reflection(state: AgentState) -> str:
    if state.get("error"):
        return "cleanup"
    
    decision = state.get("reflection_decision", "A")
    step_name = "classify_accent"
    retry_count = state["retry_count"].get(step_name, 0)
    
    if decision == "B" and retry_count < settings.MAX_RETRIES:
        return "classify_accent"
    elif decision == "C":
        return "augment_classification"
    elif decision == "D":
        state["error"] = f"Reflection decision: {decision}"
        return "cleanup"
    return "generate_summary"

workflow.add_conditional_edges(
    "reflect_classification",
    after_classification_reflection,
    {
        "classify_accent": "classify_accent",
        "augment_classification": "augment_classification",
        "generate_summary": "generate_summary",
        "cleanup": "cleanup"
    }
)
workflow.add_edge("augment_classification", "generate_summary")

def route_based_on_coaching(state: AgentState) -> str:
    if state.get("needs_coaching", False):
        return "generate_coaching"
    return "cleanup"

workflow.add_conditional_edges(
    "generate_summary",
    route_based_on_coaching,
    {
        "generate_coaching": "generate_coaching",
        "cleanup": "cleanup"
    }
)

workflow.add_edge("generate_coaching", "cleanup")
workflow.add_edge("cleanup", END)
app_graph = workflow.compile()


# FastAPI Application
app = FastAPI(
    title="REM Waste Accent Analyzer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class VideoRequest(BaseModel):
    video_url: HttpUrl
    goal: str = settings.DEFAULT_GOAL

class AccentResponse(BaseModel):
    accent: str
    confidence: float
    english_score: float
    summary: str
    request_id: str
    processing_time: float
    status: str
    plan: str

@app.post("/detect_accent/", response_model=AccentResponse)
def detect_accent(req: VideoRequest):
    """Detect accent from the entered video URL with reflection and augmentation"""
    start_time = time.time()
    request_id = hashlib.sha256(
        f"{req.video_url}{time.time()}".encode()
    ).hexdigest()[:12]
    
    initial_state = {
        "video_url": str(req.video_url),
        "goal": req.goal,
        "plan": "",
        "temp_files": [],
        "audio_features": [],
        "classification": {},
        "needs_coaching": False,
        "summary": "",
        "error": None,
        "request_id": request_id,
        "start_time": start_time,
        "retry_count": {},
        "last_step": "",
        "reflection_decision": ""
    }
    
    logger.info("Starting request %s for %s", request_id, req.video_url)
    
    try:
        result = app_graph.invoke(
            initial_state, 
            {"recursion_limit": 50}
        )
        
        if result.get("error"):
            raise RuntimeError(result["error"])
        
        classification = result["classification"]
        REQUEST_COUNT.labels(status="success").inc()
        processing_time = time.time() - start_time
        final_summary = f"Execution Plan:\n{result['plan']}\n\n" + result["summary"]
        
        return AccentResponse(
            accent=classification["accent"],
            confidence=classification["confidence"],
            english_score=classification["english_score"],
            summary=final_summary,
            request_id=request_id,
            processing_time=processing_time,
            status="success",
            plan=result["plan"]
        )
        
    except Exception as e:
        cleanup_resources(initial_state)
        
        logger.error("Request %s failed: %s", request_id, str(e))
        REQUEST_COUNT.labels(status="error").inc()
        processing_time = time.time() - start_time
        
        return AccentResponse(
            accent="",
            confidence=0.0,
            english_score=0.0,
            summary=f"Processing error: {str(e)}",
            request_id=request_id,
            processing_time=processing_time,
            status="error",
            plan=initial_state.get("plan", "")
        )

@app.get("/")
def root_endpoint():
    return {
        "service": "REM Waste Accent Analyzer",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "documentation": "/docs",
            "accent_detection": "/detect_accent",
            "health_check": "/health"
        },
        "message": "Use POST /detect_accent with a video URL for accent analysis"
    }

@app.get("/health")
def health_check():

    ffmpeg_available = shutil.which("ffmpeg") is not None
    sox_available = shutil.which("sox") is not None
    
    groq_status = "connected"
    try:
        test_response = groq_client.generate_text("Test connection", max_tokens=5)
        if "failed" in test_response.lower():
            groq_status = "error"
    except Exception:
        groq_status = "unavailable"
    
    return {
        "status": "active",
        "model": "loaded" if model_pipeline else "unavailable",
        "groq": groq_status,
        "ffmpeg": "available" if ffmpeg_available else "missing",
        "sox": "available" if sox_available else "missing",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

def create_noise_profile():
    if not os.path.exists(settings.NOISE_PROFILE):
        logger.info("Creating default noise profile")
        noise = np.random.normal(0, 0.1, settings.SAMPLE_RATE * 5)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            librosa.output.write_wav(tmp.name, noise, settings.SAMPLE_RATE)
        subprocess.run(
            ["sox", tmp.name, "-n", "noiseprof", settings.NOISE_PROFILE],
            check=True,
            capture_output=True
        )
        os.unlink(tmp.name)

create_noise_profile()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=settings.WORKERS,
        timeout_keep_alive=settings.REQUEST_TIMEOUT
    )