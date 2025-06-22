
# REM Waste Accent Analyzer

Intelligent AI (agentic system) tool for evaluating candidates' spoken English proficiency from video recordings. The agentic system use automates accent detection, English proficiency scoring, and provides hiring recommendations.

![Accent Analyzer UI](https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=1200&h=630&q=80)

## Features

-  Video URL processing (direct links)
-  Audio extraction and enhancement
-  Accent classification (American, British, Australian, Non-English)
-  English proficiency scoring (0-100%)
-  Professional hiring assessment reports
-  AI-powered coaching recommendations
-  Performance metrics and traceability

## Technology Stack

- **Backend**: Python FastAPI
- **Frontend**: Streamlit
- **ML**: Scikit-learn, Librosa
- **Audio Processing**: FFmpeg, SoX
- **LLM**: Groq API (deepseek-r1-distill-llama-70b)
- **Metrics**: Prometheus

## Setup Instructions

### Prerequisites
- Python 3.10+
- FFmpeg (`sudo apt install ffmpeg`)
- SoX (`sudo apt install sox`)
- System dependencies: `sudo apt install libsndfile1`

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create `.env` file in `backend/`:
```env
GROQ_API_KEY=Enter your_groq_api_key
```

## Running Locally

### Start Backend
```bash
cd backend
uvicorn accent_agent:app --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd frontend
streamlit run accent_ui.py
```

Access the UI at: `http://localhost:8501`

## API Documentation
The backend provides a REST API with these endpoints:

### POST `/detect_accent/`
Analyze video accent and generate report

**Request:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=example",
  "goal": "Evaluate for customer support role"
}
```

**Response:**
```json
{
  "accent": "Probably Australian English",
  "confidence": 0.56,
  "english_score": 55.87,
  "summary": "Detailed evaluation report...",
  "request_id": "abc123",
  "processing_time": 45.2,
  "status": "success",
  "plan": "Execution plan steps..."
}
```

### GET `/health`
Service health check

## Deployment

### Backend Deployment (Render)
1. Create new Web Service
2. Set environment variables:
   - `GROQ_API_KEY`
   - `WORKERS=2`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn accent_agent:app --host 0.0.0.0 --port $PORT`

### Frontend Deployment (Streamlit Cloud)
1. Connect to your GitHub repository
2. Set environment variable:
   - `BACKEND_URL=https://your-backend-service.com/detect_accent/`
3. Main file path: `frontend/accent_ui.py`

## Configuration
Customize these parameters in `backend/accent_agent.py`:

```python
class Settings:
    MIN_AUDIO_SEC = 3.0          
    MAX_AUDIO_SEC = 600.0         
    CONFIDENCE_THRESHOLD = 0.7    
    SAMPLE_RATE = 16000           
    MAX_RETRIES = 3               
```

### Example Run

Here’s a sample screenshot of the analyzer’s output when running a test video URL:

![Analyzer Output](https://github.com/HabtamuFeyera/rem-waste-accent-analyzer-/blob/main/frontend/image/Screenshot%20from%202025-06-21%2019-39-25.png?raw=true)

