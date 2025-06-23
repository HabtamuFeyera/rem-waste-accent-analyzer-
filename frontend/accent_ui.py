import streamlit as st
import requests
import time
import os
import json
import shutil 


#BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/detect_accent/")

# Corrected backend URL configuration
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://rem-waste-accent-analyzer.onrender.com")
DETECT_ENDPOINT = f"{BACKEND_BASE_URL}/detect_accent/"
HEALTH_ENDPOINT = f"{BACKEND_BASE_URL}/health"

LOGO_URL = "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=200&h=200&q=80"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root {{
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --card-bg: #f8fafc;
    }}
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .header {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
    }}
    
    .logo {{
        width: 60px;
        height: 60px;
        border-radius: 12px;
        object-fit: cover;
        border: 2px solid #e2e8f0;
    }}
    
    .title {{
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }}
    
    .subtitle {{
        color: #64748b;
        font-size: 14px;
        margin: 4px 0 0 0;
    }}
    
    .card {{
        background: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    
    .metrics-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }}
    
    .metric-card {{
        background: white;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        margin: 8px 0;
    }}
    
    .metric-label {{
        font-size: 14px;
        color: #64748b;
    }}
    
    .accent-american {{ color: #3b82f6; }}
    .accent-british {{ color: #8b5cf6; }}
    .accent-australian {{ color: #ec4899; }}
    .accent-non-english {{ color: #f97316; }}
    
    .progress-bar {{
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        margin-top: 8px;
        overflow: hidden;
    }}
    
    .progress-fill {{
        height: 100%;
        border-radius: 4px;
    }}
    
    .summary-card {{
        background: white;
        border-left: 4px solid var(--primary);
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
    }}
    
    .coaching-card {{
        background: #fff7ed;
        border-left: 4px solid var(--warning);
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
    }}
    
    .plan-card {{
        background: #f0f9ff;
        border-left: 4px solid #38bdf8;
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 14px;
        white-space: pre-wrap;
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }}
    
    .status-success {{ background: #dcfce7; color: #16a34a; }}
    .status-error {{ background: #fee2e2; color: #dc2626; }}
    
    .footer {{
        margin-top: 40px;
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
    }}
    
    @media (max-width: 768px) {{
        .header {{
            flex-direction: column;
            text-align: center;
            gap: 8px;
        }}
        .title {{
            font-size: 24px;
        }}
        .metrics-container {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown(f"""
    <div class="header">
        <img src="{LOGO_URL}" class="logo" alt="REM Waste Logo">
        <div>
            <h1 class="title">REM Waste Accent Analyzer</h1>
            <p class="subtitle">Evaluate spoken English proficiency for hiring decisions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    This agentic AI system analyzes video recordings to evaluate candidates' English speaking proficiency. 
    Upload a video URL from any supported platform (Loom, YouTube, Vimeo, or direct MP4 link) 
    to receive an accent classification, confidence score, and detailed evaluation report.
    """)
    
    # Add health check to sidebar
    with st.sidebar:
        st.header("Service Status")
        if st.button("Check Backend Health"):
            try:
                response = requests.get(HEALTH_ENDPOINT, timeout=10)
                if response.status_code == 200:
                    health = response.json()
                    st.success("✅ Backend is operational")
                    st.json(health)
                else:
                    st.error(f"❌ Backend error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
        
        st.divider()
        st.info("""
        **How to Use:**
        1. Paste a public video URL
        2. Supported platforms: YouTube, Loom, Vimeo
        3. Direct MP4 links also accepted
        4. Analysis takes 1-3 minutes
        """)
    
    with st.form("accent_form"):
        video_url = st.text_input(
            "Video URL*", 
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supported: downloadable Loom, YouTube, Vimeo, direct MP4/MOV links"
        )
        
        goal = st.text_area(
            "Evaluation Goal", 
            value="Evaluate candidate's accent for customer support position",
            help="Customize the analysis focus (e.g., 'for technical support role')"
        )
        
        submitted = st.form_submit_button("Analyze Accent", use_container_width=True)
    
    if submitted:
        if not video_url:
            st.error("Please enter a valid video URL")
            return
            
        # Show processing status with real-time updates
        status_container = st.empty()
        progress_bar = st.progress(0)
        status_messages = [
            "Validating video URL...",
            "Downloading video content...",
            "Extracting audio features...",
            "Classifying accent...",
            "Generating evaluation report..."
        ]
        
        try:
            # Simulate progress updates
            for i, message in enumerate(status_messages):
                status_container.info(f"**Step {i+1}/{len(status_messages)}:** {message}")
                progress_bar.progress((i + 1) * 20)
                time.sleep(0.5)  # Simulate processing time
            
            # Make the API request
            response = requests.post(
                DETECT_ENDPOINT,
                json={"video_url": video_url, "goal": goal},
                timeout=180
            )
            
            if response.status_code == 200:
                results = response.json()
                display_results(results)
            else:
                try:
                    error = response.json().get("detail", response.text)
                except:
                    error = response.text
                st.error(f"Analysis failed: {error[:500]}")
                
        except requests.exceptions.Timeout:
            st.error("Analysis timed out (over 3 minutes). Try a shorter video.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
        finally:
            progress_bar.empty()
            status_container.empty()


def display_results(results: dict):
    """Display analysis results in a structured format"""
    status = "success" if results.get("status") == "success" else "error"
    status_text = "Completed" if status == "success" else "Failed"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <h2>Analysis Results</h2>
        <span class="status-badge status-{status}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if status == "error":
        st.error("Processing failed. Please try another video or check the URL.")
        return
    
    # Get values safely with defaults
    accent = results.get("accent", "Unknown")
    confidence = results.get("confidence", 0.0)
    english_score = results.get("english_score", 0.0)
    processing_time = results.get("processing_time", 0.0)
    request_id = results.get("request_id", "N/A")
    summary = results.get("summary", "No summary available")
    plan = results.get("plan", "No execution plan available")
    
    # Determine accent class for styling
    accent_class = "other"
    accent_lower = accent.lower()
    if "american" in accent_lower:
        accent_class = "american"
    elif "british" in accent_lower:
        accent_class = "british"
    elif "australian" in accent_lower:
        accent_class = "australian"
    elif "non-english" in accent_lower or "non english" in accent_lower:
        accent_class = "non-english"
    
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-label">Accent Classification</div>
            <div class="metric-value accent-{accent_class}">{accent}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {confidence * 100:.1f}%; background: #3b82f6;"></div>
            </div>
            <div class="metric-label">{confidence * 100:.1f}% Confidence</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">English Proficiency</div>
            <div class="metric-value">{english_score:.1f}/100</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {english_score:.1f}%; background: {'#10b981' if english_score >= 70 else '#f59e0b'};"></div>
            </div>
            <div class="metric-label">Hiring Threshold: 70/100</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{processing_time:.1f}s</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #8b5cf6;"></div>
            </div>
            <div class="metric-label">Request ID: {request_id}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Split summary from coaching recommendations
    coaching = ""
    if "**Coaching Recommendations:**" in summary:
        parts = summary.split("**Coaching Recommendations:**", 1)
        if len(parts) > 1:
            summary, coaching = parts
            coaching = f"**Coaching Recommendations:**{coaching}"
    
    st.subheader("Evaluation Summary")
    st.markdown(f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True)
    
    if coaching:
        st.subheader("Pronunciation Coaching")
        st.markdown(f'<div class="coaching-card">{coaching}</div>', unsafe_allow_html=True)
    
    st.subheader("Execution Plan")
    st.markdown(f'<div class="plan-card">{plan}</div>', unsafe_allow_html=True)
    
    with st.expander("Technical Details"):
        st.json(results)
    
    st.markdown("---")
    st.subheader("Feedback")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("👍 Accurate", use_container_width=True, key="accurate"):
            st.toast("Thanks for your feedback!", icon="👍")
    with col2:
        if st.button("👎 Inaccurate", use_container_width=True, key="inaccurate"):
            st.toast("We'll use this to improve our system", icon="👎")
    with col3:
        if st.button("🤔 Unsure", use_container_width=True, key="unsure"):
            st.toast("We'll review this analysis", icon="🤔")
    
    st.markdown("""
    <div class="footer">
        REM Waste Hiring Toolkit • Results are estimates only • Developed by Habtamu
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()