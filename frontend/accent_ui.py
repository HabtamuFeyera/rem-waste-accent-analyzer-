import streamlit as st
import requests
import time
import os
import json


#BACKEND_URL = "https://rem-waste-accent-analyzer.onrender.com/detect_accent/"
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/detect_accent/")
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
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }}
    
    .subtitle {{
        color: #64748b;
        font-size: 16px;
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
        font-size: 28px;
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
    This agentic AI systems analyzes video recordings to evaluate candidates' English speaking proficiency. 
    Upload a video URL from any supported downloadable platform like(Loom, YouTube, Vimeo, or direct MP4 link) 
    to receive an accent classification, confidence score, and detailed evaluation report.
    """)
    
    with st.form("accent_form"):
        video_url = st.text_input(
            "Video URL*", 
            placeholder="https://www.loom.com/share/...",
            help="Supported: downloadable Loom, YouTube, Vimeo, direct MP4/MOV links"
        )
        
        goal = st.text_area(
            "Evaluation Goal", 
            value="Evaluate candidate's accent for customer support position",
            help="Customize the analysis focus (e.g., 'for technical support role')"
        )
        
        submitted = st.form_submit_button("Analyze Accent")
    
    if submitted:
        if not video_url:
            st.error("Please enter a valid video URL")
            return
            
        with st.spinner("Analyzing accent... This may take 1-2 minutes"):
            start_time = time.time()
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"video_url": video_url, "goal": goal},
                    timeout=180
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Connection to backend failed: {str(e)}")
                return
                
            processing_time = time.time() - start_time
            
        if response.status_code == 200:
            results = response.json()
            display_results(results, processing_time)
        else:
            try:
                error = response.json().get("detail", response.text)
            except:
                error = response.text
            st.error(f"Analysis failed: {error[:500]}")

def display_results(results: dict, processing_time: float):
    """Display analysis results in a structured format"""
    status = "success" if results["status"] == "success" else "error"
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
    
    accent_class = "other"
    if "american" in results["accent"].lower():
        accent_class = "american"
    elif "british" in results["accent"].lower():
        accent_class = "british"
    elif "australian" in results["accent"].lower():
        accent_class = "australian"
    elif "non-english" in results["accent"].lower():
        accent_class = "non-english"
    
    st.markdown(f"""
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-label">Accent Classification</div>
            <div class="metric-value accent-{accent_class}">{results['accent']}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {results['confidence']*100:.1f}%; background: #3b82f6;"></div>
            </div>
            <div class="metric-label">{results['confidence']*100:.1f}% Confidence</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">English Proficiency</div>
            <div class="metric-value">{results['english_score']:.1f}/100</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {results['english_score']:.1f}%; background: {'#10b981' if results['english_score'] >= 70 else '#f59e0b'};"></div>
            </div>
            <div class="metric-label">Hiring Threshold: 70/100</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{processing_time:.1f}s</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%; background: #8b5cf6;"></div>
            </div>
            <div class="metric-label">Request ID: {results['request_id']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Evaluation Summary")
    st.markdown(f'<div class="summary-card">{results["summary"]}</div>', unsafe_allow_html=True)
    
    if "Coaching Recommendations" in results["summary"] or "coaching" in results["summary"].lower():
        st.subheader("Pronunciation Coaching")
        coaching_text = results["summary"].split("**Coaching Recommendations:**")[-1]
        st.markdown(f'<div class="coaching-card">{coaching_text}</div>', unsafe_allow_html=True)
    
    st.subheader("Execution Plan")
    st.markdown(f'<div class="plan-card">{results["plan"]}</div>', unsafe_allow_html=True)
    
    with st.expander("Technical Details"):
        st.json(results)
    
    st.markdown("---")
    st.subheader("Feedback")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Accurate", use_container_width=True):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 Inaccurate", use_container_width=True):
            st.info("We'll use this to improve our system")
    with col3:
        if st.button("🤔 Unsure", use_container_width=True):
            st.info("We'll review this analysis")
    
    st.markdown("""
    <div class="footer">
        REM Waste Hiring Toolkit • Results are estimates only • Developed by Habtamu!
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()