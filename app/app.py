import os
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint config
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Session State for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Inject custom styling for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6200ea 0%, #3700b3 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(98, 0, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(98, 0, 234, 0.4);
    }
    .prediction-card-pos {
        background-color: rgba(46, 204, 113, 0.15);
        border: 2px solid #2ecc71;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.1);
    }
    .prediction-card-neg {
        background-color: rgba(231, 76, 60, 0.15);
        border: 2px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #bb86fc;
        margin-bottom: 1rem;
    }
    .history-item {
        background-color: #1f1f2e;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #bb86fc;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("##### A production-grade MLOps application powered by **DistilBERT** and **FastAPI**.")
st.markdown("---")

# Layout with two main columns
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown("### Enter Your Review")
    review_text = st.text_area(
        label="Write or paste a movie review below to analyze its sentiment:",
        placeholder="e.g. This movie was absolutely fantastic! The acting was superb and the plot kept me engaged...",
        height=200,
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

with col2:
    st.markdown("### Analysis Results")
    
    if analyze_btn:
        if not review_text.strip():
            st.warning("⚠️ Review text cannot be empty. Please type some text first!")
        else:
            with st.spinner("Analyzing text using DistilBERT model..."):
                try:
                    # Make post request to FastAPI backend
                    response = requests.post(
                        f"{FASTAPI_URL}/predict",
                        json={"text": review_text},
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        sentiment = data["sentiment"]
                        confidence = data["confidence"]
                        
                        # Save prediction history
                        st.session_state.history.insert(0, {
                            "text": review_text[:60] + "..." if len(review_text) > 60 else review_text,
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                        
                        # Render result card based on sentiment
                        if sentiment == "Positive":
                            st.markdown(
                                f"""
                                <div class="prediction-card-pos">
                                    <h4 style="color: #2ecc71; margin-top:0;">POSITIVITY DETECTED</h4>
                                    <p class="metric-value">Positive</p>
                                    <p style="margin-bottom: 5px; font-weight: 500;">Confidence score: {confidence*100:.2f}%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.progress(confidence)
                        else:
                            st.markdown(
                                f"""
                                <div class="prediction-card-neg">
                                    <h4 style="color: #e74c3c; margin-top:0;">NEGATIVITY DETECTED</h4>
                                    <p class="metric-value">Negative</p>
                                    <p style="margin-bottom: 5px; font-weight: 500;">Confidence score: {confidence*100:.2f}%</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            st.progress(confidence)
                            
                    elif response.status_code == 400:
                        st.error(f"❌ Bad Request: {response.json().get('detail', 'Empty text error')}")
                    else:
                        st.error(f"❌ Server Error: Received code {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection Error: Cannot reach the backend API.")
                    st.info(
                        f"Please check if the FastAPI application is running at **{FASTAPI_URL}**.\n\n"
                        "Run the following command to start the backend:\n"
                        "```bash\nuvicorn api.main:app --port 8000 --reload\n```"
                    )
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
    else:
        st.info("💡 Write a review on the left and click **Analyze Sentiment** to display prediction details here.")

# Sidebar for metadata and prediction history
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Application Status</p>', unsafe_allow_html=True)
    
    # Check health status of API
    try:
        health_resp = requests.get(f"{FASTAPI_URL}/", timeout=2)
        if health_resp.status_code == 200:
            st.success("🟢 API Server: Online")
            st.caption(f"Backend Model: `{health_resp.json().get('model')}`")
        else:
            st.warning("🟡 API Server: Status code " + str(health_resp.status_code))
    except requests.exceptions.RequestException:
        st.error("🔴 API Server: Offline")
        
    st.markdown("---")
    
    st.markdown('<p class="sidebar-title">🕒 Prediction History</p>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.write("No predictions run yet.")
    else:
        for idx, item in enumerate(st.session_state.history[:10]):  # Show last 10 entries
            color = "#2ecc71" if item["sentiment"] == "Positive" else "#e74c3c"
            st.markdown(
                f"""
                <div class="history-item">
                    <p style="margin-bottom:2px; font-size:0.85rem; font-style:italic;">"{item['text']}"</p>
                    <p style="margin-bottom:0; font-weight:bold; font-size:0.9rem; color: {color};">
                        {item['sentiment']} ({item['confidence']*100:.1f}%)
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        if len(st.session_state.history) > 10:
            st.caption(f"Showing last 10 of {len(st.session_state.history)} predictions.")
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
