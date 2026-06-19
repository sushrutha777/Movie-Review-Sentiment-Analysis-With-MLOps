import os
import streamlit as st
import requests

# Base configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Sentify - DistilBERT MLOps",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Main body font styling */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Title gradient */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    /* Dark/Light mode support adjustments */
    @media (prefers-color-scheme: light) {
        .glass-card {
            background: rgba(0, 0, 0, 0.03);
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.05);
        }
    }
    
    /* Sentiment badges */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-positive {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    
    .badge-negative {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    
    .sentiment-positive-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(56, 239, 125, 0.3);
    }
    
    .sentiment-negative-card {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(239, 71, 58, 0.3);
    }
    
    /* Status indicators */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-online {
        background-color: #2ecc71;
        box-shadow: 0 0 8px #2ecc71;
    }
    .status-offline {
        background-color: #e74c3c;
        box-shadow: 0 0 8px #e74c3c;
    }
    
    /* Table headers styling */
    .history-table th {
        background-color: rgba(255, 255, 255, 0.05);
        text-align: left;
        padding: 8px;
    }
    
    .history-table td {
        padding: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = ""

# Sidebar Content
st.sidebar.markdown("<h2 style='font-weight: 800;'>⚙️ System Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Backend Endpoint:** `{FASTAPI_URL}`")

# Health Check connection status
@st.cache_data(ttl=5)
def check_api_status(url):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("model", "DistilBERT Classifier")
    except Exception:
        pass
    return False, None

api_online, model_name = check_api_status(FASTAPI_URL)

if api_online:
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        f'<span class="status-dot status-online"></span>'
        f'<span style="color: #2ecc71; font-weight: 600;">API Service: Online</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(f"🤖 Loaded: **{model_name}**")
else:
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        f'<span class="status-dot status-offline"></span>'
        f'<span style="color: #e74c3c; font-weight: 600;">API Service: Offline</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.sidebar.warning("⚠️ FastAPI Backend is unreachable. Please verify that the API server is running.")

# Sample Reviews Library
positive_samples = [
    "Amazing soundtrack, perfect pacing, and visuals that made the experience feel magical and cinematic.",
    "This movie was fantastic! The acting was great and the plot was thrilling.",
    "Exceeded expectations with inspiring storytelling, top-notch acting, and a powerful emotional message.",
]

negative_samples = [
    "Visual effects were cheap, editing inconsistent, and the narrative failed to engage at all.",
    "Started strong but didn't maintain the energy or emotional impact.",
    "Terrible experience! The film dragged endlessly and made no sense at all.",
]

with st.sidebar.expander("📁 Sample Reviews"):
    st.markdown("**Positive Reviews**")
    for sample in positive_samples:
        if st.button(sample[:30] + "...", key=f"pos_{hash(sample)}"):
            st.session_state.selected_sample = sample
            st.rerun()
            
    st.markdown("**Negative Reviews**")
    for sample in negative_samples:
        if st.button(sample[:30] + "...", key=f"neg_{hash(sample)}"):
            st.session_state.selected_sample = sample
            st.rerun()

# Main Layout split
col1, col2 = st.columns([7, 5])

with col1:
    st.markdown('<h1 class="main-title">Sentify MLOps Portal</h1>', unsafe_allow_html=True)
    st.markdown("##### Real-time Sentiment Classifier served via FastAPI & DistilBERT")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Textarea input
    review_input = st.text_area(
        "💬 Movie Review Text:",
        value=st.session_state.selected_sample,
        placeholder="Type a review here... (e.g., 'An absolute cinematic masterpiece, outstanding performances!')",
        height=180
    )
    
    # Predict button
    word_count = len(review_input.strip().split())
    predict_disabled = not api_online
    
    btn_col1, btn_col2 = st.columns([3, 7])
    with btn_col1:
        predict_click = st.button(
            "Classify Sentiment 🚀",
            use_container_width=True,
            disabled=predict_disabled
        )
    with btn_col2:
        if predict_disabled:
            st.caption("🔴 Enable the FastAPI backend to activate classification.")
        elif word_count > 0 and word_count < 5:
            st.caption("⚠️ Note: Short texts (< 5 words) might have less context for the model.")
        else:
            st.caption("⚡ Submits request to FastAPI /predict endpoint")

    if predict_click:
        if not review_input.strip():
            st.warning("Review text cannot be empty!")
        elif len(review_input.split()) < 5:
            st.warning("Please enter at least 5 words for better context.")
        else:
            with st.spinner("Invoking DistilBERT via FastAPI..."):
                try:
                    payload = {"text": review_input.strip()}
                    response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=5)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        sentiment = res_data["sentiment"]
                        confidence = res_data["confidence"]
                        
                        # Add to history
                        st.session_state.history.insert(0, {
                            "text": review_input.strip(),
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                        # Limit to last 10
                        st.session_state.history = st.session_state.history[:10]
                        
                        # Save success result to display
                        st.session_state.last_result = {
                            "sentiment": sentiment,
                            "confidence": confidence
                        }
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to API server: {e}")

# Results and History
with col2:
    st.markdown("<h3 style='margin-top: 2rem;'>📊 Classification Result</h3>", unsafe_allow_html=True)
    
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        is_pos = result["sentiment"] == "Positive"
        
        # Display custom styled card
        if is_pos:
            st.markdown(
                f"""
                <div class="sentiment-positive-card">
                    <h2 style="color: white; margin: 0; font-weight: 800;">Positive 😊</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">DistilBERT classified the review as positive.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="sentiment-negative-card">
                    <h2 style="color: white; margin: 0; font-weight: 800;">Negative 😞</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">DistilBERT classified the review as negative.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Display progress bar for confidence
        st.metric(label="Model Confidence Score", value=f"{result['confidence']:.2%}")
        st.progress(result["confidence"])
    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 40px 20px;">
                <h4 style="margin: 0; opacity: 0.6;">Awaiting Review Submission</h4>
                <p style="font-size: 0.9rem; opacity: 0.5; margin-top: 10px;">
                    Type a movie review on the left and click Classify Sentiment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Prediction History Section (Bottom)
st.markdown("---")
st.markdown("### 🕒 Prediction History (Session State)")

if st.session_state.history:
    # Build clean HTML table
    table_rows = ""
    for idx, item in enumerate(st.session_state.history):
        text_truncated = item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
        badge_class = "badge-positive" if item["sentiment"] == "Positive" else "badge-negative"
        table_rows += f"""
        <tr>
            <td style="font-size: 0.9rem; width: 60%;">{text_truncated}</td>
            <td style="width: 20%;"><span class="badge {badge_class}">{item["sentiment"]}</span></td>
            <td style="font-weight: 600; width: 20%;">{item["confidence"]:.2%}</td>
        </tr>
        """
        
    st.markdown(
        f"""
        <table class="history-table" style="width:100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th>Review Fragment</th>
                    <th>Predicted Sentiment</th>
                    <th>Confidence Score</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )
    if st.button("🧹 Clear History"):
        st.session_state.history = []
        if "last_result" in st.session_state:
            del st.session_state.last_result
        st.rerun()
else:
    st.caption("No queries run in this session yet.")