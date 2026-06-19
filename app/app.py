import os
import streamlit as st
import requests

# Base configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Movie Sentiment Analysis MLOps",
    page_icon=None,
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
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #1E293B 0%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .main-title {
            background: linear-gradient(135deg, #F8FAFC 0%, #CBD5E1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    
    @media (prefers-color-scheme: light) {
        .glass-card {
            background: rgba(15, 23, 42, 0.02);
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.03);
        }
    }
    
    /* Sentiment badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.08);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.15);
    }
    
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.08);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.15);
    }
    
    .sentiment-positive-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.15);
        color: #10B981;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(16, 185, 129, 0.05);
    }
    
    .sentiment-negative-card {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(239, 68, 68, 0.05);
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
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    .status-offline {
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
    }
    
    /* Table styling */
    .history-table th {
        background-color: rgba(128, 128, 128, 0.06);
        text-align: left;
        padding: 10px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    
    .history-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        font-size: 0.9rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .history-table th {
            color: #94A3B8;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Content
st.sidebar.markdown("<h2 style='font-weight: 600; font-size: 1.5rem;'>System Control</h2>", unsafe_allow_html=True)
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
        f'<span style="color: #10B981; font-weight: 600;">API Service: Online</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.sidebar.caption(f"Model: **{model_name}**")
else:
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        f'<span class="status-dot status-offline"></span>'
        f'<span style="color: #EF4444; font-weight: 600;">API Service: Offline</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.sidebar.warning("FastAPI Backend is unreachable. Please verify that the API server is running.")

# Main Header (Full width)
st.markdown('<h1 class="main-title">End-to-End Movie Review Sentiment Analysis with MLOps</h1>', unsafe_allow_html=True)
st.markdown("##### Real-time Sentiment Classifier served via FastAPI & DistilBERT")
st.markdown("<br>", unsafe_allow_html=True)

# Main Layout split
col1, col2 = st.columns([7, 5])

with col1:
    st.markdown("<h3 style='margin-top: 0rem; font-weight: 600;'>Enter Movie Review</h3>", unsafe_allow_html=True)
    
    # Textarea input
    review_input = st.text_area(
        "Movie Review Text",
        value="",
        placeholder="Enter a movie review to analyze sentiment...",
        height=180,
        label_visibility="collapsed"
    )
    
    # Predict button
    word_count = len(review_input.strip().split())
    predict_disabled = not api_online
    
    btn_col1, btn_col2 = st.columns([3, 7])
    with btn_col1:
        predict_click = st.button(
            "Analyze Sentiment",
            use_container_width=True,
            disabled=predict_disabled
        )
    with btn_col2:
        if predict_disabled:
            st.caption("Enable the FastAPI backend to activate classification.")
        elif word_count > 0 and word_count < 5:
            st.caption("Note: Short texts (< 5 words) might have less context for the model.")

    if predict_click:
        if not review_input.strip():
            st.warning("Review text cannot be empty!")
        elif len(review_input.split()) < 5:
            st.warning("Please enter at least 5 words for better context.")
        else:
            with st.spinner("Classifying sentiment..."):
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
    st.markdown("<h3 style='margin-top: 0rem; font-weight: 600;'>Prediction Result</h3>", unsafe_allow_html=True)
    
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        is_pos = result["sentiment"] == "Positive"
        
        # Display custom styled card
        if is_pos:
            st.markdown(
                """
                <div class="sentiment-positive-card">
                    <h3 style="margin: 0; font-weight: 600; color: #10B981;">Positive</h3>
                    <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.85;">Model classified the review as positive.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="sentiment-negative-card">
                    <h3 style="margin: 0; font-weight: 600; color: #EF4444;">Negative</h3>
                    <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.85;">Model classified the review as negative.</p>
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
                <h4 style="margin: 0; opacity: 0.6; font-weight: 600;">Ready for Analysis</h4>
                <p style="font-size: 0.9rem; opacity: 0.5; margin-top: 10px;">
                    Enter a movie review and run inference to view sentiment prediction and confidence score.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Prediction History Section (Bottom)
st.markdown("---")
st.markdown("### Recent Predictions")

if st.session_state.history:
    # Build clean HTML table without newlines or spaces to prevent markdown code block rendering
    table_rows = ""
    for idx, item in enumerate(st.session_state.history):
        text_truncated = item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
        badge_class = "badge-positive" if item["sentiment"] == "Positive" else "badge-negative"
        table_rows += (
            '<tr>'
            f'<td style="width: 60%;">{text_truncated}</td>'
            f'<td style="width: 20%;"><span class="badge {badge_class}">{item["sentiment"]}</span></td>'
            f'<td style="font-weight: 600; width: 20%;">{item["confidence"]:.2%}</td>'
            '</tr>'
        )
        
    html_table = (
        '<table class="history-table" style="width:100%; border-collapse: collapse; margin-top: 10px;">'
        '<thead>'
        '<tr>'
        '<th>Review Fragment</th>'
        '<th>Predicted Sentiment</th>'
        '<th>Confidence Score</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{table_rows}</tbody>'
        '</table>'
    )
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.history = []
        if "last_result" in st.session_state:
            del st.session_state.last_result
        st.rerun()
else:
    st.caption("No queries run in this session yet.")