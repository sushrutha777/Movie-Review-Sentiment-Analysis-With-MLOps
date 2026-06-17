import os

# CRITICAL CONFIGURATION
# This MUST be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Use the project's centralized model save path from config
from src.config import MODEL_SAVE_PATH, MODEL_NAME, MAX_LENGTH

# Cache resources so they load only once
@st.cache_resource
def load_resources():
    # 1. Define the local folder where your tf_model.h5 and config.json are located
    local_model_path = str(MODEL_SAVE_PATH)
    
    # 2. Load the TOKENIZER from the Internet
    # We grab the standard one because the vocab is standard and files were missing locally
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None, None

    # 3. Load the MODEL from your Local Folder
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(local_model_path)
    except OSError:
        st.error(f" Could not find folder '{local_model_path}'. Please make sure it is in the same directory as app.py")
        return None, None
        
    return tokenizer, model

# Load resources
tokenizer, model = load_resources()

# Sample review categories (Same as before)
positive_reviews = [
    "Amazing soundtrack, perfect pacing, and visuals that made the experience feel magical and cinematic.",
    "This movie was fantastic! The acting was great and the plot was thrilling.",
    "Exceeded expectations with inspiring storytelling, top-notch acting, and a powerful emotional message.",
]

negative_reviews = [
    "Visual effects were cheap, editing inconsistent, and the narrative failed to engage at all.",
    "Started strong but didn't maintain the energy or emotional impact",
    "Terrible experience! The film dragged endlessly and made no sense at all.",
]

# Streamlit configuration
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below and let the **DistilBERT Transformer** model predict its sentiment!")

# Display sample categories
with st.expander("📁 Sample Review Library (Click to Explore)"):
    st.markdown("**Positive Reviews**")
    for rev in positive_reviews:
        st.markdown(f"- {rev}")
    st.markdown("**Negative Reviews**")
    for rev in negative_reviews:
        st.markdown(f"- {rev}")

# User input
review = st.text_area("💬 Enter your movie review (at least 5 words):")

# Prediction logic
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    elif len(review.split()) < 5:
        st.warning("Please enter at least 5 words for better context.")
    else:
        if model is None or tokenizer is None:
            st.error("Model or Tokenizer failed to load. Please check your files.")
        else:
            with st.spinner("Analyzing with DistilBERT..."):
                # 1. Preprocess using the Tokenizer (No more regex/word_index!)
                inputs = tokenizer(
                    review, 
                    return_tensors="tf", 
                    truncation=True, 
                    padding=True, 
                    max_length=MAX_LENGTH
                )

                # 2. Predict
                outputs = model(inputs)
                logits = outputs.logits
                
                # 3. Convert logits to probabilities using Softmax
                probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
                
                # Index 0 = Negative, Index 1 = Positive
                neg_score = probabilities[0]
                pos_score = probabilities[1]

                # Determine Sentiment
                if pos_score > neg_score:
                    sentiment = "Positive 😊"
                    confidence = pos_score
                    is_positive = True
                else:
                    sentiment = "Negative 😞"
                    confidence = neg_score
                    is_positive = False

                # 4. Display Result
                st.subheader("Prediction Result")
                if is_positive:
                    st.success(f"**Sentiment:** {sentiment}")
                else:
                    st.error(f"**Sentiment:** {sentiment}")
                
                st.info(f"Confidence Score: {confidence:.2f}")