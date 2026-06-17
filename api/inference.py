import os
from typing import Tuple

# Ensure TF_USE_LEGACY_KERAS is set before any TF imports
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from src.predict import SentimentPredictor
from src.utils import setup_logging

logger = setup_logging("api_inference")

# Lazy-loaded singleton predictor
_predictor: SentimentPredictor = None

def init_predictor() -> None:
    """Explicitly initializes the predictor during application startup."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing API SentimentPredictor (TensorFlow)...")
        _predictor = SentimentPredictor()
        logger.info("SentimentPredictor loaded successfully.")

def get_predictor() -> SentimentPredictor:
    """Retrieves the singleton predictor, initializing it if necessary."""
    global _predictor
    if _predictor is None:
        init_predictor()
    return _predictor

def predict_sentiment(text: str) -> Tuple[str, float]:
    """Wraps inference call and logs requests."""
    predictor = get_predictor()
    sentiment, confidence = predictor.predict(text)
    return sentiment, confidence
