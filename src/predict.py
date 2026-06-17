import os
from pathlib import Path
from typing import Tuple, List, Union

# Ensure TF_USE_LEGACY_KERAS is set before importing TF
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from src.config import MODEL_NAME, MODEL_SAVE_PATH, MAX_LENGTH, NUM_LABELS
from src.utils import setup_logging

logger = setup_logging("predict")


class SentimentPredictor:
    """Inference wrapper for TF DistilBERT Sentiment Classifier.
    
    Loads the fine-tuned TF model and tokenizer, and provides
    predict() / predict_batch() methods matching the original API.
    """
    
    def __init__(self, model_path: Union[str, Path] = MODEL_SAVE_PATH):
        self.model_path = Path(model_path)
        
        # Load local trained model if exists, otherwise load pre-trained base model
        if (self.model_path / "config.json").exists():
            logger.info(f"Loading trained TF Sentiment Analysis model from {self.model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            except Exception:
                logger.info(f"Local tokenizer files not found in {self.model_path}. Falling back to base tokenizer: {MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        else:
            logger.warning(
                f"No trained model found at {self.model_path}. "
                f"Falling back to pre-trained base model '{MODEL_NAME}' for testing/inference."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS, from_pt=True
            )

    def predict(self, text: str) -> Tuple[str, float]:
        """Predicts the sentiment and confidence score of a single text string.
        
        Args:
            text: The movie review text to classify.
            
        Returns:
            Tuple of (sentiment_label, confidence_score).
        """
        if not text.strip():
            return "Neutral/Empty", 0.0
            
        inputs = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        
        outputs = self.model(inputs, training=False)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy().flatten()
        
        pred_label_id = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_label_id])
        
        sentiment = "Positive" if pred_label_id == 1 else "Negative"
        
        logger.info(f"Inference input: '{text[:50]}...' -> Prediction: {sentiment} ({confidence:.4f})")
        return sentiment, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predicts the sentiment of a batch of text strings."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
