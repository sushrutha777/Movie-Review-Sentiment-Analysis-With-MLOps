from pathlib import Path
from typing import Dict, Union, Tuple, List
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.config import MODEL_NAME, MODEL_SAVE_PATH, MAX_LENGTH
from src.utils import setup_logging

logger = setup_logging("predict")

class SentimentPredictor:
    """Inference wrapper for DistilBERT Sentiment Classifier."""
    
    def __init__(self, model_path: Union[str, Path] = MODEL_SAVE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load local trained model if exists, otherwise load pre-trained base model
        if (self.model_path / "config.json").exists():
            logger.info(f"Loading trained Sentiment Analysis model from {self.model_path}")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_path))
            self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_path))
        else:
            logger.warning(
                f"No trained model found at {self.model_path}. "
                f"Falling back to pre-trained base model '{MODEL_NAME}' for testing/inference."
            )
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
            self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Tuple[str, float]:
        """Predicts the sentiment and confidence score of a single text string."""
        if not text.strip():
            return "Neutral/Empty", 0.0
            
        inputs = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1).flatten()
            
        pred_label_id = torch.argmax(probabilities).item()
        confidence = probabilities[pred_label_id].item()
        
        sentiment = "Positive" if pred_label_id == 1 else "Negative"
        
        logger.info(f"Inference input: '{text[:50]}...' -> Prediction: {sentiment} ({confidence:.4f})")
        return sentiment, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predicts the sentiment of a batch of text strings."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
