import pytest
from src.predict import SentimentPredictor

@pytest.fixture(scope="module")
def predictor():
    """Module-scoped fixture to instantiate the predictor once for tests."""
    return SentimentPredictor()


def test_predict_single_review(predictor):
    """Verify inference yields expected labels and valid confidence values."""
    text = "The acting was absolute brilliance."
    sentiment, confidence = predictor.predict(text)
    
    # Assert result types
    assert isinstance(sentiment, str)
    assert sentiment in ["Positive", "Negative"]
    assert isinstance(confidence, float)
    
    # Assert confidence bounds
    assert 0.0 <= confidence <= 1.0


def test_predict_empty_text(predictor):
    """Verify predictor handles empty strings gracefully."""
    sentiment, confidence = predictor.predict("   ")
    assert sentiment == "Neutral/Empty"
    assert confidence == 0.0


def test_predict_batch_reviews(predictor):
    """Verify batch predictions return a list of valid predictions."""
    texts = [
        "What a fantastic cinematic experience!",
        "I regret spending money on this waste of time."
    ]
    results = predictor.predict_batch(texts)
    
    assert len(results) == 2
    for sentiment, confidence in results:
        assert sentiment in ["Positive", "Negative"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
