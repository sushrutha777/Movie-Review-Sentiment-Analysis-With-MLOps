from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_root():
    """Verify that root endpoint returns the correct status and model info."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "running",
        "model": "DistilBERT Sentiment Classifier"
    }


def test_predict_sentiment():
    """Verify that predict endpoint returns valid sentiment classifications."""
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely amazing! I loved it."}
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["Positive", "Negative"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_validation_empty():
    """Verify that empty request text triggers Pydantic schema validation error (422)."""
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    # Pydantic min_length=1 triggers a 422 Unprocessable Entity
    assert response.status_code == 422


def test_predict_validation_whitespace():
    """Verify that whitespace-only review text triggers custom API validation error (400)."""
    response = client.post(
        "/predict",
        json={"text": "      "}
    )
    # Custom validation checks for striped string length and triggers a 400 Bad Request
    assert response.status_code == 400
    assert "Input review text cannot be empty or blank." in response.json()["detail"]
