from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Schema for prediction request containing the movie review text."""
    text: str = Field(
        ...,
        description="The movie review text to analyze.",
        min_length=1,
        example="This movie was absolutely fantastic."
    )

class PredictionResponse(BaseModel):
    """Schema for prediction response containing sentiment label and confidence score."""
    sentiment: str = Field(
        ...,
        description="The predicted sentiment label (e.g., Positive or Negative).",
        example="Positive"
    )
    confidence: float = Field(
        ...,
        description="The probability score associated with the prediction (between 0.0 and 1.0).",
        ge=0.0,
        le=1.0,
        example=0.98
    )

class StatusResponse(BaseModel):
    """Schema for the API health check response."""
    status: str = Field(..., example="running")
    model: str = Field(..., example="DistilBERT Sentiment Classifier")
