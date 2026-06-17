from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from api.schemas import PredictionRequest, PredictionResponse, StatusResponse
from api.inference import init_predictor, predict_sentiment
from src.utils import setup_logging

logger = setup_logging("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting FastAPI application...")
    start_time = time.time()
    try:
        init_predictor()
        logger.info(f"FastAPI initialization completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.critical(f"FastAPI failed to load model: {e}", exc_info=True)
    yield
    # Shutdown logic
    logger.info("Stopping FastAPI application...")


app = FastAPI(
    title="Movie Review Sentiment Analysis API",
    description="Production-grade API serving a fine-tuned DistilBERT classifier.",
    version="0.1.0",
    lifespan=lifespan
)


# Exception Handler for generic exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing request: {request.url.path} - Details: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the administrator."}
    )


# Logging middleware for incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {duration:.4f}s"
    )
    return response


@app.get("/", response_model=StatusResponse, status_code=status.HTTP_200_OK)
def read_root():
    """Health check and model status endpoint."""
    return StatusResponse(
        status="running",
        model="DistilBERT Sentiment Classifier"
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest):
    """Predicts movie review sentiment using the fine-tuned DistilBERT model."""
    text_to_predict = request.text.strip()
    if not text_to_predict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input review text cannot be empty or blank."
        )
        
    try:
        sentiment, confidence = predict_sentiment(text_to_predict)
        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference prediction failed: {str(e)}"
        )
