import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "distilbert_v1"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Model & Tokenizer Config
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

# Training Hyperparameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Dataset Configuration
# Limit samples for quick demo and testing runs. Set to -1 to use the full dataset.
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "1000"))

# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "MovieSentimentAnalysis"
REGISTERED_MODEL_NAME = "DistilBERTSentimentModel"
