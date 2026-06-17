import os
from pathlib import Path

# Force TensorFlow to use the legacy Keras 2 backend (required for TF 2.16+
# compatibility with Hugging Face Transformers TF classes)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_SAVE_PATH = BASE_DIR / "notebooks" / "distilbert_imdb_tf_model"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Model & Tokenizer Config
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

# Training Hyperparameters (matched to notebook)
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Dataset Configuration
# Limit samples for quick demo and testing runs. Set to -1 to use the full dataset.
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "-1"))

# MLflow Tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "MovieSentimentAnalysis"
REGISTERED_MODEL_NAME = "DistilBERTSentimentModel"
