import os
from pathlib import Path

# Force TensorFlow to use the legacy Keras 2 backend (required for TF 2.16+
# compatibility with Hugging Face Transformers TF classes)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_SAVE_PATH = BASE_DIR / "models" / "distilbert_imdb_tf_model"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Model & Tokenizer Config
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))

