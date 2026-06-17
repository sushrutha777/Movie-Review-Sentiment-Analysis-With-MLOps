import json
import os
import time
from pathlib import Path

# Ensure TF_USE_LEGACY_KERAS is set before importing TF
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from transformers import TFAutoModelForSequenceClassification

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MAX_SAMPLES,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    NUM_LABELS,
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
)
from src.preprocess import load_and_tokenize_data
from src.utils import set_seed, setup_logging

logger = setup_logging("train")


def main() -> None:
    """Main training pipeline — mirrors the Colab notebook approach.
    
    Pipeline:
    1. Load & tokenize IMDB dataset → tf.data.Dataset
    2. Initialize TFDistilBertForSequenceClassification
    3. Compile with Adam(lr=2e-5), SparseCategoricalCrossentropy(from_logits=True)
    4. model.fit() with validation
    5. Log metrics to MLflow
    6. Save model + tokenizer locally and to MLflow registry
    """
    logger.info("Initializing Sentiment Analysis Training Pipeline (TensorFlow)")
    set_seed(RANDOM_SEED)

    # MLflow Setup
    logger.info(f"Setting tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Using GPU: {gpus}")
        # Allow memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU found. Training on CPU.")

    # Load Data
    logger.info("Loading dataset and creating TF datasets...")
    train_tf, test_tf, tokenizer = load_and_tokenize_data(
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # Initialize Model — matches notebook exactly
    logger.info(f"Initializing TFDistilBert model: {MODEL_NAME} (num_labels={NUM_LABELS})")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        from_pt=True,
    )

    # Compile — matches notebook exactly
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary(print_fn=logger.info)

    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow Run: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "TFDistilBERT",
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "max_samples": MAX_SAMPLES,
            "num_labels": NUM_LABELS,
            "optimizer": "Adam",
            "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        })

        # Train — matches notebook: model.fit(train_tf, validation_data=test_tf, epochs=EPOCHS)
        logger.info(f"Starting Training for {EPOCHS} epochs...")
        history = model.fit(
            train_tf,
            validation_data=test_tf,
            epochs=EPOCHS,
        )

        # Log per-epoch metrics to MLflow
        for epoch_idx in range(len(history.history["loss"])):
            epoch = epoch_idx + 1
            mlflow.log_metric("train_loss", history.history["loss"][epoch_idx], step=epoch)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch_idx], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch_idx], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch_idx], step=epoch)
            
            logger.info(
                f"Epoch {epoch}/{EPOCHS} - "
                f"loss: {history.history['loss'][epoch_idx]:.4f} - "
                f"accuracy: {history.history['accuracy'][epoch_idx]:.4f} - "
                f"val_loss: {history.history['val_loss'][epoch_idx]:.4f} - "
                f"val_accuracy: {history.history['val_accuracy'][epoch_idx]:.4f}"
            )

        # Evaluate on test set
        logger.info("Evaluating on Test Set...")
        eval_results = model.evaluate(test_tf)
        test_loss, test_accuracy = eval_results[0], eval_results[1]
        logger.info(f"Final Test Loss: {test_loss:.4f}")
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })

        # Save model + tokenizer locally — matches notebook's model.save_pretrained()
        logger.info(f"Saving model to: {MODEL_SAVE_PATH}")
        model.save_pretrained(str(MODEL_SAVE_PATH))
        tokenizer.save_pretrained(str(MODEL_SAVE_PATH))

        # Write metadata file
        meta = {
            "epochs": EPOCHS,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(MODEL_SAVE_PATH / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Model metadata saved to {MODEL_SAVE_PATH / 'meta.json'}")

        # Save training history
        history_path = MODEL_SAVE_PATH / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=4)
        logger.info(f"Training history saved to {history_path}")

        # Log model artifact to MLflow
        logger.info("Logging model to MLflow...")
        mlflow.log_artifacts(str(MODEL_SAVE_PATH), artifact_path="model")
        logger.info("Model logged to MLflow successfully!")

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
