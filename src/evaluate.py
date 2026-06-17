import json
import os
from pathlib import Path

# Ensure TF_USE_LEGACY_KERAS is set before importing TF
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_SAMPLES,
    MODEL_SAVE_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    NUM_LABELS,
)
from src.preprocess import load_and_tokenize_data
from src.utils import setup_logging

logger = setup_logging("evaluate")


def plot_confusion_matrix(cm: np.ndarray, classes: list, output_path: Path) -> None:
    """Plots and saves a professional confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Sentiment Analysis")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format text inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def run_evaluation() -> dict:
    """Runs evaluation on the test split using the saved best TF model.
    
    Pipeline:
    1. Load saved TFDistilBert model + tokenizer
    2. Load test dataset as tf.data.Dataset
    3. model.evaluate() for loss/accuracy
    4. Predict on all test samples for confusion matrix + classification report
    5. Log results to MLflow
    """
    logger.info("Initializing Evaluation Script (TensorFlow)")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Check if saved model exists
    if not (MODEL_SAVE_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_SAVE_PATH}. Please run training first."
        )

    # Load Model & Tokenizer
    logger.info(f"Loading saved TF model from {MODEL_SAVE_PATH}")
    model = TFAutoModelForSequenceClassification.from_pretrained(str(MODEL_SAVE_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SAVE_PATH))

    # Load Test Data
    _, test_tf, _ = load_and_tokenize_data(
        max_samples=MAX_SAMPLES,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
    )

    # Evaluate with Keras — matches notebook: model.evaluate(test_tf)
    logger.info("Running model.evaluate() on test dataset...")
    eval_results = model.evaluate(test_tf)
    avg_loss, accuracy_keras = eval_results[0], eval_results[1]
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy (Keras): {accuracy_keras:.4f}")

    # Get predictions for detailed metrics (confusion matrix, classification report)
    logger.info("Running prediction loop for detailed metrics...")
    all_preds = []
    all_labels = []

    for batch in test_tf:
        inputs, labels = batch
        outputs = model(inputs, training=False)
        logits = outputs.logits
        preds = tf.argmax(logits, axis=1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    # Calculate sklearn metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    # Create logs directory
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute and save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = logs_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes=["Negative", "Positive"], output_path=cm_path)

    # Print Classification Report
    report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
    logger.info(f"\nClassification Report:\n{report}")

    # Save classification report to file
    report_path = logs_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Log to MLflow
    active_run = mlflow.active_run()
    if active_run:
        logger.info("Logging evaluation results to current active MLflow run.")
        _log_eval_metrics(mlflow, avg_loss, accuracy, precision, recall, f1, cm_path)
    else:
        logger.info("No active MLflow run detected. Starting a dedicated evaluation run.")
        with mlflow.start_run(run_name="Standalone_Evaluation") as run:
            mlflow.log_params({
                "eval_model_path": str(MODEL_SAVE_PATH),
                "max_samples": MAX_SAMPLES,
            })
            _log_eval_metrics(mlflow, avg_loss, accuracy, precision, recall, f1, cm_path)
            logger.info(f"Evaluation results logged to run ID: {run.info.run_id}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def _log_eval_metrics(mlflow_module, loss, accuracy, precision, recall, f1, cm_path):
    """Helper to log evaluation metrics to MLflow."""
    mlflow_module.log_metric("eval_loss", loss)
    mlflow_module.log_metric("eval_accuracy", accuracy)
    mlflow_module.log_metric("eval_precision", precision)
    mlflow_module.log_metric("eval_recall", recall)
    mlflow_module.log_metric("eval_f1_score", f1)
    mlflow_module.log_artifact(str(cm_path), artifact_path="evaluation_plots")


if __name__ == "__main__":
    run_evaluation()
