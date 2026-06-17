import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.config import (
    BATCH_SIZE,
    MAX_LENGTH,
    MAX_SAMPLES,
    MODEL_SAVE_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.preprocess import get_data_loaders
from src.utils import setup_logging

logger = setup_logging("evaluate")

def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path) -> None:
    """Plots and saves a beautiful, professional confusion matrix."""
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


def run_evaluation() -> dict[str, float]:
    """Runs evaluation on the test split using the saved best model."""
    logger.info("Initializing Evaluation Script")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check if saved model exists
    if not (MODEL_SAVE_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_SAVE_PATH}. Please run training first."
        )

    # Load Model & Tokenizer
    logger.info(f"Loading best local model from {MODEL_SAVE_PATH}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
    model.to(device)
    model.eval()

    # Load Test Data Loader
    _, _, test_loader = get_data_loaders(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        max_samples=MAX_SAMPLES
    )

    all_preds = []
    all_labels = []
    total_loss = 0.0

    logger.info("Running evaluation predictions on the test dataset...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    logger.info("Evaluation metrics computed successfully.")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")

    # Create logs directory if not exists
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = logs_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes=["Negative", "Positive"], output_path=cm_path)

    # Print Classification Report
    report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
    logger.info(f"\nClassification Report:\n{report}")

    # Log to MLflow if an active run exists, or start a new evaluation run
    active_run = mlflow.active_run()
    if active_run:
        logger.info("Logging evaluation results to current active MLflow run.")
        mlflow.log_metric("eval_loss", avg_loss)
        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_precision", precision)
        mlflow.log_metric("eval_recall", recall)
        mlflow.log_metric("eval_f1_score", f1)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
    else:
        logger.info("No active MLflow run detected. Starting a dedicated evaluation run.")
        with mlflow.start_run(run_name="Standalone_Evaluation") as run:
            mlflow.log_params({
                "eval_model_path": str(MODEL_SAVE_PATH),
                "max_samples": MAX_SAMPLES
            })
            mlflow.log_metric("eval_loss", avg_loss)
            mlflow.log_metric("eval_accuracy", accuracy)
            mlflow.log_metric("eval_precision", precision)
            mlflow.log_metric("eval_recall", recall)
            mlflow.log_metric("eval_f1_score", f1)
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
            logger.info(f"Evaluation results logged to run ID: {run.info.run_id}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":
    run_evaluation()
