import json
import os
import time
from pathlib import Path
import mlflow
import mlflow.transformers
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

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
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
)
from src.preprocess import get_data_loaders
from src.utils import set_seed, setup_logging

logger = setup_logging("train")

def train_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % 20 == 0:
            logger.info(f"  Batch {batch_idx}/{len(data_loader)} - Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[float, dict[str, float]]:
    """Evaluates the model on validation/test loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
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

    avg_loss = total_loss / len(data_loader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return avg_loss, metrics


def main() -> None:
    logger.info("Initializing Sentiment Analysis Training Pipeline")
    set_seed(RANDOM_SEED)

    # MLflow Setup
    logger.info(f"Setting tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Tokenizer & Model
    logger.info(f"Loading pre-trained DistilBERT tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    logger.info(f"Loading pre-trained DistilBERT model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Load Data Loaders
    logger.info("Loading dataset and creating PyTorch DataLoaders")
    train_loader, val_loader, test_loader = get_data_loaders(
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        max_samples=MAX_SAMPLES
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_loss = float("inf")
    best_epoch = -1

    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow Run: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "DistilBERT",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "max_samples": MAX_SAMPLES,
            "device": str(device),
            "optimizer": "AdamW"
        })

        for epoch in range(1, EPOCHS + 1):
            logger.info(f"Starting Epoch {epoch}/{EPOCHS}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)
            logger.info(f"Epoch {epoch} - Average Train Loss: {train_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Validate
            val_loss, val_metrics = evaluate(model, val_loader, device)
            logger.info(
                f"Epoch {epoch} - Val Loss: {val_loss:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.4f} | "
                f"F1 Score: {val_metrics['f1_score']:.4f}"
            )
            
            # Log validation metrics
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", val_metrics["accuracy"], step=epoch)
            mlflow.log_metric("precision", val_metrics["precision"], step=epoch)
            mlflow.log_metric("recall", val_metrics["recall"], step=epoch)
            mlflow.log_metric("f1_score", val_metrics["f1_score"], step=epoch)

            # Check if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                logger.info(f"New best model found at epoch {epoch}! Saving locally to {MODEL_SAVE_PATH}")
                
                # Save model and tokenizer locally
                model.save_pretrained(MODEL_SAVE_PATH)
                tokenizer.save_pretrained(MODEL_SAVE_PATH)
                
                # Write a tiny meta file
                meta = {
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "accuracy": val_metrics["accuracy"],
                    "f1_score": val_metrics["f1_score"],
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(MODEL_SAVE_PATH / "meta.json", "w") as f:
                    json.dump(meta, f, indent=4)

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # Run final evaluation on the test set using the best model
        logger.info("Loading best model for test evaluation...")
        best_model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        best_model.to(device)
        
        test_loss, test_metrics = evaluate(best_model, test_loader, device)
        logger.info(
            f"Test Set Evaluation -> Loss: {test_loss:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1 Score: {test_metrics['f1_score']:.4f}"
        )
        
        # Log test metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"]
        })

        # Log Hugging Face model components as an MLflow Transformer model
        logger.info("Logging best model to MLflow Model Registry...")
        
        # Define component dictionary for log_model
        transformers_components = {
            "model": best_model,
            "tokenizer": tokenizer,
        }
        
        # We also need a pipeline mapping
        # MLflow's transformers flavor simplifies loading model + tokenizer
        mlflow.transformers.log_model(
            transformers_model=transformers_components,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        logger.info("Model registered in MLflow registry successfully!")

if __name__ == "__main__":
    main()
