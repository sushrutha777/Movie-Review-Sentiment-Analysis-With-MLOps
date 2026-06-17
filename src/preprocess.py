import os
from typing import Tuple, List, Optional
import numpy as np
from transformers import AutoTokenizer
from src.utils import setup_logging
from src.config import RANDOM_SEED, MODEL_NAME, MAX_LENGTH, BATCH_SIZE

logger = setup_logging("preprocess")


def generate_mock_data(num_samples: int) -> dict:
    """Generates synthetic movie review data for offline testing and fast CI/CD pipelines.
    
    Returns a dict with 'train' and 'test' splits, each containing 'text' and 'label' lists.
    """
    logger.info(f"Generating {num_samples} mock movie reviews...")
    
    positive_templates = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged.",
        "A masterpiece of modern cinema. Beautiful cinematography and outstanding directing.",
        "Highly recommended! I loved every single second of it, especially the climax.",
        "One of the best films of the year. The lead actors did an incredible job.",
        "Amazing story and great special effects. A solid 10/10 experience.",
        "I was pleasantly surprised. The writing is sharp and the characters are very relatable.",
        "An emotional rollercoaster that delivers on every level. Brilliant performance."
    ]
    
    negative_templates = [
        "This was a terrible waste of time. The acting was wooden and the plot made no sense.",
        "Unbelievably boring. I fell asleep halfway through the movie.",
        "Very disappointing. The trailer was much better than the actual film.",
        "Horrible screenplay and awful directing. Avoid at all costs.",
        "I hated this film. It felt extremely cheap and the dialog was painful to listen to.",
        "A complete disaster. It fails to capture any of the magic of the original book.",
        "Worst movie I have seen in years. Poor acting, slow pacing, and terrible ending."
    ]

    texts = []
    labels = []
    
    for i in range(num_samples):
        # Alternate positive and negative
        idx = (RANDOM_SEED + i) % len(positive_templates)
        if i % 2 == 0:
            texts.append(positive_templates[idx])
            labels.append(1)
        else:
            texts.append(negative_templates[idx])
            labels.append(0)
            
    # Split 80/20 into train/test
    split_idx = int(num_samples * 0.8)
    
    return {
        "train": {"text": texts[:split_idx], "label": labels[:split_idx]},
        "test": {"text": texts[split_idx:], "label": labels[split_idx:]},
    }


def load_and_tokenize_data(
    max_samples: int = -1,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
):
    """Loads the IMDB dataset, tokenizes it, and returns TF datasets.
    
    This replicates the exact pipeline from the Colab notebook:
    1. Load IMDB via HuggingFace datasets
    2. Tokenize with AutoTokenizer (padding="max_length", truncation=True)
    3. Convert to tf.data.Dataset via .to_tf_dataset()
    
    Args:
        max_samples: Max training samples (-1 for full dataset).
        max_length: Tokenizer max sequence length.
        batch_size: Batch size for TF datasets.
        
    Returns:
        Tuple of (train_tf_dataset, test_tf_dataset, tokenizer)
    """
    import tensorflow as tf
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    try:
        from datasets import load_dataset, Dataset
        logger.info("Attempting to load IMDB dataset from Hugging Face datasets...")
        dataset = load_dataset("imdb")
        logger.info(f"Loaded IMDB dataset. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
        
    except Exception as e:
        from datasets import Dataset, DatasetDict
        logger.warning(f"Could not load dataset from HuggingFace ({e}). Falling back to mock data...")
        
        mock_count = 200 if max_samples <= 0 else max(50, int(max_samples * 1.5))
        mock_data = generate_mock_data(mock_count)
        
        dataset = DatasetDict({
            "train": Dataset.from_dict(mock_data["train"]),
            "test": Dataset.from_dict(mock_data["test"]),
        })
        logger.info(f"Generated mock dataset. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    # Cap samples if max_samples is set (for testing or fast CPU runs)
    if max_samples > 0:
        logger.info(f"Capping training samples to {max_samples}")
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
        test_cap = max(10, int(max_samples * 0.2))
        dataset["test"] = dataset["test"].select(range(min(test_cap, len(dataset["test"]))))

    # Tokenize — matches notebook's tokenize_batch function exactly
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    logger.info("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_batch, batched=True)
    
    # Post-process to match notebook:
    # Remove raw text column, rename label -> labels, set TF format
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="tensorflow",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Convert to tf.data.Dataset — matches notebook exactly
    logger.info("Converting to tf.data.Dataset...")
    train_tf = tokenized_datasets["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=True,
        batch_size=batch_size,
    )

    test_tf = tokenized_datasets["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols=["labels"],
        shuffle=False,
        batch_size=batch_size,
    )

    logger.info(
        f"Final dataset sizes -> Train: {len(tokenized_datasets['train'])}, "
        f"Test: {len(tokenized_datasets['test'])}"
    )

    return train_tf, test_tf, tokenizer
