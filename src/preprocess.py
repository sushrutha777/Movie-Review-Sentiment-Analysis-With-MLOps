import os
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import setup_logging
from src.config import RANDOM_SEED

logger = setup_logging("preprocess")

class MovieReviewDataset(Dataset):
    """PyTorch Dataset for Tokenized Movie Reviews."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizerFast, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def generate_mock_data(num_samples: int) -> Tuple[List[str], List[int]]:
    """Generates synthetic movie review data for offline testing and fast CI/CD pipelines."""
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
        if i % 2 == 0:
            texts.append(random_choice_with_seed(positive_templates, i))
            labels.append(1)
        else:
            texts.append(random_choice_with_seed(negative_templates, i))
            labels.append(0)
            
    return texts, labels


def random_choice_with_seed(templates: List[str], seed_offset: int) -> str:
    """Selects a template deterministically using the offset."""
    # Simple deterministic hash selection to avoid raw random side effects
    idx = (RANDOM_SEED + seed_offset) % len(templates)
    return templates[idx]


def load_imdb_data(max_samples: int = -1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the IMDB movie reviews dataset. Falls back to mock data if offline/error."""
    try:
        from datasets import load_dataset
        logger.info("Attempting to load IMDB dataset from Hugging Face datasets...")
        dataset = load_dataset("imdb", trust_remote_code=True)
        
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
        
        # Shuffle
        train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        logger.info(f"Loaded IMDB dataset. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
    except Exception as e:
        logger.warning(f"Could not load dataset from HuggingFace ({e}). Falling back to generating mock data...")
        
        # Generate 1500 mock samples to divide into train (1000), val (250), test (250)
        mock_samples = 1500 if max_samples <= 0 else int(max_samples * 1.5)
        texts, labels = generate_mock_data(mock_samples)
        
        df = pd.DataFrame({"text": texts, "label": labels})
        # Split mock data
        train_df = df.iloc[:int(mock_samples * 0.7)].reset_index(drop=True)
        test_df = df.iloc[int(mock_samples * 0.7):].reset_index(drop=True)
        
    # Validation split from training data
    val_size = int(len(train_df) * 0.2)
    val_df = train_df.iloc[:val_size].reset_index(drop=True)
    train_df = train_df.iloc[val_size:].reset_index(drop=True)
    
    # Cap samples if max_samples is set (for testing or fast CPU execution)
    if max_samples > 0:
        logger.info(f"Capping training samples to {max_samples}")
        train_df = train_df.head(max_samples).reset_index(drop=True)
        val_df = val_df.head(max(10, int(max_samples * 0.2))).reset_index(drop=True)
        test_df = test_df.head(max(10, int(max_samples * 0.2))).reset_index(drop=True)

    logger.info(f"Final dataset split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_data_loaders(
    tokenizer: DistilBertTokenizerFast,
    batch_size: int,
    max_length: int,
    max_samples: int = -1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Generates PyTorch DataLoaders for training, validation, and testing."""
    train_df, val_df, test_df = load_imdb_data(max_samples)

    train_dataset = MovieReviewDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = MovieReviewDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = MovieReviewDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Pin memory for faster GPU transfers if available
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader
