import pandas as pd
import pytest
from transformers import DistilBertTokenizerFast
import torch
from src.preprocess import load_imdb_data, MovieReviewDataset, get_data_loaders
from src.config import MODEL_NAME

def test_load_imdb_data_splits():
    """Verify data loading function splits data correctly and respects max_samples."""
    max_samples = 20
    train, val, test = load_imdb_data(max_samples=max_samples)
    
    # Assert return types are DataFrames
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
    # Verify split size respects limits
    assert len(train) == max_samples
    assert len(val) == max(10, int(max_samples * 0.2))
    assert len(test) == max(10, int(max_samples * 0.2))
    
    # Verify required columns exist
    for df in [train, val, test]:
        assert "text" in df.columns
        assert "label" in df.columns
        assert df["label"].isin([0, 1]).all()


def test_movie_review_dataset():
    """Verify PyTorch dataset tokenizes strings and formats outputs correctly."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    texts = ["This was a good movie.", "I hated this film."]
    labels = [1, 0]
    max_len = 32
    
    dataset = MovieReviewDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_len
    )
    
    assert len(dataset) == 2
    
    # Check shape and types of first item
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    
    assert item["input_ids"].shape == (max_len,)
    assert item["attention_mask"].shape == (max_len,)
    assert item["label"].item() == 1
    
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)


def test_get_data_loaders():
    """Verify data loaders are created with expected batch sizes."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = get_data_loaders(
        tokenizer=tokenizer,
        batch_size=4,
        max_length=16,
        max_samples=10
    )
    
    # Check loaders
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    # Assert we can iterate and retrieve a batch
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
    assert batch["input_ids"].shape == (4, 16)
    assert batch["label"].shape == (4,)
