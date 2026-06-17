import os
import pytest

# Ensure TF_USE_LEGACY_KERAS is set before any imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["MAX_SAMPLES"] = "20"

import numpy as np
from transformers import AutoTokenizer
from src.preprocess import load_and_tokenize_data, generate_mock_data
from src.config import MODEL_NAME


def test_generate_mock_data():
    """Verify mock data generation produces correct structure and labels."""
    mock = generate_mock_data(100)
    
    assert "train" in mock
    assert "test" in mock
    
    assert "text" in mock["train"]
    assert "label" in mock["train"]
    
    # Check all labels are 0 or 1
    for label in mock["train"]["label"] + mock["test"]["label"]:
        assert label in [0, 1]
    
    # Check split ratio (80/20)
    assert len(mock["train"]["text"]) == 80
    assert len(mock["test"]["text"]) == 20


def test_load_and_tokenize_data_shapes():
    """Verify the tokenized TF datasets have correct structure and shapes."""
    train_tf, test_tf, tokenizer = load_and_tokenize_data(
        max_samples=20,
        max_length=32,
        batch_size=4,
    )
    
    # Check that we got tf.data.Dataset objects
    import tensorflow as tf
    assert isinstance(train_tf, tf.data.Dataset)
    assert isinstance(test_tf, tf.data.Dataset)
    
    # Check that tokenizer is valid
    assert tokenizer is not None
    
    # Get one batch and verify structure
    for batch in train_tf.take(1):
        inputs, labels = batch
        
        # inputs should have input_ids and attention_mask
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        
        # Check shapes: (batch_size, max_length)
        assert inputs["input_ids"].shape[1] == 32  # max_length
        assert inputs["attention_mask"].shape[1] == 32
        
        # Labels should be 1D
        assert len(labels.shape) == 1
        
        # Labels should be 0 or 1
        for label_val in labels.numpy():
            assert label_val in [0, 1]


def test_tokenizer_output():
    """Verify the tokenizer produces expected output format."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    text = "This movie was great!"
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="np",
    )
    
    assert "input_ids" in encoding
    assert "attention_mask" in encoding
    assert encoding["input_ids"].shape == (1, 16)
    assert encoding["attention_mask"].shape == (1, 16)
    
    # First token should be [CLS] (101 for BERT tokenizers)
    assert encoding["input_ids"][0][0] == 101
