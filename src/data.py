"""
data.py

This module handles dataset loading and preprocessing for training and evaluation.
"""

import torch
from functools import partial
from transformers import AutoProcessor
from typing import Tuple
from datasets import load_dataset
from .utils import create_prompt

def preprocess_single(example: dict, processor) -> dict:
    """
    Preprocesses a single example from the dataset. This includes creating a prompt,
    tokenizing input and label, and preparing pixel values.

    Args:
        example (dict): A single example containing keys like "question", "answer", and "image".
        processor (AutoProcessor): A processor object from Hugging Face transformers.

    Returns:
        dict: A dictionary containing model inputs (pixel_values, input_ids, attention_mask) and labels.
    """
    prompt = create_prompt(example["question"])

    # Process image and text together
    inputs = processor(
        text=prompt,
        images=example["image"],
        padding="max_length",
        truncation=True,
        max_length=36,
        return_tensors="pt"
    )

    # Process answer (labels)
    labels = processor.tokenizer(
        example["answer"],
        padding="max_length",
        truncation=True,
        max_length=12,
        return_tensors="pt"
    )

    # Remove batch dimension
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    labels = labels["input_ids"].squeeze(0)

    # Combine inputs and labels
    inputs["labels"] = labels
    return inputs


def load_and_preprocess_data(args: dict, processor) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads the SLAKE dataset from Hugging Face, applies preprocessing, and returns train and test datasets.

    Args:
        args (dict): Dictionary containing configuration parameters.
        processor (AutoProcessor): The processor to use for tokenization and image processing.

    Returns:
        tuple: (train_data, test_data) as preprocessed datasets ready for training/evaluation.
    """
    # Load SLAKE dataset
    slake_data = load_dataset("mdwiratathya/SLAKE-vqa-english")

    # Create partial function with the processor
    preprocess_fn = partial(preprocess_single, processor=processor)

    # Map the preprocessing over the datasets
    train_data = slake_data['train'].map(
        preprocess_fn,
        remove_columns=slake_data['train'].column_names
    )

    test_data = slake_data['test'].map(
        preprocess_fn,
        remove_columns=slake_data['test'].column_names
    )

    # Set the format to PyTorch tensors
    train_data.set_format(type='torch', columns=['pixel_values', 'input_ids', 'attention_mask', 'labels'])
    test_data.set_format(type='torch', columns=['pixel_values', 'input_ids', 'attention_mask', 'labels'])

    return train_data, test_data
