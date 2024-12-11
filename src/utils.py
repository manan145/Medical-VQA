"""
utils.py

This module contains utility functions including prompt creation and custom collate functions.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

def create_prompt(question: str) -> str:
    """
    Creates a natural language prompt for the model given a question.

    Args:
        question (str): The question to be answered.

    Returns:
        str: A formatted prompt to be fed into the model.
    """
    return f"Answer the following question based on the image: {question}"


def custom_collate_fn(examples: list) -> dict:
    """
    Custom collate function to properly batch the preprocessed examples.

    Args:
        examples (list): A list of examples, each containing pixel_values, input_ids, attention_mask, and labels.

    Returns:
        dict: A dictionary containing batched tensors.
    """
    # Stack all tensors from the batch
    batch = {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": pad_sequence([example["input_ids"] for example in examples], batch_first=True),
        "attention_mask": pad_sequence([example["attention_mask"] for example in examples], batch_first=True),
        "labels": pad_sequence([example["labels"] for example in examples], batch_first=True)
    }
    return batch
