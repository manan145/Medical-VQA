"""
inference.py

This module provides functionality to run inference using a fine-tuned BLIP-2 model on new image-question inputs.
It defines methods to:
1. Load the saved fine-tuned model and processor.
2. Preprocess a given image and question using the same transformations and tokenization as training.
3. Generate a predicted answer from the model.

This script can be used as a standalone tool for evaluating the model on custom data or integrated into a larger
application pipeline.
"""

from torch.utils.data import DataLoader
from peft import PeftModel
import tqdm
import matplotlib.pyplot as plt

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,

)
from sklearn.metrics import f1_score
import torch
from functools import partial
from .data import preprocess_single
from .config import CONFIG


processor = AutoProcessor.from_pretrained("manan145/blip2_lora_vqa_model")

# Load the base model
model = Blip2ForConditionalGeneration.from_pretrained("manan145/blip2_lora_vqa_model")


model.eval()  # Set model to evaluation mode

# Load the SLAKE VQA dataset
slake_data = load_dataset("mdwiratathya/SLAKE-vqa-english", split="test")

# Select the first `num_examples` examples for evaluation
selected_data = slake_data.select(range(10))

# Preprocess the selected examples
preprocess_fn = partial(preprocess_single, processor=processor)
processed_data = selected_data.map(
    preprocess_fn,
    remove_columns=slake_data.column_names
)

# Set format to PyTorch tensors
processed_data.set_format(type='torch', columns=['pixel_values', 'input_ids', 'attention_mask'])

# Create a DataLoader for batching (batch size 1 for simplicity)
dataloader = DataLoader(processed_data)

# Initialize list to store predictions
predictions = []

# Disable gradient computation for inference
with torch.no_grad():
    for batch in dataloader:


        # Move tensors to the appropriate device
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Generate outputs
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_length"],
            num_beams=5,
            early_stopping=True
        )

        # Decode the generated ids to strings
        generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(generated_text)

# Display the results with images
print("\nInference Results on 10 Examples:\n")
for i in range(10):
    plt.figure(figsize=(6,6))
    plt.imshow(slake_data[i]['image'])
    plt.axis('off')
    plt.title(f"Example {i+1}:")
    plt.show()

    print(f"Question: {slake_data[i]['question']}")
    print(f"Ground Truth Answer: {slake_data[i]['answer']}")
    print(f"Model Prediction: {predictions[i]}\n")
    print("-" * 50)