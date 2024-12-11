"""
model.py

This module handles model loading and preparation. It also configures and applies LoRA.
"""

from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

def load_processor_and_model(args: dict):
    """
    Loads the model processor and the BLIP-2 model. Applies LoRA configuration to the model.

    Args:
        args (dict): Configuration dictionary containing model and cache details.

    Returns:
        tuple: (processor, model) where processor is the AutoProcessor and model is the modified BLIP-2 model.
    """
    processor = AutoProcessor.from_pretrained(args["model_name"], cache_dir=args["cache_dir"])
    model = Blip2ForConditionalGeneration.from_pretrained(args["model_name"], cache_dir=args["cache_dir"])

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    return processor, model
