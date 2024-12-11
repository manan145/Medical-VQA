"""
config.py

This module stores configuration variables and constants used throughout the training process.
"""

MOUNT_PATH = "./stat-sw-proj"

CONFIG = {
    "model_name": "Salesforce/blip2-flan-t5-xl",
    "do_train": True,
    "do_eval": True,
    "learning_rate": 2e-3,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "num_train_epochs": 10,
    "logging_steps": 100,
    "save_steps": 100,
    "output_dir": f"{MOUNT_PATH}/blip2_lora_vqa_model",
    "cache_dir": f"{MOUNT_PATH}/cache_dir",
    "max_length": 128
}
