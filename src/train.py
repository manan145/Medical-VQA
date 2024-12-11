"""
train.py

This module orchestrates the training and evaluation of the BLIP-2 model on the SLAKE dataset,
using parameter-efficient fine-tuning with LoRA.
"""

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from .config import CONFIG
from .data import load_and_preprocess_data
from .model import load_processor_and_model
from .metrics import compute_metrics
from .utils import custom_collate_fn

def run_training():
    """
    Runs the training and evaluation process for the BLIP-2 model on the SLAKE dataset.
    This includes:
    - Loading data and processor.
    - Preparing the model with LoRA.
    - Configuring training arguments.
    - Initializing the Trainer and running training/evaluation.
    """
    args = CONFIG
    processor, model = load_processor_and_model(args)

    # Load and preprocess datasets
    train_data, test_data = load_and_preprocess_data(args, processor)

    # Verify shapes
    print(train_data[0]['input_ids'].shape, train_data[0]['pixel_values'].shape)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args["output_dir"],
        eval_strategy="epoch" if args["do_eval"] else "no",
        save_strategy="epoch",
        per_device_train_batch_size=args["train_batch_size"],
        per_device_eval_batch_size=args["eval_batch_size"],
        num_train_epochs=args["num_train_epochs"],
        logging_steps=args["logging_steps"],
        save_steps=args["save_steps"],
        learning_rate=args["learning_rate"],
        save_total_limit=2,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    tokenizer = processor.tokenizer

    # Initialize Trainer with custom collate function
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if args["do_train"] else None,
        eval_dataset=test_data if args["do_eval"] else None,
        processing_class=processor.tokenizer,  # Optional attribute, may remove if not needed
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        data_collator=custom_collate_fn
    )

    # Train and evaluate
    if args["do_train"]:
        trainer.train()
        trainer.push_to_hub("manan145/medical-VQA")
        processor.save_pretrained("path/to/your/fine-tuned-model")

    if args["do_eval"]:
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
