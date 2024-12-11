"""
metrics.py

This module defines the metric computation functions for model evaluation, including exact match and F1 scores.
"""

from evaluate import load

def compute_metrics(pred, tokenizer):
    """
    Computes Exact Match (EM) and F1 score using the SQuAD v2 metric.

    Args:
        pred (PredictionOutput): A named tuple with `predictions` and `label_ids` from the Trainer.
        tokenizer: The tokenizer used for decoding predictions and labels.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    # Preprocess: strip whitespace and convert to lowercase
    preds = [p.strip().lower() for p in preds]
    labels = [l.strip().lower() for l in labels]

    # Format predictions/references for SQuAD v2 style
    formatted_preds = [{"id": str(i), "prediction_text": p, "no_answer_probability": 0.0} for i, p in enumerate(preds)]
    formatted_refs = [
        {"id": str(i), "answers": {"text": [l], "answer_start": [0]}} for i, l in enumerate(labels)
    ]

    # Load the SQuAD metric
    squad_metric = load("squad_v2")
    results = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
    return results
