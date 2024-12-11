# vqa_app/app/inference.py
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import os 

def load_model_and_processor():
    

    processor = AutoProcessor.from_pretrained('manan145/blip2_lora_vqa_model')
    model = Blip2ForConditionalGeneration.from_pretrained('manan145/blip2_lora_vqa_model') 

    model.eval()
    return processor, model

def run_inference(image: Image.Image, question: str):
    # Preprocess the input

    processor, model = load_model_and_processor()

    inputs = processor(image, question, return_tensors="pt", max_length=50)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            num_beams=5,
        )
    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

