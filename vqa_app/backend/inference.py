import io
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import PeftModel

class VQAModel:
    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load the base model and processor (adjust this to the appropriate base model)
        base_model_name = "Salesforce/blip2-flan-t5-xl"
        base_model = Blip2ForConditionalGeneration.from_pretrained(base_model_name)

        # Load the LoRA adapter on top of the base model
        self.model = PeftModel.from_pretrained(base_model, model_name)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    def answer_question(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=50,
                num_beams=5,
            )
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

class ImageStore:
    def __init__(self):
        self._store = {}

    def add_image(self, image_id: str, image: Image.Image):
        self._store[image_id] = image

    def get_image(self, image_id: str) -> Image.Image:
        return self._store.get(image_id, None)

# Utility function to convert raw bytes to a PIL image
def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
