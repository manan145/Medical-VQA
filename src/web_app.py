import streamlit as st
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Load processor and model
@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained("manan145/blip2_lora_vqa_model")
    model = Blip2ForConditionalGeneration.from_pretrained("manan145/blip2_lora_vqa_model")
    model.eval()
    return processor, model

# Load the processor and model
processor, model = load_model_and_processor()
tokenizer = processor.tokenizer

# Title
st.title("Visual Question Answering (VQA)")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Store the image state so the user doesn't need to upload it every time
if "image" not in st.session_state:
    st.session_state["image"] = None
    st.session_state["question"] = None
    st.session_state["answer"] = None

# If the user uploads a new image, update the session state
if uploaded_image:
    st.session_state["image"] = Image.open(uploaded_image)
    st.session_state["question"] = None
    st.session_state["answer"] = None
    st.image(st.session_state["image"], caption="Uploaded Image",use_container_width=True)

# Ask a question if the image is uploaded
if st.session_state["image"]:
    question = st.text_input("Enter your question about the image:")

    if question:
        # Preprocess the input
        inputs = processor(st.session_state["image"], question, return_tensors="pt")

        # Generate the answer
        with st.spinner("Generating answer..."):
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=inputs.pixel_values,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=50,
                    num_beams=5,
                )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Store the answer
        st.session_state["question"] = question
        st.session_state["answer"] = generated_text

        # Display the result
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {generated_text}")

    # Provide an option to ask another question or upload a new image
    if st.button("Ask Another Question"):
        st.session_state["question"] = None
        st.session_state["answer"] = None

    if st.button("Upload New Image"):
        st.session_state["image"] = None
        st.session_state["question"] = None
        st.session_state["answer"] = None
