import streamlit as st
import requests
from io import BytesIO

# Replace this with your Google Cloud Run container endpoint
API_BASE = "https://medical-vqa-955125037464.us-central1.run.app"

st.title("Visual Question Answering (VQA)")

if "image_id" not in st.session_state:
    st.session_state["image_id"] = None
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None and st.session_state["image_id"] is None:
    # Read the uploaded image as bytes so we can display it later
    image_bytes = uploaded_image.read()
    # Reset the file pointer for the request
    uploaded_image.seek(0)
    
    files = {"image_file": (uploaded_image.name, image_bytes, uploaded_image.type)}
    response = requests.post(f"{API_BASE}/upload_image", files=files)
    if response.status_code == 200:
        data = response.json()
        st.session_state["image_id"] = data["image_id"]
        st.session_state["image_bytes"] = image_bytes
        st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    else:
        st.error("Failed to upload image. Please try again.")

# If we have an image_id, allow asking questions
if st.session_state["image_id"] is not None:
    # Always display the image before asking questions
    st.image(st.session_state["image_bytes"], caption="Uploaded Image", use_container_width=True)

    question = st.text_input("Enter your question about the image:")
    if question and st.button("Ask"):
        response = requests.post(f"{API_BASE}/ask_question", data={"image_id": st.session_state["image_id"], "question": question})
        if response.status_code == 200:
            data = response.json()
            st.write(f"**Question:** {data['question']}")
            st.write(f"**Answer:** {data['answer']}")
        else:
            st.error("Failed to get an answer from the backend.")

    if st.button("Ask Another Question"):
        # Just clears the question input to allow a new question
        st.experimental_rerun()

    if st.button("Upload New Image"):
        # Reset the state to allow a new image to be uploaded
        st.session_state["image_id"] = None
        st.session_state["image_bytes"] = None
        st.experimental_rerun()
