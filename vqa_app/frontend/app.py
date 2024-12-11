# vqa_app/streamlit_app/app.py
import streamlit as st
import requests
from PIL import Image
import io

st.title("Medical Visual Question Answering (VQA)")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Session state to remember the uploaded image and Q&A
if "image_data" not in st.session_state:
    st.session_state["image_data"] = None
    st.session_state["question"] = None
    st.session_state["answer"] = None

if uploaded_image:
    # Read and store image data in session state
    image_bytes = uploaded_image.read()
    st.session_state["image_data"] = image_bytes
    st.session_state["question"] = None
    st.session_state["answer"] = None
    # Display the image
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, caption="Uploaded Image")

if st.session_state["image_data"]:
    question = st.text_input("Enter your question about the image:")

    if st.button("Get Answer"):
        if question:
            # Send request to FastAPI backend
            files = {"file": st.session_state["image_data"]}
            data = {"question": question}
            with st.spinner("Generating answer..."):
                response = requests.post("http://backend:8080/predict", files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                st.session_state["question"] = question
                st.session_state["answer"] = answer
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")
            else:
                st.error("Error from backend.")
        else:
            st.warning("Please enter a question.")

    if st.button("Ask Another Question"):
        st.session_state["question"] = None
        st.session_state["answer"] = None

    if st.button("Upload New Image"):
        st.session_state["image_data"] = None
        st.session_state["question"] = None
        st.session_state["answer"] = None


