from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import uvicorn

from inference import VQAModel, ImageStore, load_image_from_bytes

app = FastAPI(title="Visual Question Answering (VQA) API")

# Initialize the model and image store once at startup
model_name = "manan145/blip2_lora_vqa_model"
vqa_model = VQAModel(model_name=model_name)
image_store = ImageStore()

@app.post("/upload_image")
async def upload_image_endpoint(image_file: UploadFile = File(...)):
    """
    Upload an image and store it temporarily in memory.
    Returns an image_id that can be used to ask multiple questions about this image.
    """
    image_bytes = await image_file.read()
    try:
        image = load_image_from_bytes(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    image_id = str(uuid.uuid4())
    image_store.add_image(image_id, image)
    return JSONResponse({"image_id": image_id})

@app.post("/ask_question")
async def ask_question_endpoint(
    image_id: str = Form(...),
    question: str = Form(...)
):
    """
    Ask a question about a previously uploaded image using its image_id.
    """
    image = image_store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    answer = vqa_model.answer_question(image, question)
    return JSONResponse({"question": question, "answer": answer})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)