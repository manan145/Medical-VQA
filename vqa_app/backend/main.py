from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn
from inference import run_inference

app = FastAPI()

import os 

@app.post("/predict")
async def predict(file: UploadFile = File(...), question: str = ""):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    answer = run_inference(image, question)
    return {"answer": answer}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)