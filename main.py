from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import logging
from io import BytesIO


app = FastAPI()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


try:
    model = load_model("best_model.hdf5")
except Exception as e:
    raise RuntimeError("Failed to load the model. Ensure the model file exists and is correctly formatted.") from e

def preprocess_image(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        raise ValueError("Failed to preprocess the image.") from e

@app.post("/predict-image-upload")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")

    try:
        image_bytes = await file.read()
        image = np.fromstring(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode the image. Ensure the file is a valid image.")

        preprocessed_image = preprocess_image(image)
        
        prediction = model.predict(preprocessed_image)
        probability = float(np.max(prediction))
        label = 'Pneumonia' if np.argmax(prediction) == 1 else 'Normal'
        
        return JSONResponse(content={"prediction": label, "probability": probability})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

class ImageUrl(BaseModel):
    url: str




@app.post("/predict-url")
async def predict_url(image_url: ImageUrl):
    try:
        response = requests.get(image_url.url)
        if response.status_code != 200:
            raise ValueError("Could not download the image. Ensure the URL is correct and the image is accessible.")

        image_bytes = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode the image. Ensure the URL points to a valid image.")

        preprocessed_image = preprocess_image(image)
        
        prediction = model.predict(preprocessed_image)
        probability = float(np.max(prediction))
        label = 'Pneumonia' if np.argmax(prediction) == 1 else 'Normal'
        
        return JSONResponse(content={"prediction": label, "probability": probability})
    except ValueError as ve:
        logger.error(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
