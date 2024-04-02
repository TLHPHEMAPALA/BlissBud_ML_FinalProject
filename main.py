from keras.preprocessing import image
import numpy as np
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
from keras.models import load_model
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
import tensorflow as tf
from io import BytesIO
import io
from fastapi.middleware.cors import CORSMiddleware
# Download NLTK resources during application startup
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Emotion
# Load the saved emotion detection model
emotion_model = load_model('emotion_model.h5')

# Load the pre-trained face detector (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class_labels = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprise']


@app.get("/")
def home():
    return {"API_health_check": "OK", "model_version": "0.1.0"}


@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        # Read image as grayscale
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # Resize the input image to match the expected input shape of the model (224x224)
        img = cv2.resize(img, (224, 224))

        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to match the input shape of the model
        img_rgb = np.expand_dims(img_rgb, axis=0)

        # Normalize the image data
        img_rgb = img_rgb.astype('float32') / 255.0

        # Perform emotion prediction
        emotion_predictions = emotion_model.predict(img_rgb)
        predicted_emotion_index = np.argmax(emotion_predictions)
        predicted_emotion_label = class_labels[predicted_emotion_index]

        return {"predicted_emotion": str(predicted_emotion_label)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
# Function to read uploaded file as an image


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


myconfig = r"--psm 11 --oem 3"


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)
