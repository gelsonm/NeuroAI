from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model('modelVGG.h5')

def predict_image(image):
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    image = keras.applications.vgg16.preprocess_input(image)

    # Run prediction on the image
    prediction = model.predict(image)

    # Convert the predicted probabilities to a class label
    class_label = "Yes Tumour" if prediction[0][0] > 0.5 else "No Tumour"

    return class_label

@app.post('/')
async def predict(request: Request, selected_image: str = None, image: UploadFile = File(...)):
    prediction = ''

    if image is None:
        # Load the selected predefined image
        with open('static/image_' + selected_image + '.jpg', 'rb') as f:
            image_data = f.read()
            image_data = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)

    else:
        # Read the image data from the FileStorage object
        contents = await image.read()
        np_image = np.frombuffer(contents, np.uint8)

        # Decode the image data into a OpenCV image
        image_data = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Perform prediction
    prediction = predict_image(image_data)

    # Encode the image as a base64-encoded string
    image_data = base64.b64encode(cv2.imencode('.jpg', image_data)[1]).decode('utf-8')

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "image_data": image_data})

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ''})
