from flask import Flask,request,jsonify,render_template

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import base64

app = Flask(__name__)

model = tf.keras.models.load_model('modelVGG.h5')

# def predict(img):
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     prediction = model.predict(img)
#     return prediction
#
# @app.route('/predict', methods=['POST'])
# def predict_route():
#     try:
#         img = request.files['image'].read()
#         prediction = predict(img)
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)})

def predict_image(image):
    # Load and preprocess the image
    # image = cv2.imread(image_path)
    # image = cv2.imread(image)
    print(type(image))
    # image = np.array(image, dtype=np.float32)
    image = cv2.resize(image, (224,224))
    image = np.expand_dims(image, axis=0)
    image = keras.applications.vgg16.preprocess_input(image)

    # Run prediction on the image
    prediction = model.predict(image)

    # Convert the predicted probabilities to a class label
    class_label = "Yes Tumour" if prediction[0][0] > 0.5 else "No Tumour"

    return class_label

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = ''

    if request.method == 'POST':
        selected_image = request.form.get('selected_image')
        if selected_image is not None:
            # Load the selected predefined image
            with open('static/image_' + selected_image + '.jpg', 'rb') as f:
                image_data = f.read()
                image_data = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
                # image = np.fromstring(image_data, np.uint8)
                # image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

        else:
            # Get the uploaded image
            image = request.files['image']
            # Read the image data from the FileStorage object
            np_image = np.fromstring(image.read(), np.uint8)

            # Decode the image data into a OpenCV image
            image_data = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Perform prediction
        prediction = predict_image(image_data)

        # Encode the image as a base64-encoded string
        image_data = base64.b64encode(image_data).decode('utf-8')
        # print('Image',image_data)
        # print('Image_Type',type(image_data))
        return render_template('index.html', prediction=prediction, image_data=image_data)

    return render_template('index.html',prediction='')

if __name__ == '__main__':
    app.run()
