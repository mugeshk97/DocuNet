from flask import Flask, request, jsonify
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)
import numpy as np
import cv2
import base64

app = Flask(__name__)

def preprocess_input(input_image, shape, data_format=None):
    try:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_image = cv2.resize(input_image, shape)
        _, input_image = cv2.threshold(input_image, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        input_image = input_image / 255.0
        input_image = np.expand_dims(input_image, axis = -1)
        input_image = np.expand_dims(input_image, axis = 0)
        return input_image
    except Exception as e:
        return None

def base64_to_image(image_b64):
    try:
        # decode the base64 encoded string
        decoded_string = base64.b64decode(image_b64)
        # convert the decoded string to numpy array
        string_as_np = np.frombuffer(decoded_string, dtype=np.uint8)
        # convert the numpy array to image
        image = cv2.imdecode(string_as_np, flags=1)
        return image
    except Exception as e:
        return None

    
model = tf.keras.models.load_model('/home/mugesh/IB/DocuNet/models/1/DocNet_v_1.h5') # path to the model to be tested

@app.route('/predict', methods=['POST'])
def predict():
    # get the image from the request as base64 encoded string
    image_b64 = request.json['image']
    # convert the base64 encoded string to an image
    image = base64_to_image(image_b64)
    if image is not None:
        # preprocess the image
        image = preprocess_input(image, (360, 360))
        # predict the image
        if image is not None:
            output  = model.predict(image)
            #  convert the output dict with ndarray 
            output = {key: [np.round(value.tolist()[0][0], 3)] for key, value in output.items()}
            return jsonify(output)


if __name__ == '__main__':
    app.run(debug= True)
