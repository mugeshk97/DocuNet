import requests
import cv2
import base64
import json
import numpy as np
import os

# Set the URL
url = "http://127.0.0.1:5000/predict"



# single image
image = cv2.imread("/home/mugesh/IB/DocuNet/raw_data/black_border/00001417.tif")
# convert image to base64
_, img_encoded = cv2.imencode('.jpg', image)
# encode to base64 string
data = base64.b64encode(img_encoded).decode('utf-8')
# create a dict with the data
payload = {'image': data}
# send the request to the server with json payload and get the response as a json
response = requests.post(url, json=payload)
# convert the response to json
response_json = json.loads(response.text)
# print the response
print(response_json)

def image_to_base64(image):
    retval, buffer = cv2.imencode('.jpg', image) # encode the image as jpg
    jpg_as_text = base64.b64encode(buffer) # encode the jpg as base64
    return jpg_as_text.decode('utf-8') # decode the base64 encoded string and return it


def predict_image(image):
    url = "http://127.0.0.1:5000/predict"
    # encode the image as base64
    image_b64 = image_to_base64(image)
    # create the payload
    payload = {'image': image_b64}
    # make the request
    r = requests.post(url, json = payload)
    if r.status_code == 200:
        # convert the response to json
        response_json = json.loads(r.text)
        return response_json
    else:
        return None

# directory of images
file_directory = "/home/mugesh/IB/DocuNet/raw_data/black_border" # path to the folder containing the images to be tested
filenames = os.listdir(file_directory)
for file_ in filenames:
    if file_.split('.')[-1] in ['TIF', 'tif']:
        file_path = os.path.join(file_directory, file_)
        img = cv2.imread(file_path)
        output = predict_image(img)
        if output is not None:
            print(output)
