import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)
import numpy as np
import os
import cv2
import pandas as pd

def preprocess_input(input_image, shape, data_format=None):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.resize(input_image, shape)
    _, input_image = cv2.threshold(input_image, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis = -1)
    input_image = np.expand_dims(input_image, axis = 0)
    return input_image


file_directory = "/home/mugesh/IB/DocuNet/raw_data/black_border" # path to the folder containing the images to be tested
filenames = os.listdir(file_directory)
model = tf.keras.models.load_model('/home/mugesh/IB/DocuNet/models/DocNet_v_1.h5') # path to the model to be tested
df = pd.DataFrame() 
for file_ in filenames:
    if file_.split('.')[-1] in ['TIF', 'tif']:
        file_path = os.path.join(file_directory, file_)
        img = cv2.imread(file_path)
        img = preprocess_input(img, (360, 360))
        output  = model.predict(img)
        for i in output.keys():
            output[i] = np.round(output[i][0], 3)  
        output['abs_filename'] = file_path
        comment = '' 
        if output['black_border'] > 0.95:
            comment += 'Black Border, '
        if output['good'] > 0.95:
            comment += 'Normal, '
        if output['gridline'] > 0.95:
            comment += 'Gridline, '
        if output['shaded'] > 0.95:
            comment += 'Shaded, '
        output['image_quality'] = comment
        df = df.append(output, ignore_index= True)
        df.to_csv("src/results/report.csv", index= False)