import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

import numpy as np
import os
import cv2
import pandas as pd

def start_prediction(model, file_directory, xcel_filename):
    data_frame = pd.DataFrame()
    df = {}
    filenames = os.listdir(file_directory)
    for file_ in filenames:
        if file_.split('.')[-1] in ['TIF', 'tif']:
            file_path = os.path.join(file_directory, file_)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img , (360,360))
            _, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img = img / 255.0
            img = np.expand_dims(img, axis = -1)
            img = np.expand_dims(img, axis = 0)
            black_border_op, good_op, grid_op, shad_op  = model.predict(img)
            df['abs_filename'] = file_path
            df['black_border'] = np.round(black_border_op[0][0], 3)
            df['normal'] = np.round(good_op[0][0], 3)
            df['gridline']= np.round(grid_op[0][0], 3)
            df['shaded']=np.round(shad_op[0][0], 3)
            comment = ''
            if np.round(black_border_op[0][0], 3) > 0.95:
                comment += 'Black Border, '
            if np.round(good_op[0][0], 3) > 0.95:
                comment += 'Normal, '
            if np.round(grid_op[0][0], 3) > 0.95:
                comment += 'Gridline, '
            if np.round(shad_op[0][0], 3) > 0.95:
                comment += 'Shaded, '
            df['image_quality'] = comment
            data_frame = data_frame.append(df, ignore_index= True)
            data_frame.to_excel(xcel_filename, index= False)

if __name__ == '__main__':
    loaded_model = tf.keras.models.load_model('mtm5.h5') # load the model from the h5 file
     # start the prediction process with the model and the directory of the images to be predicted and the name of the excel file to save the results
    start_prediction(model= loaded_model, file_directory= 'document_data/shaded', xcel_filename= "report_cls.xlsx")
