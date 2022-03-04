import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu, True)
from utils import DataLoader
import json
import os

BATCH_SIZE = 32

data_loader = DataLoader(shape=(360,360), batch_size = BATCH_SIZE) 
data_loader.load_from_directory("processed_data/")
data_gen = data_loader.data_generator()

## Model Architecture
input_ = tf.keras.layers.Input(shape=(360,360,1), name ="input")

conv_1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same", name="conv_1")(input_)
act_1 = tf.keras.layers.Activation('relu', name='act_1')(conv_1)
pool_1 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name = "pool_1")(act_1)

conv_2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "same" , name="conv_2")(pool_1)
act_2 = tf.keras.layers.Activation('relu', name='act_2')(conv_2)
pool_2 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name = "pool_2")(act_2)

conv_3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = "same",  name="conv_3")(pool_2)
act_3 = tf.keras.layers.Activation('relu', name='act_3')(conv_3)
pool_3 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name = "pool_3")(act_3)

conv_4 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = "same", name="conv_4")(pool_3)
act_4 = tf.keras.layers.Activation('relu', name='act_4')(conv_4)
pool_4 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name = "pool_4")(act_4)
pool_5 = tf.keras.layers.MaxPool2D(pool_size = (2,2), name = "pool_5")(pool_4)

flatten = tf.keras.layers.Flatten(name="flatten")(pool_5)


dense_1 = tf.keras.layers.Dense(64, activation='relu', name = "dense_1")(flatten)
dense_2 = tf.keras.layers.Dense(32, activation='relu', name = "dense_2")(dense_1)

shaded = tf.keras.layers.Dense(1,activation='sigmoid', name="shaded")(dense_2)
gridline = tf.keras.layers.Dense(1,activation='sigmoid', name="gridline")(dense_2)
good = tf.keras.layers.Dense(1,activation='sigmoid', name="good")(dense_2)
black_border = tf.keras.layers.Dense(1,activation='sigmoid', name="black_border")(dense_2)

model_outputs = {'black_border' : black_border, 'good' : good, 'gridline' : gridline, 'shaded' : shaded}


model = tf.keras.models.Model(input_, model_outputs, name ="DockNet")
model.summary()

## Compile 
model.compile(
    loss={
        "shaded": 'binary_crossentropy',
        "gridline": "binary_crossentropy",
        "good": "binary_crossentropy",
        "black_border": "binary_crossentropy"
    },
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    data_gen,
    steps_per_epoch= data_loader.num_files // BATCH_SIZE,
    epochs=10)

model_version = max([int(i) for i in os.listdir('Model/') + [0] ] ) + 1 # get the latest model version


model.save(f"Model/DocNet_v_{str(model_version)}.h5")
print(f'saved model version {model_version}')