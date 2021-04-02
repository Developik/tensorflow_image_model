from operator import add
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import csv

# load numpy array from csv file
from numpy import loadtxt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

img_height = 180
img_width = 180

objNames = []

with open('namesArr.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    for i in range(len(data)):
        objNames.append(data[i][0])
#objNames = np.genfromtxt('namesArr.csv', delimiter = ' ')

print(objNames)

new_model = tf.keras.models.load_model('./models/model_2')

# Check its architecture
new_model.summary()

sunflower_url = os.getcwd() + "\sec_model\mockup-graphics-iUGPq02__Gc-unsplash.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(objNames[np.argmax(score)], 100 * np.max(score))
)

print(score)