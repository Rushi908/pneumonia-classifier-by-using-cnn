# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:22:31 2023

@author: Rushi
"""
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

os.chdir("D:\\Deep Learning\\Dog_cat_Classifier\\archive")

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

mypath = "Prediction/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

# predicting images
dog_counter = 0
cat_counter = 0

for file in onlyfiles:
    # Load an image using load_img and resize it
    img = load_img(join(mypath, file), target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=2)
    predicted_class = classes[0][0]

    if predicted_class < 0:
        print(file + ": cat")
        cat_counter += 1
    else:
        print(file + ": dog")
        dog_counter += 1

print("Total Dogs:", dog_counter)
print("Total Cats:", cat_counter)