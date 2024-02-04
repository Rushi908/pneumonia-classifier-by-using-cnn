# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:56:01 2023

@author: Rushi
"""

import tensorflow as tf
from tensorflow import keras

# Load your pre-trained CNN model
model = keras.models.load_model('your_cnn_model.h5')

# Load and preprocess your testing data
 test_data = ...

# Make predictions on the testing data
predictions = model.predict(test_data)

# Convert predicted probabilities to class labels (e.g., using argmax)
predicted_labels = tf.argmax(predictions, axis=1)

# Calculate accuracy
correct_predictions = tf.equal(predicted_labels, test_labels)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(f"Testing Accuracy: {accuracy.numpy() * 100:.2f}%")
