# -*- coding: utf-8 -*-
"""neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O2oGbGEnBgCoPonatQMjSvQ_3ZFL4ltG

# Import TensorFlow and other libraries
"""
"""You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order."""

import init

import os

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

train_ds = init.train_ds
val_ds = init.val_ds
class_names = train_ds.class_names
print(class_names)

"""## Predict on new data

Finally, let's use our model to classify an image that wasn't included in the training or validation sets.

Note: Data augmentation and dropout layers are inactive at inference time.
"""

model = Sequential([
  layers.Rescaling(1./255, input_shape=(init.img_height, init.img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5)
])

for image, label in val_ds.take(8):
  pass
image = image[0]
label = label[0]

image = tf.expand_dims(image, 0) # Create a batch

predictions = model.predict(image)
score = tf.nn.softmax(predictions[0])


print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
print("Ground truth label: {}".format(class_names[label]))