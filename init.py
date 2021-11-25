
import matplotlib.pyplot as plt
import gdown
import os

import tensorflow as tf

"""# Download and explore the dataset"""

if not os.path.isfile("dataset.zip"):
    url = 'https://drive.google.com/uc?id=1sQEIPh3bdKQ_1J3g0Z8CRqD6uU7v746l'
    output = 'dataset.zip'
    gdown.download(url, output, quiet=False)

import pathlib
data_dir = pathlib.Path("./flowers")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

"""# Load data using a Keras utility

## Create a dataset

Define some parameters for the loader:
"""

global batch_size
global img_height
global img_width

batch_size = 32
img_height = 180
img_width = 180

"""It's good practice to use a validation split when developing your model. Let's use 80% of the images for training, and 20% for validation."""

global train_ds
global val_ds

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)