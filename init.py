import os
import pathlib

import gdown
import numpy as np
from PIL import Image

if not os.path.isfile("dataset.zip"):
    url = 'https://drive.google.com/uc?id=1sQEIPh3bdKQ_1J3g0Z8CRqD6uU7v746l'
    output = 'dataset.zip'
    gdown.download(url, output, quiet=False)

print("test")

data_dir = pathlib.Path("./flowers")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

global dataset 

dataset= []
count = 0
for label in os.listdir("./flowers/"):
  for filename in os.listdir(os.path.join("./flowers/",label)):
    count = count + 1
    print(str(count) + " ---loading " + filename)
    image = Image.open(os.path.join("./flowers/",label,filename))
    image.load()
    image = np.asarray(image, dtype="float32" )
    dataset.append((image, label))

import random

print(random.sample(dataset, 10))

import random

random.shuffle(dataset)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()

for i, ax in enumerate(axs.flat):
  ax.imshow(dataset[i][0].astype("int32"))
  ax.axis('off')
  ax.set_title(dataset[i][1])
plt.show()