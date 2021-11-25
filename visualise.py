import matplotlib.pyplot as plt
import init

class_names = init.train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in init.train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

"""You will train a model using these datasets by passing them to `Model.fit` in a moment. If you like, you can also manually iterate over the dataset and retrieve batches of images:"""

for image_batch, labels_batch in init.train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
