import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import glob

from training_loop import train

# from discriminator import make_discriminator_model

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# print(train_images.shape, train_labels.shape)
# print(train_images[0].shape)

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 10

# Batch and shuffle the data
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# train(train_dataset, EPOCHS)

def create_image_generator(path="./data/images/images/**/"):
    size = (28, 28)
    images = glob.glob(path + "*.jpg", recursive=True)
    dataset_size = len(images)
    def gen():
        for i in range(dataset_size):
            # _, axs = plt.subplots(1, 2)
            img = PIL.Image.open(images[i])
            # axs[0].imshow(img)
            img = img.resize(size).convert("RGB")
            img = np.array(img, dtype=np.float32)
            img -= 127.5
            img /= 255
            # axs[1].imshow(img)
            # plt.show()
            yield img
    return gen, dataset_size

img_gen, dataset_size = create_image_generator("./data/images/images/**/")

print("Dataset size is {}".format(dataset_size))

image_dataset = tf.data.Dataset.from_generator(generator=img_gen, output_types=tf.float32).batch(BATCH_SIZE) #.shuffle(BUFFER_SIZE)
train(image_dataset, EPOCHS, dataset_size, BATCH_SIZE)

# for img in image_generator():
#     print("yey")



##############################################
##############################################

# generator = make_generator_model()

# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

# discriminator = make_discriminator_model()
# decision = discriminator(generated_image)
# print(decision)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# display_image(EPOCHS)
