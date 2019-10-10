from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import backend
import numpy as np
import random

# assign value
batch_size = 128
epochs = 3
num_classes = 10
image_row = 28
image_col = 28
color_channels = 1

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# coerce to float, deal with rgb, coerce to categorical class
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# noise functions
def noisy_gauss(image):
    row, col, ch = input_shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def noisy_poisson(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy


def noisy_speckle(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss
    return noisy


def noisy_salt_pepper(image, noise_percentage, salt_portion):
    out = np.copy(image)
    out = out.reshape(1, image.size)
    ran_seq = random.sample([n for n in range(image.size)], np.int(noise_percentage * image.size))
    # salt mode
    num_salt = np.int(noise_percentage * image.size * salt_portion)
    coords_salt = random.sample(ran_seq, k=num_salt)
    out[0, coords_salt] = 1
    # # pepper mode
    # num_pepper = np.int(noise_percentage * image.size * (1. - salt_portion))
    # coords_pepper = [item for item in ran_seq if item not in coords_salt]
    # out[0, coords_pepper] = 0
    out = out.reshape(image_row, image_col)
    return out

#def noisy_salt_pepper(image, noise_percentage, salt_portion):
    # out = np.copy(image)
    # # salt mode
    # num_salt = np.ceil(noise_percentage * image.size * salt_portion)
    # coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    # out[coords] = 1
    # # pepper mode
    # num_pepper = np.ceil(noise_percentage * image.size * (1. - salt_portion))
    # coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    # out[coords] = 0
    # return out


# add noise
for i in range(len(x_train)):
    x_train[i] = noisy_salt_pepper(image=x_train[i], noise_percentage=0.1, salt_portion=1.0)
for i in range(len(x_test)):
    x_test[i] = noisy_salt_pepper(image=x_test[i], noise_percentage=0.1, salt_portion=1.0)

# dimension ordering
if backend.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], color_channels, image_row, image_col)
    x_test = x_test.reshape(x_test.shape[0], color_channels, image_row, image_col)
    input_shape = (color_channels, image_row, image_col)
elif backend.image_data_format() == "channels_last":
    x_train = x_train.reshape(x_train.shape[0], image_row, image_col, color_channels)
    x_test = x_test.reshape(x_test.shape[0], image_row, image_col, color_channels)
    input_shape = (image_row, image_col, color_channels)

# cnn model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# apply data to cnn model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
result = model.evaluate(x_train, y_train, verbose=1)
print("train accuracy (noise=10%): ", result[1])
result = model.evaluate(x_test, y_test, verbose=1)
print("test accuracy (noise=10%): ", result[1])
