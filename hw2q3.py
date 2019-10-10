from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D
from keras import backend as tensorflow_backend
import numpy as np
import random
import matplotlib.pyplot as plt

tensorflow_backend.tensorflow_backend._get_available_gpus()

# assign value
batch_size = 128
epochs = 2
num_classes = 10
image_row = 28
image_col = 28
color_channels = 1
noise_factor = 0.1

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# coerce to float, deal with rgb, coerce to categorical class
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


# # noise functions
# def noisy_salt_pepper(image, noise_percentage, salt_portion):
#     out = np.copy(image)
#     out = out.reshape(1, image.size)
#     ran_seq = random.sample([n for n in range(image.size)], np.int(noise_percentage * image.size))
#     # salt mode
#     num_salt = np.int(noise_percentage * image.size * salt_portion)
#     coords_salt = random.sample(ran_seq, k=num_salt)
#     out[0, coords_salt] = 1
#     # # pepper mode
#     # num_pepper = np.int(noise_percentage * image.size * (1. - salt_portion))
#     # coords_pepper = [item for item in ran_seq if item not in coords_salt]
#     # out[0, coords_pepper] = 0
#     out = out.reshape(image_row, image_col)
#     return out
#
#
# # add noise
# for i in range(len(x_train)):
#     x_train[i] = noisy_salt_pepper(image=x_train[i], noise_percentage=0.4, salt_portion=1.0)
# for i in range(len(x_test)):
#     x_test[i] = noisy_salt_pepper(image=x_test[i], noise_percentage=0.4, salt_portion=1.0)

# dimension ordering
if tensorflow_backend.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], color_channels, image_row, image_col)
    x_test = x_test.reshape(x_test.shape[0], color_channels, image_row, image_col)
    input_shape = (color_channels, image_row, image_col)
elif tensorflow_backend.image_data_format() == "channels_last":
    x_train = x_train.reshape(x_train.shape[0], image_row, image_col, color_channels)
    x_test = x_test.reshape(x_test.shape[0], image_row, image_col, color_channels)
    input_shape = (image_row, image_col, color_channels)

# ##########
# inputs = Input(shape=(784, ))
# enc_fc = Dense(32, activation='relu')
# encoded = enc_fc(inputs)
#
# dec_fc = Dense(784, activation='sigmoid')
# decoded = dec_fc(encoded)
#
# # build the model to train
# autoencoder = Model (inputs, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def make_convolutional_autoencoder():
    # encoding
    inputs = Input(shape=input_shape)
    x = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D( 8, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(padding='same')(x)
    x = Conv2D( 8, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(padding='same')(x)

    # decoding
    x = Conv2D(8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling2D()(x)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, activation='relu')(x)  # padding='valid'
    x = UpSampling2D()(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    # autoencoder
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    return autoencoder


# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
#######################################################
# autoencoder = make_convolutional_autoencoder()
# autoencoder.summary()
# autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)
# x_test_decoded = autoencoder.predict(x_test)


def show_images(before_images):
    plt.figure(figsize=(10, 2))
    for i in range(10):
        # before
        plt.subplot(2, 10, i+1)
        plt.imshow(before_images[i].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # # after
        # plt.subplot(2, 10, 10+i+1)
        # plt.imshow(after_images[i].reshape(28, 28), cmap='gray')
        # plt.xticks([])
        # plt.yticks([])
    plt.show()


def add_noise(x):
    x = x+np.random.randn(*x.shape) * noise_factor
    x = x.clip(0., 1.)
    return x


x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)
autoencoder = make_convolutional_autoencoder()

autoencoder.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size)
result = autoencoder.evaluate(x_train_noisy, x_train, verbose=1)
print("train mean absolute error(noise=10%): ", result[1])

autoencoder.fit(x_test_noisy, x_test, epochs=epochs, batch_size=batch_size)
result = autoencoder.evaluate(x_test_noisy, x_test, verbose=1)
print("test mean absolute error(noise=10%): ", result[1])

x_train_decoded = autoencoder.predict(x_train_noisy)
show_images(x_train_noisy)
x_test_decoded = autoencoder.predict(x_test_noisy)
show_images(x_test_noisy)