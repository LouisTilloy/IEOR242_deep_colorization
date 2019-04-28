import os
import tensorflow as tf
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, Deconv2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils import data_generator

IMAGE_FOLDER = "val"
BATCH_SIZE = 16
N_BINS = 13*13

image_paths = [IMAGE_FOLDER + "/" + name for name in next(os.walk(IMAGE_FOLDER))[2]]


def batch_generator():
    while True:
        gen = data_generator(image_paths)
        for features, labels in gen:
            inputs = []
            targets = []
            for i in range(BATCH_SIZE):
                inputs.append(np.expand_dims(features, -1))
                targets.append(np.expand_dims(labels, -1))
            yield np.array(inputs), np.array(targets)


def get_model():
    model = Sequential([
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=3, padding="same",
               input_shape=(104, 104, 1)),
        Activation('relu'),
        Conv2D(filters=64, kernel_size=3, strides=2, padding="same"),
        Activation('relu'),
        BatchNormalization(),

        # conv 2: (128, 128, 64) -> (64, 64, 128)
        Conv2D(filters=128, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same"),
        Activation('relu'),
        BatchNormalization(),

        # conv 3: (64, 64, 128) -> (32, 32, 256)
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, strides=2, padding="same"),
        Activation('relu'),
        BatchNormalization(),

        # conv 4: (32, 32, 256) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        BatchNormalization(),

        # conv5: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        BatchNormalization(),

        # conv6: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2),
        Activation('relu'),
        BatchNormalization(),

        # conv7: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=512, kernel_size=3, padding="same"),
        Activation('relu'),
        BatchNormalization(),

        # conv8: (32, 32, 512) -> (64, 64, 256)
        Deconv2D(filters=256, kernel_size=4, strides=2, padding="same"),
        Activation('relu'),
        Deconv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Deconv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),

        # FixMe: this layer is NOT in the article
        # conv9 (64, 64, 256) -> (256, 256, 256)
        Deconv2D(filters=256, kernel_size=4, strides=4, padding="same"),
        Activation('relu'),
        Deconv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Deconv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),

        # prediction (256, 256, 256) -> (256, 256, N_BINS)
        Conv2D(filters=N_BINS, kernel_size=1, padding="same"),
        Activation('softmax')
    ])
    return model


if __name__ == "__main__":
    batch_gen = batch_generator()

    model = get_model()
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',)
                  #  metrics=['sparse_categorical_accuracy'])


    mc = ModelCheckpoint('weights{epoch:08d}.h5',
                         save_weights_only=True, period=1)

    last_epoch = 10  # change this value to fit your training situation
    model.load_weights('weights.h5')

    model.fit_generator(batch_gen, steps_per_epoch=120000/BATCH_SIZE, verbose=1,
                        callbacks=[mc], epochs=10, initial_epoch=last_epoch + 1)

    model.save_weights("weights.h5")
