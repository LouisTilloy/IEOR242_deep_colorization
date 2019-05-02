from keras.models import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization, Conv2DTranspose

N_BINS = 13*13

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
        Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),

        # FixMe: this layer is NOT in the article
        # conv9 (64, 64, 256) -> (256, 256, 256)
        Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same"),
        Activation('relu'),

        # prediction (256, 256, 256) -> (256, 256, N_BINS)
        Conv2D(filters=N_BINS, kernel_size=1, padding="same"),
        Activation('softmax')
    ])
    return model


def get_small_model():
    model = Sequential([
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=3, padding="same",
               kernel_initializer='random_uniform',
               input_shape=(104, 104, 1)),
        Activation('relu'),
        Conv2D(filters=64, kernel_size=3, strides=2, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv 2: (128, 128, 64) -> (64, 64, 128)
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv 3: (64, 64, 128) -> (32, 32, 256)
        Conv2D(filters=256, kernel_size=3, strides=2, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv 4: (32, 32, 256) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv5: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2,
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv6: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same", dilation_rate=2,
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv7: (32, 32, 512) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv8: (32, 32, 512) -> (64, 64, 256)
        Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),
        Conv2D(filters=256, kernel_size=3, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),

        # FixMe: this layer is NOT in the article
        # conv9 (64, 64, 256) -> (256, 256, 256)
        Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),

        # prediction (256, 256, 256) -> (256, 256, N_BINS)
        Conv2D(filters=N_BINS, kernel_size=1, padding="same",
               kernel_initializer='random_uniform',),
        Activation('softmax')
    ])
    return model


def get_tiny_model():
    model = Sequential([
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=5, strides=2, padding="same",
               kernel_initializer='random_uniform',
               input_shape=(32, 32, 1)),
        Activation('relu'),
        BatchNormalization(),

        # conv 2: (128, 128, 64) -> (64, 64, 128)
        Conv2D(filters=128, kernel_size=3, strides=2, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv 3: (64, 64, 128) -> (32, 32, 256)
        Conv2D(filters=256, kernel_size=3, strides=2, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv 4: (32, 32, 256) -> (32, 32, 512)
        Conv2D(filters=512, kernel_size=3, padding="same",
               kernel_initializer='random_uniform',),
        Activation('relu'),
        BatchNormalization(),

        # conv8: (32, 32, 512) -> (64, 64, 256)
        Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),

        # FixMe: this layer is NOT in the article
        # conv9 (64, 64, 256) -> (256, 256, 256)
        Conv2DTranspose(filters=256, kernel_size=4, strides=4, padding="same",
                 kernel_initializer='random_uniform',),
        Activation('relu'),

        # prediction (256, 256, 256) -> (256, 256, N_BINS)
        Conv2DTranspose(filters=N_BINS, kernel_size=1, padding="same",
                        kernel_initializer='random_uniform',),
        Activation('softmax')
    ])
    return model