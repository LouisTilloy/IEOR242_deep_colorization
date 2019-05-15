from keras.models import Sequential
from keras.layers import Activation, Conv2D, BatchNormalization, Conv2DTranspose,\
    UpSampling2D


def get_model(resolution, n_classes=None, is_regression=False):
    body = [
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=3, padding="same",
               input_shape=(resolution, resolution, 1)),
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
        Activation('relu')]

    if is_regression:
        head = [
            # conv9: (64, 64, 256) -> (128, 128, 128)
            Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same",
                            kernel_initializer='random_uniform'),
            Activation('relu'),
            Conv2D(filters=128, kernel_size=3, padding="same"),
            Activation('relu'),
            Conv2D(filters=128, kernel_size=3, padding="same"),
            Activation('relu'),

            # conv10: (128, 128, 128) -> (256, 256, 64)
            Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same",
                            kernel_initializer='random_uniform'),
            Activation('relu'),
            Conv2D(filters=64, kernel_size=3, padding="same"),
            Activation('relu'),
            Conv2D(filters=64, kernel_size=3, padding="same"),
            Activation('relu'),

            # predictions: (256, 256, 2)
            Conv2D(filters=2, kernel_size=1, padding="same")
            ]
    else:
        head = [
            # unary predictions: (64, 64, 256) -> (64, 64, n_classes)
            Conv2D(filters=n_classes, kernel_size=1, padding="same",
                   kernel_initializer='random_uniform'),
            Activation('softmax'),

            # bilinear upsampling: (64, 64, n_classes) -> (256, 256, n_classes)
            UpSampling2D(size=(4, 4), data_format="channels_last", interpolation="bilinear")
        ]

    return Sequential(body + head)


def get_small_model(resolution, n_classes=None, is_regression=False):
    model = Sequential([
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=3, padding="same",
               kernel_initializer='random_uniform',
               input_shape=(resolution, resolution, 1)),
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

        # unary predictions: (64, 64, 256) -> (64, 64, n_classes)
        Conv2D(filters=n_classes, kernel_size=1, padding="same",
               kernel_initializer='random_uniform',),
        Activation('softmax'),

        # bilinear upsampling: (64, 64, n_classes) -> (256, 256, n_classes)
        UpSampling2D(size=(4, 4), data_format="channels_last", interpolation="bilinear")
    ])
    return model


def get_tiny_model(resolution, n_classes=None, is_regression=False):
    """
    OBSOLETE
    """
    model = Sequential([
        # conv 1: (256, 256, 1) -> (128, 128, 64)
        Conv2D(filters=64, kernel_size=5, strides=2, padding="same",
               kernel_initializer='random_uniform',
               input_shape=(resolution, resolution, 1)),
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

        # prediction (256, 256, 256) -> (256, 256, n_classes)
        Conv2DTranspose(filters=n_classes, kernel_size=1, padding="same",
                        kernel_initializer='random_uniform',),
        Activation('softmax')
    ])
    return model