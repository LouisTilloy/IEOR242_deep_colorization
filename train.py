import argparse
import os
import pickle
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard
from models import get_model, get_small_model, get_tiny_model
from utils import data_generator, cifar_10_train_data_generator


IMAGENET_FOLDER = "val"
CIFAR10_FOLDER = "cifar-10-batches-py"


def batch_generator(generator, batch_size):
    while True:
        gen = generator()
        for features, labels in gen:
            inputs = []
            targets = []
            for i in range(batch_size):
                inputs.append(np.expand_dims(features, -1))
                targets.append(np.expand_dims(labels, -1))
            yield np.array(inputs), np.array(targets)


if __name__ == "__main__":
    # ********** PARSER **********
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="tiny",
                        help="'paper', 'small' or 'tiny'")
    parser.add_argument('--data', type=str, default="imagenet",
                        help="'imagenet' or 'cifar10")
    parser.add_argument('-n', '--n_data', type=int, default=50000,
                        help="number of images in train dataset")
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help="batch size for training")
    args = parser.parse_args()

    # Get batch generator
    if args.data == "imagenet":
        batch_gen = batch_generator(lambda: data_generator(IMAGENET_FOLDER),
                                    args.batch_size)
    elif args.data == "cifar10":
        batch_gen = batch_generator(lambda: cifar_10_train_data_generator(CIFAR10_FOLDER),
                                    args.batch_size)
    else:
        raise ValueError("--data argument should be either 'imagenet' or 'cifar10")

    # Get model
    if args.model == "paper":
        model = get_model()
    elif args.model == "small":
        model = get_small_model()
    elif args.model == "tiny":
        model = get_tiny_model()
    else:
        raise ValueError("--model argument should be 'paper', 'small', or 'tiny'")

    # ********** TRAIN **********
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',)
                  #  metrics=['sparse_categorical_accuracy'])

    mc = ModelCheckpoint(args.model + '_weights{epoch:08d}.h5',
                         save_weights_only=True, period=1)

    last_epoch = -1  # change this value to fit your training situation
    #model.load_weights('weights.h5')

    model.fit_generator(batch_gen, steps_per_epoch=args.n_data/args.batch_size, verbose=1,
                        callbacks=[mc], epochs=50, initial_epoch=last_epoch + 1)

    model.save_weights("{}_weights.h5".format(args.model))
