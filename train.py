import argparse
import os
import pickle
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
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
                inputs.append(features)
                targets.append(labels)
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
    parser.add_argument('-b', '--batch_size', type=int, default=40,
                        help="batch size for training (40 in the paper)")
    parser.add_argument('-r', '--resolution', type=int, default=104,
                        help="resolution size in the pre-process")
    parser.add_argument('--n_bins', type=int, default=13,
                        help="Number of bins (1 dimension, so in 2D the number will be squared")
    parser.add_argument('--regression', action="store_true",
                        help="use this flag to do regression instead of classification")
    parser.add_argument('--logdir', type=str, default="tensorboard",
                        help="Tensorboard log directory")
    args = parser.parse_args()

    # Get batch generator
    if args.data == "imagenet":
        batch_gen = batch_generator(lambda: data_generator(IMAGENET_FOLDER, resolution=args.resolution,
                                                           n_bins=args.n_bins, is_regression=args.regression),
                                    args.batch_size)
    elif args.data == "cifar10":
        batch_gen = batch_generator(lambda: cifar_10_train_data_generator(CIFAR10_FOLDER, n_bins=args.n_bins,
                                                                          is_regression=args.regression),
                                    args.batch_size)
    else:
        raise ValueError("--data argument should be either 'imagenet' or 'cifar10")

    # Get model
    if args.model == "paper":
        model = get_model(resolution=args.resolution,
                          n_classes=args.n_bins**2,
                          is_regression=args.regression)
    elif args.model == "small":
        if args.regression:
            raise NotImplementedError
        model = get_small_model(resolution=args.resolution,
                                n_classes=args.n_bins**2,
                                is_regression=args.regression)
    elif args.model == "tiny":
        if args.regression:
            raise NotImplementedError
        model = get_tiny_model(resolution=args.resolution,
                               n_classes=args.n_bins**2,
                               is_regression=args.regression)
    else:
        raise ValueError("--model argument should be 'paper', 'small', or 'tiny'")

    # ********** TRAIN **********
    # Article optimizer

    optimizer = Adam(lr=3.16e-5, beta_1=0.9, beta_2=0.99, decay=0.001)
    if args.regression:
        prefix = args.model + "_reg"
        model.compile(optimizer=optimizer,
                      loss="mean_squared_error")
    else:
        prefix = args.model + "_cla"
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy')

    logging = TensorBoard(log_dir=args.logdir)
    mc = ModelCheckpoint(prefix + '{epoch:04d}_loss{loss:.0f}.h5',
                         save_weights_only=True, period=1)

    last_epoch = 0  # change this value to fit your training situation
    # model.load_weights("{}_weights.h5".format(prefix))
    model.fit_generator(batch_gen, steps_per_epoch=args.n_data/args.batch_size, verbose=1,
                        callbacks=[logging, mc], epochs=50, initial_epoch=last_epoch)

    model.save_weights("{}_weights.h5".format(prefix))
