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
    parser.add_argument('-b', '--batch_size', type=int, default=40,
                        help="batch size for training (40 in the paper)")
    parser.add_argument('-r', '--resolution', type=int, default=104,
                        help="resolution size in the pre-process")
    args = parser.parse_args()

    # Get batch generator
    if args.data == "imagenet":
        batch_gen = batch_generator(lambda: data_generator(IMAGENET_FOLDER, args.resolution),
                                    args.batch_size)
    elif args.data == "cifar10":
        batch_gen = batch_generator(lambda: cifar_10_train_data_generator(CIFAR10_FOLDER, args.resolution),
                                    args.batch_size)
    else:
        raise ValueError("--data argument should be either 'imagenet' or 'cifar10")

    # Get model
    if args.model == "paper":
        model = get_model(args.resolution)
    elif args.model == "small":
        model = get_small_model(args.resolution)
    elif args.model == "tiny":
        model = get_tiny_model(args.resolution)
    else:
        raise ValueError("--model argument should be 'paper', 'small', or 'tiny'")

    # ********** TRAIN **********
    # Article optimizer
    optimizer = Adam(lr=3.16e-5, beta_1=0.9, beta_2=0.99, decay=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',)
                  #  metrics=['sparse_categorical_accuracy'])

    mc = ModelCheckpoint(args.model + '_weights{epoch:08d}.h5',
                         save_weights_only=True, period=1)

    last_epoch = 0  # change this value to fit your training situation
    #model.load_weights("{}_weights.h5".format(args.model))

    model.fit_generator(batch_gen, steps_per_epoch=args.n_data/args.batch_size, verbose=1,
                        callbacks=[mc], epochs=50, initial_epoch=last_epoch)

    model.save_weights("{}_weights.h5".format(args.model))
