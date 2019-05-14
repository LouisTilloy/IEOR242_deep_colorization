from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import argparse

from tqdm import tqdm_notebook as tqdm
from copy import copy, deepcopy
from tensorflow.keras import datasets, layers, models, Sequential
from skimage import color
from tensorflow.keras.layers import *

from utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard

BATCH_SIZE = 16
N_BINS = 313
EPOCHS = 5

load_data = tfds.load("cifar10")
train, test = load_data["train"], load_data["test"]

rgb_images = [np.array(data["image"])
                for data in train]

lab_images = [np.array(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB))
                for rgb_img in rgb_images]


def data_generator(images):
    for image in images:
        try:
            lab_image = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
            yield lab_image[:,:,0], lab_image[:,:,1], lab_image[:,:,2]
        except cv2.error:
            print("/!\\ CV2 ERROR /!\\")
            

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        #print(self.dtype)
        ############################
        #########  Conv 1  #########
        ############################

        # (batch_size, 32, 32, 1) --> (batch_size, 16, 16, 8)
        self.conv_1_1 = Conv2D(filters=8, kernel_size=3,
                               padding='same',
                               activation='relu',
                               input_shape=(32, 32, 1))
        self.conv_1_2 = Conv2D(filters=8, kernel_size=3,
                               strides=(2, 2),
                               padding='same',
                               activation='relu')
        self.bn_1 = BatchNormalization()

        ############################
        #########  Conv 2  #########
        ############################

        # (batch_size, 16, 16, 8) --> (batch_size, 8, 8, 16)
        self.conv_2_1 = Conv2D(filters=16, kernel_size=3,
                               padding='same',
                               activation='relu')
        self.conv_2_2 = Conv2D(filters=16, kernel_size=3,
                               strides=(2, 2),
                               padding='same',
                               activation='relu')
        self.bn_2 = BatchNormalization()

        ############################
        #########  Conv 3  #########
        ############################

        # (batch_size, 8, 8, 16)  --> (batch_size, 4, 4, 32)
        self.conv_3_1 = layers.Conv2D(filters=32, kernel_size=3,
                                      padding='same',
                                      activation='relu')
        self.conv_3_2 = layers.Conv2D(filters=32, kernel_size=3,
                                      padding='same',
                                      activation='relu')
        self.conv_3_3 = layers.Conv2D(filters=32, kernel_size=3,
                                      strides=(2, 2),
                                      padding='same',
                                      activation='relu')
        self.bn_3 = BatchNormalization()

        ############################
        #########  Conv 4  #########
        ############################

        # (batch_size, 4, 4, 32) --> (batch_size, 4, 4, 64)
        self.conv_4_1 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu')
        self.conv_4_2 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu')
        self.conv_4_3 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu')
        self.bn_4 = BatchNormalization()

        ############################
        #########  Conv 5  #########
        ############################

        # (batch_size, 4, 4, 64) --> (batch_size, 4, 4, 64)
        self.conv_5_1 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               dilation_rate=2)
        self.conv_5_2 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               dilation_rate=2)
        self.conv_5_3 = Conv2D(filters=64, kernel_size=3,
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               dilation_rate=2)
        self.bn_5 = BatchNormalization()

        ############################
        #########  Conv 6  #########
        ############################

        # (batch_size, 4, 4, 64) --> (batch_size, 4, 4, 64)
        self.conv_6_1 = Conv2D(filters=64, kernel_size=3,
                               padding='same',
                               activation='relu',
                               dilation_rate=2)
        self.conv_6_2 = Conv2D(filters=64, kernel_size=3,
                               padding='same',
                               activation='relu',
                               dilation_rate=2)
        self.conv_6_3 = Conv2D(filters=64, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  dilation_rate=2)
        self.bn_6 = BatchNormalization()

        ############################
        #########  Conv 7  #########
        ############################

        # (batch_size, 4, 4, 64) --> (batch_size, 4, 4, 64)
        self.conv_7_1 = Conv2D(filters=64, kernel_size=3,
                               padding='same',
                               activation='relu',
                               dilation_rate=1)
        self.conv_7_2 = Conv2D(filters=64, kernel_size=3,
                               padding='same',
                               activation='relu',
                               dilation_rate=1)
        self.conv_7_3 = Conv2D(filters=64, kernel_size=3,
                               padding='same',
                               activation='relu',
                               dilation_rate=1)
        self.bn_7 = BatchNormalization()

        ############################
        #########  Deconv  #########
        ############################

        # (batch_size, 4, 4, 64) --> (batch_size, 32, 32, 32)
        self.deconv_1_1 = Conv2DTranspose(filters=32, kernel_size=4,
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu',
                                          dilation_rate=1)
        self.deconv_1_2 = Conv2DTranspose(filters=32, kernel_size=3,
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu',
                                          dilation_rate=1)
        self.deconv_1_3 = Conv2DTranspose(filters=32, kernel_size=3,
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu',
                                          dilation_rate=1)

        ############################
        ####  Unary prediction  ####
        ############################

        # (batch_size, 32, 32, 32) --> (batch_size, 32, 32, 1)
        self.conv_a = Conv2D(filters=1,
                             kernel_size=1,
                             strides=(1, 1),
                             dilation_rate=1)
        self.conv_b = Conv2D(filters=1,
                             kernel_size=1,
                             strides=(1, 1),
                             dilation_rate=1)
        
        self.seq_layers = [self.conv_1_1, self.conv_1_2, self.bn_1,
                           self.conv_2_1, self.conv_2_2, self.bn_2,
                           self.conv_3_1, self.conv_3_2, self.conv_3_3, self.bn_3,
                           self.conv_4_1, self.conv_4_2, self.conv_4_3, self.bn_4,
                           self.conv_5_1, self.conv_5_2, self.conv_5_3, self.bn_5,
                           self.conv_6_1, self.conv_6_2, self.conv_6_3, self.bn_6,
                           self.conv_7_1, self.conv_7_2, self.conv_7_3, self.bn_7,
                           self.deconv_1_1, self.deconv_1_2, self.deconv_1_3]

    def call(self, inputs):
        x = inputs
        for layer in self.seq_layers:
            x = layer(x)
        probs_a = self.conv_a(x)
        probs_b = self.conv_b(x)
        return probs_a, probs_b
        

train_loss = tf.keras.metrics.Mean()
train_accuracy_a = tf.keras.metrics.MeanSquaredError(name='train_accuracy_a')
train_accuracy_b = tf.keras.metrics.MeanSquaredError(name='train_accuracy_b')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy_a = tf.keras.metrics.MeanSquaredError(name='test_accuracy_a')
test_accuracy_b = tf.keras.metrics.MeanSquaredError(name='test_accuracy_b')
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(image, labels_a, labels_b, model):
    with tf.GradientTape() as tape:
        probs_a, probs_b = model(image)
        loss_a = loss_object(labels_a, probs_a)
        loss_b = loss_object(labels_b, probs_b)
        loss = loss_a + loss_b
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy_a(labels_a, probs_a)
    train_accuracy_b(labels_b, probs_b)
    
@tf.function
def test_step(image, labels_a, labels_b, model):
    probs_a, probs_b = model(image)
    t_loss_a = loss_object(labels_a, probs_a)
    t_loss_b = loss_object(labels_b, probs_b)
    t_loss = t_loss_a + t_loss_b

    test_loss(t_loss)
    test_accuracy_a(labels_a, probs_a)
    test_accuracy_b(labels_b, probs_b)
    
def train_generator():
    gen = data_generator(rgb_images)
    for features, labels_a, labels_b in gen:
        inputs = []
        targets_a = []
        targets_b = []
        for i in range(BATCH_SIZE):
            #print(np.expand_dims(labels, -1)[:,:,0].shape)
            inputs.append(np.expand_dims(features, -1))
            targets_a.append(np.expand_dims(labels_a, -1))
            targets_b.append(np.expand_dims(labels_b, -1))
        yield np.array(inputs).astype('float32'), \
                np.array(targets_a).astype('float32'), \
                    np.array(targets_b).astype('float32')

def train(model):
    for epoch in range(EPOCHS):
        train_gen = train_generator()
        for index, (batch_luminance, batch_a, batch_b) in enumerate(train_gen):
            train_step(batch_luminance, batch_a, batch_b, model)
            if index%1000==0:
                print('Epoch {}, iteration {}'.format(epoch, index))
        print('Epoch {} is done'.format(epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str,
                        help="Path to file where weights will be saved")
    args = parser.parse_args()
    
    model_reg = Model()
    train(model_reg)

    model.save_weights(args.weight_path)