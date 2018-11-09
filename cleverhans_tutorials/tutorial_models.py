"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import math

import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Model

from cleverhans.model import Model
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear
from cleverhans.picklable_model import Softmax


class ModelBasicCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, input_shape))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=HeReLuNormalInitializer)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            yy = my_conv(x, self.nb_filters, 8, strides=2, padding='same')
            yy = my_conv(yy, 2 * self.nb_filters, 6, strides=2, padding='valid')
            yy = my_conv(yy, 2 * self.nb_filters, 5, strides=1, padding='valid')
            logits = tf.layers.dense(
                tf.layers.flatten(yy), self.nb_classes,
                kernel_initializer=HeReLuNormalInitializer)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class ModelAE(Model):
    O_ENCODED = 'encoded'

    def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

        # Do a dummy run of fprop to make sure the variables are created from
        # the start
        self.fprop(tf.placeholder(tf.float32, input_shape))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            ### Encoder
            conv1 = tf.layers.conv2d(x, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 28x28x32
            maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')
            # Now 14x14x32
            conv2 = tf.layers.conv2d(maxpool1, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 14x14x32
            maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')
            # Now 7x7x32
            conv3 = tf.layers.conv2d(maxpool2, 16, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 7x7x16
            encoded = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same')
            # Now 4x4x16

            ### Decoder
            upsample1 = tf.image.resize_nearest_neighbor(encoded, (7, 7))
            # Now 7x7x16
            conv4 = tf.layers.conv2d(upsample1, 16, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 7x7x16
            upsample2 = tf.image.resize_nearest_neighbor(conv4, (14, 14))
            # Now 14x14x16
            conv5 = tf.layers.conv2d(upsample2, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 14x14x32
            upsample3 = tf.image.resize_nearest_neighbor(conv5, (28, 28))
            # Now 28x28x32
            conv6 = tf.layers.conv2d(upsample3, 32, (3, 3), padding='same', activation=tf.nn.relu)
            # Now 28x28x32

            logits = tf.layers.conv2d(conv6, 1, (3, 3), padding='same', activation=None)
            # Now 28x28x1

            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits),
                    self.O_ENCODED: encoded}

    def get_encoded(self, x, **kwargs):
        return self.fprop(x, **kwargs)[self.O_ENCODED]


class ModelAllConvolutional(Model):
    def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters
        self.input_shape = input_shape

        # Do a dummy run of fprop to create the variables from the start
        self.fprop(tf.placeholder(tf.float32, input_shape))
        # Put a reference to the params in self so that the params get pickled
        self.params = self.get_params()

    def fprop(self, x, **kwargs):
        del kwargs
        conv_args = dict(
            activation=tf.nn.leaky_relu,
            kernel_initializer=HeReLuNormalInitializer,
            kernel_size=3,
            padding='same')
        y = x

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            log_resolution = int(round(math.log(self.input_shape[1]) / math.log(2)))
            for scale in range(log_resolution - 2):
                y = tf.layers.conv2d(y, self.nb_filters << scale, **conv_args)
                y = tf.layers.conv2d(y, self.nb_filters << (scale + 1), **conv_args)
                y = tf.layers.average_pooling2d(y, 2, 2)
            y = tf.layers.conv2d(y, self.nb_classes, **conv_args)
            logits = tf.reduce_mean(y, [1, 2])
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class HeReLuNormalInitializer(tf.initializers.random_normal):
    def __init__(self, dtype=tf.float32):
        super(HeReLuNormalInitializer, self).__init__(dtype=dtype)

    def get_config(self):
        return dict(dtype=self.dtype.name)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        dtype = self.dtype if dtype is None else dtype
        std = tf.rsqrt(tf.cast(tf.reduce_prod(shape[:-1]), tf.float32) + 1e-7)
        return tf.random_normal(shape, stddev=std, dtype=dtype)


def make_basic_picklable_cnn(nb_filters=64, nb_classes=10,
                             input_shape=(None, 28, 28, 1)):
    """The model for the picklable models tutorial.
    """
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]
    model = MLP(layers, input_shape)
    return model
