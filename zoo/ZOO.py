"""
ZOO
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans import dataset
from cleverhans.attacks import Zoo
from cleverhans.loss import CrossEntropy
from cleverhans.utils import AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import ModelBasicCNN, ModelAE, ModelAllConvolutional

FLAGS = flags.FLAGS

SOLVER = 'adam'
DATASET = 'CIFAR10'
BATCH_SIZE = 128
NB_EPOCHS = 6 if DATASET == 'MNIST' else 40
NB_EPOCHS_AE = 100
SOURCE_SAMPLES = 10
LEARNING_RATE = .001
ZOO_LEARNING_RATE = .01
ATTACK_ITERATIONS = 3000  # 1000
INIT_CONST = 0.01
BINARY_SEARCH_STEPS = 9
TARGETED = True
MODEL_PATH = os.path.join(os.path.join('models', DATASET.lower()), DATASET.lower())


def zoo(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
        source_samples=SOURCE_SAMPLES,
        learning_rate=LEARNING_RATE,
        attack_iterations=ATTACK_ITERATIONS,
        model_path=MODEL_PATH,
        targeted=TARGETED):
    """
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    if DATASET == 'MNIST':
        train_start = 0
        train_end = 60000
        test_start = 0
        test_end = 10000
        ds = dataset.MNIST(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
                           center=False)
    elif DATASET == 'SVHN':
        train_start = 0
        train_end = 73257
        test_start = 0
        test_end = 26032
        ds = dataset.SVHN(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
    elif DATASET == 'CIFAR10':
        train_start = 0
        train_end = 60000
        test_start = 0
        test_end = 10000
        ds = dataset.CIFAR10(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end,
                             center=False)

    x_train, y_train, x_test, y_test = ds.get_set('train') + ds.get_set('test')

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    x_pie = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    nb_filters = 64

    # Define TF model graph
    model = ModelBasicCNN(DATASET, nb_classes, nb_filters,
                          (None, img_rows, img_cols, nchannels)) if DATASET == 'MNIST' else ModelAllConvolutional(
        DATASET, nb_classes, nb_filters, (None, img_rows, img_cols, nchannels))
    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    modelAE = ModelAE(DATASET + 'AE', nb_classes, nb_filters, (None, img_rows, img_cols, nchannels))
    lossAE = CrossEntropy(modelAE, smoothing=0)
    print("Defined AE.")
    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'filename': os.path.split(model_path)[-1]
    }

    rng = np.random.RandomState([2018, 10, 22])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        print('Starting to train blackbox model')
        train(sess, loss, x, y, x_train, y_train, args=train_params, rng=rng,
              var_list=tf.trainable_variables(scope=DATASET))

        # print('Starting to train AE')
        # train_params = {
        #     'nb_epochs': NB_EPOCHS_AE,
        #     'batch_size': batch_size,
        #     'learning_rate': learning_rate,
        #     'filename': os.path.split(model_path)[-1]
        # }
        # # Add random noise to the input images
        # x_noisy_train = x_train + 0.5 * np.random.randn(*x_train.shape)
        # # Clip the images to be between 0 and 1
        # x_noisy_train = np.clip(x_noisy_train, 0., 1.)
        # train(sess, lossAE, x, x_pie, x_noisy_train, x_train, init_all=False, args=train_params, rng=rng,
        #       var_list=tf.trainable_variables(scope=DATASET + 'AE'))

        os.makedirs(os.path.split(model_path)[0], exist_ok=True)

        saver = tf.train.Saver()
        saver.save(sess, model_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    assert x_test.shape[0] == test_end - test_start, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples
    ###########################################################################

    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample + ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a Zoo attack object
    zoo = Zoo(DATASET + '_' + SOLVER, model, modelAE, sess=sess)

    assert source_samples == nb_classes
    idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0] for i in range(nb_classes)]
    if targeted:
        adv_inputs = np.array([[instance] * nb_classes for instance in x_test[idxs]], dtype=np.float32)

        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

        adv_inputs = adv_inputs.reshape((source_samples * nb_classes, img_rows, img_cols, nchannels))
        adv_ys = np.array([one_hot] * source_samples, dtype=np.float32).reshape(
            (source_samples * nb_classes, nb_classes))
        yname = "y_target"
    else:
        adv_inputs = x_test[idxs]

        adv_ys = None
        yname = "y"

    zoo_params = {'binary_search_steps': BINARY_SEARCH_STEPS,
                  yname: adv_ys,
                  'max_iterations': attack_iterations,
                  'learning_rate': ZOO_LEARNING_RATE,
                  'batch_size': source_samples * nb_classes if targeted else source_samples,
                  'initial_const': INIT_CONST,
                  'solver': SOLVER,
                  'image_shape': [img_rows, img_cols, nchannels],
                  'nb_classes': nb_classes}

    attack_time = time.time()

    adv = zoo.generate_np(adv_inputs, **zoo_params)

    attack_time = time.time() - attack_time

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    if targeted:
        adv_accuracy = model_eval(sess, x, y, preds, adv, adv_ys, args=eval_params)
    else:
        adv_accuracy = 1 - model_eval(sess, x, y, preds, adv, y_test[idxs], args=eval_params)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs) ** 2, axis=(1, 2, 3)) ** .5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    print('Avg. attack time {0:.4f}'.format(attack_time / adv_inputs.shape[0]))

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    zoo(nb_epochs=FLAGS.nb_epochs,
        batch_size=FLAGS.batch_size,
        source_samples=FLAGS.source_samples,
        learning_rate=FLAGS.learning_rate,
        attack_iterations=FLAGS.attack_iterations,
        model_path=FLAGS.model_path,
        targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_integer('source_samples', SOURCE_SAMPLES, 'Number of test inputs to attack')
    flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate for training')
    flags.DEFINE_string('model_path', MODEL_PATH, 'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', ATTACK_ITERATIONS, 'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', TARGETED, 'Run the tutorial in targeted mode?')
    flags.DEFINE_string('solver', SOLVER, 'Adam or Newton?')
    flags.DEFINE_string('dataset', DATASET, 'MNIST or SVHN or CIFAR10?')

    tf.app.run()
