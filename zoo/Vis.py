from __future__ import absolute_import, print_function, division

import os
import re

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

event_file = 'attack_log_MNIST_CW/events.out.tfevents.1542931138.j2xpnero'
original_class, target_class, image_w, image_h, image_c = 10, 10, 28, 28, 1
attacker = 'CW'

image_tensors = np.zeros([original_class, target_class], dtype=np.object)
perturbation_tensors = np.zeros([original_class, target_class], dtype=np.object)

images = np.zeros([original_class, target_class, image_w, image_h, image_c], dtype=np.uint8)
perturbations = np.zeros([original_class, target_class, image_w, image_h, image_c], dtype=np.uint8)

if attacker == 'CW':
    image_label = 'adv/image/([0-9]+)'
    perturbation_label = 'perturbation/image/([0-9]+)'
else:
    image_label = 'TC_([0-9])_M\/TC_[0-9]_M\/OC\/([0-9])\/image'
    perturbation_label = 'TC_([0-9])_P\/TC_[0-9]_P\/OC\/([0-9])\/image'

with tf.Session() as sess:
    for e in tf.train.summary_iterator(event_file):
        for v in e.summary.value:
            m = re.search(image_label, v.tag)
            if m:
                if attacker == 'CW':
                    image_tensors[int(m.group(1)) % original_class, math.floor(
                        int(m.group(1)) / original_class)] = tf.image.decode_png(v.image.encoded_image_string)
                else:
                    image_tensors[int(m.group(1)), int(m.group(2))] = tf.image.decode_png(v.image.encoded_image_string)

            m = re.search(perturbation_label, v.tag)
            if m:
                if attacker == 'CW':
                    perturbation_tensors[int(m.group(1)) % original_class, math.floor(
                        int(m.group(1)) / original_class)] = tf.image.decode_png(
                        v.image.encoded_image_string)
                else:
                    perturbation_tensors[int(m.group(1)), int(m.group(2))] = tf.image.decode_png(
                        v.image.encoded_image_string)

    images = np.array(sess.run(image_tensors.tolist()))
    perturbations = np.array(sess.run(perturbation_tensors.tolist()))

fig, ax = plt.subplots(original_class, target_class, sharey='none', figsize=(10, 10))

for row in range(original_class):
    for col in range(target_class):
        if image_c == 1:
            ax[row, col].imshow(images[row, col, :, :, 0], cmap='gray')
            ax[row, col].axis('off')
        else:
            ax[row, col].imshow(images[row, col, :, :, :])
            ax[row, col].axis('off')
plt.savefig(fname=os.path.split(event_file)[0] + 'Adversarial')

fig, ax = plt.subplots(original_class, target_class, sharey='none', figsize=(10, 10))

for row in range(original_class):
    for col in range(target_class):
        if image_c == 1:
            ax[row, col].imshow(perturbations[row, col, :, :, 0], cmap='gray')
            ax[row, col].axis('off')
        else:
            ax[row, col].imshow(perturbations[row, col, :, :, :])
            ax[row, col].axis('off')

plt.savefig(fname=os.path.split(event_file)[0] + 'Perturbation')
plt.show()
