from __future__ import absolute_import, print_function, division

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

event_file = 'attack_log_MNIST_adam/events.out.tfevents.1541715884.j2xpnero'
original_class, target_class, image_w, image_h, image_c = 10, 10, 28, 28, 1

figure = plt.figure(figsize=(16, 8))
figure.canvas.set_window_title('Attack Visualization')

images = np.zeros([original_class, target_class, image_w, image_h, image_c], dtype=np.uint8)
perturbations = np.zeros([original_class, target_class, image_w, image_h, image_c], dtype=np.uint8)

image_label = 'TC_([0-9])_M\/TC_[0-9]_M\/OC\/([0-9])\/image'
perturbation_label = 'TC_([0-9])_P\/TC_[0-9]_P\/OC\/([0-9])\/image'

with tf.Session() as sess:
    for e in tf.train.summary_iterator(event_file):
        for v in e.summary.value:
            m = re.search(image_label, v.tag)
            if m:
                images[int(m.group(1)), int(m.group(2))] = tf.image.decode_png(v.image.encoded_image_string).eval(
                    session=sess)

            m = re.search(perturbation_label, v.tag)
            if m:
                perturbations[int(m.group(1)), int(m.group(2))] = tf.image.decode_png(
                    v.image.encoded_image_string).eval(
                    session=sess)

for row in range(original_class):
    for col in range(target_class):
        figure.add_subplot(original_class * 2, target_class, row * 2 * target_class + col + 1)
        plt.axis('off')

        if image_c == 1:
            plt.imshow(images[row, col, :, :, 0], cmap='gray')
        else:
            plt.imshow(images[row, col, :, :, :])

        figure.add_subplot(original_class * 2, target_class, (row + 1) * 2 * target_class + col + 1)
        plt.axis('off')

        if image_c == 1:
            plt.imshow(perturbations[row, col, :, :, 0], cmap='gray')
        else:
            plt.imshow(perturbations[row, col, :, :, :])

plt.tight_layout(pad=0.1)
plt.savefig(fname=os.path.split(event_file)[0] + '.png')
