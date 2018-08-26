import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data


NAME = 'sep_cnn_a'


def k_in(stddev):
    """ Kernel initializer with given standard deviation. """
    return tf.truncated_normal_initializer(mean=0, stddev=stddev, dtype=tf.float32)


def graph(x, drop_prob, wd):
    tf.summary.image('input_image', x)

    conv1 = tf.layers.conv2d(x, 256, [5, 5], kernel_initializer=k_in(5e-2))
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2, 2], padding='valid')

    sepconv1 = tf.layers.separable_conv2d(conv1, 256, [3, 3], activation=tf.nn.relu)
    sepconv1 = tf.layers.batch_normalization(sepconv1)

    sepconv2 = tf.layers.separable_conv2d(sepconv1, 512, [3, 3], activation=tf.nn.relu)
    sepconv2 = tf.layers.batch_normalization(sepconv2)

    sepconv3 = tf.layers.separable_conv2d(sepconv2, 1024, [2, 2], strides=[2, 2], activation=tf.nn.relu)
    sepconv3 = tf.layers.batch_normalization(sepconv3)
    sepconv3 = tf.layers.max_pooling2d(sepconv3, 3, strides=[3, 3], padding='valid')

    conv_flat = tf.reshape(sepconv3, [-1, np.product(sepconv3.shape[1:])])

    dense1 = tf.layers.dense(conv_flat, 1024, activation=tf.nn.relu, kernel_initializer=k_in(.004))
    dense1 = tf.layers.batch_normalization(dense1)

    dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu, kernel_initializer=k_in(.004))
    dense2 = tf.layers.batch_normalization(dense2)

    logits = tf.layers.dense(dense2, data.NUM_CLASSES, kernel_initializer=k_in(1./200))
    softmax = tf.nn.softmax(logits, axis=1, name='softmax')

    return logits, softmax


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    loss_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    return loss_mean


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='acc')

