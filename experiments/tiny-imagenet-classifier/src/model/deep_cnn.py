import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data


NAME = 'deep_cnn_b'


def k_in(stddev):
    """ Kernel initializer with given standard deviation. """
    return tf.truncated_normal_initializer(mean=0, stddev=stddev, dtype=tf.float32)


def graph(x, drop_prob, wd):
    tf.summary.image('input_image', x)

    conv1 = tf.layers.conv2d(x, 256, [5, 5], kernel_initializer=k_in(5e-2))
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2, 2], padding='valid')

    conv2 = tf.layers.conv2d(conv1, 256, [4, 4], padding='valid', kernel_initializer=k_in(5e-2))
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.layers.batch_normalization(conv2)

    conv3 = tf.layers.conv2d(conv2, 512, [3, 3], padding='valid', kernel_initializer=k_in(5e-2))
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.layers.batch_normalization(conv3)

    conv4 = tf.layers.conv2d(conv3, 1024, [2, 2], padding='valid', kernel_initializer=k_in(5e-2))
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.layers.batch_normalization(conv4)
    conv4 = tf.layers.max_pooling2d(conv4, 3, strides=[2, 2], padding='valid')

    conv_flat = tf.reshape(conv4, [-1, np.product(conv4.shape[1:])])

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

