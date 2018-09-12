import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data


def model(x):
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[7, 7], strides=[2, 2], padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding='valid')

    # residual block #1
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same')
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same')
    res1 = conv1 + conv3
    res1 = tf.nn.relu(res1)
    res1 = tf.layers.batch_normalization(res1)

    # residual block #2
    conv4 = tf.layers.conv2d(res1, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same')
    conv4 = tf.nn.relu(conv4)

    conv5 = tf.layers.conv2d(conv4, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same')
    res2 = res1 + conv5
    res2 = tf.nn.relu(res2)
    res2 = tf.layers.batch_normalization(res2)

    # 'adapter' layer
    conv6 = tf.layers.conv2d(res2, filters=64, kernel_size=[3, 3], strides=[2, 2], padding='valid')
    conv6 = tf.nn.relu(conv6)

    # residual block #3
    conv7 = tf.layers.conv2d(conv6, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same')
    conv7 = tf.nn.relu(conv7)

    conv8 = tf.layers.conv2d(conv7, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same')
    res3 = conv6 + conv8
    res3 = tf.nn.relu(res3)
    res3 = tf.layers.max_pooling2d(res3, pool_size=[2, 2], strides=[2, 2])

    res3_flat = tf.reshape(res3, [-1, np.product(res3.shape[1:])])

    logits = tf.layers.dense(res3_flat, units=data.NUM_CLASSES)
    softmax = tf.nn.softmax(logits, axis=1, name='softmax')

    return logits, softmax


def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32) , logits=logits)
    loss_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    return loss_mean


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='accuracy')
