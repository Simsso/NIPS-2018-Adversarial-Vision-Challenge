import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data


NAME = 'incepish_001a'


def get_params(x):
    """ Extracts kernel and bias params from a tf.layers.dense or tf.layers.conv2d tensor. """
    name = x.name.split('/')[0:-1]
    return [tf.get_variable('/'.join(name) + '/kernel'), tf.get_variable('/'.join(name) + '/bias')]


def k_in(stddev):
    """ Kernel initializer with given standard deviation. """
    return tf.truncated_normal_initializer(mean=0, stddev=stddev, dtype=tf.float32)


def add_wd(op, wd):
    """
    Extracts the parameters from an op and adds weight decay to the losses collection.
    Return the op for simple chaining.
    """
    params = get_params(op)
    for param in params:
        weight_decay = tf.multiply(tf.nn.l2_loss(param), wd)
        tf.add_to_collection('losses', weight_decay)
    return op


def inception_layer(value, name, wd):
    conva = add_wd(tf.layers.conv2d(value, filters=64, kernel_size=[1, 1], name='%s/conva' % name), wd)
    convb = add_wd(tf.layers.conv2d(value, filters=64, kernel_size=[1, 1], name='%s/convb1' % name), wd)
    convb = add_wd(tf.layers.conv2d(convb, filters=64, kernel_size=[3, 3], padding='same', name='%s/convb2' % name), wd)
    convc = add_wd(tf.layers.conv2d(value, filters=32, kernel_size=[1, 1], name='%s/convc1' % name), wd)
    convc = add_wd(tf.layers.conv2d(convc, filters=32, kernel_size=[5, 5], padding='same', name='%s/convc2' % name), wd)
    poola = tf.layers.max_pooling2d(value, pool_size=[3, 3], strides=[1, 1], padding='same')
    convd = add_wd(tf.layers.conv2d(poola, filters=64, kernel_size=[1, 1], name='%s/convd' % name), wd)
    pre_act = tf.concat([conva, convb, convc, convd], axis=3, name=name)
    post_act = tf.nn.relu(pre_act)
    return tf.layers.batch_normalization(post_act)


def graph(x, drop_prob, wd):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        tf.summary.image('input_image', x)

        conv1 = add_wd(tf.layers.conv2d(x, 256, [5, 5]), wd)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.batch_normalization(conv1)

        incep1 = inception_layer(conv1, 'incep1', wd)
        incep2 = inception_layer(incep1, 'incep2', wd)
        incep3 = inception_layer(incep2, 'incep3', wd)
        incep4 = inception_layer(incep3, 'incep4', wd)

        pool1 = tf.layers.max_pooling2d(incep4, (3, 3), (2, 2))

        conv2 = add_wd(tf.layers.conv2d(pool1, 256, [5, 5], strides=(2, 2)), wd)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.batch_normalization(conv2)

        pool2 = tf.layers.max_pooling2d(conv2, (3, 3), (2, 2))

        conv_flat = tf.reshape(pool2, [-1, np.product(pool2.shape[1:])])
        conv_flat = tf.layers.dropout(conv_flat, rate=drop_prob)

        dense1 = add_wd(tf.layers.dense(conv_flat, 1024, activation=tf.nn.relu, kernel_initializer=k_in(.004)), wd)
        dense1 = tf.layers.batch_normalization(dense1)
        dense1 = tf.layers.dropout(dense1, rate=drop_prob)

        dense2 = add_wd(tf.layers.dense(dense1, 512, activation=tf.nn.relu, kernel_initializer=k_in(.004)), wd)
        dense2 = tf.layers.batch_normalization(dense2)
        dense2 = tf.layers.dropout(dense2, rate=drop_prob)

        logits = add_wd(tf.layers.dense(dense2, data.NUM_CLASSES, kernel_initializer=k_in(1. / data.NUM_CLASSES)), wd)
        softmax = tf.nn.softmax(logits, axis=1, name='softmax')

    return logits, softmax


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    tf.add_to_collection('losses', cross_entropy_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')  # includes weight decay loss terms
    return total_loss


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='acc')
