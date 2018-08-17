import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data

NAME = 'tiny_inception_001-aux'


def get_params(x):
    """ Extracts kernel params (not the biases) from a tf.layers.dense or tf.layers.conv2d tensor. """
    name = x.name.split('/')[0:-1]
    return [tf.get_variable('/'.join(name) + '/kernel')] #, tf.get_variable('/'.join(name) + '/bias')]


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


def inception_layer(value, name, wd, reduction_filter_counts, conv_filter_counts):
    """
    Creates an inception layer.

    Inputs:
        - value: previous layer tensor
        - name: the name of the inception layer (block)
        - wd: how much to weight the weight decay
        - reduction_filter_counts: array of the number of filters used in the dimensionality reduction 1x1 convolutions
                                   where the order is as follows: [#(3x3 reduce), #(5x5 reduce), #(after-max-pooling)]
        - conv_filter_counts: array of the number of filters used for the non-reduction (1x1, 3x3, 5x5) convolutions
                              in the order of [#(1x1 direct), #(3x3), #(5x5)]
    """

    assert (len(reduction_filter_counts) == 3)
    assert (len(conv_filter_counts) == 3)

    # 1x1 direct
    conva = add_wd(tf.layers.conv2d(value, filters=conv_filter_counts[0], kernel_size=[1, 1], name='%s/conv-1x1-direct' % name), wd)

    # 1x1 -> 3x3
    convb = add_wd(tf.layers.conv2d(value, filters=reduction_filter_counts[0], kernel_size=[1, 1], name='%s/conv-1x1-reduce-3x3' % name), wd)
    convb = add_wd(tf.layers.conv2d(convb, filters=conv_filter_counts[1], kernel_size=[3, 3], padding='same', name='%s/conv-3x3' % name), wd)

    # 1x1 -> 5x5
    convc = add_wd(tf.layers.conv2d(value, filters=reduction_filter_counts[1], kernel_size=[1, 1], name='%s/conv-1x1-reduce-5x5' % name), wd)
    convc = add_wd(tf.layers.conv2d(convc, filters=conv_filter_counts[2], kernel_size=[5, 5], padding='same', name='%s/convc2' % name), wd)

    # 3x3 max pooling -> 1x1
    poola = tf.layers.max_pooling2d(value, pool_size=[3, 3], strides=[1, 1], padding='same')
    convd = add_wd(tf.layers.conv2d(poola, filters=reduction_filter_counts[2], kernel_size=[1, 1], name='%s/conv-1x1-reduce-maxpool' % name), wd)

    # add activations
    pre_act = tf.concat([conva, convb, convc, convd], axis=3, name=name)
    post_act = tf.nn.relu(pre_act)

    return post_act


def auxiliary_softmax_branch(inputs, name, is_training, avg_pool_size=[5, 5], avg_pool_strides=3):
    softmax_aux = tf.layers.average_pooling2d(inputs, pool_size=avg_pool_size, strides=avg_pool_strides)
    softmax_aux = tf.layers.conv2d(softmax_aux, filters=256, kernel_size=[1, 1], strides=1, name=("%s/conv-1x1" % name))
    softmax_aux = tf.nn.relu(softmax_aux)
    print(("after %s-pool: " % name), softmax_aux.get_shape().as_list())
    softmax_aux = tf.layers.flatten(softmax_aux)
    softmax_aux = tf.layers.dense(softmax_aux, units=256, activation=tf.nn.relu, name=("%s/dense-1-256" % name))
    softmax_aux = tf.layers.dropout(softmax_aux, rate=0.7, training=is_training)
    softmax_aux = tf.layers.dense(softmax_aux, units=data.NUM_CLASSES, activation=tf.nn.relu, name=("%s/dense-out" % name))
    softmax_aux = tf.nn.softmax(softmax_aux, axis=1, name=("%s/softmax" % name))
    return softmax_aux

def graph(inputs, dropout_prob, is_training, wd):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        tf.summary.image('input_image', inputs)

        # initial conv layer 1
        conv1 = add_wd(tf.layers.conv2d(inputs, filters=64, kernel_size=[5, 5], strides=2, name="conv-initial-1"), wd)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=1)

        print("after conv1: ", conv1.get_shape().as_list())

        # initial conv layer 2 (with 1x1 reduction as in GoogLeNet)
        conv2 = add_wd(tf.layers.conv2d(conv1, filters=64, kernel_size=[1, 1], name="conv-1x1-reduce-initial-2"), wd)
        conv2 = add_wd(tf.layers.conv2d(conv2, filters=192, kernel_size=[3, 3], name="conv-initial-2"), wd)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=1)

        print("after conv2: ", conv2.get_shape().as_list())

        # Let's go deeper!
        incep3a = inception_layer(conv2,   'incep3a', wd, reduction_filter_counts=[96, 16, 32],  conv_filter_counts=[64, 128, 32])
        incep3b = inception_layer(incep3a, 'incep3b', wd, reduction_filter_counts=[128, 32, 64], conv_filter_counts=[128, 192, 96])
        maxpool1 = tf.layers.max_pooling2d(incep3b, pool_size=[3, 3], strides=2)
        print("after incep3a / 3b / maxpool: ", maxpool1.get_shape().as_list())

        # Some more inception goodness
        incep4a = inception_layer(maxpool1, 'incep4a', wd, reduction_filter_counts=[96, 16, 64],  conv_filter_counts=[192, 208, 48])
        # -------------- auxiliary softmax output branch 1 ---------------
        softmax_aux_1 = auxiliary_softmax_branch(incep4a, name="softmax_aux_1", is_training=is_training)
        # ----------------------------------------------------------------
        incep4b = inception_layer(incep4a,  'incep4b', wd, reduction_filter_counts=[112, 24, 64], conv_filter_counts=[160, 224, 64])
        incep4c = inception_layer(incep4b,  'incep4c', wd, reduction_filter_counts=[128, 24, 64], conv_filter_counts=[128, 256, 64])
        maxpool2 = tf.layers.max_pooling2d(incep4c, pool_size=[3, 3], strides=2)
        print("after incep4a / 4b / 4c / maxpool: ", maxpool2.get_shape().as_list())

        incep5a = inception_layer(maxpool2, 'incep5a', wd, reduction_filter_counts=[160, 32, 128], conv_filter_counts=[256, 320, 128])
        # -------------- auxiliary softmax output branch 2 ---------------
        softmax_aux_2 = auxiliary_softmax_branch(incep5a, name="softmax_aux_2", is_training=is_training,
                                                 avg_pool_size=[4, 4], avg_pool_strides=1)
        # ----------------------------------------------------------------
        incep5b = inception_layer(incep5a,  'incep5b', wd, reduction_filter_counts=[192, 48, 128], conv_filter_counts=[384, 384, 128])
        # output has 384 + 384 + 128 + 128 = 1024 channels

        # average pooling (reduce to spatial 1x1)
        avgpool = tf.layers.average_pooling2d(incep5b, pool_size=[4, 4], strides=1)
        print("after incep5a / 5b / avgpool: ", avgpool.get_shape().as_list())
        flat = tf.layers.flatten(avgpool)
        
        # dropout
        dropout = tf.layers.dropout(flat, rate=dropout_prob, training=is_training)

        # one dense layer
        logits = add_wd(tf.layers.dense(dropout, units=data.NUM_CLASSES), wd)
        softmax = tf.nn.softmax(logits, axis=1)

        # weight output with auxiliary softmax outputs, but only at training time, not at inference time
        softmax = tf.cond(pred=is_training,
                          true_fn=lambda: (0.7 * softmax + 0.15 * (softmax_aux_1 + softmax_aux_2)),
                          false_fn=lambda: softmax)
        softmax = tf.identity(softmax, name="softmax")

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
