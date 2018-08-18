import numpy as np
import tensorflow as tf
import data.tiny_imagenet as data
import util.tf_summary as summary_util

NAME = 'tiny_inception_002'


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


def inception_layer(value, name, wd, reduction_filter_counts, conv_filter_counts, batch_norm=False, is_training=None):
    """
    Creates an inception layer.

    Inputs:
        - value: previous layer tensor
        - name: the name of the inception layer (block)
        - wd: how much to weight the weight decay
        - reduction_filter_counts: array of the number of filters used in the dimensionality reduction 1x1 convolutions
                                   where the order is as follows: [#(3x3 reduce), #(5x5 reduce), #(after-max-pooling)]
        - conv_filter_counts: array of the number of filters used for the non-reduction (1x1, 3x3, 5x5) convolutions
                              in the order of [#(1x1 direct), #(3x3), #(5x5)
        - batch_norm: whether or not to apply batch norm before the final relu activations
        - is_training: boolean tensor (only needed if batch_norm==True)
    """

    assert (len(reduction_filter_counts) == 3)
    assert (len(conv_filter_counts) == 3)

    # 1x1 direct
    conva = add_wd(tf.layers.conv2d(value, filters=conv_filter_counts[0], kernel_size=[1, 1], name='%s/conv-1x1-direct' % name), wd)

    # 1x1 -> 3x3
    convb = add_wd(tf.layers.conv2d(value, filters=reduction_filter_counts[0], kernel_size=[1, 1], name='%s/conv-1x1-reduce-3x3' % name), wd)
    convb = tf.nn.relu(convb)
    convb = add_wd(tf.layers.conv2d(convb, filters=conv_filter_counts[1], kernel_size=[3, 3], padding='same', name='%s/conv-3x3' % name), wd)

    # 1x1 -> 5x5
    convc = add_wd(tf.layers.conv2d(value, filters=reduction_filter_counts[1], kernel_size=[1, 1], name='%s/conv-1x1-reduce-5x5' % name), wd)
    convc = tf.nn.relu(convc)
    convc = add_wd(tf.layers.conv2d(convc, filters=conv_filter_counts[2], kernel_size=[5, 5], padding='same', name='%s/convc2' % name), wd)

    # 3x3 max pooling -> 1x1
    poola = tf.layers.max_pooling2d(value, pool_size=[3, 3], strides=[1, 1], padding='same')
    convd = add_wd(tf.layers.conv2d(poola, filters=reduction_filter_counts[2], kernel_size=[1, 1], name='%s/conv-1x1-reduce-maxpool' % name), wd)

    # add activations
    pre_act = tf.concat([conva, convb, convc, convd], axis=3, name=name)
    if batch_norm:
        pre_act = tf.layers.batch_normalization(pre_act, training=is_training)

    activations = tf.nn.relu(pre_act, name=("%s-activation" % name))
    summary_util.activation_summary(activations)

    return activations


def auxiliary_softmax_branch(inputs, name, is_training, avg_pool_size=[5, 5], avg_pool_strides=3):
    softmax_aux = tf.layers.average_pooling2d(inputs, pool_size=avg_pool_size, strides=avg_pool_strides)
    softmax_aux = tf.layers.conv2d(softmax_aux, filters=128, kernel_size=[1, 1], strides=1, name=("%s/conv-1x1" % name))
    softmax_aux = tf.layers.batch_normalization(softmax_aux, training=is_training)
    softmax_aux = tf.nn.relu(softmax_aux)
    print(("after %s-pool: " % name), softmax_aux.get_shape().as_list())

    softmax_aux = tf.layers.flatten(softmax_aux)
    softmax_aux = tf.layers.dense(softmax_aux, units=256, activation=tf.nn.relu, name=("%s/dense-1-256" % name))
    softmax_aux = tf.layers.dropout(softmax_aux, rate=0.7, training=is_training)

    logits = tf.layers.dense(softmax_aux, units=data.NUM_CLASSES, name=("%s/logits-out" % name))
    softmax = tf.nn.softmax(logits, axis=1, name=("%s/softmax" % name))

    summary_util.activation_summary(logits)
    summary_util.activation_summary(softmax)

    return logits, softmax

def graph(inputs, is_training, dropout_prob, wd):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        tf.summary.image('input_image', inputs)

        # initial conv layer 1
        conv1 = add_wd(tf.layers.conv2d(inputs, filters=128, kernel_size=[5, 5], strides=2, name="conv-initial-1"), wd)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=1)
        summary_util.activation_summary(conv1)

        print("after conv1: ", conv1.get_shape().as_list())

        # initial conv layer 2 (with 1x1 reduction as in GoogLeNet)
        conv2 = add_wd(tf.layers.conv2d(conv1, filters=64, kernel_size=[1, 1], name="conv-1x1-reduce-initial-2"), wd)
        conv2 = tf.nn.relu(conv2)
        conv2 = add_wd(tf.layers.conv2d(conv2, filters=128, kernel_size=[3, 3], name="conv-initial-2"), wd)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=1)
        summary_util.activation_summary(conv2)

        print("after conv2: ", conv2.get_shape().as_list())

        # Let's go deeper!
        incep3a = inception_layer(conv2,   'incep3a', wd, reduction_filter_counts=[96, 16, 32],  conv_filter_counts=[64, 128, 32])
        incep3b = inception_layer(incep3a, 'incep3b', wd, reduction_filter_counts=[128, 32, 64], conv_filter_counts=[128, 64, 96])
        maxpool1 = tf.layers.max_pooling2d(incep3b, pool_size=[3, 3], strides=2)
        print("after incep3a / 3b / maxpool: ", maxpool1.get_shape().as_list())

        # Some more inception goodness
        incep4a = inception_layer(maxpool1, 'incep4a', wd, reduction_filter_counts=[96, 16, 64],  conv_filter_counts=[192, 208, 48])
        # -------------- auxiliary softmax output branch 1 ---------------
        logits_aux_1, _ = auxiliary_softmax_branch(incep4a, name="softmax_aux_1", is_training=is_training)
        # ----------------------------------------------------------------
        incep4b = inception_layer(incep4a,  'incep4b', wd, reduction_filter_counts=[112, 24, 64], conv_filter_counts=[128, 128, 64])
        incep4c = inception_layer(incep4b,  'incep4c', wd, reduction_filter_counts=[128, 24, 64], conv_filter_counts=[128, 256, 64],
                                  batch_norm=True, is_training=is_training)
        maxpool2 = tf.layers.max_pooling2d(incep4c, pool_size=[3, 3], strides=2)
        print("after incep4a / 4b / 4c / maxpool: ", maxpool2.get_shape().as_list())

        incep5a = inception_layer(maxpool2, 'incep5a', wd, reduction_filter_counts=[160, 32, 128], conv_filter_counts=[128, 256, 128])
        # -------------- auxiliary softmax output branch 2 ---------------
        logits_aux_2, _ = auxiliary_softmax_branch(incep5a, name="softmax_aux_2", is_training=is_training,
                                                   avg_pool_size=[4, 4], avg_pool_strides=1)
        # ----------------------------------------------------------------  
        incep5b = inception_layer(incep5a,  'incep5b', wd, reduction_filter_counts=[128, 48, 128], conv_filter_counts=[256, 256, 128], 
                                  batch_norm=True, is_training=is_training)
        # output has 256 + 256 + 128 + 128 = 768 channels

        # average pooling (reduce to spatial 1x1)
        avgpool = tf.layers.average_pooling2d(incep5b, pool_size=[4, 4], strides=1)
        print("after incep5a / 5b / avgpool: ", avgpool.get_shape().as_list())
        flat = tf.layers.flatten(avgpool)
        
        # dropout
        dropout = tf.layers.dropout(flat, rate=dropout_prob, training=is_training)

        # one dense layer
        logits = add_wd(tf.layers.dense(dropout, units=data.NUM_CLASSES), wd)
        softmax = tf.nn.softmax(logits, axis=1, name="softmax")

        summary_util.activation_summary(logits)
        summary_util.weight_summary_for_all()

    return (logits, logits_aux_1, logits_aux_2), softmax


def loss(labels, logits):
    logits_main, logits_aux_1, logits_aux_2 = logits
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)

    cross_entropy_main = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits_main)
    cross_entropy_aux_1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits_aux_1)
    cross_entropy_aux_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits_aux_2)

    # add auxiliary branches-loss weighted by 0.2 to normal 'out' cross entropy
    cross_entropy_loss = tf.reduce_mean(cross_entropy_main, name='cross_entropy_loss')
    cross_entropy_loss = cross_entropy_loss + 0.2 * tf.reduce_mean(cross_entropy_aux_1, name='cross_entropy_loss_aux_1')
    cross_entropy_loss = cross_entropy_loss + 0.2 * tf.reduce_mean(cross_entropy_aux_2, name='cross_entropy_loss_aux_2')

    tf.add_to_collection('losses', cross_entropy_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')  # includes weight decay loss terms
    return total_loss


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='acc')
