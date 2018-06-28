import tensorflow as tf

from data.tiny_image_net import CLASS_COUNT
from data.tiny_image_net import IMAGE_SIZE


# helper function to construct a convolution layer followed by a pooling layer
def conv_pool_layer(inputs, filters=64, kernel_size=[5, 5], padding="same",
                    activation=tf.nn.relu, pool_size=[2, 2], strides=2):
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation
    )
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, strides=strides)
    return pool


"""
Defines a simple CNN to classify images of shape [IMAGE_SIZE, IMAGE_SIZE, 3] in CLASS_COUNT classes

Inputs:
- input_x should be a tensor of shape [N, IMAGE_SIZE, IMAGE_SIZE, 3] and type tf.float32
- labels  should be a tensor of shape [N] and type tf.int32
- mode

Return Value:
- if mode == TRAIN      returns the loss so it can be optimized
- if mode == PREDICT    returns a dictionary with classes (argmax) and probabilities (softmax over logits)
- if mode == EVAL       returns a dictionary with the loss and the accuracy
"""
def simple_cnn(input_x, labels, mode):
    conv1 = conv_pool_layer(input_x, filters=32)
    conv2 = conv_pool_layer(conv1, filters=64)
    conv3 = conv_pool_layer(conv2, filters=64)

    image_size = int(IMAGE_SIZE / 8)   # after 2x2 pooling from conv3, 2 and 1
    conv3_flat = tf.reshape(conv3, [-1, image_size * image_size * 64])

    dense = tf.layers.dense(
        inputs=conv3_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    logits = tf.layers.dense(
        inputs=dropout,
        units=CLASS_COUNT
    )

    classes = tf.argmax(input=logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return {
            "classes": classes,
            "probabilities": tf.nn.softmax(logits)
        }

    # for TRAIN and EVAL we need the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return {
            "loss": loss,
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)
        }

    # otherwise: TRAIN mode => return loss only
    return loss
