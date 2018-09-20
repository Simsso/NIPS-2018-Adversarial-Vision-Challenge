import tensorflow as tf
import data.tiny_imagenet as data


def model(x):
    tf.summary.image('input_image', x)
    x_flat = tf.reshape(x, [-1, data.IMG_DIM*data.IMG_DIM*data.IMG_CHANNELS])
    logits = tf.layers.dense(x_flat, data.NUM_CLASSES, use_bias=True, name='dense1')
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

