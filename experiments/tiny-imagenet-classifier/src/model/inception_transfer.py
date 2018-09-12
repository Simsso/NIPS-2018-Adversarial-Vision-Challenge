import tensorflow as tf
import data.tiny_imagenet as data
from util.weight_decay import add_wd
import util.tf_summary as summary_util

NAME = "inception_transfer_001"


def graph(transfer_values, is_training, weight_decay, dropout_rate):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # simple 2-layer graph
        transfer_values = tf.layers.batch_normalization(transfer_values, training=is_training)

        fc1 = add_wd(tf.layers.dense(transfer_values, units=512, name="classifier/fc1"), weight_decay)
        fc1 = tf.nn.relu(fc1)

        fc1 = tf.layers.batch_normalization(fc1, training=is_training)
        logits = add_wd(tf.layers.dense(fc1, units=data.NUM_CLASSES, name="classifier/logits"), weight_decay)
        softmax = tf.nn.softmax(logits, axis=1, name='classifier/softmax')

        summary_util.activation_summary(logits)
        summary_util.weight_summary_for_all()

    return logits, softmax


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy_loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits, label_smoothing=0.1)

    tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_loss)
    total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')  # includes weight decay loss terms
    return total_loss


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='accuracy')

