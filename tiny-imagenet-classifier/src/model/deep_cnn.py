import tensorflow as tf
import data.tiny_imagenet as data


def model(x):
    tf.summary.image('input_image', x)

    conv1 = tf.layers.conv2d(x, 256, [5, 5], activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=[2, 2], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 3, strides=[2, 2])

    conv3 = tf.layers.conv2d(pool2, 64, [6, 6], strides=[2, 2], activation=tf.nn.relu)

    conv_flat = tf.reshape(conv3, [-1, conv3.shape[1] * conv3.shape[2] * conv3.shape[3]])

    dense1 = tf.layers.dense(conv_flat, 1024, activation=tf.nn.relu)

    logits = tf.layers.dense(dense1, data.NUM_CLASSES, use_bias=True, name='dense1')
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

