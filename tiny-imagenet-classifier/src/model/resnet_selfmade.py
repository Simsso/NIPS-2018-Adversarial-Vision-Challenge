import tensorflow as tf
import data.tiny_imagenet as data


NAME = 'resnet_x01'


def conv2d(inputs, filters, kernel_size, strides):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding='same', use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer())


def block_layer(x, filters, blocks, strides, is_training):
    def projection_shortcut(proj_inputs):
        return conv2d(proj_inputs, filters, kernel_size=1, strides=strides)

    # Only the first block per block_layer uses projection_shortcut and strides
    x = building_block_v2(x, filters, is_training, projection_shortcut, strides)

    for _ in range(1, blocks):
        x = building_block_v2(x, filters, is_training, None, 1)
    return x


def batch_norm(inputs, is_training):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                         training=is_training, fused=True)


def building_block_v2(x, filters, is_training, projection_shortcut, strides):
    shortcut = x
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(x)

    x = conv2d(x, filters, kernel_size=3, strides=strides)

    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    x = conv2d(x, filters, kernel_size=3, strides=1)

    return x + shortcut


def graph(x, is_training, drop_prob, wd):
    # parametrization
    num_filters_base = 64
    kernel_size = 3
    conv_stride = 1
    first_pool_size = 0
    first_pool_stride = 2
    block_sizes = [2, 2, 2, 2]
    block_strides = [1, 2, 2, 2]

    x = conv2d(x, num_filters_base, kernel_size, conv_stride)
    if first_pool_size:
        x = tf.layers.max_pooling2d(x, first_pool_size, first_pool_stride, padding='same')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters_base * (2 ** i)
        x = block_layer(x, num_filters, num_blocks, block_strides[i], is_training)

    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)

    x = tf.reduce_mean(x, [1, 2], keepdims=True)  # global average pooling
    x = tf.layers.flatten(x)

    x = tf.layers.dropout(x, rate=drop_prob, training=is_training)
    x = tf.layers.dense(x, units=data.NUM_CLASSES)

    return x, tf.nn.softmax(x, axis=1)


def loss(labels, logits):
    labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_one_hot, logits=logits)
    loss_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    return loss_mean


def accuracy(labels, softmax):
    correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
    return tf.reduce_mean(correct, name='acc')
