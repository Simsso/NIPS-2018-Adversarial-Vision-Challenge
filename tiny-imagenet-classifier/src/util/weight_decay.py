import tensorflow as tf

def get_params(x):
    """ Extracts kernel params (not the biases) from a tf.layers.dense or tf.layers.conv2d tensor. """
    name = x.name.split('/')[0:-1]
    return [tf.get_variable('/'.join(name) + '/kernel')]


def get_all_params(x):
    """ Extracts all params (including the biases) from a tf.layers.dense or tf.layers.conv2d tensor. """
    name = x.name.split('/')[0:-1]
    return [tf.get_variable('/'.join(name) + '/kernel'), tf.get_variable('/'.join(name) + '/bias')]


def add_wd(op, wd):
    """
    Extracts the parameters from an op and adds weight decay to the losses collection.
    Return the op for simple chaining.
    """
    params = get_params(op)
    for param in params:
        weight_decay = tf.multiply(tf.nn.l2_loss(param), wd)
        tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay)
    return op