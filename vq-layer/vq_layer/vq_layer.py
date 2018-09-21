import numpy as np
import tensorflow as tf
from typing import Union
from collections import namedtuple

VQEndpoints = namedtuple('VQEndpoints', ['layer_out', 'emb_space', 'access_count', 'distance', 'emb_spacing',
                                         'replace_embeds'])


def vector_quantization(x: tf.Tensor, n: int, alpha: Union[float, tf.Tensor] = 0.1,
                        beta: Union[float, tf.Tensor] = 1e-4, gamma: Union[float, tf.Tensor] = 1e-6,
                        lookup_ord: int = 2,
                        embedding_initializer: tf.keras.initializers.Initializer = tf.random_normal_initializer,
                        num_splits: int = 1, num_embeds_replaced: int = 0, return_endpoints: bool = False)\
        -> Union[tf.Tensor, VQEndpoints]:
    """
    Vector quantization layer.
    :param x: Tensor of shape [batch, r, q], where this function quantizes along dimension q
    :param n: Size of the embedding space (number of contained vectors)
    :param alpha: Weighting of the alpha-loss term (lookup vector distance penalty)
    :param beta: Weighting of the beta-loss term (all vectors distance penalty)
    :param gamma: Weighting of the coulomb-loss term (embedding space spacing)
    :param lookup_ord: Order of the distance function; one of [np.inf, 1, 2]
    :param embedding_initializer: Initializer for the embedding space variable
    :param num_splits: Number of splits along the input dimension q (defaults to 1)
    :param num_embeds_replaced: If not equal to 0, this adds an op to the endpoints tuple which replaces the respective
    number of least used embedding vectors in the batch with the batch activations most distant from the embedding
    vectors. If 'return_endpoints' is False, changing this to a number != 0 will not result in anything.
    :param return_endpoints: Whether or not to return a plurality of endpoints (defaults to False)
    :return: Only the layer output if return_endpoints is False
             5-tuple with the values:
                Layer output
                Embedding space
                Access counter with integral values indicating how often each vector in the embedding space was used
                Distance of inputs from the embedding space vectors
                Embedding spacing vector where each entry indicates the distance between embedding space vectors
    """
    if n <= 0:
        raise ValueError("Parameter 'n' must be greater than 0.")

    if num_embeds_replaced < 0:
        raise ValueError("Parameter 'num_embeds_replaced' must be greater than or equal to 0.")

    in_shape = x.get_shape().as_list()
    if not len(in_shape) == 3:
        raise ValueError("Parameter 'x' must be a tensor of shape [batch, a, q]. Got {}.".format(in_shape))
    in_shape[0] = in_shape[0] if in_shape[0] is not None else -1  # allow for variable-sized batch dimension

    valid_lookup_ord_values = [1, 2, np.inf]
    if lookup_ord not in valid_lookup_ord_values:
        raise ValueError("Parameter 'lookup_ord' must be one of {}. Got '{}'."
                         .format(valid_lookup_ord_values, lookup_ord))

    if num_splits <= 0:
        raise ValueError("Parameter 'num_splits' must be greater than 0. Got '{}'.".format(num_splits))

    if not in_shape[2] % num_splits == 0:
        raise ValueError("Parameter 'num_splits' must be a divisor of the third axis of 'x'. Got {} and {}."
                         .format(num_splits, in_shape[2]))

    vec_size = in_shape[2] // num_splits
    x = tf.reshape(x, [in_shape[0], in_shape[1] * num_splits, vec_size])

    with tf.variable_scope('vq'):
        # embedding space
        emb_space = tf.get_variable('emb_space', shape=[n, vec_size], dtype=x.dtype, initializer=embedding_initializer,
                                    trainable=True)

        # map x to y, where y is the vector from emb_space that is closest to x
        # distance of x from all vectors in the embedding space
        diff = tf.expand_dims(x, axis=2) - emb_space
        dist = tf.norm(diff, lookup_ord, axis=3)  # distance between x and all vectors in emb
        emb_index = tf.argmin(dist, axis=2)
        y = tf.gather(emb_space, emb_index, axis=0)

        # update access counter
        one_hot_access = tf.one_hot(emb_index, depth=n)
        access_count = tf.reduce_sum(one_hot_access, axis=[0, 1], name='access_count')

        if alpha != 0:
            # closest embedding update loss (alpha-loss)
            nearest_loss = tf.reduce_mean(alpha * tf.norm(y - x, lookup_ord, axis=2), axis=[0, 1])
            tf.add_to_collection(tf.GraphKeys.LOSSES, nearest_loss)

        if beta != 0:
            # all embeddings update loss (beta-loss)
            all_loss = tf.reduce_mean(beta * tf.reduce_sum(dist, axis=2), axis=[0, 1])
            tf.add_to_collection(tf.GraphKeys.LOSSES, all_loss)

        emb_spacing = None
        if gamma != 0 or return_endpoints:
            # all embeddings distance from each other (coulomb-loss)
            # pair-wise diff vectors (n x n x vec_size)
            pdiff = tf.expand_dims(emb_space, axis=0) - tf.expand_dims(emb_space, axis=1)
            pdist = tf.norm(pdiff, lookup_ord, axis=2)  # pair-wise distance scalars (n x n)
            emb_spacing = strict_upper_triangular_part(pdist)
            if gamma != 0:
                coulomb_loss = tf.reduce_sum(-gamma * tf.reduce_mean(pdist, axis=1), axis=0)
                tf.add_to_collection(tf.GraphKeys.LOSSES, coulomb_loss)

        # TODO what happens if num_embeds_replaced > batch_size?
        # this is difficult, as the batch size can only be known at runtime, and according to the code,
        # tf.top_k crashes in this case:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op.cc#L66
        replace_embeds = None
        if num_embeds_replaced > 0 and return_endpoints:
            # this returns the indices of the k largest values, so we negate the count to get the smallest values
            _, least_used_indices = tf.nn.top_k(-access_count, k=num_embeds_replaced)

            # now find the activations in the batch that were furthest away from the embedding vectors
            max_dist_activations = tf.squeeze(tf.reduce_max(dist, axis=2))
            furthest_away_activations, _ = tf.nn.top_k(max_dist_activations, k=num_embeds_replaced)

            # create assign-op that replaces the least used embedding vectors with the furthest away activations
            replace_embeds = tf.scatter_update(ref=emb_space, indices=least_used_indices,
                                               updates=furthest_away_activations)

        # return selection in original size
        # skip this layer when doing back-prop
        layer_out = tf.reshape(tf.stop_gradient(y - x) + x, in_shape)

        if return_endpoints:
            return VQEndpoints(layer_out, emb_space, access_count, dist, emb_spacing, replace_embeds)
        return layer_out


def strict_upper_triangular_part(matrix: tf.Tensor) -> tf.Tensor:
    """
    Converts the strict upper triangular part of a matrix into a flat vector.
    Partly taken from this SO answer: https://stackoverflow.com/a/46614084/3607984
    :param matrix: Square matrix
    :return: Vector of the elements in the strict upper triangular part of the input matrix
    """
    ones = tf.ones_like(matrix)
    mask_a = tf.matrix_band_part(ones, 0, -1)  # upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # bool mask (strict upper triangular part is True)
    upper_triangular_flat = tf.boolean_mask(matrix, mask)
    return upper_triangular_flat
