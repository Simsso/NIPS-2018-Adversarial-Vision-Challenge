import numpy as np
import tensorflow as tf
from typing import Tuple, Union
from collections import namedtuple

VQEndpoints = namedtuple('VQEndpoints', ['layer_out', 'emb_space', 'access_count', 'distance', 'emb_spacing',
                                         'emb_closest_spacing', 'replace_embeds', 'emb_space_batch_init'])

__valid_lookup_ord_values = [1, 2, np.inf]


def vector_quantization(x: tf.Tensor, n: int, alpha: Union[float, tf.Tensor] = 0.1,
                        beta: Union[float, tf.Tensor] = 1e-4, gamma: Union[float, tf.Tensor] = 1e-6,
                        lookup_ord: int = 2, embedding_initializer: Union[str, tf.keras.initializers.Initializer] =
                        tf.random_normal_initializer, num_splits: int = 1, num_embeds_replaced: int = 0,
                        is_training: Union[bool, tf.Tensor] = False, return_endpoints: bool = False, name: str = 'vq') \
        -> Union[tf.Tensor, VQEndpoints]:
    """
    Vector quantization layer.
    :param x: Tensor of shape [batch, r, q], where this function quantizes along dimension q
    :param n: Size of the embedding space (number of contained vectors)
    :param alpha: Weighting of the alpha-loss term (lookup vector distance penalty)
    :param beta: Weighting of the beta-loss term (all vectors distance penalty)
    :param gamma: Weighting of the coulomb-loss term (embedding space spacing)
    :param lookup_ord: Order of the distance function; one of [np.inf, 1, 2]
    :param embedding_initializer: Initializer for the embedding space variable or 'batch'
    :param num_splits: Number of splits along the input dimension q (defaults to 1)
    :param num_embeds_replaced: If greater than 0, this adds an op to the endpoints tuple which replaces the respective
           number of least used embedding vectors since the last replacement with vectors distant from the embedding
           vectors. If 'return_endpoints' is False, changing this to a number != 0 will not result in anything.
    :param is_training: Whether or not to update replacement accumulators.
    :param return_endpoints: Whether or not to return a plurality of endpoints (defaults to False)
    :param name: Name to use for the variable scope
    :return: Only the layer output if return_endpoints is False
             VQEndpoints-tuple with the values:
                layer_out: Layer output
                emb_space: Embedding space
                access_count: Access counter with integral values indicating how often each embedding vector was used
                distance: Distance of inputs from the embedding space vectors
                emb_spacing: Embedding spacing vector where each entry indicates the distance between embedding vectors
                emb_closest_spacing: Distance of embedding vectors to the closest other embedding vector
                replace_embeds: Op that replaces the least used embedding vectors with the most distant input vectors
                emb_space_batch_init: Embedding space batch init op (is set if embedding_initializer is 'batch')
    """
    if n <= 0:
        raise ValueError("Parameter 'n' must be greater than 0.")

    if num_embeds_replaced < 0:
        raise ValueError("Parameter 'num_embeds_replaced' must be greater than or equal to 0.")

    in_shape = x.get_shape().as_list()
    if not len(in_shape) == 3:
        raise ValueError("Parameter 'x' must be a tensor of shape [batch, a, q]. Got {}.".format(in_shape))
    in_shape[0] = in_shape[0] if in_shape[0] is not None else -1  # allow for variable-sized batch dimension

    if lookup_ord not in __valid_lookup_ord_values:
        raise ValueError("Parameter 'lookup_ord' must be one of {}. Got '{}'."
                         .format(__valid_lookup_ord_values, lookup_ord))

    if num_splits <= 0:
        raise ValueError("Parameter 'num_splits' must be greater than 0. Got '{}'.".format(num_splits))

    if not in_shape[2] % num_splits == 0:
        raise ValueError("Parameter 'num_splits' must be a divisor of the third axis of 'x'. Got {} and {}."
                         .format(num_splits, in_shape[2]))

    dynamic_emb_space_init = (embedding_initializer == 'batch')
    if dynamic_emb_space_init:
        embedding_initializer = tf.zeros_initializer

    vec_size = in_shape[2] // num_splits
    x = tf.reshape(x, [in_shape[0], in_shape[1] * num_splits, vec_size])

    with tf.variable_scope(name):
        # embedding space
        emb_space = tf.get_variable('emb_space', shape=[n, vec_size], dtype=x.dtype, initializer=embedding_initializer,
                                    trainable=True)

        # map x to y, where y is the vector from emb_space that is closest to x
        # distance of x from all vectors in the embedding space
        diff = tf.expand_dims(tf.stop_gradient(x), axis=2) - emb_space
        dist = tf.norm(diff, lookup_ord, axis=3)  # distance between x and all vectors in emb
        emb_index = tf.argmin(dist, axis=2)
        y = tf.gather(emb_space, emb_index, axis=0)

        # update access counter
        one_hot_access = tf.one_hot(emb_index, depth=n)
        access_count = tf.reduce_sum(one_hot_access, axis=[0, 1], name='access_count')

        if alpha != 0:
            # closest embedding update loss (alpha-loss)
            nearest_loss = tf.reduce_mean(alpha * tf.norm(y - tf.stop_gradient(x), lookup_ord, axis=2), axis=[0, 1],
                                          name='alpha_loss')
            tf.add_to_collection(tf.GraphKeys.LOSSES, nearest_loss)

        if beta != 0:
            # all embeddings update loss (beta-loss)
            all_loss = tf.reduce_mean(beta * tf.reduce_sum(dist, axis=2), axis=[0, 1], name='beta_loss')
            tf.add_to_collection(tf.GraphKeys.LOSSES, all_loss)

        emb_spacing, emb_closest_spacing = None, None
        if gamma != 0 or return_endpoints:
            # embeddings distance from closest other embedding (coulomb-loss)
            # pair-wise diff vectors (n x n x vec_size)
            pdiff = tf.expand_dims(emb_space, axis=0) - tf.expand_dims(emb_space, axis=1)
            pdist = tf.norm(pdiff, lookup_ord, axis=2)  # pair-wise distance scalars (n x n)
            emb_spacing = strict_upper_triangular_part(pdist)
            max_identity_matrix = tf.eye(n) * tf.reduce_max(pdist, axis=[0, 1])  # removes the diagonal zeros
            assert max_identity_matrix.shape == pdist.shape
            emb_closest_spacing = tf.reduce_min(pdist + max_identity_matrix, axis=1)
            if gamma != 0:
                coulomb_loss = tf.reduce_sum(-gamma * emb_closest_spacing, axis=0, name='coulomb_loss')
                tf.add_to_collection(tf.GraphKeys.LOSSES, coulomb_loss)

        replace_embeds_and_reset = None
        if num_embeds_replaced > 0 and return_endpoints:
            # accumulators for embedding space vector usage count, inputs, and input distances
            accumulated_usage_count = tf.get_variable('accumulated_usage_count', shape=access_count.shape,
                                                      dtype=tf.float32, initializer=tf.zeros_initializer,
                                                      trainable=False)
            distant_inputs = tf.get_variable('distant_inputs', shape=[num_embeds_replaced, vec_size], dtype=tf.float32,
                                             initializer=tf.zeros_initializer, trainable=False)
            # smallest distance between input i and any vector in the embedding space is stored at position i
            input_distances = tf.get_variable('input_distances', shape=[num_embeds_replaced], dtype=tf.float32,
                                              initializer=tf.zeros_initializer, trainable=False)

            # replacement op
            # this returns the indices of the k largest values, so we negate the count to get the smallest values
            _, least_used_indices = tf.nn.top_k(-accumulated_usage_count, k=num_embeds_replaced)
            # now find the inputs in the batch that were furthest away from the embedding vectors
            _, furthest_away_indices = tf.nn.top_k(input_distances, k=num_embeds_replaced)
            furthest_away_inputs = tf.gather(distant_inputs, indices=furthest_away_indices)
            # create assign-op that replaces the least used embedding vectors with the furthest away inputs
            replace_embeds = tf.scatter_update(ref=emb_space, indices=least_used_indices, updates=furthest_away_inputs)

            with tf.control_dependencies([replace_embeds]):
                # accumulator reset ops (zeroing)
                zero_accumulator = tf.assign(accumulated_usage_count, tf.zeros_like(accumulated_usage_count))
                zero_distances = tf.assign(input_distances, tf.zeros_like(input_distances))
                replace_embeds_and_reset = tf.group(zero_accumulator, zero_distances)

            def y_with_update():
                # accumulator update ops (must be placed inside the function; otherwise tf.cond does not work)
                assert accumulated_usage_count.shape == access_count.shape
                add_access = tf.assign_add(accumulated_usage_count, access_count)
                # accumulator and current batch concatenated
                distant_inputs_concat = tf.concat([distant_inputs, tf.reshape(x, [-1, vec_size])], axis=0)
                input_distances_concat = tf.concat(
                    [input_distances, tf.reshape(tf.reduce_min(dist, axis=2), shape=[-1])],
                    axis=0)
                new_input_distances, distant_input_indices = tf.nn.top_k(input_distances_concat, k=num_embeds_replaced)
                new_distant_inputs = tf.gather(distant_inputs_concat, indices=distant_input_indices)
                assign_distant_inputs = tf.assign(distant_inputs, new_distant_inputs)
                assign_input_distances = tf.assign(input_distances, new_input_distances)
                update_accumulators_op = tf.group([add_access, assign_distant_inputs, assign_input_distances],
                                                  name='update_replacement_accumulators')
                with tf.control_dependencies([update_accumulators_op]):
                    return tf.identity(y)

            # execute update accumulator op on forward pass if is_training is true
            y = tf.cond(is_training, true_fn=y_with_update, false_fn=lambda: y)

        emb_space_batch_init = None
        if dynamic_emb_space_init:
            replace_value = tf.slice(tf.reshape(x, [-1, vec_size]),
                                     begin=tf.zeros_like(emb_space.shape), size=emb_space.shape)
            emb_space_batch_init = tf.assign(emb_space, value=replace_value, validate_shape=True)

        # return selection in original size
        # skip this layer when doing back-prop
        layer_out = tf.reshape(tf.stop_gradient(y - x) + x, in_shape)

        if return_endpoints:
            return VQEndpoints(layer_out, emb_space, access_count, dist, emb_spacing, emb_closest_spacing,
                               replace_embeds_and_reset, emb_space_batch_init)
        return layer_out


def vec_lookup(x: np.ndarray, emb_space: np.ndarray, norm_ord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the vector lookup with a normal Python function (non TF) which can be used in the TF graph.
    Its memory requirements are lower than the TF version with broadcasting, because the data is processed sequentially.

    Addition of the following lines of code is sufficient to make use of it.
        def vec_lookup_op(x_val: np.ndarray, emb_space_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            batch_size = x_val.shape[0]
            if num_embeds_replaced > batch_size:
                raise ValueError('Number of embedding replacements must be less than or equal to the batch size.')
            return vec_lookup(x_val, emb_space_val, lookup_ord)
        y, dist, emb_index = tf.py_func(vec_lookup_op, [x, emb_space], [tf.float32, tf.float32, tf.int32],
                                        stateful=False, name='emb_lookup_py_op')
    :param x: Tensor of shape [batch, r, q]
    :param emb_space: Embedding space values [n, vec_size]
    :param norm_ord: Order of the distance norm.
    :return: Tuple with (quantized x, distance between xs and embedding vectors, chosen embedding indices)
    """
    in_shape = x.shape
    x_val = np.reshape(x, [in_shape[0] * in_shape[1], in_shape[2]])
    y, dist, emb_index = [], [], []
    for vec in x_val:
        vec = np.expand_dims(vec, 0)
        diff = vec - emb_space
        vec_dist = np.linalg.norm(diff, norm_ord, axis=1)
        dist.append(vec_dist)
        closest_emb_index = np.argmin(vec_dist, axis=0)
        y.append(emb_space[closest_emb_index])
        emb_index.append(closest_emb_index)
    y = np.asarray(y, np.float32)
    y_out = np.reshape(y, in_shape)
    dist_out = np.reshape(np.asarray(dist, np.float32), [in_shape[0], in_shape[1], emb_space.shape[0]])
    emb_index_out = np.reshape(np.asarray(emb_index, np.int32), in_shape[:-1])
    return y_out, dist_out, emb_index_out


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
