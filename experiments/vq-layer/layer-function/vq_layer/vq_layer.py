import tensorflow as tf


def vector_quantization(x: tf.Tensor, n, alpha=0.1, beta=1e-4, gamma=1e-6, lookup_ord='euclidean',
                        embedding_initializer=tf.random_normal_initializer, num_splits=1, return_endpoints=False):
    # shape of x is [batch, , q], where this function quantizes along dimension q
    in_shape = x.get_shape().as_list()
    in_shape[0] = in_shape[0] if in_shape[0] is not None else -1  # allow for variable-sized batch dimension

    assert in_shape[2] % num_splits == 0
    vec_size = in_shape[2] // num_splits
    x = tf.reshape(x, [in_shape[0], in_shape[1] * num_splits, vec_size])

    with tf.variable_scope('vq'):
        # embedding space
        emb_space = tf.get_variable('emb_space', shape=[n, vec_size], dtype=x.dtype, initializer=embedding_initializer,
                                    trainable=True)

        # map x to y, where y is the vector from emb_space that is closest to x
        diff = tf.expand_dims(x, axis=2) - emb_space  # distance of x from all vectors in the embedding space
        dist = tf.norm(diff, lookup_ord, axis=3)  # distance between x and all vectors in emb
        emb_index = tf.argmin(dist, axis=2)
        y = tf.gather(emb_space, emb_index, axis=0)

        # update access counter
        one_hot_access = tf.one_hot(emb_index, depth=n)
        access_count = tf.reduce_sum(one_hot_access, axis=[0, 1], name='access_count')

        # closest embedding update loss (alpha-loss)
        nearest_loss = tf.reduce_mean(alpha * tf.norm(y - x, lookup_ord, axis=2), axis=[0, 1])
        tf.add_to_collection(tf.GraphKeys.LOSSES, nearest_loss)

        # all embeddings update loss (beta-loss)
        all_loss = tf.reduce_mean(beta * tf.reduce_sum(dist, axis=2), axis=[0, 1])
        tf.add_to_collection(tf.GraphKeys.LOSSES, all_loss)

        # all embeddings distance from each other (coulomb-loss)
        # pair-wise diff vectors (n x n x vec_size)
        pdiff = tf.expand_dims(emb_space, axis=0) - tf.expand_dims(emb_space, axis=1)
        pdist = tf.norm(pdiff, lookup_ord, axis=2)  # pair-wise distance scalars (n x n)
        coulomb_loss = tf.reduce_sum(-gamma * tf.reduce_mean(pdist, axis=1), axis=0)
        tf.add_to_collection(tf.GraphKeys.LOSSES, coulomb_loss)

        # return selection in original size
        # skip this layer when doing back-prop
        layer_out = tf.reshape(tf.stop_gradient(y - x) + x, in_shape)

        if return_endpoints:
            return layer_out, emb_space, access_count
        return layer_out
