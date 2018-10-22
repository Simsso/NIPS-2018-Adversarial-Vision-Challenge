import numpy as np
import tensorflow as tf
from typing import Tuple, Union, List, Optional
from collections import namedtuple

VQEndpoints = namedtuple('VQEndpoints', ['layer_out', 'emb_space', 'access_count', 'distance', 'emb_spacing',
                                         'emb_closest_spacing', 'replace_embeds', 'emb_space_batch_init'])
CosineVQEndpoints = namedtuple('CosineVQEndpoints', ['layer_out', 'emb_space', 'percentage_identity_mapped',
                                                     'similarity_values', 'identity_mapping_mask', 'most_common_label'])

__valid_lookup_ord_values = [1, 2, np.inf]
__valid_dim_reduction_values = ['pca-batch', 'pca-emb-space']


def vector_quantization(x: tf.Tensor, n: int, alpha: Union[float, tf.Tensor] = 0.1,
                        beta: Union[float, tf.Tensor] = 1e-4, gamma: Union[float, tf.Tensor] = 1e-6,
                        lookup_ord: int = 2, dim_reduction: str = None, num_dim_reduction_components: int = -1,
                        embedding_initializer: Union[str, tf.keras.initializers.Initializer] =
                        tf.random_normal_initializer, constant_init: bool = False, num_splits: int = 1,
                        num_embeds_replaced: int = 0, is_training: Union[bool, tf.Tensor] = False,
                        return_endpoints: bool = False, name: str = 'vq') -> Union[tf.Tensor, VQEndpoints]:
    """
    Vector quantization layer.
    :param x: Tensor of shape [batch, r, q], where this function quantizes along dimension q
    :param n: Size of the embedding space (number of contained vectors)
    :param alpha: Weighting of the alpha-loss term (lookup vector distance penalty)
    :param beta: Weighting of the beta-loss term (all vectors distance penalty)
    :param gamma: Weighting of the coulomb-loss term (embedding space spacing)
    :param dim_reduction: If not None, will use the given technique to reduce the dimensionality of inputs and
           embedding vectors before comparing them using the distance measure given by lookup_ord; one of
           ['pca-batch', 'pca-emb-space'].
    :param num_dim_reduction_components: When using dimensionality reduction, this specifies the number of components
           (dimensions) that each embedding vector (and corresponding input) is reduced to.
    :param lookup_ord: Order of the distance function; one of [np.inf, 1, 2]
    :param embedding_initializer: Initializer for the embedding space variable or 'batch'
    :param constant_init: Whether the initializer is constant (in this case, the shape will not be passed to
                          'get_variable' explicitly.
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
    dynamic_emb_space_init = (embedding_initializer == 'batch')
    if dynamic_emb_space_init:
        embedding_initializer = tf.zeros_initializer

    in_shape, vec_size = __extract_vq_dimensions(x, num_splits)
    __validate_vq_parameters(n, vec_size, lookup_ord, dim_reduction, num_dim_reduction_components, num_embeds_replaced)

    x = tf.reshape(x, [in_shape[0], in_shape[1] * num_splits, vec_size])
    with tf.variable_scope(name):
        emb_space = __create_embedding_space(x, constant_init, embedding_initializer, n, vec_size)

        adjusted_x = x
        adjusted_emb_space = emb_space
        if dim_reduction is not None:
            adjusted_x, adjusted_emb_space = __transform_lookup_space(x, emb_space, dim_reduction, in_shape, n,
                                                                      vec_size, num_dim_reduction_components)

        # map x to y, where y is the vector from emb_space that is closest to x
        # distance of (adjusted) x from all vectors in the (adjusted) embedding space
        diff = tf.expand_dims(tf.stop_gradient(adjusted_x), axis=2) - adjusted_emb_space
        dist = tf.norm(diff, lookup_ord, axis=3)  # distance between x and all vectors in emb
        emb_index = tf.argmin(dist, axis=2)
        y = tf.gather(emb_space, emb_index, axis=0)

        # update access counter
        one_hot_access = tf.one_hot(emb_index, depth=n)
        access_count = tf.reduce_sum(one_hot_access, axis=[0, 1], name='access_count')

        # add losses
        if alpha != 0:
            __add_alpha_loss(x, y, lookup_ord, alpha)

        if beta != 0:
            __add_beta_loss(dist, beta)

        emb_spacing, emb_closest_spacing = None, None
        if gamma != 0 or return_endpoints:
            emb_spacing, emb_closest_spacing = __calculate_emb_spacing(emb_space, n, lookup_ord)
            if gamma != 0:
                __add_coulomb_loss(emb_closest_spacing, gamma)

        replace_embeds_and_reset = None
        if num_embeds_replaced > 0 and return_endpoints:
            y, replace_embeds_and_reset = __create_embedding_space_replacement_op(x, y, access_count, emb_space, dist,
                                                                                  num_embeds_replaced, vec_size,
                                                                                  is_training)

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


def cosine_vector_quantization(x: tf.Tensor, n: int, dim_reduction: str = None, num_dim_reduction_components: int = -1,
                               embedding_initializer: Union[str, tf.keras.initializers.Initializer] =
                               tf.random_normal_initializer, constant_init: bool = False, num_splits: int = 1,
                               return_endpoints: bool = False, name: str = 'vq') -> Union[tf.Tensor, CosineVQEndpoints]:
    """
    Vector quantization layer performing the lookup based on the largest cosine similarity (argmax of dot product).
    :param x: Tensor of shape [batch, r, q], where this function quantizes along dimension q
    :param n: Size of the embedding space (number of contained vectors)
    :param dim_reduction: If not None, will use the given technique to reduce the dimensionality of inputs and
           embedding vectors before comparing them using the distance measure given by lookup_ord; one of
           ['pca-batch', 'pca-emb-space'].
    :param num_dim_reduction_components: When using dimensionality reduction, this specifies the number of components
           (dimensions) that each embedding vector (and corresponding input) is reduced to.
    :param embedding_initializer: Initializer for the embedding space variable or 'batch'
    :param constant_init: Whether the initializer is constant (in this case, the shape will not be passed to
                          'get_variable' explicitly).
    :param num_splits: Number of splits along the input dimension q (defaults to 1)
    :param return_endpoints: Whether or not to return a plurality of endpoints (defaults to False)
    :param name: Name to use for the variable scope
    :return: Only the layer output if return_endpoints is False
             CosineVQEndpoints-tuple with the values:
                layer_out: Layer output tensor
                emb_space: Embedding space tensor
                percentage_identity_mapped: A float scalar tensor describing the percentage of inputs identity-mapped,
                                            will be None if identity_mapping_threshold < 0
                similarity_values: A rank-1 tensor containing all maximum cosine similarity values for a given batch
                                   (used to calculate a similarity-histogram); of shape [batch * r].
    """
    def perform_projection(emb_space: tf.Tensor, dot_product: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        # dot_product is of shape [n, m, batch]
        emb_index = tf.transpose(tf.argmax(dot_product, axis=0), perm=[1, 0])  # shape [batch, m]

        # the percentage of inputs identity-mapped is a constant 0 here
        return tf.gather(emb_space, emb_index, axis=0), tf.constant(0, dtype=x.dtype), None, None

    return __abstract_cosine_vector_quantization(x, perform_projection, n, dim_reduction, num_dim_reduction_components,
                                                 embedding_initializer, constant_init, num_splits, return_endpoints,
                                                 name)


def cosine_knn_vector_quantization(x: tf.Tensor, emb_labels: tf.Tensor, num_classes: int, k: int, n: int,
                                   dim_reduction: str = None, num_dim_reduction_components: int = -1,
                                   embedding_initializer: Union[str, tf.keras.initializers.Initializer] =
                                   tf.random_normal_initializer, num_splits: int = 1,
                                   return_endpoints: bool = False, majority_threshold: float = -1,
                                   name: str = 'vq') -> Union[tf.Tensor, CosineVQEndpoints]:
    """
    Vector quantization layer performing the lookup based on an emb_label majority vote (k-nearest-neighbors)
    of the k embedding vectors with largest cosine similarity.
    :param x: Tensor of shape [batch, r, q], where this function quantizes along dimension q
    :param emb_labels: The labels to which the embedding vectors correspond. This only makes sense if the embedding
           space is pre-initialized. It must be a rank-1 tensor of shape [n].
    :param num_classes: The number of classes (with respect to the emb_labels).
    :param k: The number of embedding vectors among which a label-majority vote is performed; must be <= n
    :param n: Size of the embedding space (number of contained vectors)
    :param dim_reduction: If not None, will use the given technique to reduce the dimensionality of inputs and
           embedding vectors before comparing them using the distance measure given by lookup_ord; one of
           ['pca-batch', 'pca-emb-space'].
    :param num_dim_reduction_components: When using dimensionality reduction, this specifies the number of components
           (dimensions) that each embedding vector (and corresponding input) is reduced to.
    :param embedding_initializer: Initializer for the embedding space variable or 'batch'
    :param num_splits: Number of splits along the input dimension q (defaults to 1)
    :param return_endpoints: Whether or not to return a plurality of endpoints (defaults to False)
    :param majority_threshold: If >= 0, maps inputs to their identity if the fraction of the most common class under the
           k most similar embedding vectors is less than this value.
    :param name: Name to use for the variable scope
    :return: Only the layer output if return_endpoints is False
             CosineVQEndpoints-tuple with the values:
                layer_out: Layer output tensor
                emb_space: Embedding space tensor
                percentage_identity_mapped: A float scalar tensor describing the percentage of inputs identity-mapped,
                                            will be None if identity_mapping_threshold < 0
                similarity_values: A rank-1 tensor containing all cosine similarity values for a given batch (used to
                                   calculate a similarity-histogram)
    """
    if k > n:
        raise ValueError("Parameter 'k' must be smaller than n. Got {} > {}.".format(k, n))

    def perform_projection(emb_space: tf.Tensor, dot_product: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # dot_product is of shape [n, m, batch], but top_k calculates the max values for each vector in the last dim
        dot_product = tf.transpose(dot_product, perm=[2, 1, 0])  # now we have [batch, m, n] => find top_k over n-dim
        _, top_k_indices = tf.nn.top_k(dot_product, k=k)                # shape [batch, m, k]

        # now do a majority-vote based on the labels
        top_k_labels = tf.gather(emb_labels, indices=top_k_indices)
        top_k_labels = tf.one_hot(top_k_labels, depth=num_classes)      # [batch, m, k, num_classes]

        summed_top_k_labels = tf.reduce_sum(top_k_labels, axis=2)       # [batch, m, num_classes]
        most_common_label = tf.argmax(summed_top_k_labels, axis=2, output_type=emb_labels.dtype)  # [batch, m]

        # do explicit broadcasting here since m==1 might introduce ambiguities
        quantization_shape = most_common_label.get_shape().as_list()    # [batch, m]
        most_common_label_bc = tf.broadcast_to(most_common_label, shape=[n] + quantization_shape)
        broadcasted_emb_labels = tf.broadcast_to(emb_labels, shape=quantization_shape + [n])
        broadcasted_emb_labels = tf.transpose(broadcasted_emb_labels, perm=[2, 0, 1])

        # find out which embedding vectors are possible (i.e. have the most common label among the top-k)
        possible_emb_indices = tf.equal(broadcasted_emb_labels, most_common_label_bc)  # [n, batch, m]

        # choose any of the possible embeddings (here, argmax uses the first 'True' => 1)
        first_emb_index = tf.argmax(tf.cast(possible_emb_indices, dtype=tf.int8), axis=0)   # [batch, m]
        y = tf.gather(emb_space, first_emb_index)

        identity_mapping_mask = None
        # perform identity-mapping for inputs whose top-class constitutes fraction that is below the majority_threshold
        percentage_identity_mapped = tf.constant(0, x.dtype)
        if majority_threshold >= 0:
            fraction_of_top_class = tf.reduce_max(summed_top_k_labels, axis=2) / k          # [batch, m]
            identity_mapping_mask = tf.less(fraction_of_top_class, majority_threshold)

            # broadcast to equal the shape of x / y
            identity_mapping_mask = tf.broadcast_to(tf.expand_dims(identity_mapping_mask, axis=2), shape=y.shape)
            y = tf.where(identity_mapping_mask, x=x, y=y)

            # count how many inputs in the batch were identity-mapped
            number_of_inputs_mapped = tf.reduce_sum(tf.cast(identity_mapping_mask, dtype=tf.float32))
            number_of_inputs = quantization_shape[0] * quantization_shape[1]
            percentage_identity_mapped = number_of_inputs_mapped / number_of_inputs

        return y, percentage_identity_mapped, identity_mapping_mask, most_common_label

    return __abstract_cosine_vector_quantization(x, perform_projection, n, dim_reduction, num_dim_reduction_components,
                                                 embedding_initializer, constant_init=False, num_splits=num_splits,
                                                 return_endpoints=return_endpoints, name=name)


def __abstract_cosine_vector_quantization(x: tf.Tensor, perform_projection, n: int, dim_reduction: str = None,
                                          num_dim_reduction_components: int = -1,
                                          embedding_initializer: Union[str, tf.keras.initializers.Initializer] =
                                          tf.random_normal_initializer, constant_init: bool = False,
                                          num_splits: int = 1, return_endpoints: bool = False,
                                          name: str = 'vq') -> \
        Union[tf.Tensor, CosineVQEndpoints]:
    """
    Vector quantization layer performing the lookup based on cosine similarity (dot product magnitude).
    The embedding indices are chosen according to the choose_emb_index function. Parameters are defined analogously
    to the cosine_vector_quantization function.
    :param perform_projection: A function that chooses the indices of the embeddings to be used and applies the
           projection. It's signature must be:
           (emb_space: tf.Tensor, dot_product: tf.Tensor) ->
                y: tf.Tensor, percentage_identity_mapped: tf.Tensor where y is of shape [batch, m, vec_size]
                percentage_identity_mapped is a scalar tensor
                identity_mapping_mask: Mask that marks elements which are being identity-mapped (may be None)
                most_common_label: The chosen label for the given vectors (may be None)
    """
    dynamic_emb_space_init = (embedding_initializer == 'batch')
    if dynamic_emb_space_init:
        embedding_initializer = tf.zeros_initializer

    in_shape, vec_size = __extract_vq_dimensions(x, num_splits)
    __validate_vq_parameters(n, vec_size, lookup_ord=1, dim_reduction=dim_reduction,
                             num_dim_reduction_components=num_dim_reduction_components, num_embeds_replaced=0)

    x = tf.reshape(x, [in_shape[0], in_shape[1] * num_splits, vec_size])
    with tf.variable_scope(name):
        emb_space = __create_embedding_space(x, constant_init, embedding_initializer, n, vec_size)

        adjusted_x = x
        adjusted_emb_space = emb_space
        if dim_reduction is not None:
            adjusted_x, adjusted_emb_space = __transform_lookup_space(x, emb_space, dim_reduction, in_shape, n,
                                                                      vec_size, num_dim_reduction_components)

        # normalize the spaces to have unit norm
        adjusted_x = tf.nn.l2_normalize(adjusted_x, axis=2)
        adjusted_emb_space = tf.nn.l2_normalize(adjusted_emb_space, axis=1)

        # note: adjusted_x has shape            [batch, m, adjusted_vec_size]
        #       adjusted_emb_space has shape    [n, adjusted_vec_size]
        # compute dot-product of x with embedding space over the vec_size dimension -> shape [n, m, batch]
        dot_product = tf.tensordot(adjusted_emb_space, tf.transpose(adjusted_x, perm=[2, 1, 0]), axes=1)
        y, percentage_identity_mapped, identity_mapping_mask, most_common_label = \
            perform_projection(emb_space, dot_product)

    if return_endpoints:
        similarity_values = tf.reshape(tf.reduce_max(dot_product, axis=0), shape=[-1])  # flatten to rank-1 tensor
        return CosineVQEndpoints(y, emb_space, percentage_identity_mapped, similarity_values, identity_mapping_mask,
                                 most_common_label)
    return y


def __extract_vq_dimensions(x: tf.Tensor, num_splits: int) -> Tuple[List[int], int]:
    """
    Extracts and validates the shape and resulting quantization vector size used in a vq-layer. The parameter definition
    is analogous to the parameter definition indicated in the vector_quantization function.
    :return: A tuple of the input shape (as list of ints) and the quantization vector size (as int)
    """
    in_shape = x.get_shape().as_list()
    if not len(in_shape) == 3:
        raise ValueError("Parameter 'x' must be a tensor of shape [batch, a, q]. Got {}.".format(in_shape))

    in_shape[0] = in_shape[0] if in_shape[0] is not None else -1  # allow for variable-sized batch dimension

    if num_splits <= 0:
        raise ValueError("Parameter 'num_splits' must be greater than 0. Got '{}'.".format(num_splits))

    if not in_shape[2] % num_splits == 0:
        raise ValueError("Parameter 'num_splits' must be a divisor of the third axis of 'x'. Got {} and {}."
                         .format(num_splits, in_shape[2]))

    vec_size = in_shape[2] // num_splits
    return in_shape, vec_size


def __validate_vq_parameters(n: int, vec_size: int, lookup_ord: int, dim_reduction: str,
                             num_dim_reduction_components: int, num_embeds_replaced: int) -> None:
    """
    Validates the given vq-layer parameters. The parameter definition is analogous to the parameter definition indicated
    in the vector_quantization function.
    """
    if n <= 0:
        raise ValueError("Parameter 'n' must be greater than 0.")

    if num_embeds_replaced < 0:
        raise ValueError("Parameter 'num_embeds_replaced' must be greater than or equal to 0.")

    if lookup_ord not in __valid_lookup_ord_values:
        raise ValueError("Parameter 'lookup_ord' must be one of {}. Got '{}'."
                         .format(__valid_lookup_ord_values, lookup_ord))

    if dim_reduction is not None and dim_reduction not in __valid_dim_reduction_values:
        raise ValueError("Parameter 'dim_reduction' must be either None or one of {}. Got '{}'."
                         .format(__valid_dim_reduction_values, dim_reduction))

    if dim_reduction is not None and num_dim_reduction_components <= 0:
        raise ValueError("Parameter 'num_dim_reduction_components' must be > 0 when 'dim_reduction' is not None." +
                         "Got '{}'.".format(num_dim_reduction_components))

    if not num_dim_reduction_components <= vec_size:
        raise ValueError("Parameter 'num_dim_reduction_components' must be smaller than or equal to the embedding"
                         "vector size. Got {} > {}".format(num_dim_reduction_components, vec_size))


def __create_embedding_space(x: tf.Tensor, constant_init: bool, initializer: tf.keras.initializers.Initializer, n: int,
                             vec_size: int) -> tf.Tensor:
    """
    Constructs an embedding space variable according to the given parameters (defined analogously to the
    vector_quantization function).
    :return: A tensor describing an embedding space variable
    """
    get_variable_args = {
        'name': 'emb_space',
        'dtype': x.dtype,
        'initializer': initializer,
        'trainable': True
    }
    if not constant_init:
        get_variable_args['shape'] = [n, vec_size]

    return tf.get_variable(**get_variable_args)


def __add_alpha_loss(x: tf.Tensor, y: tf.Tensor, lookup_ord: int, alpha: Union[tf.Tensor, float]) -> None:
    """
    Adds an 'alpha'-loss (closest embedding update loss) term to the tf.GraphKeys.LOSSES collection.
    Parameters are defined analogously to the vector_quantization function.
    """
    nearest_loss = tf.reduce_mean(alpha * tf.norm(y - tf.stop_gradient(x), lookup_ord, axis=2), axis=[0, 1],
                                  name='alpha_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, nearest_loss)


def __add_beta_loss(dist: tf.Tensor, beta: Union[tf.Tensor, float]) -> None:
    """
    Adds a 'beta'-loss (all embeddings update loss) term to the tf.GraphKeys.LOSSES collection.
    :param dist: Tensor describing the distance between the input x and all vectors in the embedding space;
                 of shape [batch, r, n]
    """
    all_loss = tf.reduce_mean(beta * tf.reduce_sum(dist, axis=2), axis=[0, 1], name='beta_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, all_loss)


def __add_coulomb_loss(emb_closest_spacing: tf.Tensor, gamma: Union[tf.Tensor, float]) -> None:
    """
    Adds a 'gamma'-loss (all embeddings update loss) term to the tf.GraphKeys.LOSSES collection.
    :param emb_closest_spacing: Tensor describing the smallest spacing of two embedding vectors.
    """
    coulomb_loss = tf.reduce_sum(-gamma * emb_closest_spacing, axis=0, name='coulomb_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, coulomb_loss)


def __calculate_emb_spacing(emb_space: tf.Tensor, n: int, lookup_ord: int) -> Tuple:
    """
    Calculates the embeddings' distance from the closest other embedding vector. Parameters defined analogously to the
    vector_quantization function.
    :param emb_space: A tensor describing the embedding space, of shape [n, vec_size].
    :return: A tuple of the emb_spacing and emb_closest_spacing tensors
    """
    # pair-wise diff vectors (n x n x vec_size)
    pdiff = tf.expand_dims(emb_space, axis=0) - tf.expand_dims(emb_space, axis=1)
    pdist = tf.norm(pdiff, lookup_ord, axis=2)  # pair-wise distance scalars (n x n)
    emb_spacing = strict_upper_triangular_part(pdist)
    max_identity_matrix = tf.eye(n) * tf.reduce_max(pdist, axis=[0, 1])  # removes the diagonal zeros

    assert max_identity_matrix.shape == pdist.shape
    emb_closest_spacing = tf.reduce_min(pdist + max_identity_matrix, axis=1)

    return emb_spacing, emb_closest_spacing


def pca_reduce_dims(x: tf.Tensor, num_components: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reduces the dimensionality of given input vectors to the given number of components using principal component
    analysis (PCA).
    :param x: Tensor of shape [num_vecs = n, dim], where this function reduces 'dim' to 'num_components'
    :param num_components: The number of components this function reduces the vectors in the input to
    :return: The reduced-dimension tensor of the input, of shape [num_vecs, num_components] and the principal components
             (eigenvectors corresponding to the num_components largest eigenvalues) of shape [num_vecs, num_components]
    """
    # subtract mean to get zero-mean data
    x -= tf.reduce_mean(x, axis=0, keepdims=True)

    # calculate the SVD (singular value decomposition) to get the left and right singular values u and v
    # shapes (with p := min(num_vecs, dim)): u: [num_vecs, p]; sigma: [p]; v: [dim, p]
    sigma, u, v = tf.svd(x, full_matrices=False, compute_uv=True)

    # sigma, u and v are already sorted in descending order of magnitude of the singular values => take the top
    # num_components 'eigenvectors' and project the data using the top num_components values of sigma
    projection = tf.matmul(u[:, :num_components], tf.matrix_diag(sigma[:num_components]))
    principal_components = v[:, :num_components]

    return projection, principal_components


def __transform_lookup_space(x: tf.Tensor, emb_space: tf.Tensor, mode: str, in_shape: List[int], n: int,
                             vec_size: int, num_dim_reduction_components: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Transforms the lookup space (i.e. the input batch x and the embedding space) using the given mode
    (one of ['pca-batch', 'pca-emb-space']). Parameters are defined analogously to the vector_quantization function.
    :return: A tuple of two tensors: the transformed x-space and the transformed embedding space.
    """
    adjusted_x, adjusted_emb_space = x, emb_space
    x_shape = x.get_shape().as_list()
    if mode == 'pca-batch':
        # batch-concatenated mode (calculate principal components based on batch and embedding space)
        x_concat_space = tf.reshape(x, shape=[in_shape[0] * x_shape[1], vec_size])
        concat_space = tf.concat([emb_space, x_concat_space], axis=0)
        projection, _ = pca_reduce_dims(concat_space, num_dim_reduction_components)

        # re-extract embedding space and x => will calculate distance based on projection
        adjusted_emb_space = projection[:n, :]
        adjusted_x = projection[n:, :]
        adjusted_x = tf.reshape(adjusted_x, [in_shape[0], x_shape[1], num_dim_reduction_components])
    elif mode == 'pca-emb-space':
        # embedding-space-only mode (calculate principal components only based on the embedding space)
        adjusted_emb_space, principal_components = pca_reduce_dims(emb_space, num_dim_reduction_components)

        # now use the principal components derived from the emb space to project the batch
        x_concat_space = tf.reshape(x, shape=[in_shape[0] * x_shape[1], vec_size])
        x_projection = tf.matmul(x_concat_space, principal_components)
        adjusted_x = tf.reshape(x_projection, [in_shape[0], x_shape[1], num_dim_reduction_components])
    else:
        raise ValueError("Invalid parameter 'mode': '{}'. Must be one of {}."
                         .format(mode, __valid_dim_reduction_values))

    return adjusted_x, adjusted_emb_space


def __create_embedding_space_replacement_op(x: tf.Tensor, y: tf.Tensor, access_count: tf.Tensor, emb_space: tf.Tensor,
                                            dist: tf.Tensor, num_embeds_replaced: int, vec_size: int,
                                            is_training: Union[bool, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates a TensorFlow op that replaces the given number of embedding vectors per batch. Also applies this
    replacement op automatically after each batch, is is_training is True. Parameters are defined analogously to the
    vector_quantization function.
    :param access_count: Tensor describing how often each of the embedding vectors has been used
    :param dist: Tensor describing the distance between the embeddings and the input
    :return: A tuple of two tensors: a transformed y that applies the replacement op after each training batch and
             the replacement op itself.
    """
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

    return y, replace_embeds_and_reset


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
