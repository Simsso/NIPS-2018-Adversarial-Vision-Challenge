import tensorflow as tf
import numpy as np
import h5py
from typing import Tuple, List
from vq_layer import cosine_knn_vector_quantization as cos_knn_vq
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline

FLAGS = tf.flags.FLAGS


def lesci_layer(x: tf.Tensor, shape: List[int], activation_size: int, proj_thres: float, k: int)\
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Scales the input vector down using a loaded PCA matrix and compares it to a loaded embedding space (k-NN).
    :param x: Activations of the previous layer
    :param shape: Size of the embedding space (num_samples x code_size)
    :param activation_size: Size of the flattened input vector x
    :param proj_thres: Majority projection threshold for the VQ-layer
    :param k: k-parameter for the VQ-layer (k-nearest-neighbor vote)
    :return: Tuple with three tensors:
            - Mask indicating whether a sample was identity mapped
            - Label that is suggested for the sample (only meaningful, where no identity mapping was used)
            - Percentage of identity mapped samples
    """

    assert len(shape) == 2
    num_samples = shape[0]
    code_size = shape[1]

    x = tf.reshape(x, [-1, activation_size])
    pca_mat = tf.get_variable('pca_mat', dtype=tf.float32,
                              initializer=__make_init(FLAGS.pca_compression_file,
                                                      shape=[activation_size, code_size],
                                                      dtype=tf.float32, mat_name='pca_out'),
                              trainable=False)

    x = tf.matmul(x, pca_mat)
    x = tf.expand_dims(x, axis=1)
    label_variable = tf.get_variable('lesci_labels', dtype=tf.int32,
                                     initializer=__make_init(FLAGS.lesci_emb_space_file, [num_samples],
                                                             tf.int32, mat_name='labels'), trainable=False)

    embedding_init = __make_init(FLAGS.lesci_emb_space_file, shape, tf.float32, mat_name='act_compressed')
    vq = cos_knn_vq(x, emb_labels=label_variable, num_classes=TinyImageNetPipeline.num_classes, k=k,
                    n=num_samples, embedding_initializer=embedding_init, constant_init=True,
                    num_splits=1, return_endpoints=True, majority_threshold=proj_thres, name='cos_knn_vq')

    identity_mask = vq.identity_mapping_mask
    label = vq.most_common_label
    percentage_identity_mapped = vq.percentage_identity_mapped

    return tf.reshape(identity_mask, [-1]), tf.reshape(label, [-1]), percentage_identity_mapped


def __make_init(mat_file_path: str, shape: List[int], dtype=tf.float32, mat_name: str = 'emb_space') -> tf.Tensor:
    """
    Creates an initializer from a matrix file. If the file cannot be found the method returns a placeholder. This
    case occurs if the weights are being loaded from a checkpoint instead.
    :param mat_file_path: File to load the data from
    :param shape: Shape of the variable
    :param dtype: Data type of the resulting tensor
    :param mat_name: Name of the entry in the dictionary data matrix with shape 'shape'
    :return: Initializer for the variable
    """
    try:
        tf.logging.info("Trying to load variable values space from '{}'.".format(mat_file_path))

        # need to use h5py for MATLAB v7.3 files
        f = h5py.File(mat_file_path)

        var_val = None
        for k, v in f.items():
            if k == mat_name:
                var_val = np.array(v)
                break
        if var_val is None:
            raise FileNotFoundError()

        tf.logging.info("Loaded variable with shape {}".format(var_val.shape))
        var_val = np.reshape(var_val, shape)
        var_placeholder = tf.placeholder(dtype, var_val.shape)
        # init_feed_dict[var_placeholder] = var_val
        return var_placeholder
    except FileNotFoundError:
        tf.logging.info("Could not load variable values; model should be initialized from a checkpoint")
        return tf.placeholder(dtype, shape)