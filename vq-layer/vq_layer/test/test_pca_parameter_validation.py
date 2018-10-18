import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


class TestPCAParameterValidation(TFTestCase):
    """
    Ensures that the parameter validation for PCA-related parameters in the vector_quantization function is done
    properly.
    """

    def test_vec_size_vs_num_components(self):
        """
        Makes sure a ValueError is raised when the resulting vector size is smaller than the number of dimensions the
        vectors are projected down to.
        """
        for x in [
            tf.placeholder(tf.float32, shape=[None, 1, 10]),
            tf.placeholder(tf.float32, shape=[100,  1, 3]),
            tf.placeholder(tf.float32, shape=[42,   1, 1]),
        ]:
            with self.assertRaises(ValueError):
                # here, num_dim_reduction_components > vec_size for all of the placeholders above
                vq(x, n=10, dim_reduction='pca-batch', num_dim_reduction_components=11)

    def test_num_components_too_small(self):
        """
        Makes sure a ValueError is raised when the given num_dim_reduction_components is smaller than one.
        """
        x = tf.placeholder(tf.float32, shape=[None, 1, 10])

        with self.assertRaises(ValueError):
            vq(x, n=10, dim_reduction='pca-embed', num_dim_reduction_components=0)
        with self.assertRaises(ValueError):
            vq(x, n=10, dim_reduction='pca-embed', num_dim_reduction_components=-10)
        with self.assertRaises(ValueError):
            vq(x, n=10, dim_reduction='pca-batch', num_dim_reduction_components=-1)

