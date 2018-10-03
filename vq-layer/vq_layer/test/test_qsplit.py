import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization
from vq_layer.vq_layer import VQEndpoints
from typing import List


class TestSplitProjection(TFTestCase):
    """
    Test the splitting along the q-dimension. That way the embedding lookup happens several times for a single input
    vector, because the vector is split along its q-dimension num_split times.
    """

    def get_endpoints(self, x_val: List, emb_space_val: List, num_splits: int) -> VQEndpoints:
        x_val = np.array(x_val, dtype=np.float32)
        x = tf.placeholder_with_default(x_val, shape=[None, 4])
        x_reshaped = tf.expand_dims(x, axis=1)

        emb_space_val = np.array(emb_space_val, dtype=np.float32)
        emb_init = tf.constant_initializer(emb_space_val)
        endpoints = vector_quantization(x_reshaped, n=len(emb_space_val), num_splits=num_splits,
                                        embedding_initializer=emb_init, return_endpoints=True)
        self.init_vars()
        return endpoints

    def test_split_projection(self) -> None:
        """
        Test whether the projection is correct when splitting.
        """
        x_val = [[1, 1, 4.5, 5], [2, 2, -5, -3], [2.2, 2.2, 1, 1]]
        emb_space_val = [[1.1, 1], [2.1, 2.1], [5, 5], [6, 7], [-3, -3]]
        endpoints = self.get_endpoints(x_val, emb_space_val, num_splits=2)

        y_val = self.sess.run(endpoints.layer_out)
        self.assert_numerically_equal(y_val, [[[1.1, 1, 5, 5]], [[2.1, 2.1, -3, -3]], [[2.1, 2.1, 1.1, 1]]])

    def test_split_usage_count(self) -> None:
        """
        Test whether the usage counter is correct when splitting.
        """
        x_val = [[1, 1, 4.5, 5], [2, 2, -5, -3], [2.2, 2.2, 1, 1]]
        emb_space_val = [[1.1, 1], [2.1, 2.1], [5, 5], [6, 7], [-3, -3]]
        endpoints = self.get_endpoints(x_val, emb_space_val, num_splits=2)

        access_count_val = self.sess.run(endpoints.access_count)
        self.assert_numerically_equal(access_count_val, [2, 2, 1, 0, 1])

    def test_split_projection_vector_size_1(self) -> None:
        """
        Test whether the projection and usage counter are correct when splitting with the same number of dimensions
        as the inputs themselves, i.e. num_splits = q, vector_size = 1
        """
        x_val = [[1, 2.3, 4.5, 5], [2, 2, -5, -3], [2.2, 10, 23.234, 3.2]]
        emb_space_val = [1, 5, 10]
        endpoints = self.get_endpoints(x_val, emb_space_val, num_splits=4)

        y_val, access_count_val = self.sess.run([endpoints.layer_out, endpoints.access_count])

        self.assert_numerically_equal(y_val, [[[1, 1, 5, 5]], [[1, 1, 1, 1]], [[1, 10, 10, 5]]])
        self.assert_numerically_equal(access_count_val, [7, 3, 2])
