import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestSplitProjection(TFTestCase):
    """
    Test the splitting along the q-dimension. That way the embedding lookup happens several times for a single input
    vector, because the vector is split along its q-dimension num_split times.
    """

    def setUp(self) -> None:
        super(TestSplitProjection, self).setUp()
        self.x_val = np.array([[1, 1, 4.5, 5], [2, 2, -5, -3], [2.2, 2.2, 1, 1]], dtype=np.float32)
        self.x = tf.placeholder_with_default(self.x_val, shape=[None, 4])
        x_reshaped = tf.expand_dims(self.x, axis=1)
        emb_space_val = np.array([[1.1, 1], [2.1, 2.1], [5, 5], [6, 7], [-3, -3]], dtype=np.float32)
        emb_init = tf.constant_initializer(emb_space_val)
        endpoints = vector_quantization(x_reshaped, len(emb_space_val), num_splits=2,
                                        embedding_initializer=emb_init, return_endpoints=True)
        self.y = endpoints.layer_out
        self.access_count = endpoints.access_count
        self.init_vars()

    def test_split_projection(self) -> None:
        """
        Test whether the projection is correct when splitting.
        """
        y_val = self.sess.run(self.y)
        self.assert_output(y_val, [[[1.1, 1, 5, 5]], [[2.1, 2.1, -3, -3]], [[2.1, 2.1, 1.1, 1]]])

    def test_split_usage_count(self) -> None:
        """
        Test whether the usage counter is correct when splitting.
        """
        access_count_val = self.sess.run(self.access_count)
        self.assert_output(access_count_val, [2, 2, 1, 0, 1])
