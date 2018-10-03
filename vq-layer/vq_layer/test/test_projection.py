import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestProjection(TFTestCase):
    """
    Test the projection feature of the layer, i.e. the mapping of inputs to the closest vector in the embedding space.
    Test the usage count of embedding vectors.
    """

    def setUp(self) -> None:
        super(TestProjection, self).setUp()
        self.x_val = np.array([[1, 1], [2, 2], [2.2, 2.2]], dtype=np.float32)
        self.x = tf.placeholder_with_default(self.x_val, shape=[None, 2])
        x_reshaped = tf.expand_dims(self.x, axis=1)
        emb_space_val = np.array([[1.1, 1], [2.1, 2.1], [5, 5], [6, 7]], dtype=np.float32)
        emb_init = tf.constant_initializer(emb_space_val)
        endpoints = vector_quantization(x_reshaped, len(emb_space_val),
                                        embedding_initializer=emb_init, return_endpoints=True)
        self.y = endpoints.layer_out
        self.access_count = endpoints.access_count
        self.init_vars()

    def test_projection(self) -> None:
        """
        Test whether inputs are being projected to embedding space vectors.
        """
        y_val = self.sess.run(self.y)
        self.assert_numerically_equal(y_val, [[[1.1, 1]], [[2.1, 2.1]], [[2.1, 2.1]]])

    def test_usage_count(self) -> None:
        """
        Test whether the usage count corresponds to the number of times inputs were mapped to a given embedding space
        vector.
        """
        access_count_val = self.sess.run(self.access_count)
        self.assert_numerically_equal(access_count_val, [1, 2, 0, 0])
