import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestEmbeddingSpacing(TFTestCase):
    """
    The embedding spacing is an indicator for the distance of the embedding vectors between each other.
    This test validates its correctness numerically.
    """

    def setUp(self) -> None:
        super(TestEmbeddingSpacing, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)
        self.x_in = [[-0.5, 3], [3, -3]]

    def feed(self, emb_space_val, emb_spacing_target) -> None:
        emb_space_val = np.array(emb_space_val, dtype=np.float32)
        x_val = np.array(self.x_in, dtype=np.float32)
        endpoints = vector_quantization(self.x_reshaped, len(emb_space_val), lookup_ord=1, return_endpoints=True,
                                        embedding_initializer=tf.constant_initializer(emb_space_val))
        self.init_vars()
        emb_spacing_val = self.sess.run(endpoints.emb_spacing, feed_dict={self.x: x_val})
        self.assert_output(emb_spacing_val, emb_spacing_target)

    def test_two_embedding_vectors(self) -> None:
        """
        Spacing between two embedding vectors
        """
        self.feed([[3, 3], [0, 0]], [6])

    def test_multiple_embedding_vectors(self) -> None:
        """
        Spacing between multiple embeddings in the space
        """
        self.feed([[3, 3], [1, 2], [5, -1], [0, 0]], [3, 6, 6, 7, 3, 6])
