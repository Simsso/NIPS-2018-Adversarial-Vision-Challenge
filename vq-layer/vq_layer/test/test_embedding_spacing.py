import numpy as np
import tensorflow as tf
from typing import List
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


class TestEmbeddingSpacing(TFTestCase):
    """
    The embedding spacing is an indicator for the distance of the embedding vectors between each other.
    This test validates its correctness numerically.
    There are two spacing measures: (1: embedding_spacing) spacing between all embedding vectors and
    (2 emb_closest_spacing) minimal distance from each embedding vector to any other embedding vector.
    """

    lookup_ord = 1

    def setUp(self) -> None:
        super(TestEmbeddingSpacing, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)
        self.x_in = [[-0.5, 3], [3, -3]]

    def feed(self, emb_space_val: List, emb_spacing_target: List, emb_closest_spacing_target: List) -> None:
        emb_space_val = np.array(emb_space_val, dtype=np.float32)
        x_val = np.array(self.x_in, dtype=np.float32)
        endpoints = vq(self.x_reshaped, n=len(emb_space_val), lookup_ord=self.lookup_ord, return_endpoints=True,
                       embedding_initializer=tf.constant_initializer(emb_space_val))
        self.init_vars()
        emb_spacing_val, emb_closest_spacing_val = self.sess.run([endpoints.emb_spacing, endpoints.emb_closest_spacing],
                                                                 feed_dict={self.x: x_val})
        self.assert_numerically_equal(emb_spacing_val, emb_spacing_target)
        self.assert_numerically_equal(emb_closest_spacing_val, emb_closest_spacing_target)

    def test_single_embedding_vector(self) -> None:
        """
        Spacing does not exist if there is only a single embedding vector.
        """
        self.feed(emb_space_val=[[1, -5]], emb_spacing_target=[], emb_closest_spacing_target=[])

    def test_two_embedding_vectors(self) -> None:
        """
        Spacing between two embedding vectors
        """
        self.feed(emb_space_val=[[3, 3], [0, 0]],
                  emb_spacing_target=[6],
                  emb_closest_spacing_target=[6, 6])

    def test_multiple_embedding_vectors_1(self) -> None:
        """
        Spacing between multiple embeddings in the space
        """
        self.feed(emb_space_val=[[3, 3], [1, 2], [5, -1], [0, 0]],
                  emb_spacing_target=[3, 6, 6, 7, 3, 6],
                  emb_closest_spacing_target=[3, 3, 6, 3])

    def test_multiple_embedding_vectors_2(self) -> None:
        """
        Spacing between multiple embeddings in the space
        """
        self.feed(emb_space_val=[[1, -.5], [1, 0], [1, 1], [2, 2]],
                  emb_spacing_target=[.5, 1.5, 3.5, 1, 3, 2],
                  emb_closest_spacing_target=[.5, .5, 1, 2])

    def test_identical_values(self) -> None:
        """
        Spacing between identical values must be 0.
        """
        self.feed(emb_space_val=[[5, 4.3], [5, 4.3], [5, 4.3]],
                  emb_spacing_target=[0, 0, 0],
                  emb_closest_spacing_target=[0, 0, 0])
