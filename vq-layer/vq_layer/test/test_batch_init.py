from typing import List
import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestEmbeddingSpaceBatchInit(TFTestCase):
    """
    Test for the batch initialization feature which initializes the embedding space with the first n values fed into the
    VQ layer.
    """

    def setUp(self) -> None:
        super(TestEmbeddingSpaceBatchInit, self).setUp()

        self.n = 4

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

    def feed(self, x_in: List, emb_target: List) -> None:
        x_val = np.array(x_in, dtype=np.float32)
        endpoints = vector_quantization(self.x_reshaped, self.n, embedding_initializer='batch',
                                        return_endpoints=True)
        self.init_vars()

        self.sess.run(endpoints.emb_space_batch_init, feed_dict={self.x: x_val})
        emb_val = self.sess.run(endpoints.emb_space)

        self.assert_numerically_equal(emb_val, emb_target)

    def test_batch_init(self) -> None:
        """
        The vectors in the embedding space must be replaced with the first n vectors from the input (x_in).
        """
        x_in = [[0.1, -0.1], [0., 0.3], [1.1, 0.9], [-4., -4.], [20., 20.]]
        emb_target = x_in[:self.n]

        self.feed(x_in=x_in, emb_target=emb_target)

    def test_too_few_samples_in_batch(self) -> None:
        """
        Number of samples in the first batch is smaller than the embedding space. Therefore the embedding space cannot
        be initialized completely and an error should be raised.
        """
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.feed(x_in=[[0.1, -0.1]], emb_target=[])
