import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestEmbeddingSpaceBatchInit(TFTestCase):
    def setUp(self):
        super(TestEmbeddingSpaceBatchInit, self).setUp()

        self.n = 4

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

    def feed(self, x_in, emb_target):
        x_val = np.array(x_in, dtype=np.float32)
        endpoints = vector_quantization(self.x_reshaped, self.n, embedding_initializer='emb_space_batch_init',
                                        return_endpoints=True)
        self.init_vars()

        self.sess.run(endpoints.emb_space_batch_init, feed_dict={self.x: x_val})
        emb_val = self.sess.run(endpoints.emb_space, feed_dict={self.x: x_val})

        self.assert_output(emb_val, emb_target)

    def test_batch_init(self):
        x_in = [[0.1, -0.1], [0, 0.3], [1.1, 0.9], [-4, -4], [20, 20]]
        emb_target = x_in[:self.n]

        self.feed(x_in=x_in, emb_target=emb_target)

    def test_too_few_samples_in_batch(self):
        """
        Number of samples in the first batch is samller than the embedding space.
        """
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.feed(x_in=[[0.1, -0.1]], emb_target=None)


if __name__ == '__main__':
    unittest.main()
