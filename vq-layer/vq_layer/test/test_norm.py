import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestNorm(TFTestCase):
    def setUp(self):
        super(TestNorm, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)
        self.emb_space_val = np.array([[1, .5], [0, 0]], dtype=np.float32)

    def feed(self, x_in, y_target, lookup_ord):
        x_val = np.array(x_in, dtype=np.float32)
        y, _, _, dist = vector_quantization(self.x_reshaped, len(self.emb_space_val), lookup_ord=lookup_ord,
                                            embedding_initializer=tf.constant_initializer(self.emb_space_val),
                                            return_endpoints=True)
        self.init_vars()
        y_val, dist_val = self.sess.run([y, dist], feed_dict={self.x: x_val})
        self.assert_output(y_val, y_target)

    def test_ord_1(self):
        """L1 distance test"""
        self.feed(x_in=[[-0.5, 3], [3, -3]], y_target=[[[0, 0]], [[1, .5]]], lookup_ord=1)

    def test_ord_2(self):
        """L2 distance test"""
        self.feed(x_in=[[-0.5, 3], [3, -3]], y_target=[[[1, .5]], [[1, .5]]], lookup_ord=2)

    def test_ord_inf(self):
        """L infinity distance test"""
        self.feed(x_in=[[0, 5], [3, -3]], y_target=[[[1, .5]], [[0, 0]]], lookup_ord=np.inf)


if __name__ == '__main__':
    unittest.main()
