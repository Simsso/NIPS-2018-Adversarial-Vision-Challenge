import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


class TestMemoryConsumption(TFTestCase):
    """
    This test case serves the purpose of testing the memory requirements of the vq layer function.
    The test case is always true and makes sense in combination with profiling tools.
    """

    def setUp(self) -> None:
        super(TestMemoryConsumption, self).setUp()

    def feed(self, batch_size: int, r: int, q: int, n: int, num_splits: int = 1) -> None:
        vec_size = q // num_splits
        emb_space_val = np.random.normal(size=[n, vec_size])
        x_val = np.random.normal(size=[batch_size, r, q])

        x = tf.placeholder(tf.float32, shape=[batch_size, r, q])
        endpoints = vq(x, n, lookup_ord=2, embedding_initializer=tf.constant_initializer(emb_space_val),
                       alpha=0, beta=0, gamma=0, num_splits=num_splits, return_endpoints=True)
        self.init_vars()

        self.sess.run(endpoints.layer_out, feed_dict={x: x_val})
        self.assertTrue(True)

    def test_performance(self) -> None:
        """
        Uses the VQ layer without any loss or training to test the memory requirements.
        """
        self.feed(batch_size=16, r=16, q=64, n=16, num_splits=2)
