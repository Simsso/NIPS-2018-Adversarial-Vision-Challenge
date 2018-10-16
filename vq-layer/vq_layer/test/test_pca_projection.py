import numpy as np
import tensorflow as tf
from typing import List, Union
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


class TestPCAProjection(TFTestCase):
    """
    Test the vq-layer projection with PCA dimensionality reduction enabled.
    """

    def setUp(self) -> None:
        super(TestPCAProjection, self).setUp()
        np.random.seed(42)

    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray], num_components: int) -> List:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

        endpoints = vq(self.x_reshaped, n=len(emb_space_val), lookup_ord=1,
                       dim_reduction='pca', num_dim_reduction_components=num_components, return_endpoints=True,
                       embedding_initializer=tf.constant_initializer(np.array(emb_space_val, dtype=np.float32)))
        self.init_vars()

        y = self.sess.run(endpoints.layer_out)
        return y

    def test_projection1(self):
        rand_x = np.random.uniform(low=0, high=10, size=10)
        rand_y = rand_x * 10 + np.random.normal(loc=0, scale=.5, size=10)
        rand_z = rand_x * .1 + np.random.normal(loc=0, scale=.5, size=10)
        x_val = np.stack((rand_x, rand_y, rand_z), axis=1)

        emb_space = [[1, -1, 1], [1, 10, .1], [-2, 3, 20], [100, 200, 300]]
        layer_out = self.feed(emb_space, x_val, num_components=2)

        expected = np.array([1, 10, .1] * 10, dtype=np.float32)
        expected = np.reshape(expected, newshape=[10, 1, 3])

        self.assert_numerically_equal(layer_out, expected)

