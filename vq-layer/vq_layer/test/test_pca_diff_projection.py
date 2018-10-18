import numpy as np
import tensorflow as tf
from typing import List, Union
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


class TestPCADiffProjection(TFTestCase):
    """
    Test the vq-layer embedding lookup with PCA dimensionality reduction enabled.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray], num_components: int,
             dim_reduction_mode: str = 'pca-batch') -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

        endpoints = vq(self.x_reshaped, n=len(emb_space_val), lookup_ord=1, dim_reduction=dim_reduction_mode,
                       num_dim_reduction_components=num_components, return_endpoints=True,
                       embedding_initializer=tf.constant_initializer(np.array(emb_space_val, dtype=np.float32)))
        self.init_vars()

        y = self.sess.run(endpoints.layer_out)
        return y

    def test_batch_projection1(self):
        """
        Creates an input batch that is a noisy input of [x, 10x, .1x]. Then uses the PCA-batch mode to select
        the corresponding embedding vector, which should be the one resembling this distribution ([1, 10, .1]).
        """
        rand_x = np.random.uniform(low=0, high=10, size=10)
        rand_y = rand_x * 10 + np.random.normal(loc=0, scale=.5, size=10)
        rand_z = rand_x * .1 + np.random.normal(loc=0, scale=.5, size=10)
        x_val = np.stack((rand_x, rand_y, rand_z), axis=1)

        emb_space = [[1, -1, 1], [1, 10, .1], [-2, 3, 20], [100, 200, 300]]

        # expecting all 10 samples in the batch to be mapped to embedding vec [1, 10, .1]
        expected = np.array([1, 10, .1] * 10, dtype=np.float32)
        expected = np.reshape(expected, newshape=[10, 1, 3])

        layer_out_batch = self.feed(emb_space, x_val, num_components=2, dim_reduction_mode='pca-batch')
        self.assert_numerically_equal(layer_out_batch, expected)

    def test_emb_space_projection1(self):
        """
        Creates an input batch that is a noisy input of [x, 10x, .1x]. Then uses the PCA-embedding-only mode to select
        the corresponding embedding vector, which should be the one resembling this distribution ([1, 10, .1]).
        """
        rand_x = np.random.uniform(low=0, high=10, size=10)
        rand_y = rand_x * 10 + np.random.normal(loc=0, scale=.5, size=10)
        rand_z = rand_x * .1 + np.random.normal(loc=0, scale=.5, size=10)
        x_val = np.stack((rand_x, rand_y, rand_z), axis=1)

        emb_space = [[1, -1, 1], [1, 10, .1], [-2, 3, 20], [100, 200, 300]]

        # expecting all 10 samples in the batch to be mapped to embedding vec [1, 10, .1]
        expected = np.array([1, 10, .1] * 10, dtype=np.float32)
        expected = np.reshape(expected, newshape=[10, 1, 3])

        layer_out_emb = self.feed(emb_space, x_val, num_components=2, dim_reduction_mode='pca-emb-space')
        self.assert_numerically_equal(layer_out_emb, expected)

    def test_batch_projection2(self):
        """
        Tests the PCA-batch mode embedding lookup. Here, another lookup strategy would yield another chosen output
        vector. TODO should be explicitly tested!
        """
        x_val = np.array([[1, -1, 1, -1, 1, -1]])
        emb_space = np.array([[2, -2, 2, -2, 2, -2],
                              [.02, .2, 2, 20, 200, 2000],
                              [10, 20, 40, 80, 160, 320],
                              [1, 1, 1, 1, 1, 1]])
        expected = np.array([[2, -2, 2, -2, 2, -2]])

        layer_out_batch = self.feed(emb_space, x_val, num_components=3, dim_reduction_mode='pca-batch')
        self.assert_numerically_equal(layer_out_batch, expected)

    def test_emb_space_projection2(self):
        """
        Tests the PCA-embedding-only mode embedding lookup. Here, another lookup strategy would yield another chosen
        output vector. TODO should be explicitly tested!
        """
        x_val = np.array([[1, -1, 1, -1, 1, -1]])
        emb_space = np.array([[2, -2, 2, -2, 2, -2],
                              [.02, .2, 2, 20, 200, 2000],
                              [10, 20, 40, 80, 160, 320],
                              [1, 1, 1, 1, 1, 1]])
        expected = np.array([[2, -2, 2, -2, 2, -2]])

        layer_out_emb = self.feed(emb_space, x_val, num_components=3, dim_reduction_mode='pca-emb-space')
        self.assert_numerically_equal(layer_out_emb, expected)
