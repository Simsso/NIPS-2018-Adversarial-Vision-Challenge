import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List
from vq_layer.vq_layer import cosine_vector_quantization


class TestCosineProjection(TFTestCase):
    """
    Test the projection feature of the cosine-similarity based vq-layer, i.e. the mapping of inputs to the embedding
    vector with the largest cosine similarity value.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray]) -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        projection = cosine_vector_quantization(self.x_reshaped, n=len(emb_space_val),
                                                embedding_initializer=emb_space_init)

        self.init_vars()

        y = self.sess.run(projection)
        return y

    def test_projection1(self):
        """
        Tests a simple projection on toy values with a batch size of just one.
        """
        x_val = [[1, 2, 3, 4]]
        emb_space = [[2, 4, 6, 8],  # this is to be expected
                     [1, 1, 1, 1],
                     [4, 3, 2, 1],
                     [10, 10, 10, 10],
                     [0, 100, 0, 100]]
        expected = np.array([[[2, 4, 6, 8]]])

        y = self.feed(emb_space, x_val)
        self.assert_numerically_equal(y, expected)

    def test_projection2(self):
        """
        Tests a simple projection on toy values with a batch size of > 1.
        """
        x_val = [[1, 2, 3, 4],      # should be mapped to [2, 4, 6, 8]
                 [0, 1, 0, 0]]      # should be mapped to [5, 100, 5, 5]
        emb_space = [[2, 4, 6, 8],
                     [1, 1, 1, 1],
                     [4, 3, 2, 1],
                     [10, 10, 10, 10],
                     [5, 100, 5, 5]]
        expected = np.array([[[2, 4, 6, 8]], [[5, 100, 5, 5]]])

        y = self.feed(emb_space, x_val)
        self.assert_numerically_equal(y, expected)
