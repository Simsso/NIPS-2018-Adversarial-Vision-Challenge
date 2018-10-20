import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List, Tuple
from vq_layer.vq_layer import cosine_vector_quantization


class TestCosineSimilarityHistogram(TFTestCase):
    """
    Tests the similarity values returned by the cosine-similarity based vq-function.
    """
    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray]) -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        endpoints = cosine_vector_quantization(self.x_reshaped, n=len(emb_space_val),
                                               embedding_initializer=emb_space_init, return_endpoints=True)

        self.init_vars()

        similarity_values = self.sess.run(endpoints.similarity_values)
        return similarity_values

    def test_similarity_values(self):
        """
        Tests the similarity values for a toy batch and embedding space.
        """
        x_val = np.eye(3)               # a batch of 3 one-hot vectors
        emb_space = [[1, .5, .5],       # highest similarity with the first input vector [1, 0, 0]
                     [1,  1,  3],       # highest similarity with the third input vector [0, 0, 1]
                     [1,  5,  1]]       # highest similarity with the second input vector [0, 1, 0]

        expected = [1 / np.sqrt(1.5), 5 / np.sqrt(27), 3 / np.sqrt(11)]

        similarity_values = self.feed(emb_space, x_val)
        self.assert_numerically_equal(similarity_values, expected)
