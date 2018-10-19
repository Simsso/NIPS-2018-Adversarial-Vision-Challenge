import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List
from vq_layer.vq_layer import cosine_vector_quantization


class TestCosineIdentityMapping(TFTestCase):
    """
    Test the projection feature of the cosine-similarity based vq-layer, with the identity mapping below a certain
    threshold.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray],
             identity_threshold: float) -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        endpoints = cosine_vector_quantization(self.x_reshaped, n=len(emb_space_val),
                                               embedding_initializer=emb_space_init,
                                               abs_identity_mapping_threshold=identity_threshold, return_endpoints=True)

        self.init_vars()

        y = self.sess.run(endpoints.layer_out)
        return y

    def test_identity_mapping1(self):
        """
        Tests the identity mapping for a toy batch and random embedding space where the threshold is infinity, i.e.
        the complete batch should be mapped to its identity.
        """
        x_val = np.array([[1, 1, 1, 1], [2, 3, 10, 0], [-1, 3, 0.4, 4.3]])
        emb_space = np.random.randn(10, 4)

        # as the identity threshold is np.inf, all input vectors should be identity-mapped
        expected = np.expand_dims(x_val, axis=1)
        y = self.feed(emb_space, x_val, identity_threshold=np.inf)
        self.assert_numerically_equal(y, expected)