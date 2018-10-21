import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List, Tuple
from vq_layer.vq_layer import cosine_vector_quantization


@unittest.skip("This feature is no longer available. It has been replaced with a label-based majority threshold.")
class TestCosineIdentityMapping(TFTestCase):
    """
    Test the projection feature of the cosine-similarity based vq-layer, with the identity mapping below a certain
    threshold. Also tests the corresponding identity-mapped percentage.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray],
             identity_threshold: float) -> Tuple[np.ndarray, float]:
        x_val = np.array(x_val, dtype=np.float32)
        x_val = np.expand_dims(x_val, axis=1)
        return self.feed_wo_reshape(emb_space_val, x_val, identity_threshold)

    def feed_wo_reshape(self, emb_space_val: Union[List, np.ndarray], x_val: Union[List, np.ndarray],
                        identity_threshold: float) -> Tuple[np.ndarray, float]:
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        endpoints = cosine_vector_quantization(self.x, n=len(emb_space_val),
                                               embedding_initializer=emb_space_init,
                                               identity_mapping_threshold=identity_threshold, return_endpoints=True)

        self.init_vars()

        y, percentage_identity_mapped = self.sess.run([endpoints.layer_out, endpoints.percentage_identity_mapped])
        return y, percentage_identity_mapped

    def test_identity_mapping_all_identity(self):
        """
        Tests the identity mapping for a toy batch and random embedding space where the threshold is > 1, i.e.
        the complete batch should be mapped to its identity.
        """
        x_val = np.array([[[1, 1, 1, 1], [2, 3, 10, 0], [-1, 3, 0.4, 4.3]],
                          [[1, 1, 2, 3], [2, 3, 10, 0], [-4, 3, 0.2, 4.3]]])
        emb_space = np.random.randn(10, 4)

        # as the identity threshold is 1.1 > 1, all input vectors should be identity-mapped
        expected = x_val
        y, percentage_identity_mapped = self.feed_wo_reshape(emb_space, x_val, identity_threshold=1.1)
        self.assert_numerically_equal(y, expected)

        # the percentage should be one, as all inputs have been mapped to their identity
        self.assert_scalar_numerically_equal(percentage_identity_mapped, 1.0)

    def test_identity_mapping_mixed(self):
        """
        Tests the identity mapping for a toy batch and embedding space where the threshold should make some inputs be
        projected and others be identity-mapped instead.
        """
        x_val = [[1, 1, 1],  # high similarity to embedding space ([.5, .5, .5])
                 [-1, 1, 1],  # high similarity to embedding space ([-.9,  .8,  .8])
                 [1, -1, 1],  # low similarity => should be identity-mapped
                 [1, 1, -1]]  # low similarity => should be identity-mapped
        emb_space = [[.5, .5, .5],
                     [-.9, .8, .8]]
        expected = np.array([[[.5, .5, .5]],
                             [[-.9, .8, .8]],
                             [[1, -1, 1]],
                             [[1, 1, -1]]], dtype=np.float32)

        y, percentage_identity_mapped = self.feed(emb_space, x_val, identity_threshold=.5)
        self.assert_numerically_equal(y, expected)

        # the percentage should be 0.5, because 2 of 4 input vectors have been identity-mapped
        self.assert_scalar_numerically_equal(percentage_identity_mapped, 0.5)
