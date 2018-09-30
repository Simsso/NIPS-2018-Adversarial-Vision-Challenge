import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestInputValidation(TFTestCase):
    def test_x_is_tensor(self):
        """Input x must be a tensor."""
        with self.assertRaises(AttributeError):
            vector_quantization([1, 2], 2)

    def test_x_shape(self):
        """Input x must be 3-dimensional."""
        # invalid inputs
        for x in [
            tf.placeholder(tf.float32, [None, 5]),
            tf.placeholder(tf.float32, [None, 5, 4, 8]),
            tf.placeholder(tf.float32, [1, 5, 4, 8]),
            tf.placeholder(tf.float32, ())
        ]:
            with self.assertRaises(ValueError):
                vector_quantization(x, 2)

        # valid input
        vector_quantization(tf.placeholder(tf.float32, [None, 1, 5]), 5)

    def test_n_is_positive(self):
        """Input n (number of vectors in the embedding space) must be greater than 0."""
        for n in [-10000, -5, -1, 0]:
            with self.assertRaises(ValueError):
                vector_quantization(tf.placeholder(tf.float32, [None, 1, 3]), n)

    def test_lookup_ord_is_valid(self):
        """Input lookup_ord must be an input that is valid for tf.norm's order parameter."""
        # valid inputs
        n = 5
        for lookup_ord in [1, 2, np.inf]:
            self.setUp()
            vector_quantization(tf.placeholder(tf.float32, [None, 1, 3]), n, lookup_ord=lookup_ord)

        # invalid inputs
        for lookup_ord in ['fro', 'euclidean', -1, 0, 5]:
            with self.assertRaises(ValueError):
                vector_quantization(tf.placeholder(tf.float32, [None, 1, 3]), n, lookup_ord=lookup_ord)

    def test_num_splits_gt_zero(self):
        """Input num_splits must be greater than 0."""
        # invalid inputs
        for num_splits in [-5, -1, 0]:
            with self.assertRaises(ValueError):
                vector_quantization(tf.placeholder(tf.float32, [None, 1, 3]), 4, num_splits=num_splits)

        # valid input (1 is always valid)
        vector_quantization(tf.placeholder(tf.float32, [None, 1, 3]), 4, num_splits=1)

    def test_num_splits_is_divisor(self):
        # invalid input
        with self.assertRaises(ValueError):
            vector_quantization(tf.placeholder(tf.float32, [None, 5, 200]), 4, num_splits=21)

        # valid input
        vector_quantization(tf.placeholder(tf.float32, [None, 5, 200]), 4, num_splits=20)


if __name__ == '__main__':
    unittest.main()
