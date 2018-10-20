import numpy as np
import tensorflow as tf
import unittest

from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List
from vq_layer.vq_layer import cosine_knn_vector_quantization as cos_knn_vq


class TestCosineKNNProjection(TFTestCase):
    """
    Test the k-NN-based projection of the cosine-similarity vq-layer.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], emb_labels: Union[List, np.ndarray], k: int,
             num_classes: int, x_val: Union[List, np.ndarray]) -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)
        self.x = tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1) if len(x_val.shape) == 2 else self.x

        n = len(emb_labels)
        emb_labels = np.array(emb_labels, dtype=np.int32)
        self.emb_labels = tf.placeholder_with_default(emb_labels, shape=emb_labels.shape)

        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        endpoints = cos_knn_vq(self.x_reshaped, self.emb_labels, num_classes=num_classes,
                               k=k, n=n, embedding_initializer=emb_space_init, return_endpoints=True)

        self.init_vars()

        y = self.sess.run(endpoints.layer_out)
        return y

    def test_projection1(self):
        """
        Tests a simple projection on toy values with a batch size of two.
        """
        x_val = [[1, 2, 3, 4],
                 [0, 1, 0, 0]]

        emb_space = [[2, 4,   6, 8],    # closest to first input (dot product of 1)
                     [1, 2.2, 3, 4],    # second-closest to first input
                     [1, 3,   3, 4],    # third-closest to first input
                     [0, 1,   0, .5],   # third-closest to second input
                     [0, 1,   0, .1],   # second-closest to second input
                     [0, 2,   0, 0]]    # closest to second input

        emb_labels = [0, 1, 1, 2, 2, 3]
        num_classes = 4
        k = 3

        # The first input should be projected based on a label majority vote in [0, 1, 1] => label 1 => *not* the vector
        # [2, 4, 6, 8] which has the highest dot product.
        # The second input should be projected based on a label majority vote in [2, 2, 3] => label 2 => *not* the
        # vector [0, 2, 0, 0] which has the highest dot product
        expected = [[[1, 2.2, 3, 4]], [[0, 1,   0, .5]]]

        y = self.feed(emb_space_val=emb_space, emb_labels=emb_labels, k=k, num_classes=num_classes, x_val=x_val)
        self.assert_numerically_equal(y, expected)

    def test_projection2(self):
        """
        Tests a projection on toy values.
        Uses the same values as the test_projection1 test case, but here, the batch size is 1 and instead the second
        dimension of x is 2.
        """
        x_val = [[[1, 2, 3, 4],
                 [0, 1, 0, 0]]]  # 3-D!

        # otherwise exactly the same as the values in test_projection1
        emb_space = [[2, 4,   6, 8],    # closest to first input (dot product of 1)
                     [1, 2.2, 3, 4],    # second-closest to first input
                     [1, 3,   3, 4],    # third-closest to first input
                     [0, 1,   0, .5],   # third-closest to second input
                     [0, 1,   0, .1],   # second-closest to second input
                     [0, 2,   0, 0]]    # closest to second input

        emb_labels = [0, 1, 1, 2, 2, 3]
        num_classes = 4
        k = 3

        expected = [[[1, 2.2, 3, 4], [0, 1, 0, .5]]]    # notice shape [1, 1, 2] instead of [1, 2, 1]

        y = self.feed(emb_space_val=emb_space, emb_labels=emb_labels, k=k, num_classes=num_classes, x_val=x_val)
        self.assert_numerically_equal(y, expected)