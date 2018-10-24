import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from typing import Union, List
from vq_layer.vq_layer import cosine_knn_vector_quantization as cos_knn_vq


class TestCosineKNNProjection(TFTestCase):
    """
    Test the k-NN-based projection of the cosine-similarity vq-layer.
    """

    def feed(self, emb_space_val: Union[List, np.ndarray], emb_labels: Union[List, np.ndarray], k: int,
             num_classes: int, x_val: Union[List, np.ndarray], majority_threshold: float = -1) -> np.ndarray:
        x_val = np.array(x_val, dtype=np.float32)

        x_shape = x_val.shape
        placeholder_shape = [None, 1, x_shape[1]] if len(x_shape) == 2 else [None, x_shape[1], x_shape[2]]
        self.x = tf.placeholder(tf.float32, shape=placeholder_shape) #tf.placeholder_with_default(x_val, shape=x_val.shape)
        self.x_reshaped = tf.expand_dims(self.x, axis=1) if len(x_val.shape) == 2 else self.x

        n = len(emb_labels)
        emb_labels = np.array(emb_labels, dtype=np.int32)
        self.emb_labels = tf.placeholder_with_default(emb_labels, shape=emb_labels.shape)

        emb_space_init = tf.constant_initializer(np.array(emb_space_val), dtype=tf.float32)
        endpoints = cos_knn_vq(self.x_reshaped, self.emb_labels, num_classes=num_classes,
                               k=k, n=n, embedding_initializer=emb_space_init, majority_threshold=majority_threshold,
                               return_endpoints=True)

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

    def test_projection_with_threshold1(self):
        """
        Tests the same projection as the test_projection1 function, but in this case uses the majority_threshold
        argument.
        """
        x_val = [[1, 2, 3, 4],
                 [0, 1, 0, 0]]

        emb_space = [[2, 4, 6, 8],  # closest to first input (dot product of 1)
                     [1, 2.2, 3, 4],  # second-closest to first input
                     [1, 3, 3, 4],  # third-closest to first input
                     [0, 1, 0, .5],  # third-closest to second input
                     [0, 1, 0, .1],  # second-closest to second input
                     [0, 2, 0, 0]]  # closest to second input

        emb_labels = [0, 1, 1, 2, 2, 2]  # second input has unambiguous vote for class 2
        majority_threshold = .7
        num_classes = 4
        k = 3

        # When majority_threshold < 0, then as in test_projection1: expected = [[[1, 2.2, 3, 4]], [[0, 1, 0, .5]]]
        # but now when the threshold is 0.7, the majority of labels, which is 2/3 and 1 in the inputs respectively
        # only the second input should be projected, but the first should be identity-mapped. Therefore:
        expected = [[[1, 2, 3, 4]], [[0, 1, 0, .5]]]

        y = self.feed(emb_space_val=emb_space, emb_labels=emb_labels, k=k, num_classes=num_classes, x_val=x_val,
                      majority_threshold=majority_threshold)
        self.assert_numerically_equal(y, expected)

    def test_projection_with_threshold2(self):
        """
        Tests another toy projection using the majority_threshold.
        """
        x_val = [[[0, 1, 0], [1, 2, 3]]]

        emb_space = [[1, 2.2, 3],   # the first three are closest to the second input
                     [1.2, 2, 3],
                     [1.1, 2, 3.1],

                     [1, 5, 0.5],   # the second three are closest to the first input
                     [0.1, .7, .1],
                     [2, 10, 2]]

        # for the first input, the fraction of the majority-class is 2/3 => should be projected to [1, 5, 0.5]
        # for the second input, the fraction of the majority-class is only 1/3 => should be identity-mapped
        emb_labels = [0, 1, 2, 3, 3, 1]
        majority_threshold = .5
        num_classes = 4
        k = 3

        expected = [[[1, 5, 0.5], [1, 2, 3]]]

        y = self.feed(emb_space_val=emb_space, emb_labels=emb_labels, k=k, num_classes=num_classes, x_val=x_val,
                      majority_threshold=majority_threshold)
        self.assert_numerically_equal(y, expected)

    def test_dynamic_batch_size(self):
        """
        Tests with variable batch size that is unknown at graph construction time.
        """
        x_val = np.array([[[0, 1, 0]], [[1, 2, 3]], [[1, 2, 3]]], dtype=np.float32)
        emb_space_val = np.array([[1, 2.2, 3], [1.2, 2, 3], [1.1, 2, 3.1], [1, 5, 0.5], [0.1, .7, .1], [2, 10, 2]])
        emb_labels_val = np.array([0, 1, 2, 3, 3, 1], dtype=np.int32)
        self.x = tf.placeholder(tf.float32, shape=[None, 1, 3])
        self.emb_labels = tf.placeholder_with_default(emb_labels_val, shape=emb_labels_val.shape)

        emb_space_init = tf.constant_initializer(emb_space_val, dtype=tf.float32)
        endpoints = cos_knn_vq(self.x, self.emb_labels, num_classes=4, k=3, n=len(emb_labels_val), return_endpoints=True,
                               embedding_initializer=emb_space_init, majority_threshold=.5)

        self.init_vars()

        y = self.sess.run(endpoints.layer_out, feed_dict={self.x: x_val})

        expected = [[[1, 5, 0.5]], [[1, 2, 3]], [[1, 2, 3]]]
        self.assert_numerically_equal(y, expected)
