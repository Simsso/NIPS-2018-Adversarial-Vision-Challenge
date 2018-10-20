import numpy as np
import tensorflow as tf
import unittest

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
        endpoints = cosine_vector_quantization(self.x_reshaped, n=len(emb_space_val),
                                               embedding_initializer=emb_space_init, return_endpoints=True)

        self.init_vars()

        y = self.sess.run(endpoints.layer_out)
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

    @unittest.skip("This takes a long time and a lot of RAM, so it should only be evaluated manually.")
    def test_space_requirements(self):
        """
        Uses a large embedding space and input batch with the cosine vq-layer. Used to verify the space requirement.
        Expected space usage:
            O(N) where N = max(n * vec_size, batch_size * r * vec_size, batch_size * n * r)
            which means here N = max(2**28, 2**24, 2**32) = 2**32. For float32's, this means around 16 GiB.

        Evaluation:
            Maximum usage during empirical evaluation was around 12 GiB (only in the beginning) with an average of
            around 8.5 GiB, which is reasonable.
        """
        n = 2**18               # number of embedding vectors
        vec_size = 2**10
        r = 2**6                # depth (second dimension of x)
        batch_size = 2**8

        x_val = np.random.randn(batch_size, r, vec_size)
        emb_space = np.random.randn(n, vec_size)

        x = tf.placeholder(dtype=tf.float32, shape=x_val.shape)
        emb_space_init = tf.constant_initializer(emb_space, dtype=tf.float32)
        endpoints = cosine_vector_quantization(x, n, embedding_initializer=emb_space_init, num_splits=1,
                                               return_endpoints=True)
        self.init_vars()

        # the result is not relevant, only the space requirement (needs to be tracked manually)
        _ = self.sess.run(endpoints.layer_out, feed_dict={x: x_val})

    def test_noisy_onehot1(self):
        """
        Does the noisy onehot-test using a batch size of 2 and a vec size of 5.
        """
        self.__noisy_onehot_test(batch_size=2, vec_size=5)

    def test_noisy_onehot2(self):
        """
        Does the noisy onehot-test using a batch size of 10 and a vec size of 100.
        """
        self.__noisy_onehot_test(batch_size=10, vec_size=100)

    def test_noisy_onehot3(self):
        """
        Does the noisy onehot-test using a batch size of 100, a vec size of 1000 and a noise stddev of 3.
        The fact that this still works in 1000 dimensions shows that the noise (at this standard deviation) does
        not impact the projection, which might not be the case for other distance metrics.
        """
        self.__noisy_onehot_test(batch_size=100, vec_size=1000, noise_stddev=3)

    def __noisy_onehot_test(self, batch_size: int, vec_size: int, noise_stddev: float = 1.0):
        """
        Tests a projection that works as follows:
        - the input batch is the first 'batch_size' one-hot vectors of dimension 'vec_size'
        - the embedding space is a noisy version of a diagonal matrix of dimension 'vec_size x vec_size' with random
          ints on the diagonal, which are significantly larger than the noise
        - the expected projection is the first 'batch_size' vectors of the embedding space, as these have the largest
          cosine similarity with the input batch
        :param batch_size: The number of onehot-vectors used as the input batch (must be <= vec_size)
        :param vec_size: The dimensionality of the input vectors and embedding space vectors (must be >= batch_size)
        """
        assert batch_size <= vec_size

        # corresponds to a batch of the first batch_size one-hot vectors of dimension dim
        x_val = np.zeros([batch_size, vec_size])
        x_val[:batch_size, :batch_size] = np.eye(batch_size)

        noise = np.random.normal(loc=0, scale=noise_stddev, size=vec_size ** 2)  # vec_size**2 random noise values
        noise = np.reshape(noise, newshape=[vec_size, vec_size])

        emb_space = np.eye(vec_size)                                    # vec_size one-hot vectors
        emb_space *= np.random.randint(low=5, high=200, size=vec_size)  # ... rows multiplied by random ints
        emb_space += noise                                              # ... plus small noise

        # as the noise is minor and because we use cosine similarity, the first batch_size embedding vectors should be
        # chosen by the projection
        expected = np.expand_dims(emb_space[:batch_size], axis=1)

        y = self.feed(emb_space, x_val)
        self.assert_numerically_equal(y, expected)
