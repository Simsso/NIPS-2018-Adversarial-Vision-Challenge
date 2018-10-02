from typing import List, Union

import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestEmbeddingReplace(TFTestCase):
    """
    Tests for the replacement features, which replaces the n least frequently used vectors in the embedding space with
    the most distant input vectors.
    """

    def setUp(self) -> None:
        super(TestEmbeddingReplace, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)
        self.emb_space_val = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)

    def feed(self, x_in: List, emb_target: Union[List, np.ndarray], lookup_ord: Union[int, str],
             num_embeds_replaced: int) -> None:
        x_val = np.array(x_in, dtype=np.float32)
        endpoints = vector_quantization(self.x_reshaped, len(self.emb_space_val), lookup_ord=lookup_ord,
                                        embedding_initializer=tf.constant_initializer(self.emb_space_val),
                                        num_embeds_replaced=num_embeds_replaced, return_endpoints=True)
        self.init_vars()

        # run the replacement op (if existent) and then check the embedding space
        if endpoints.replace_embeds is not None:
            self.sess.run(endpoints.replace_embeds, feed_dict={self.x: x_val})
        emb_val = self.sess.run(endpoints.emb_space, feed_dict={self.x: x_val})

        self.assert_output(emb_val, emb_target)

    def test_no_action_for_no_replacements(self) -> None:
        """
        Embedding space is preserved, if the number of replacements is set to 0.
        """
        emb_target = self.emb_space_val
        self.feed(x_in=[[1, 2], [4, 5]], emb_target=emb_target, lookup_ord=1, num_embeds_replaced=0)

    def test_single_replacement(self) -> None:
        """
        Test replacing a single vector in the embedding space.

        initial emb space: [0, 0], [1, 1], [2, 2]
        input x_in should be mapped to the 2nd and 3rd embeddings, and [5, 5] should be the most distant input,
        therefore replace the [0, 0] embedding
        """
        x_in = [[1.1, 0.9], [2.1, 1.8], [5, 5]]
        emb_target = [[5, 5], [1, 1], [2, 2]]

        self.feed(x_in=x_in, emb_target=emb_target, lookup_ord=1, num_embeds_replaced=1)

    def test_error_if_batch_too_small(self) -> None:
        """
        Test the case where the number of replacements is larger than the number of inputs.
        """
        x_in = [[1.1, 0.9], [2.1, 1.8], [5, 5]]
        emb_target = [[5, 5], [1, 1], [2, 2]]

        with self.assertRaises(ValueError):
            # num_embeds_replaced = 4 > 3 = batch_size ==> should raise error
            self.feed(x_in=x_in, emb_target=emb_target, lookup_ord=1, num_embeds_replaced=len(x_in)+1)

    def test_two_replacements(self) -> None:
        """
        Test the replacement feature with two replacements.

        initial emb space: [0, 0], [1, 1], [2, 2]
        usage count:       [    3,      1,     1]
        least used (2):    [     ,      x,     x]
        """

        x_in = [[0.1, -0.1], [0, 0.3], [1.1, 0.9], [-4, -4], [20, 20]]
        emb_target = [[0, 0], [20, 20], [-4, -4]]

        self.feed(x_in=x_in, emb_target=emb_target, lookup_ord=2, num_embeds_replaced=2)


if __name__ == '__main__':
    unittest.main()
