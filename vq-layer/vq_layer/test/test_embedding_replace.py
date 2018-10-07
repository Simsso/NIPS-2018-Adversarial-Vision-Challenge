from typing import List

import numpy as np
import tensorflow as tf
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization as vq


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
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    def init_vq(self, num_embeds_replaced: int) -> None:
        self.vq_endpoints = vq(self.x_reshaped, len(self.emb_space_val), lookup_ord=1,
                               num_embeds_replaced=num_embeds_replaced,
                               embedding_initializer=tf.constant_initializer(self.emb_space_val),
                               is_training=self.is_training, return_endpoints=True)
        self.assertIsNotNone(self.vq_endpoints.replace_embeds)
        self.init_vars()

    def feed(self, x_val: List, is_training_val: bool = True) -> None:
        self.sess.run(self.vq_endpoints.layer_out, feed_dict={self.x: x_val, self.is_training: is_training_val})

    def perform_replacement(self) -> None:
        self.sess.run(self.vq_endpoints.replace_embeds)

    def assert_emb_space(self, expected_val: List) -> None:
        emb_space_val = self.sess.run(self.vq_endpoints.emb_space)
        self.assert_numerically_equal(emb_space_val, expected_val)

    def test_single_batch_single_replacement_1(self) -> None:
        """
        Replacement of a single vector after feeding a single batch.
        """
        self.init_vq(num_embeds_replaced=1)
        self.feed([[.1, .1], [1.1, 1.1], [10, 10]])
        self.perform_replacement()
        self.assert_emb_space([[10, 10], [1, 1], [2, 2]])

    def test_single_batch_single_replacement_2(self) -> None:
        """
        Replacement of a single vector after feeding a single batch.
        """
        self.init_vq(num_embeds_replaced=1)
        self.feed([[100, 100], [1.1, 0.9], [0, 0], [0, 0], [.6, .6]])
        self.perform_replacement()
        self.assert_emb_space([[0, 0], [1, 1], [100, 100]])

    def test_multiple_batches_single_replacement(self) -> None:
        """
        Replacement of a single vector after feeding multiple batches.
        Usage count:
        [[0, 0], [1, 1], [2, 2]]
             4       2       3
        """
        self.init_vq(num_embeds_replaced=1)
        self.feed([[.1, .1], [1.1, 1.1], [10, 10]])
        self.feed([[.1, .1], [1.1, 1.1], [20, 20]])
        self.feed([[.1, .1], [2, 2], [-3, -3]])
        self.perform_replacement()
        self.assert_emb_space([[0, 0], [20, 20], [2, 2]])

    def test_multiple_batches_multiple_replacements(self) -> None:
        """
        Replacement of multiple vectors after feeding multiple batches.
        """
        self.init_vq(num_embeds_replaced=2)
        self.feed([[4, 4], [5, 5], [6, 6], [-10, -10]])
        self.feed([[4, 4], [5, 5], [-11, -11], [6, 6]])
        self.perform_replacement()
        self.assert_emb_space([[-10, -10], [-11, -11], [2, 2]])

    def test_is_training_consideration_1(self) -> None:
        """
        Test that no replacement is done when feeding data through the VQ layer with is_training set to False.
        """
        self.init_vq(num_embeds_replaced=1)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [5, 5]], is_training_val=True)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [10, 10]], is_training_val=False)
        self.perform_replacement()
        self.assert_emb_space([[0, 0], [1, 1], [5, 5]])

    def test_is_training_consideration_2(self) -> None:
        """
        Test that no replacement is done when feeding data through the VQ layer with is_training set to False.
        """
        self.init_vq(num_embeds_replaced=1)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [5, 5]], is_training_val=True)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [10, 10]], is_training_val=False)
        self.feed([[-5, 38], [.5, .5], [-3, .2], [6.4, 32], [-20, -20]], is_training_val=False)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [6, 6]], is_training_val=True)
        self.feed([[.1, .1], [.2, -.2], [1, 1], [1.1, 1.1], [4, 4]], is_training_val=True)
        self.perform_replacement()
        self.assert_emb_space([[0, 0], [1, 1], [6, 6]])

    def test_no_op_for_zero_replacements(self) -> None:
        """
        Test that the endpoint `replace_embeds` is `None` when setting `num_embeds_replaced=0`
        """
        endpoints = vq(self.x_reshaped, len(self.emb_space_val), num_embeds_replaced=0, return_endpoints=True)
        self.assertIsNone(endpoints.replace_embeds)
