from typing import List

import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestGradientMagnitude(TFTestCase):
    """
    Testing the behavior of the gradient / embedding space update depending on batch size, embedding space size, ...
    """

    def setUp(self) -> None:
        super(TestGradientMagnitude, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 2])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)

    def feed(self, x_in1: List, emb_space_val: List, alpha: float = 1., beta: float = 1., gamma: float = 1.) -> List:
        """
        :return: New value of the embedding space
        """
        x_val1 = np.array(x_in1, dtype=np.float32)

        emb_space_val = np.array(emb_space_val, dtype=np.float32)
        vq_endpoints = vector_quantization(self.x_reshaped, len(emb_space_val), lookup_ord=1, return_endpoints=True,
                                           embedding_initializer=tf.constant_initializer(emb_space_val),
                                           alpha=alpha, beta=beta, gamma=gamma, num_embeds_replaced=0)
        emb_space = vq_endpoints.emb_space
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
        loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
        train_op = optimizer.minimize(loss)

        self.init_vars()
        emb_space_val_updated, _ = self.sess.run([emb_space, train_op], feed_dict={self.x: x_val1})
        return emb_space_val_updated

    def test_batch_size_irrelevant(self) -> None:
        """
        Larger batches must not lead to larger updates of the trained variables (here the embedding space).
        """
        batch1 = [[30, 30], [0.3, 0.3]]
        batch2 = [[30, 30], [0.3, 0.3], [30, 30], [0.3, 0.3]]
        emb_space_initial = [[1, 2], [3, 4], [5, 6]]

        emb_space1 = self.feed(batch1, emb_space_initial)
        self.setUp()  # reset embedding space and optimizer
        emb_space2 = self.feed(batch2, emb_space_initial)

        self.assert_output(emb_space1, emb_space2)

    def test_unused_embedding_vectors_irrelevant(self) -> None:
        """
        Having multiple vectors in the embedding space which are unused must not affect the update of the other vectors,
        if the coulomb loss is disabled.
        """
        batch = [[1, 1], [2, 2], [3, 3]]
        emb_space_initial1 = [[.5, .5], [2.2, 2.2], [4, 4]]
        emb_space_initial2 = [[.5, .5], [2.2, 2.2], [4, 4], [100, 100], [200, 200]]

        emb_space1 = self.feed(batch, emb_space_initial1, gamma=0)
        self.setUp()
        emb_space2 = self.feed(batch, emb_space_initial2, gamma=0)

        self.assert_output(emb_space1, emb_space2[:len(emb_space1)])

    def test_coulomb_loss_agnostic_to_input(self) -> None:
        """
        When training with the coulomb loss only, the inputs must not have an effect.
        """
        batch1 = [[1, 1], [2, 2], [4, 4]]
        batch2 = [[2, 5], [1, 5], [6, 7], [2, 1]]
        emb_space_initial = [[2, 3], [1.3, 32.2], [5, 5], [7, 7]]

        emb_space1 = self.feed(batch1, emb_space_initial, alpha=0, beta=0, gamma=1)
        self.setUp()
        emb_space2 = self.feed(batch2, emb_space_initial, alpha=0, beta=0, gamma=1)

        self.assert_output(emb_space1, emb_space2)
