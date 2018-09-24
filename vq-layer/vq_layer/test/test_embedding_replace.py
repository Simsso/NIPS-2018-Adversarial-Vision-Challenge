import numpy as np
import tensorflow as tf
import unittest
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import vector_quantization


class TestEmbeddingReplace(TFTestCase):
    def setUp(self):
        super(TestEmbeddingReplace, self).setUp()

        self.x = tf.placeholder(tf.float32, shape=[None, 3])
        self.x_reshaped = tf.expand_dims(self.x, axis=1)
        self.emb_space_val = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

    def feed(self, x_in, emb_target, lookup_ord, num_embeds_replaced):
        x_val = np.array(x_in, dtype=np.float32)
        endpoints = vector_quantization(self.x_reshaped, len(self.emb_space_val), lookup_ord=lookup_ord,
                                        embedding_initializer=tf.constant_initializer(self.emb_space_val),
                                        num_embeds_replaced=num_embeds_replaced, return_endpoints=True)
        self.init_vars()

        # run the replacement op (if existent) and then check the embedding space
        if endpoints.replace_embeds is not None:
            emb_val = self.sess.run(endpoints.replace_embeds, feed_dict={self.x: x_val})
        else:
            emb_val = self.sess.run(endpoints.emb_space, feed_dict={self.x: x_val})

        sorted_target = np.sort(np.array(emb_target, dtype=np.float32), axis=0)
        sorted_actual = np.sort(emb_val, axis=0)

        print("expected: {}".format(sorted_target))
        print("actual: {}".format(sorted_actual))
        self.assert_output(sorted_actual, sorted_target)

    def testNoActionForNoReplacements(self):
        emb_target = self.emb_space_val
        self.feed(x_in=[[1, 2, 3], [4, 5, 6]], emb_target=emb_target, lookup_ord=1, num_embeds_replaced=0)

    def test1Replacement(self):
        # current emb space: [0, 0, 0], [1, 1, 1], [2, 2, 2]

        # should be mapped to the 2nd and 3rd embeddings, and [5, 5, 5] should be the furthest activation away,
        # therefore replace the [0, 0, 0] embedding
        x_in = [[1.1, 1.1, 0.9], [2.1, 2.1, 1.8], [5, 5, 5]]
        emb_target = [[5, 5, 5], [1, 1, 1], [2, 2, 2]]

        self.feed(x_in=x_in, emb_target=emb_target, lookup_ord=1, num_embeds_replaced=1)


if __name__ == '__main__':
    unittest.main()
