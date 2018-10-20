import os
from typing import List
import numpy as np
from resnet_base.model.resnet import ResNet
import scipy.io
import tensorflow as tf
from vq_layer import cosine_vector_quantization as cvq

tf.flags.DEFINE_string("lesci_emb_space_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'data_lesci_emb_space_small.mat')),
                       "Path to the file (*.mat) where embedding space values are being stored.")
FLAGS = tf.flags.FLAGS


class LESCIResNet(ResNet):

    def _lesci_layer(self, x: tf.Tensor, shape: List[int], mat_name: str) -> tf.Tensor:
        assert len(shape) == 2
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            return cvq(x, shape[0], constant_init=True, num_splits=1, return_endpoints=False,
                       embedding_initializer=self._make_init(FLAGS.lesci_emb_space_file, shape=shape,
                                                             mat_name=mat_name))

    def _make_init(self, mat_file_path: str, shape: List[int], mat_name: str = 'emb_space'):
        """
        Creates an embedding space initializer. If the file cannot be found the method returns a placeholder. This case
        occurs when the weights will be loaded from a checkpoint instead.
        :param mat_file_path: File to load the data from
        :param shape: Shape of the embedding space
        :param mat_name: Name of the entry in the dictionary from 'mat_file_path' that contains the data matrix with
                         shape 'shape'
        :return: Initializer for the embedding space
        """
        try:
            tf.logging.info("Tying to load embedding space from '{}'.".format(mat_file_path))
            emb_space_val = scipy.io.loadmat(mat_file_path)[mat_name]
            tf.logging.info("Loaded embedding space with shape {}".format(emb_space_val.shape))
            emb_space_val = np.reshape(emb_space_val, shape)
            emb_space_placeholder = tf.placeholder(tf.float32, emb_space_val.shape)
            self.init_feed_dict[emb_space_placeholder] = emb_space_val
            return emb_space_placeholder
        except FileNotFoundError:
            tf.logging.info("Could not load embedding space; model should be initialized from a checkpoint")
            return tf.placeholder(tf.float32, shape)

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)  # 2x2x1024
        x = tf.reshape(x, [-1, 1, 2 * 2 * 1024])
        x = self._lesci_layer(x, shape=[32768, 2 * 2 * 1024], mat_name='act5_block3_small')
        x = tf.reshape(x, [-1, 2, 2, 1024])
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)
