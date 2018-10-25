import os
from typing import List, Tuple
import numpy as np
from resnet_base.model.resnet import ResNet
import scipy.io
import tensorflow as tf
from vq_layer import cosine_knn_vector_quantization as cos_knn_vq
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline

tf.flags.DEFINE_string("lesci_emb_space_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'data_lesci_emb_space_small.mat')),
                       "Path to the file (*.mat) where embedding space values ('act_compressed') and labels ('labels') "
                       "are being stored.")
tf.flags.DEFINE_string("pca_compression_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'pca.mat')),
                       "Path to the file (*.mat) where the PCA compression matrix ('pca_out') is stored.")
FLAGS = tf.flags.FLAGS


class LESCIResNet(ResNet):

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)  # 2x2x1024

        identity_mask, knn_label, percentage_identity_mapped = self._lesci_layer(x, shape=[74246, 64])
        self.percentage_identity_mapped = percentage_identity_mapped

        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        resnet_out = self.global_avg_pooling(x)
        knn_label_one_hot = tf.one_hot(knn_label, depth=TinyImageNetPipeline.num_classes)

        self.__log_projection_identity_accuracy(identity_mask, resnet_out, knn_label)
        return tf.where(identity_mask, x=resnet_out, y=knn_label_one_hot)

    def _lesci_layer(self, x: tf.Tensor, shape: List[int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        assert len(shape) == 2
        num_samples = shape[0]
        code_size = shape[1]

        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            x = tf.reshape(x, [-1, 2 * 2 * 1024])
            pca_mat = tf.get_variable('pca_mat', dtype=tf.float32,
                                      initializer=self._make_init(FLAGS.pca_compression_file,
                                                                  shape=[2 * 2 * 1024, code_size],
                                                                  dtype=tf.float32, mat_name='pca_out'),
                                      trainable=False)
            x = tf.matmul(x, pca_mat)
            x = tf.expand_dims(x, axis=1)
            label_variable = tf.get_variable('lesci_labels', dtype=tf.int32,
                                             initializer=self._make_init(FLAGS.lesci_emb_space_file, [num_samples],
                                                                         tf.int32, mat_name='labels'), trainable=False)
            embedding_init = self._make_init(FLAGS.lesci_emb_space_file, shape, tf.float32, mat_name='act_compressed')
            vq = cos_knn_vq(x, emb_labels=label_variable, num_classes=TinyImageNetPipeline.num_classes, k=10,
                            n=num_samples, embedding_initializer=embedding_init, constant_init=True,
                            num_splits=1, return_endpoints=True, majority_threshold=.5, name='cos_knn_vq')
            identity_mask = vq.identity_mapping_mask
            label = vq.most_common_label
            percentage_identity_mapped = vq.percentage_identity_mapped

            return tf.reshape(identity_mask, [-1]), tf.reshape(label, [-1]), percentage_identity_mapped

    def _make_init(self, mat_file_path: str, shape: List[int], dtype=tf.float32, mat_name: str = 'emb_space'):
        """
        Creates an initializer from a matrix file. If the file cannot be found the method returns a placeholder. This
        case occurs if the weights are being loaded from a checkpoint instead.
        :param mat_file_path: File to load the data from
        :param shape: Shape of the variable
        :param mat_name: Name of the entry in the dictionary data matrix with shape 'shape'
        :return: Initializer for the variable
        """
        try:
            tf.logging.info("Tying to load variable values space from '{}'.".format(mat_file_path))
            var_val = scipy.io.loadmat(mat_file_path)[mat_name]
            tf.logging.info("Loaded variable with shape {}".format(var_val.shape))
            var_val = np.reshape(var_val, shape)
            var_placeholder = tf.placeholder(dtype, var_val.shape)
            self.init_feed_dict[var_placeholder] = var_val
            return var_placeholder
        except FileNotFoundError:
            tf.logging.info("Could not load variable values; model should be initialized from a checkpoint")
            return tf.placeholder(dtype, shape)

    def __log_projection_identity_accuracy(self, identity_mask: tf.Tensor, resnet_out: tf.Tensor,
                                           projection_labels: tf.Tensor):
        """
        Calculates the classification accuracy for both the identity-mapped inputs and the projected inputs and logs
        them.
        """
        labels = tf.cast(self.labels, tf.int64)
        identity_softmax = tf.nn.softmax(resnet_out)

        # adding .001 so we don't divide by zero
        num_identity_mapped = tf.reduce_sum(tf.cast(identity_mask, dtype=tf.float32)) + .001
        num_projected = tf.reduce_sum(tf.cast(tf.logical_not(identity_mask), dtype=tf.float32)) + .001

        correct_identity = tf.equal(tf.argmax(identity_softmax, axis=1), labels)
        # only include the correctly classified inputs that are identity-mapped
        correct_identity = tf.logical_and(correct_identity, identity_mask)
        accuracy_identity = tf.reduce_sum(tf.cast(correct_identity, dtype=tf.float32)) / num_identity_mapped
        self.accuracy_identity = accuracy_identity
        self.logger_factory.add_scalar('accuracy_identity_mapping', accuracy_identity, log_frequency=10)

        correct_projection = tf.equal(tf.cast(projection_labels, tf.int64), labels)
        # only include the correctly classified inputs that are *not* identity-mapped, i.e. projected
        correct_projection = tf.logical_and(correct_projection, tf.logical_not(identity_mask))
        accuracy_projection = tf.reduce_sum(tf.cast(correct_projection, dtype=tf.float32)) / num_projected
        self.accuracy_projection = accuracy_projection
        self.logger_factory.add_scalar('accuracy_projection', accuracy_projection, log_frequency=10)






