import numpy as np
import os
from resnet_base.model.resnet import ResNet
import scipy.io
import tensorflow as tf

tf.flags.DEFINE_string("pca_mat_file", os.path.expanduser(os.path.join('~', '.data', 'activations', 'pca.mat')),
                       "Path to the file (*.mat) where the PCA matrices are being stored.")
FLAGS = tf.flags.FLAGS


class PCAResNet(ResNet):
    """
    Modification of the ResNet where activations of certain layers/blocks are being compressed with a PCA matrix and
    subsequently scaled back the the size required by the next layer. The compression is typically lossy and does
    therefore affect the model's accuracy and robustness.
    """

    def _pca_layer(self, x: tf.Tensor, mat_val: np.ndarray, name: str = 'pca_layer') -> tf.Tensor:
        """
        PCA layer: Multiplies the input with the given transformation matrix and subsequently the Moore-Penrose inverse.
        :param x: Layer input
        :param mat_val: Matrix to multiply with
        :param name: Name of the layer
        :return: Layer output; y = x * mat_val * (mat_val)^-1 (where '*' is matrix multiplication)
        """
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            mat_inv_val = np.linalg.pinv(mat_val)

            mat_ph = tf.placeholder(tf.float32, mat_val.shape)
            mat = tf.get_variable('{}/mat'.format(name), dtype=tf.float32, initializer=mat_ph)
            mat_inv_ph = tf.placeholder(tf.float32, mat_inv_val.shape)
            mat_inv = tf.get_variable('{}/mat_inv'.format(name), dtype=tf.float32, initializer=mat_inv_ph)

            self.init_feed_dict[mat_ph] = mat_val
            self.init_feed_dict[mat_inv_ph] = mat_inv_val

            code = tf.matmul(x, mat)
            tf.logging.info("Compression matrix shape: {}".format(mat.shape))
            tf.logging.info("Code shape: {}".format(code.shape))
            output = tf.matmul(code, mat_inv)

            return output

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        self._imported_dict = None
        x = ResNet._first_conv(x)  # 16x16x64
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)  # 4x4x512
        x = tf.reshape(x, [-1, 4*4*512])
        x = self._pca_layer(x, self._get_matrix('pca_out'))
        x = tf.reshape(x, [-1, 4, 4, 512])
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)  # 2x2x1024
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)

    def _load_matrices_file(self) -> None:
        """
        Loads the .mat-file containing matrices and stores it as an attribute. The path is stored in 'FLAGS.pca_mat_file'.
        """
        file = FLAGS.pca_mat_file
        self._imported_dict = scipy.io.loadmat(file)

    def _get_matrix(self, name: str = 'pca_out') -> np.ndarray:
        """
        Returns the matrix with the given name, loaded from the .mat-file
        :param name: Name of the matrix
        :return: The matrix itself
        """
        if self._imported_dict is None:
            self._load_matrices_file()
        return self._imported_dict[name]