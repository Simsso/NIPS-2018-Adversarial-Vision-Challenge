import numpy as np
from resnet_base.model.resnet import ResNet
import tensorflow as tf


class PCAResNet(ResNet):
    def pca_layer(self, x: tf.Tensor, mat_val: np.ndarray, name: str = 'pca_layer') -> tf.Tensor:
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            mat_inv_val = np.linalg.pinv(mat_val)

            mat_ph = tf.placeholder(tf.float32, mat_val.shape)
            mat = tf.get_variable('{}/mat'.format(name), dtype=tf.float32, initializer=mat_ph)
            mat_inv_ph = tf.placeholder(tf.float32, mat_inv_val.shape)
            mat_inv = tf.get_variable('{}/mat_inv'.format(name), dtype=tf.float32, initializer=mat_inv_ph)

            self.init_feed_dict[mat_ph] = mat_val
            self.init_feed_dict[mat_inv_ph] = mat_inv_val

            code = tf.matmul(x, mat)
            output = tf.matmul(code, mat_inv)

            return output

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = tf.reshape(x, [-1, 64])
        x = self.pca_layer(x, np.random.rand(64, 62))
        x = tf.reshape(x, [-1, 16, 16, 64])
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)
