import tensorflow as tf

from resnet_base.model.resnet import ResNet
from vq_layer import vector_quantization as vq


class VQResNet(ResNet):
    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            with tf.device("/cpu:0"):
                x = tf.reshape(x, [-1, 256, 64])
                x = vq(x, n=256, alpha=1e-1, beta=1e-2, gamma=0, num_splits=8)
                x = tf.reshape(x, [-1, 16, 16, 64])
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)