import tensorflow as tf
from resnet_base.model.resnet import ResNet
from vq_layer import vector_quantization as vq


class VQResNet(ResNet):
    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            x = tf.reshape(x, [-1, 256, 64])
            vq_endpoints = vq(x, n=128, alpha=.1, beta=.1, gamma=.1, num_splits=16, lookup_ord=1, return_endpoints=True,
                              embedding_initializer=tf.random_normal_initializer(0, stddev=1 / 32., seed=15092017),
                              num_embeds_replaced=1)
            x = vq_endpoints.layer_out
            access_count = vq_endpoints.access_count
            self.logger_factory.add_histogram('access_count', access_count, log_frequency=50)
            self.logger_factory.add_scalar('unused_embeddings', tf.nn.zero_fraction(access_count), log_frequency=5)
            with tf.control_dependencies([vq_endpoints.replace_embeds]):
                x = tf.reshape(x, [-1, 16, 16, 64])
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)
