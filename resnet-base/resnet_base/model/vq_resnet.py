import numpy as np
import tensorflow as tf
from resnet_base.model.resnet import ResNet
from vq_layer import vector_quantization as vq


class VQResNet(ResNet):
    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            log = self.logger_factory
            log.add_histogram('vq_in_activations', x, log_frequency=5)

            x = tf.reshape(x, [-1, 256, 64])
            vq_endpoints = vq(x, n=4096, alpha=5e-3, beta=4e-6, gamma=0, num_splits=16, lookup_ord=1,
                              num_embeds_replaced=0, return_endpoints=True,
                              embedding_initializer=tf.random_normal_initializer(mean=.5, stddev=.5, seed=15092017))

            def x_with_update():
                if vq_endpoints.replace_embeds is None:
                    return vq_endpoints.layer_out
                with tf.control_dependencies([vq_endpoints.replace_embeds]):
                    return vq_endpoints.layer_out

            x = tf.cond(self.is_training, true_fn=x_with_update, false_fn=lambda: vq_endpoints.layer_out)
            access_count = vq_endpoints.access_count
            x = tf.reshape(x, [-1, 16, 16, 64])

            log.add_histogram('vq_out', x, log_frequency=5)
            log.add_histogram('access_count', access_count, is_sum_value=True, log_frequency=50)
            log.add_histogram('embedding_space', vq_endpoints.emb_space, log_frequency=5)
            log.add_histogram('embedding_spacing', vq_endpoints.emb_spacing, log_frequency=5)
            log.add_scalar('unused_embeddings', tf.nn.zero_fraction(access_count), log_frequency=10)
            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            [log.add_scalar(loss.name, loss, log_frequency=10) for loss in losses]

        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)
