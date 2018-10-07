import tensorflow as tf
from resnet_base.model.resnet import ResNet
from vq_layer import vector_quantization as vq
from vq_layer import VQEndpoints


class ParallelVQResNet(ResNet):
    def __parallel_vqs(self, x: tf.Tensor) -> tf.Tensor:
        self._post_gradient_ops = []

        x = tf.reshape(x, [-1, 256, 64])

        log = self.logger_factory
        log.add_histogram('vq_in_activations', x, log_frequency=5)

        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            parallel_xs = tf.split(x, 16, axis=2)
            parallel_vq_out = []
            for i, depthwise_x in enumerate(parallel_xs):
                vq_endp = vq(depthwise_x, n=128, alpha=1e-1, beta=0, gamma=5e-4, lookup_ord=1, return_endpoints=True,
                             embedding_initializer=tf.random_uniform_initializer(minval=-.2, maxval=1.5, seed=15092017),
                             is_training=self.is_training, num_embeds_replaced=2, name='vq_{}'.format(i))
                parallel_vq_out.append(vq_endp.layer_out)
                self.__add_logging(vq_endp, i)
                self._post_gradient_ops.append(vq_endp.replace_embeds)

            x = tf.concat(parallel_vq_out, axis=2)

        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        [log.add_scalar('{}'.format(loss.op.name), loss, log_frequency=10) for loss in losses]

        log.add_histogram('vq_out', x, log_frequency=5)

        x = tf.reshape(x, [-1, 16, 16, 64])
        return x

    def __add_logging(self, vq_endpoints: VQEndpoints, i: int) -> None:
        log = self.logger_factory
        log.add_histogram('vq_parallel_{}/access_count'.format(i), vq_endpoints.access_count, is_sum_value=True,
                          log_frequency=16)
        log.add_histogram('vq_parallel_{}/embedding_space'.format(i), vq_endpoints.emb_space, log_frequency=16)
        log.add_histogram('vq_parallel_{}/embedding_spacing'.format(i), vq_endpoints.emb_spacing, log_frequency=16)
        log.add_histogram('vq_parallel_{}/embedding_closest_spacing'.format(i), vq_endpoints.emb_closest_spacing,
                          log_frequency=16)
        log.add_scalar('vq_parallel_{}/used_embed'.format(i), 1 - tf.nn.zero_fraction(vq_endpoints.access_count),
                       log_frequency=1)

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = self.__parallel_vqs(x)
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)

    def post_gradient_application(self, sess: tf.Session) -> None:
        super().post_gradient_application(sess)
        sess.run(self._post_gradient_ops)
