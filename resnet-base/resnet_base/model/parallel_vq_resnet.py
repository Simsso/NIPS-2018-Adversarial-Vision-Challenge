import tensorflow as tf
from resnet_base.model.resnet import ResNet
from vq_layer import vector_quantization as vq
from vq_layer import VQEndpoints


class ParallelVQResNet(ResNet):
    def __parallel_vqs(self, x: tf.Tensor, num_parallel: int, n: int, name: str) -> tf.Tensor:
        self._post_gradient_ops = []
        in_shape = x.shape.as_list()
        x = tf.reshape(x, [-1, in_shape[1]*in_shape[2], in_shape[3]])

        log = self.logger_factory
        log.add_histogram('vq_in_activations', x, log_frequency=5)

        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            parallel_xs = tf.split(x, num_parallel, axis=2)
            parallel_vq_out = []
            for i, depthwise_x in enumerate(parallel_xs):
                vq_endp = vq(depthwise_x, n, alpha=.5e-2, beta=0, gamma=4e-4, lookup_ord=1, return_endpoints=True,
                             embedding_initializer=tf.random_uniform_initializer(minval=-.2, maxval=1.5, seed=15092017),
                             is_training=self.is_training, num_embeds_replaced=1, name='{}/{}'.format(name, i))
                parallel_vq_out.append(vq_endp.layer_out)
                self.__add_logging(vq_endp, i, name)
                self._post_gradient_ops.append(vq_endp.replace_embeds)

            x = tf.concat(parallel_vq_out, axis=2)

        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        [log.add_scalar('{}/{}'.format(name, loss.op.name.split("/")[-1]), loss, log_frequency=10) for loss in losses]

        log.add_histogram('{}/out'.format(name), x, log_frequency=5)

        x = tf.reshape(x, [-1] + in_shape[1:])
        return x

    def __add_logging(self, vq_endpoints: VQEndpoints, i: int, name: str) -> None:
        log = self.logger_factory
        log.add_histogram('{}/{}/access_count'.format(name, i), vq_endpoints.access_count, is_sum_value=True,
                          log_frequency=16)
        log.add_histogram('{}/{}/embedding_space'.format(name, i), vq_endpoints.emb_space, log_frequency=16)
        log.add_histogram('{}/{}/embedding_spacing'.format(name, i), vq_endpoints.emb_spacing, log_frequency=16)
        log.add_histogram('{}/{}/embedding_closest_spacing'.format(name, i), vq_endpoints.emb_closest_spacing,
                          log_frequency=16)
        log.add_scalar('{}/{}/used_embed'.format(name, i), 1 - tf.nn.zero_fraction(vq_endpoints.access_count),
                       log_frequency=1)

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = self.__parallel_vqs(x, num_parallel=32, n=256, name='vq_post_conv1')
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)  # 8x8x256
        x = self.__parallel_vqs(x, num_parallel=64, n=512, name='vq_post_block1')
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)

    def post_gradient_application(self, sess: tf.Session) -> None:
        super().post_gradient_application(sess)
        sess.run(self._post_gradient_ops)
