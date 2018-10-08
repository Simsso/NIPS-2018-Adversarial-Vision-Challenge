import tensorflow as tf
from resnet_base.model.resnet import ResNet


class VQVAEResNet(ResNet):
    def __ae_layer(self, activations: tf.Tensor) -> tf.Tensor:
        """
        Auto-encoder layer along the channel dimension of x
        :param activations: Tensor of size batchx16x16x64
        """
        x = activations

        # encoder
        x = tf.layers.conv2d(x, filters=48, kernel_size=[1, 1])
        code = tf.nn.relu(x)

        # decoder
        x = code
        x = tf.layers.conv2d(x, filters=64, kernel_size=[1, 1])

        output = x
        assert output.shape.as_list() == activations.shape.as_list()
        mse_reconstruction_loss = tf.reduce_mean(tf.square(output - activations))
        tf.add_to_collection(tf.GraphKeys.LOSSES, 1e-3*mse_reconstruction_loss)

        self.logger_factory.add_scalar('ae/mse', mse_reconstruction_loss, 8)
        self.logger_factory.add_histogram('ae/input', activations, 16)
        self.logger_factory.add_histogram('ae/code', code, 16)
        self.logger_factory.add_histogram('ae/output', output, 16)

        return x

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            x = self.__ae_layer(x)
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)
