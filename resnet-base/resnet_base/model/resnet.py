import os
from resnet_base.model.base_model import BaseModel
import tensorflow as tf
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline as Data

slim = tf.contrib.slim

# define flags
tf.flags.DEFINE_string("pretrained_checkpoint", os.path.expanduser('~/.models/tiny_imagenet_alp05_2018_06_26.ckpt'),
                       "Checkpoint path of pre-trained weights.")
tf.flags.DEFINE_string("custom_checkpoint", "", "Checkpoint path of custom-tuned weights.")

FLAGS = tf.flags.FLAGS


class ResNet(BaseModel):

    def __init__(self, x: tf.Tensor = None, labels: tf.Tensor = None):
        super().__init__()

        self.accuracy: tf.Tensor = None
        self.loss: tf.Tensor = None
        self.logits: tf.Tensor = None
        self.softmax: tf.Tensor = None
        self.x: tf.Tensor = tf.placeholder(tf.float32, shape=[None, Data.img_width, Data.img_height, Data.img_channels],
                                           name='x') if x is None else x
        self.is_training: tf.Tensor = tf.placeholder_with_default(False, (), 'is_training')
        self.num_classes: int = 200
        self.labels = tf.placeholder(tf.uint8, shape=[None], name='labels') if labels is None else labels

        self.pretrained_saver: tf.train.Saver = None
        self.custom_saver: tf.train.Saver = None
        with tf.variable_scope('custom') as scope:
            self.custom_scope: tf.VariableScope = scope

        self.build_model()
        self.init_saver()
        self.init_loss()
        self.init_accuracy()

    def init_saver(self) -> None:
        # global saver (complete graph)
        self.saver = BaseModel._create_saver('')
        self.pretrained_saver = BaseModel._create_saver('resnet_v2_50')
        self.custom_saver = BaseModel._create_saver(self.custom_scope.name)

    def save(self, sess: tf.Session):
        super().load(sess)
        BaseModel._save_to_path(sess, self.pretrained_saver, self.global_step, FLAGS.pretrained_checkpoint)
        BaseModel._save_to_path(sess, self.custom_saver, self.global_step, FLAGS.custom_checkpoint)

    def load(self, sess: tf.Session):
        super().load(sess)
        BaseModel._restore_checkpoint(self.pretrained_saver, sess, FLAGS.pretrained_checkpoint)
        BaseModel._restore_checkpoint(self.custom_saver, sess, FLAGS.custom_checkpoint)

    def build_model(self) -> None:
        with tf.contrib.framework.arg_scope(ResNet.resnet_arg_scope()):
            with tf.variable_scope('resnet_v2_50', 'resnet_v2', [self.x], reuse=tf.AUTO_REUSE) as sc:
                with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                    x = self.x
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        x = ResNet.conv2d_same(x, 64, 7, stride=2, scope='conv1')
                    x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')
                    x = ResNet.v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
                    x = ResNet.v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
                    x = ResNet.v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
                    x = ResNet.v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)

                    x = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')

                    # global average pooling
                    x = tf.reduce_mean(x, [1, 2], name='pool5', keepdims=True)
                    x = slim.conv2d(x, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')
                    self.logits = x
                    self.softmax = slim.softmax(x, scope='predictions')

    def init_loss(self) -> None:
        labels_one_hot = tf.one_hot(self.labels, depth=Data.num_classes)
        self.loss = tf.losses.softmax_cross_entropy(labels_one_hot, self.logits)

    def init_accuracy(self) -> None:
        correct = tf.cast(tf.equal(tf.argmax(self.softmax, axis=1), tf.cast(self.labels, tf.int64)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(correct, name='accuracy')

    @staticmethod
    def conv2d_same(inputs: tf.Tensor, num_outputs: int, kernel_size: int, stride: int, rate: int = 1,
                    scope: str = None) -> tf.Tensor:
        if stride == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                               rate=rate, padding='VALID', scope=scope)

    @staticmethod
    def v2_block(x: tf.Tensor, scope: str, base_depth: int, num_units: int, stride: int) -> tf.Tensor:
        args = [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride
        }]

        with tf.variable_scope(scope, 'block', [x]):
            block_stride = 1
            for i, unit in enumerate(args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[x]):
                    x = ResNet.bottleneck(x, rate=1, **unit)

            x = ResNet.subsample(x, block_stride)
        return x

    @staticmethod
    @slim.add_arg_scope
    def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
                   outputs_collections=None, scope=None):
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = ResNet.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            residual = ResNet.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

    @staticmethod
    def subsample(inputs, factor, scope=None):
        """Subsamples the input along the spatial dimensions.
        Args:
          inputs: A `Tensor` of size [batch, height_in, width_in, channels].
          factor: The subsampling factor.
          scope: Optional variable_scope.
        Returns:
          output: A `Tensor` of size [batch, height_out, width_out, channels] with the
            input, either intact (if factor == 1) or subsampled (if factor > 1).
        """
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

    @staticmethod
    def resnet_arg_scope(weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True,
                         activation_fn=tf.nn.relu,
                         use_batch_norm=True,
                         batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
        """Defines the default ResNet arg scope.
        TODO(gpapan): The batch-normalization related default values above are
        appropriate for use in conjunction with the reference ResNet models
        released at https://github.com/KaimingHe/deep-residual-networks. When
        training ResNets from scratch, they might need to be tuned.
        Args:
        weight_decay: The weight decay to use for regularizing the model.
        batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
        batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
        batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
        activation_fn: The activation function which is used in ResNet.
        use_batch_norm: Whether or not to use batch normalization.
        batch_norm_updates_collections: Collection for the update ops for
        batch norm.
        Returns:
        An `arg_scope` to use for the resnet models.
        """
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': batch_norm_updates_collections,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm if use_batch_norm else None,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc
